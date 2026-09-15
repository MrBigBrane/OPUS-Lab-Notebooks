"""Offline SM80/86/89 compile gate. This never executes a CUDA kernel."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path


def compile_plan(contexts=(8192,32768), budgets=(32,64,128,256,512,1024),
                 dtypes=('float16',)):
    """Signatures mirror runtime.py; CPU-testable without importing Triton."""
    cases=[]
    def add(name, signature, constants):
        case=(name,signature,constants)
        if case not in cases:
            cases.append(case)
    def pointers(names,typ):
        return {n:typ for n in names.split()}
    for n in contexts:
        for k in budgets:
            t=((max(n//4,k+1)+63)//64)*64
            add('compact_plan',{
                **pointers('TopValues Logits LogP','*fp32'),
                **pointers('Lengths Counts Selected Prefix RowCounts Stamps','*i32'),
                'TopIndices':'*i64','epoch':'i32'},
                dict(T=t,G=7,K=k,BK=1<<(k-1).bit_length()))
            add('diagnostics',{
                **pointers('Lengths Counts Stamps Rows','*i32'),
                'LogP':'*fp32','Out':'*fp64','epoch':'i32'},
                dict(T=t,K=k,G=7,BT=1<<(t-1).bit_length(),
                     BSEL=1<<(7*k-1).bit_length(),BG=8))
            for dtype in dtypes:
                dt='*fp16' if dtype=='float16' else '*bf16'
                add('grouped_route',{
                    **pointers('Q Leaders',dt), **pointers('Lengths Counts','*i32'),
                    **pointers('Logits Priorities Uniforms','*fp32'),
                    'seed':'u64','layer':'i32','epoch':'i32'},
                    dict(T=t,D=128,G=7,BD=128,BM=16,BN=64,USE_UNIFORMS=False))
                add('selected_attention',{
                    **pointers('Q PK PV FullK FullV',dt),
                    **pointers('Starts Counts Selected Prefix RowCounts','*i32'),
                    **pointers('LogP PM PL PA','*fp32'),
                    **{x:'i32' for x in ('total_tokens','full_k_head_stride','full_k_token_stride',
                                         'full_v_head_stride','full_v_token_stride')}},
                    dict(N=n,T=t,D=128,G=7,K=k,SPLITS=16,BD=128,BR=32,SEARCH=k.bit_length()))
                add('reduce_partials',{**pointers('PM PL PA','*fp32'),'Out':dt},
                    dict(D=128,BD=128,SPLITS=16,BS=16))
                add('refresh_packed',{
                    **pointers('PK PV FullK FullV',dt),'Inverse':'*i32',
                    **{x:'i32' for x in ('start','end','kh','kt','vh','vt')}},
                    dict(N=n,D=128,BD=128))
    return cases


def compile_portability(output: Path, *, targets=(86,89), contexts=(8192,32768),
                        budgets=(32,64,128,256,512,1024), dtypes=('float16',)):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from . import kernels
    output.mkdir(parents=True,exist_ok=True)
    report={'status':'running','executed_on_gpu':False,'triton':triton.__version__,'cases':[]}
    try:
        for target in targets:
            if target not in (80,86,89):
                raise ValueError('Supported offline targets are 80, 86, 89')
            for index,(name,sig,const) in enumerate(compile_plan(contexts,budgets,dtypes)):
                fn=getattr(kernels,name)
                if set(fn.arg_names) != set(sig)|set(const):
                    raise AssertionError(f'Signature mismatch for {name}')
                kwargs={'signature':sig}
                kwargs['constexprs' if 'constexprs' in inspect.signature(ASTSource).parameters else 'constants']=const
                compiled=triton.compile(ASTSource(fn,**kwargs),target=GPUTarget('cuda',target,32),
                    options={'num_warps':4,'num_stages':2,'num_ctas':1})
                ptx=compiled.asm['ptx']
                shared=int(compiled.metadata.shared)
                if shared>99*1024:
                    raise AssertionError(f'{name}: {shared} shared bytes exceeds conservative SM86 budget')
                filename=f'sm{target}_{index:03d}_{name}.ptx'
                (output/filename).write_text(ptx)
                report['cases'].append({'target':target,'kernel':name,'constants':const,
                    'shared_bytes':shared,'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),
                    'cubin_generated':'cubin' in compiled.asm})
                print(f'COMPILE PASS sm{target} {name} {const}',flush=True)
        report['status']='passed'
        return report
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output/'compile_report.json').write_text(json.dumps(report,indent=2)+'\n')
