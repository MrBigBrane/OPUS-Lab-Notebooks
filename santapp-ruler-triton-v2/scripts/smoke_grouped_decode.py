#!/usr/bin/env python3
"""R006 source-bound CPU compile / real GPU correctness gate and small handoff."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from santapp_ruler.runtime_defaults import configure_allocator
configure_allocator()
from santapp_ruler.attention.triton_decode.identity import source_digest


def write(path,data):
    path.write_text(json.dumps(data,indent=2,default=str)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compile-only',action='store_true')
    p.add_argument('--targets',nargs='+',type=int,default=[86,89])
    p.add_argument('--contexts',nargs='+',type=int,default=[8192,32768])
    p.add_argument('--sample-budgets',nargs='+',type=int,default=[128,256,512,1024,2048,4096])
    p.add_argument('--dtypes',nargs='+',choices=['float16','bfloat16'],default=['float16'])
    p.add_argument('--run-tests',action='store_true')
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--gate-file',type=Path)
    p.add_argument('--require-compile-gate',type=Path)
    args=p.parse_args()
    from santapp_ruler.attention.triton_decode.packing import requested_teams
    budgets=[requested_teams(s,16,4) for s in args.sample_budgets]
    if any(n<1 for n in args.contexts):
        p.error('contexts must be positive')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    name=('compile' if args.compile_only else 'gpu')+'-'+stamp
    out=args.output_root/name
    out.mkdir(parents=True,exist_ok=False)
    digest=source_digest()
    report={'status':'running','source_digest':digest,'mode':name.split('-')[0],
            'nominal_sample_budgets':args.sample_budgets,'requested_teams':budgets,
            'contexts':args.contexts,'cases':[]}
    code=1
    try:
        if args.compile_only:
            from santapp_ruler.attention.triton_decode.compile_check import compile_portability
            if args.run_tests:
                subprocess.run([sys.executable,'-m','pytest','-q','-m','not gpu',
                                '--junitxml',str(out/'cpu-tests.xml')],cwd=ROOT,check=True)
            report['compile']=compile_portability(out/'compiler',targets=args.targets,
                contexts=args.contexts,budgets=budgets,dtypes=args.dtypes)
        else:
            if args.require_compile_gate:
                gate=json.loads(args.require_compile_gate.read_text())
                if gate.get('status')!='passed' or gate.get('mode')!='compile' or gate.get('source_digest')!=digest:
                    raise RuntimeError('The offline compile gate has not passed for this source/image')
            if os.environ.get('TRITON_INTERPRET')=='1':
                raise RuntimeError('Compiled real CUDA execution is required, not TRITON_INTERPRET')
            from santapp_ruler.attention.prefill_runtime import require_triton_environment
            report['environment']=require_triton_environment()
            write(out/'environment.json',report['environment'])
            if args.run_tests:
                subprocess.run([sys.executable,'-m','pytest','-q','tests/test_grouped_decode_gpu.py',
                                '--junitxml',str(out/'gpu-tests.xml')],cwd=ROOT,check=True)
            import torch
            from santapp_ruler.attention.triton_decode.validation import validate_case
            for dtype in args.dtypes:
                for context in args.contexts:
                    for budget in budgets:
                        result=validate_case(context,budget,dtype=getattr(torch,dtype),
                                             suffixes=(0,1,17),irregular=False)
                        report['cases'].append(result)
                        write(out/'partial.json',report)
                        print(f'GPU PASS context={context} S={4*budget} teams={budget} dtype={dtype}',flush=True)
        report['status']='passed'
        code=0
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        write(out/'summary.json',report)
        # Never leave a stale success marker after a failed retry.
        if args.gate_file:
            args.gate_file.parent.mkdir(parents=True,exist_ok=True)
            write(args.gate_file,report)
        entries=[]
        for file in sorted(out.rglob('*')):
            if file.is_file():
                entries.append(hashlib.sha256(file.read_bytes()).hexdigest()+'  '+str(file.relative_to(out)))
        (out/'MANIFEST.sha256').write_text('\n'.join(entries)+'\n')
        archive=args.output_root/(name+'.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for file in sorted(out.rglob('*')):
                if file.is_file():z.write(file,str(file.relative_to(out.parent)))
        print(f'GATE_STATUS={report["status"]}',flush=True)
        print(f'UPLOAD_ARCHIVE={archive.resolve()}',flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
