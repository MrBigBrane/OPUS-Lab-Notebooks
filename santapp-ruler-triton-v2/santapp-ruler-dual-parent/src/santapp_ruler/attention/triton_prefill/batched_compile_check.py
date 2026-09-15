"""Offline SM86 compile gate for every batched numerical entry point."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import re

from .compile_check import CompileCase, compile_plan as single_plan


def compile_plan(contexts=(8192, 32768), dimensions=128):
    cases = []
    for c in single_plan(contexts, dimensions):
        if c.module != 'kmeans':
            continue
        total = int(c.tag.split('_')[0][1:])
        clusters = min(total, max(2, total // 16))
        constants, sig = dict(c.constants), dict(c.signature)
        name = c.kernel
        if name == 'gather_rows':
            constants['N'] = total
        elif name == 'first_center':
            constants['K'] = clusters
            del sig['first_id']
            sig['FirstIds'] = '*i64'
        elif name == 'scan_closest':
            constants['NB'] = (constants['N'] + constants['B'] - 1) // constants['B']
        elif name in {'sample_candidates', 'choose_and_commit'}:
            constants['K'] = clusters
        elif name == 'trial_distances':
            constants['NB'] = (constants['N'] + constants['BN'] - 1) // constants['BN']
        elif name in {'assignment_partials', 'finish_assignment', 'sum_scalar', 'accumulate_batch'}:
            sig['Active'] = '*i1'
            if name == 'finish_assignment':
                constants['NP'] = (constants['N'] + constants['BM'] - 1) // constants['BM']
            if name == 'accumulate_batch':
                constants['K'] = clusters
        elif name == 'apply_reassignment':
            constants['M'] = min(total, 4096)
            constants['ROW_STRIDE'] = max(1, min(clusters, constants['M'] // 2))
        cases.append(CompileCase('batched_kmeans', name, sig, constants, c.tag))
        if name == 'gather_rows':
            init = min(total, max(3 * min(total, 4096), 3 * clusters))
            if init != constants['M']:
                cases.append(CompileCase('batched_kmeans', name, dict(sig),
                                         {**constants, 'M': init}, c.tag + '_init'))
    return cases


def compile_portability(output: Path, targets=(86,), contexts=(8192, 32768), dimensions=128):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from .kernels import p007_batched_kmeans

    output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'execution': False, 'triton': triton.__version__,
              'targets': list(targets), 'cases': []}
    try:
        for target in targets:
            if target not in (80, 86, 89):
                raise ValueError('Supported offline targets: 80, 86, 89')
            for case in compile_plan(contexts, dimensions):
                fn = getattr(p007_batched_kmeans, case.kernel)
                assert set(fn.arg_names) == set(case.signature) | set(case.constants)
                kwargs = {'signature': case.signature}
                arg = 'constexprs' if 'constexprs' in inspect.signature(ASTSource).parameters else 'constants'
                kwargs[arg] = case.constants
                compiled = triton.compile(
                    ASTSource(fn, **kwargs), target=GPUTarget('cuda', target, 32),
                    options={'num_warps': 4, 'num_ctas': 1, 'enable_fp_fusion': False})
                ptx = compiled.asm['ptx']
                assert re.search(rf'\.target\s+sm_{target}\b', ptx)
                assert not re.search(r'wgmma\.|tcgen05\.|cp\.async\.bulk\.tensor', ptx)
                assert int(compiled.metadata.shared) <= 99 * 1024
                report['cases'].append({'kernel': case.kernel, 'case': case.tag,
                    'target': target, 'shared_bytes': int(compiled.metadata.shared),
                    'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest()})
                (output / f'sm{target}_{case.kernel}_{case.tag}.ptx').write_text(ptx)
                print(f'BATCHED COMPILE PASS sm{target} {case.kernel} {case.tag}', flush=True)
        report['status'] = 'passed'
        return report
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output / 'compile_report.json').write_text(json.dumps(report, indent=2) + '\n')
