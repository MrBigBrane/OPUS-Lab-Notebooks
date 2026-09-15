#!/usr/bin/env python3
"""Run both source-bound R007 gates in one CPU or GPU Job, preserving console logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['compile', 'gpu'], required=True)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--contexts', type=int, nargs='+', default=[8192, 32768])
    p.add_argument('--sample-budgets', type=int, nargs='+', default=[128, 256, 512, 1024, 2048, 4096])
    p.add_argument('--fit-batch-size', type=int, default=8)
    args = p.parse_args()
    root = args.results_root
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == 'gpu':
        from santapp_ruler.attention.prefill_runtime import require_triton_environment
        env = require_triton_environment()
        if env['gpu'].lower().replace('-', '').replace(' ', '') != 'nvidiaa10':
            raise RuntimeError(f"R007 smoke requires NVIDIA A10, got {env['gpu']}")
    decoder = [sys.executable, str(ROOT/'scripts/smoke_grouped_decode.py'), '--run-tests',
        '--contexts', *map(str, args.contexts), '--sample-budgets', *map(str, args.sample_budgets),
        '--output-root', str(root/('decode-compile' if args.mode == 'compile' else 'decode-kernels'))]
    kmeans = [sys.executable, str(ROOT/'scripts/smoke_batched_kmeans.py'),
        '--contexts', *map(str, args.contexts), '--fit-batch-size', str(args.fit_batch_size),
        '--output-root', str(root/'batched-kmeans')]
    if args.mode == 'compile':
        decoder += ['--compile-only', '--targets', '86', '--gate-file', str(root/'gates/compile.json')]
        kmeans += ['--compile-only', '--targets', '86', '--gate-file', str(root/'gates/kmeans-compile.json')]
    else:
        decoder += ['--require-compile-gate', str(root/'gates/compile.json'),
                    '--gate-file', str(root/'gates/decode.json')]
        kmeans += ['--run-tests', '--require-a10', '--require-compile-gate', str(root/'gates/kmeans-compile.json'),
                   '--gate-file', str(root/'gates/kmeans.json')]
    # On a rerun, invalidate BOTH previous gates before executing either stage.
    for command in (decoder, kmeans):
        gate = Path(command[command.index('--gate-file')+1])
        gate.parent.mkdir(parents=True, exist_ok=True)
        gate.write_text(json.dumps({'status': 'pending', 'mode': args.mode}) + '\n')
    with (root/f'{args.mode}-gate-console.log').open('a') as log:
        for command in (decoder, kmeans):
            print('RUN ' + ' '.join(command), flush=True)
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
            if status:
                return status
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
