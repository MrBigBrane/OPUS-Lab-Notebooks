#!/usr/bin/env python3
"""R007 offline compiler / real CUDA correctness and paired fit timing gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from santapp_ruler.runtime_defaults import configure_allocator
configure_allocator()
from santapp_ruler.attention.triton_decode.identity import source_digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def normalize_gpu(name):
    return name.lower().replace('-', '').replace(' ', '')


def benchmark_case(tokens, fits, repeats, output):
    import torch
    from santapp_ruler.attention.triton_prefill.batched_kmeans import BatchedTritonMiniBatchKMeans
    from santapp_ruler.attention.triton_prefill.prefill_kmeans import TritonMiniBatchKMeans
    from santapp_ruler.attention.triton_prefill.batched_validation import validate_fitted_batch
    from santapp_ruler.gpu_telemetry import GpuTelemetry, summarize_windows

    k = max(2, tokens // 16)
    x = (torch.randn(fits, tokens, 128, generator=torch.Generator().manual_seed(1307 + tokens))
         * 0.35).half().to(device='cuda', dtype=torch.float32)
    params = dict(n_clusters=k, batch_size=4096, n_init=1, max_iter=100,
                  max_no_improvement=10, random_state=0, dot_precision='ieee')
    # Compile/warm up these same input shapes OUTSIDE timing and telemetry windows.
    warm = {**params, 'max_iter': 1}
    BatchedTritonMiniBatchKMeans(**warm).fit_predict(x)
    TritonMiniBatchKMeans(**warm).fit_predict(x[0])
    torch.cuda.synchronize()
    trials = []
    validated = None
    csv_path = output / f'gpu-utilization-N{tokens}.csv'
    with GpuTelemetry(csv_path):
        for repeat in range(repeats):
            # Alternate order to reduce systematic clock/order bias.
            order = ('batched', 'sequential') if repeat % 2 == 0 else ('sequential', 'batched')
            for mode in order:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start_unix, start = time.time(), time.perf_counter()
                if mode == 'batched':
                    estimator = BatchedTritonMiniBatchKMeans(**params)
                    labels = estimator.fit_predict(x)
                    inertia = estimator.inertia_.tolist()
                    steps = estimator.n_steps_.tolist()
                else:
                    inertia, steps = [], []
                    for f in range(fits):
                        old = TritonMiniBatchKMeans(**params)
                        old.fit_predict(x[f])
                        inertia.append(old.inertia_)
                        steps.append(old.n_steps_)
                    del old
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                end_unix = time.time()
                row = {'mode': mode, 'repeat': repeat, 'seconds': seconds,
                       'started_at_unix': start_unix, 'finished_at_unix': end_unix,
                       'inertia': inertia, 'n_steps': steps,
                       'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                       'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3}
                trials.append(row)
                print(f'FIT TIMING N={tokens} fits={fits} {mode} {seconds:.6f}s', flush=True)
                if mode == 'batched':
                    if validated is None:
                        validated = validate_fitted_batch(x, estimator, labels)
                        row['validation'] = validated
                    row['kernel_launch_counts'] = estimator.launch_counts_
                    row['minibatch_host_control_transfers'] = estimator.minibatch_host_control_transfers_
                    expected = 4 * (k - 1) + 1
                    measured = sum(estimator.launch_counts_.get(name, 0) for name in (
                        'first_center', 'scan_closest', 'sample_candidates',
                        'trial_distances', 'choose_and_commit'))
                    assert measured == expected, (measured, expected)
                    row['initialization_launcher_calls'] = measured
                    del estimator, labels
                write(output / f'partial-N{tokens}.json', trials)
    medians = {mode: statistics.median(r['seconds'] for r in trials if r['mode'] == mode)
               for mode in ('batched', 'sequential')}
    for row in trials:
        row['utilization'] = summarize_windows(csv_path,
            [(row['started_at_unix'], row['finished_at_unix'])])
    result = {'status': 'passed', 'tokens': tokens, 'fits': fits, 'clusters': k,
              'dimension': 128, 'minibatch_tokens': 4096, 'n_init': 1, 'max_iter': 100,
              'trials': trials, 'median_seconds': medians,
              'sequential_over_batched_time': medians['sequential'] / medians['batched'],
              'initialization_calls_per_batch': 1 + 4 * (k-1),
              'sequential_initialization_calls_for_same_fits': fits * (1 + 4 * (k-1)),
              'numeric_fidelity': 'Same independent MiniBatchKMeans mechanics; no exact fitted-label requirement',
              'note': 'No model resident in this microbenchmark. Real 24GB peak memory is tested by model canary/matrix.'}
    del x
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compile-only', action='store_true')
    p.add_argument('--run-tests', action='store_true')
    p.add_argument('--require-a10', action='store_true')
    p.add_argument('--targets', type=int, nargs='+', default=[86])
    p.add_argument('--contexts', type=int, nargs='+', default=[8192, 32768])
    p.add_argument('--fit-batch-size', type=int, default=8)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--gate-file', type=Path, required=True)
    p.add_argument('--require-compile-gate', type=Path)
    args = p.parse_args()
    if args.fit_batch_size < 1 or args.repeats < 1 or any(n < 32 for n in args.contexts):
        p.error('fit-batch-size/repeats must be positive and contexts >= 32')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    mode = 'compile' if args.compile_only else 'gpu'
    output = args.output_root / (mode + '-' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'mode': mode, 'source_digest': source_digest(),
              'contexts': args.contexts, 'fit_batch_size': args.fit_batch_size,
              'dimensions': 128, 'dot_precision': 'ieee', 'cases': []}
    code = 1
    try:
        if args.compile_only:
            if args.run_tests:
                subprocess.run([sys.executable, '-m', 'pytest', '-q', '-m', 'not gpu',
                    '--junitxml', str(output/'cpu-tests.xml')], cwd=ROOT, check=True)
            from santapp_ruler.attention.triton_prefill.batched_compile_check import compile_portability
            report['batched_compile'] = compile_portability(output/'compiler',
                targets=args.targets, contexts=args.contexts)
            # Native team builders are unchanged but still part of this delivery.
            from santapp_ruler.attention.triton_prefill.compile_check import compile_portability as single_compile
            report['existing_prefill_compile'] = single_compile(output/'existing-prefill-compiler',
                targets=[86], contexts=args.contexts)
        else:
            from santapp_ruler.attention.prefill_runtime import require_triton_environment
            environment = require_triton_environment()
            report['environment'] = environment
            write(output/'environment.json', environment)
            if args.require_a10 and normalize_gpu(environment['gpu']) != 'nvidiaa10':
                raise RuntimeError(f"A10-only acceptance run; got {environment['gpu']}")
            if args.require_compile_gate:
                gate = json.loads(args.require_compile_gate.read_text())
                if (gate.get('status') != 'passed' or gate.get('mode') != 'compile'
                        or gate.get('source_digest') != report['source_digest']
                        or not set(args.contexts) <= set(gate.get('contexts', []))):
                    raise RuntimeError('Batched compile gate has not passed for this source/context')
            if args.run_tests:
                subprocess.run([sys.executable, '-m', 'pytest', '-q',
                    'tests/test_batched_kmeans_gpu.py', '--junitxml', str(output/'gpu-tests.xml')],
                    cwd=ROOT, check=True)
            import torch
            old_precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision('highest')
            try:
                for tokens in args.contexts:
                    report['cases'].append(benchmark_case(tokens, args.fit_batch_size,
                                                          args.repeats, output))
                    write(output/'partial.json', report)
            finally:
                torch.set_float32_matmul_precision(old_precision)
        report['status'] = 'passed'
        code = 0
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        write(output/'summary.json', report)
        write(args.gate_file, report)  # Failed retries overwrite stale successes.
        print(f'BATCHED_KMEANS_GATE={report["status"]} REPORT={output}', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
