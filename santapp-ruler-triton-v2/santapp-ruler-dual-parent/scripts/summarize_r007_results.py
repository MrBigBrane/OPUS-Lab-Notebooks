#!/usr/bin/env python3
"""Write an auditable matrix CSV/acceptance JSON, including failed/absent cases.

A software pass is distinct from utilization >40%. Boundary samples are retained;
one NIAH prompt is a smoke test, not evidence of full-benchmark accuracy parity.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def summarize(root, contexts=(8192, 32768), budgets=(128, 256, 512, 1024, 2048, 4096),
              fit_batch_size=8, generated_tokens=128):
    rows, errors = [], []
    found = {}
    for path in sorted(root.rglob('shard.json')):
        if 'matrix' not in path.relative_to(root).parts:
            continue
        try:
            shard = load_json(path)
            key = (int(shard['context_length']), shard['backend'], int(shard['sample_budget']))
            found.setdefault(key, []).append((path.parent, shard))
        except Exception as exc:
            errors.append(f'{path.relative_to(root)}: {type(exc).__name__}: {exc}')
    for context in contexts:
        for budget in budgets:
            for backend in ('hierarchical', 'santapp'):
                key = (context, backend, budget)
                row = {'context': context, 'backend': backend, 'nominal_samples': budget,
                       'requested_teams': budget//4, 'status': 'missing', 'software_pass': False}
                problems = []
                matches = found.pop(key, [])
                if len(matches) != 1:
                    problems.append(f'Expected exactly one shard; found {len(matches)}')
                else:
                    run, shard = matches[0]
                    row['run_directory'] = str(run.relative_to(root))
                    row['status'] = shard.get('status', 'unknown')
                    try:
                        runtime = load_json(run/'runtime.json')
                        row['gpu'] = runtime.get('gpu')
                        if str(row['gpu']).lower().replace('-', '').replace(' ', '') != 'nvidiaa10':
                            problems.append('Runtime GPU is not verified NVIDIA A10')
                        if row['status'] != 'complete':
                            problems.append('Shard is not complete')
                        if shard.get('prefill_backend') != 'triton' or shard.get('decode_backend') != 'grouped_triton':
                            problems.append('Unexpected prefill/decode backend')
                        predictions = []
                        for path in sorted((run/'predictions').rglob('*.jsonl')):
                            predictions += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                        if len(predictions) != 1:
                            problems.append(f'Expected one smoke prediction, found {len(predictions)}')
                        if predictions:
                            m = predictions[0].get('metrics', {})
                            for field in ('prompt_tokens', 'generated_tokens', 'peak_allocated_gib', 'peak_reserved_gib',
                                          'parent_build_seconds', 'kmeans_fit_seconds', 'team_construction_seconds',
                                          'decode_prepare_seconds', 'decode_seconds', 'total_seconds',
                                          'example_ruler_score', 'mean_selected_teams_per_head_call',
                                          'kmeans_fit_batch_size_used', 'kmeans_fit_chunk_count'):
                                row[field] = m.get(field)
                            if m.get('nominal_sample_budget_per_head') != budget or m.get('requested_teams_per_head') != budget//4:
                                problems.append('Budget/team mapping mismatch')
                            if m.get('generated_tokens') != generated_tokens:
                                problems.append('Unexpected number of generated tokens')
                            if not (0 < m.get('prompt_tokens', 0) <= context):
                                problems.append('Invalid prompt context length')
                            if m.get('configured_context_length') != context:
                                problems.append('Configured context mismatch')
                            if m.get('prefill_reference_replay') is not False:
                                problems.append('Native no-replay flag not verified')
                            mean_teams = m.get('mean_selected_teams_per_head_call')
                            if mean_teams is None or abs(mean_teams - budget//4) > 1e-5:
                                problems.append('Observed selected-team count mismatch')
                            if backend == 'santapp':
                                chunks = m.get('kmeans_fit_chunks', [])
                                sizes = [c.get('fit_count', 0) for c in chunks]
                                if (m.get('kmeans_fit_execution') != 'triton_batched_independent_fits'
                                        or m.get('kmeans_fit_batch_size_used') != fit_batch_size
                                        or not sizes or max(sizes) > fit_batch_size
                                        or sum(sizes) != 112 or len(sizes) != math.ceil(112/fit_batch_size)):
                                    problems.append('Expected 112 independent fits in bounded GPU batches not verified')
                            for field in ('peak_allocated_gib', 'peak_reserved_gib', 'example_ruler_score'):
                                value = row.get(field)
                                if value is not None and not math.isfinite(value):
                                    problems.append(f'Nonfinite {field}')
                            if not predictions[0].get('pred'):
                                problems.append('Empty prediction')
                        # Use existing telemetry; generate from raw files if an early exit missed finalization.
                        phase = load_json(run/'phase-utilization.json')
                        if not phase and (run/'gpu-utilization.csv').exists():
                            from santapp_ruler.gpu_telemetry import summarize_phase_telemetry
                            phase = summarize_phase_telemetry(run)
                        case = (phase.get('cases') or [{}])[0]
                        phases = dict(case.get('phases', {}))
                        phases['process'] = phase.get('monitored_process_lifetime', {})
                        for name in ('generation', 'kmeans_fit', 'parent_build', 'team_build', 'decode_prepare', 'decode', 'process'):
                            stats = phases.get(name, {})
                            value, count = stats.get('mean_device_gpu_util_pct'), stats.get('device_utilization_samples', 0)
                            row[f'{name}_gpu_util_pct'] = value
                            row[f'{name}_util_samples'] = count
                            row[f'{name}_above_40pct'] = value > 40 if value is not None else None
                            row[f'{name}_util_at_least_3_samples'] = count >= 3
                        row['software_pass'] = not problems
                    except Exception as exc:
                        problems.append(f'{type(exc).__name__}: {exc}')
                        row['software_pass'] = False
                row['issues'] = '; '.join(problems)
                if problems:
                    errors.append(f'{context}/{backend}/S{budget}: {row["issues"]}')
                rows.append(row)
    for key in found:
        errors.append(f'Unexpected matrix shard: {key}')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (root/'MATRIX_SUMMARY.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    known = [row for row in rows if row.get('generation_gpu_util_pct') is not None]
    report = {
        'expected_cases': len(rows), 'software_pass_cases': sum(bool(r['software_pass']) for r in rows),
        'software_matrix_pass': not errors, 'errors': errors,
        'whole_generation_utilization_above_40_cases': sum(r['generation_gpu_util_pct'] > 40 for r in known),
        'whole_generation_utilization_known_cases': len(known),
        'whole_generation_utilization_all_above_40': len(known) == len(rows) and all(
            r['generation_gpu_util_pct'] > 40 for r in known),
        'cases_with_fewer_than_3_generation_samples': sum(r.get('generation_util_samples', 0) < 3 for r in rows),
        'gates': {name: load_json(root/'gates'/name).get('status', 'missing') for name in
                  ('compile.json', 'kmeans-compile.json', 'decode.json', 'kmeans.json')},
        'boundary_samples_removed': False,
        'qualification': ('Sampled device utilization is not SM occupancy or the Nautilus policy window. '
                          'Software pass and utilization pass are separate. One NIAH prompt per case '
                          'does not establish full-RULER or identical-RNG accuracy equivalence.'),
        'rows': rows,
    }
    (root/'MATRIX_ACCEPTANCE.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--contexts', type=int, nargs='+', default=[8192, 32768])
    p.add_argument('--sample-budgets', type=int, nargs='+', default=[128, 256, 512, 1024, 2048, 4096])
    p.add_argument('--fit-batch-size', type=int, default=8)
    p.add_argument('--generated-tokens', type=int, default=128)
    p.add_argument('--require-complete', action='store_true')
    args = p.parse_args()
    if not args.root.is_dir():
        p.error('Results root does not exist')
    report = summarize(args.root, args.contexts, args.sample_budgets, args.fit_batch_size, args.generated_tokens)
    print(f'SOFTWARE_MATRIX_PASS={report["software_matrix_pass"]} '
          f'CASES={report["software_pass_cases"]}/{report["expected_cases"]}', flush=True)
    print(f'GENERATION_UTIL_ABOVE_40={report["whole_generation_utilization_above_40_cases"]}/'
          f'{report["expected_cases"]} KNOWN={report["whole_generation_utilization_known_cases"]}', flush=True)
    print(f'SUMMARY={args.root/"MATRIX_SUMMARY.csv"}', flush=True)
    return 1 if args.require_complete and not report['software_matrix_pass'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
