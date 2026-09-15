"""Low-rate device telemetry. No GPU work is generated to influence utilization."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import threading
import time


class GpuTelemetry:
    def __init__(self,path: Path):
        self.path=path
        self.stop=threading.Event()
        self.thread=None

    def __enter__(self):
        self.thread=threading.Thread(target=self._sample,daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        command=['nvidia-smi','--query-gpu=uuid,utilization.gpu,utilization.memory,memory.used,memory.total',
                 '--format=csv,noheader,nounits']
        self.path.parent.mkdir(parents=True,exist_ok=True)
        try:
            with self.path.open('w',newline='') as handle:
                writer=csv.writer(handle)
                writer.writerow(['unix_time','gpu_uuid','gpu_util_pct','memory_util_pct','memory_used_mib','memory_total_mib'])
                while not self.stop.is_set():
                    sample=subprocess.run(command,text=True,capture_output=True,check=True,timeout=5)
                    now=time.time()
                    for row in csv.reader(sample.stdout.splitlines()):
                        if row:writer.writerow([now]+[s.strip() for s in row])
                    handle.flush()
                    self.stop.wait(1.0)
        except Exception as exc:
            self.path.with_suffix('.error.txt').write_text(f'{type(exc).__name__}: {exc}\n')

    def __exit__(self,*_):
        self.stop.set()
        if self.thread:self.thread.join(timeout=7)


def summarize_decode_telemetry(run_dir: Path) -> dict:
    """Join NVML device samples to actual per-prompt decode windows, not cold setup."""
    path=run_dir/'gpu-utilization.csv'
    samples=[]
    if path.exists():
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                try:
                    samples.append((float(row['unix_time']),row['gpu_uuid'],float(row['gpu_util_pct'])))
                except (ValueError,KeyError):
                    continue
    cases=[]
    for file in sorted((run_dir/'predictions').rglob('*.jsonl')):
        for line in file.read_text().splitlines():
            record=json.loads(line)
            metrics=record.get('metrics',{})
            start=metrics.get('decode_started_at_unix')
            end=metrics.get('decode_finished_at_unix')
            if start is None or end is None:continue
            values=[v for t,_,v in samples if start<=t<=end]
            cases.append({'prediction_file':str(file.relative_to(run_dir)),
                'decode_backend':metrics.get('decode_backend'),
                'nominal_sample_budget':metrics.get('nominal_sample_budget_per_head'),
                'generated_tokens':metrics.get('generated_tokens'),
                'decode_seconds':metrics.get('decode_seconds'),
                'device_utilization_samples_in_decode':len(values),
                'mean_device_gpu_util_pct':sum(values)/len(values) if values else None,
                'min_device_gpu_util_pct':min(values) if values else None,
                'max_device_gpu_util_pct':max(values) if values else None,
                'fraction_samples_at_least_40pct':sum(v>=40 for v in values)/len(values) if values else None})
    result={'cases':cases,'total_device_samples':len(samples),
        'measurement':'NVIDIA device utilization, not SM occupancy or a reproduction of Nautilus warning windows',
        'note':'Decode-only timestamps; no result is inferred when no samples fall in the interval.'}
    (run_dir/'decode-utilization.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def _read_device_samples(path: Path) -> list[tuple[float, str, float]]:
    samples = []
    if path.exists():
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                try:
                    timestamp, value = float(row['unix_time']), float(row['gpu_util_pct'])
                    if 0 <= value <= 100:
                        samples.append((timestamp, row['gpu_uuid'], value))
                except (ValueError, KeyError):
                    continue
    return samples


def summarize_windows(path: Path, windows) -> dict:
    """Timestamp-matched device samples, kept intact; never pool distinct GPUs."""
    windows = sorted((float(a), float(b)) for a, b in windows if b >= a)
    merged = []
    for a, b in windows:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    groups = {}
    for timestamp, gpu, value in _read_device_samples(path):
        if any(a <= timestamp <= b for a, b in merged):
            groups.setdefault(gpu, []).append(value)
    devices = {}
    for gpu, values in groups.items():
        devices[gpu] = {
            'device_utilization_samples': len(values),
            'mean_device_gpu_util_pct': sum(values)/len(values),
            'min_device_gpu_util_pct': min(values),
            'max_device_gpu_util_pct': max(values),
            'fraction_samples_at_least_40pct': sum(v >= 40 for v in values)/len(values),
        }
    primary = next(iter(devices.values())) if len(devices) == 1 else {
        'device_utilization_samples': 0,
        'mean_device_gpu_util_pct': None, 'min_device_gpu_util_pct': None,
        'max_device_gpu_util_pct': None, 'fraction_samples_at_least_40pct': None,
    }
    return {**primary, 'window_count': len(merged),
            'window_seconds': sum(b-a for a, b in merged), 'per_gpu': devices,
            'ambiguous_multiple_devices': len(devices) > 1}


def _logged_windows(path: Path):
    import re
    pending, windows = {}, {}
    if not path.exists():
        return windows
    for line in path.read_text(errors='replace').splitlines():
        match = re.search(r'PHASE_(START|END)=(\w+)\s+unix=([0-9.]+)', line)
        if not match:
            continue
        kind, name, timestamp = match.group(1), match.group(2), float(match.group(3))
        chunk_match = re.search(r'\bchunk=(\d+)', line)
        key = (name, chunk_match.group(1) if chunk_match else None)
        if kind == 'START':
            pending[key] = timestamp
        elif key in pending:
            windows.setdefault(name, []).append((pending.pop(key), timestamp))
    return windows


def summarize_phase_telemetry(run_dir: Path) -> dict:
    """Whole generation plus fitting/team/preparation/decode phase windows.

    Boundaries are not trimmed. This is sampled device utilization, not the
    cluster's policy calculation or SM occupancy. Short phases can have no data.
    """
    csv_path = run_dir/'gpu-utilization.csv'
    logged = _logged_windows(run_dir/'console.log')
    cases = []
    fields = {'generation': ('generation_started_at_unix', 'generation_finished_at_unix'),
              'dense_prefill': ('prefill_started_at_unix', 'prefill_finished_at_unix'),
              'parent_build': ('parent_build_started_at_unix', 'parent_build_finished_at_unix'),
              'decode': ('decode_started_at_unix', 'decode_finished_at_unix')}
    for file in sorted((run_dir/'predictions').rglob('*.jsonl')):
        for line in file.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            metrics = record.get('metrics', {})
            windows = {}
            for phase, (start, end) in fields.items():
                if start in metrics and end in metrics:
                    windows[phase] = [(metrics[start], metrics[end])]
            for phase, field in (('kmeans_fit', 'kmeans_fit_chunks'), ('team_build', 'team_build_chunks')):
                if field in metrics:
                    windows[phase] = [(r['started_at_unix'], r['finished_at_unix'])
                                      for r in metrics[field]]
            generation = windows.get('generation', [])
            if generation:
                start, end = generation[0]
                for phase, phase_windows in logged.items():
                    if phase not in windows:
                        windows[phase] = [(a, b) for a, b in phase_windows if start <= a <= b <= end]
            cases.append({'uid': record.get('uid'), 'backend': metrics.get('backend'),
                          'nominal_sample_budget': metrics.get('nominal_sample_budget_per_head'),
                          'phases': {p: summarize_windows(csv_path, w) for p, w in windows.items()}})
    samples = _read_device_samples(csv_path)
    lifetime = [(min(r[0] for r in samples), max(r[0] for r in samples))] if samples else []
    report = {'cases': cases, 'monitored_process_lifetime': summarize_windows(csv_path, lifetime),
              'measurement': 'NVIDIA sampled device utilization; not SM occupancy or Nautilus warning-window reproduction',
              'boundary_samples_removed': False,
              'note': 'Cold setup and all boundaries are retained. No samples means unknown, not zero utilization.'}
    (run_dir/'phase-utilization.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
