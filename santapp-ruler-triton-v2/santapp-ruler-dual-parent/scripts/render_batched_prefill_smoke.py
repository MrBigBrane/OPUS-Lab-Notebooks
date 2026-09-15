#!/usr/bin/env python3
"""Render the R007 A10-only, batched-prefill smoke. Does not access Kubernetes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]


def env_set(container, **values):
    env = {item['name']: item for item in container.get('env', [])}
    for key, value in values.items():
        env[key] = {'name': key, 'value': str(value)}
    container['env'] = list(env.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--namespace', default='ucsb-opus-lab')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--parallelism', type=int, default=8)
    p.add_argument('--fit-batch-size', type=int, default=8)
    p.add_argument('--contexts', nargs='+', type=int, default=[8192, 32768])
    p.add_argument('--sample-budgets', nargs='+', type=int, default=[128, 256, 512, 1024, 2048, 4096])
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--cpu', default='4')
    p.add_argument('--memory', default='32Gi')
    args = p.parse_args()
    if args.parallelism < 1 or not 1 <= args.fit_batch_size <= 65535:
        p.error('parallelism must be positive; fit-batch-size must be 1..65535')
    if not args.contexts or any(n < 256 for n in args.contexts):
        p.error('contexts must be at least 256')
    if any(s <= 0 or s % 4 or s > 4096 for s in args.sample_budgets):
        p.error('nominal budgets must be positive multiples of four, at most 4096')
    if len(set(args.contexts)) != len(args.contexts) or len(set(args.sample_budgets)) != len(args.sample_budgets):
        p.error('contexts and budgets must not contain duplicates')
    if args.max_new_tokens < 1:
        p.error('max-new-tokens must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('--output must be new or empty')
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp)/'r006'
        subprocess.run([sys.executable, str(ROOT/'scripts/render_grouped_decode_smoke.py'),
            '--owner', args.owner, '--image', args.image, '--namespace', args.namespace,
            '--output', str(out), '--parallelism', str(args.parallelism), '--gpu-product', 'NVIDIA-A10',
            '--contexts', *map(str, args.contexts), '--sample-budgets', *map(str, args.sample_budgets),
            '--tasks', 'niah_single_1', '--max-new-tokens', str(args.max_new_tokens),
            '--cpu', args.cpu, '--memory', args.memory], check=True)
        docs = {f.name: yaml.safe_load(f.read_text()) for f in out.glob('*.yaml')}
        info = json.loads((out/'RUN_INFO.json').read_text())
        index = json.loads((out/'INDEX_MAP.json').read_text())
    prefix, results = info['prefix'], info['results']
    # CPU-only compile/prefetch; every actual GPU Pod has required A10 node affinity.
    common = ['--contexts', *map(str, args.contexts), '--sample-budgets', *map(str, args.sample_budgets),
              '--fit-batch-size', str(args.fit_batch_size), '--results-root', results]
    compiler = docs['02-offline-compile-job.yaml']['spec']['template']['spec']['containers'][0]
    compiler['args'] = ['/workspace/scripts/smoke_r007.py', '--mode', 'compile', *common]
    kernel = docs.pop('04-decode-kernel-job.yaml')
    kernel['metadata']['name'] = prefix+'-kernels'
    kernel['metadata']['labels']['santapp-stage'] = 'kernels'
    kernel['spec']['template']['metadata']['labels']['santapp-stage'] = 'kernels'
    kernel['spec']['template']['spec']['containers'][0]['args'] = [
        '/workspace/scripts/smoke_r007.py', '--mode', 'gpu', *common]
    docs['04-gpu-kernel-job.yaml'] = kernel
    for name in ('05-ruler-canary-job.yaml', '06-ruler-matrix-job.yaml', '08-torch-reference-job.yaml'):
        c = docs[name]['spec']['template']['spec']['containers'][0]
        env_set(c, SMOKE_KMEANS_FIT_BATCH_SIZE=args.fit_batch_size,
                SANTAPP_REQUIRE_KMEANS_GATE=results+'/gates/kmeans.json',
                SANTAPP_REQUIRE_GPU_PRODUCT='NVIDIA-A10')
    canary = docs['05-ruler-canary-job.yaml']['spec']['template']['spec']['containers'][0]
    env_set(canary, SMOKE_CONTEXTS=max(args.contexts), SMOKE_SAMPLE_BUDGETS=max(args.sample_budgets),
            SMOKE_BACKENDS='santapp')
    reference = docs.pop('08-torch-reference-job.yaml')
    env_set(reference['spec']['template']['spec']['containers'][0], SMOKE_BACKENDS='santapp',
            SMOKE_DECODE_BACKEND='grouped_triton', SMOKE_KMEANS_FIT_BATCH_SIZE=1)
    docs['08-single-fit-reference-job.yaml'] = reference
    cp = docs['07-copy-pod.yaml']['spec']
    cp['activeDeadlineSeconds'] = 7200
    cp['containers'][0]['resources'] = {
        'requests': {'cpu': '1', 'memory': '512Mi'}, 'limits': {'cpu': '1', 'memory': '512Mi'}}
    cp['containers'][0]['args'] = [
        "echo 'CPU-only transfer pod: 1 CPU; expires after 7200 seconds'; sleep 7200"]
    # Record the exact fitting batch in the mounted config, not only in env overrides.
    cm = docs['01-configmap.yaml']
    for key, text in cm['data'].items():
        config = yaml.safe_load(text)
        if isinstance(config, dict) and 'santapp' in config:
            config['santapp'].setdefault('kmeans', {})['fit_batch_size'] = args.fit_batch_size
            cm['data'][key] = yaml.safe_dump(config, sort_keys=False)
    info.update(release='R007', package_version='1.5.0', fit_batch_size=args.fit_batch_size,
                gpu_products=['NVIDIA-A10'], canary={'context': max(args.contexts),
                'parent_method': 'santapp', 'nominal_tokens': max(args.sample_budgets)},
                optional_reference='Native batched fitter with batch size 1; grouped decode; not Torch',
                contexts=args.contexts, nominal_sample_budgets=args.sample_budgets)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, doc in docs.items():
        (args.output/name).write_text(yaml.safe_dump(doc, sort_keys=False))
    for name, value in (('RUN_INFO.json', info), ('INDEX_MAP.json', index)):
        (args.output/name).write_text(json.dumps(value, indent=2)+'\n')
    (args.output/'APPLY_ORDER.txt').write_text(
        '00 PVC (wait Bound), 01 ConfigMap.\n'
        '02 CPU compile: decoder + batched/original prefill SM86; 03 CPU HF prefetch. Wait for BOTH.\n'
        '04 A10 GPU correctness + paired original-single vs batch-8 fitting benchmark. Wait Complete.\n'
        '05 A10 full-model K-means 32k/S4096 canary by default. Wait Complete.\n'
        '06 24-case A10 Indexed matrix, parallelism 8 by default. Wait for ALL completions.\n'
        '07 temporary CPU transfer Pod; 08 optional single-fit native control (not Torch).\n'
        'Do not apply the entire directory at once. Stages are not ordered by Kubernetes.\n'
        'Do not delete the PVC during cleanup.\n')
    print('OUTPUT_DIRECTORY='+str(args.output.resolve()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
