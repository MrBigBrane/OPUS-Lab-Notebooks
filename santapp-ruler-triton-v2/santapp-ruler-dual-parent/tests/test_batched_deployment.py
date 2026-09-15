"""CPU release checks. These do not claim that Triton compiled or CUDA ran."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


@pytest.fixture()
def rendered(tmp_path):
    out = tmp_path/'rendered'
    subprocess.run([sys.executable, str(ROOT/'scripts/render_batched_prefill_smoke.py'),
                    '--owner', 'test-r007', '--image', 'example/research:r007', '--output', str(out)],
                   check=True, capture_output=True, text=True)
    return out, {p.name: yaml.safe_load(p.read_text()) for p in out.glob('*.yaml')}


def test_a10_only_parallel_8_and_equal_resources(rendered):
    out, docs = rendered
    assert len(docs) == 9
    assert docs['06-ruler-matrix-job.yaml']['spec']['parallelism'] == 8
    assert docs['06-ruler-matrix-job.yaml']['spec']['completions'] == 24
    assert docs['06-ruler-matrix-job.yaml']['spec']['completionMode'] == 'Indexed'
    for name, doc in docs.items():
        pod = (doc['spec']['template']['spec'] if doc['kind'] == 'Job'
               else doc.get('spec') if doc['kind'] == 'Pod' else None)
        if not pod:
            continue
        gpu = False
        for c in pod.get('containers', []) + pod.get('initContainers', []):
            assert c['resources']['requests'] == c['resources']['limits']
            gpu |= bool(c['resources']['limits'].get('nvidia.com/gpu', 0))
        if gpu:
            terms = pod['affinity']['nodeAffinity']['requiredDuringSchedulingIgnoredDuringExecution']['nodeSelectorTerms']
            for term in terms:
                product = [e for e in term['matchExpressions'] if e['key'] == 'nvidia.com/gpu.product']
                assert len(product) == 1 and product[0]['operator'] == 'In' and product[0]['values'] == ['NVIDIA-A10']
        else:
            assert 'affinity' not in pod
    assert json.loads((out/'RUN_INFO.json').read_text())['fit_batch_size'] == 8
    assert len(json.loads((out/'INDEX_MAP.json').read_text())) == 24


def test_canary_worst_case_and_native_reference(rendered):
    _, docs = rendered
    def env(name):
        c = docs[name]['spec']['template']['spec']['containers'][0]
        return {e['name']: e.get('value') for e in c['env']}
    canary = env('05-ruler-canary-job.yaml')
    assert canary['SMOKE_CONTEXTS'] == '32768'
    assert canary['SMOKE_SAMPLE_BUDGETS'] == '4096'
    assert canary['SMOKE_BACKENDS'] == 'santapp'
    for name in ('05-ruler-canary-job.yaml', '06-ruler-matrix-job.yaml', '08-single-fit-reference-job.yaml'):
        e = env(name)
        assert e['SANTAPP_REQUIRE_KMEANS_GATE'].endswith('/gates/kmeans.json')
        assert e['SANTAPP_REQUIRE_DECODE_GATE'].endswith('/gates/decode.json')
        assert e['SANTAPP_REQUIRE_GPU_PRODUCT'] == 'NVIDIA-A10'
        assert e['SMOKE_DECODE_BACKEND'] == 'grouped_triton'
        assert e['SMOKE_STOP_ON_EOS'] == 'false'
        assert e['SMOKE_MAX_NEW_TOKENS'] == '128'
    assert env('08-single-fit-reference-job.yaml')['SMOKE_KMEANS_FIT_BATCH_SIZE'] == '1'
    assert env('06-ruler-matrix-job.yaml')['SMOKE_KMEANS_FIT_BATCH_SIZE'] == '8'
    copy = docs['07-copy-pod.yaml']['spec']
    assert copy['containers'][0]['resources']['limits']['cpu'] == '1'
    assert copy['activeDeadlineSeconds'] == 7200


def test_compile_and_gpu_both_stages_have_source_gates(rendered):
    _, docs = rendered
    for name, mode in (('02-offline-compile-job.yaml', 'compile'), ('04-gpu-kernel-job.yaml', 'gpu')):
        args = docs[name]['spec']['template']['spec']['containers'][0]['args']
        assert args[:3] == ['/workspace/scripts/smoke_r007.py', '--mode', mode]
        assert args[args.index('--fit-batch-size')+1] == '8'
    source = (ROOT/'scripts/smoke_r007.py').read_text()
    for file in ('compile.json', 'kmeans-compile.json', 'decode.json', 'kmeans.json'):
        assert file in source


def _write_case(root, context, backend, budget, util):
    run = root/'smoke'/f'S{budget}'/'matrix'/str(context)/backend
    (run/'predictions'/backend).mkdir(parents=True)
    def write(name, obj):
        (run/name).write_text(json.dumps(obj)+'\n')
    write('shard.json', {'context_length': context, 'backend': backend, 'sample_budget': budget,
                        'status': 'complete', 'prefill_backend': 'triton', 'decode_backend': 'grouped_triton'})
    write('runtime.json', {'gpu': 'NVIDIA A10'})
    m = {'prompt_tokens': context-256, 'generated_tokens': 128, 'configured_context_length': context,
         'nominal_sample_budget_per_head': budget, 'requested_teams_per_head': budget//4,
         'mean_selected_teams_per_head_call': float(budget//4), 'prefill_reference_replay': False,
         'kmeans_fit_execution': 'triton_batched_independent_fits', 'kmeans_fit_batch_size_used': 8,
         'kmeans_fit_chunk_count': 14, 'kmeans_fit_chunks': [{'fit_count': 8}]*14,
         'example_ruler_score': 100., 'peak_reserved_gib': 21.2}
    write(f'predictions/{backend}/niah_single_1.jsonl', {'pred': 'answer', 'metrics': m})
    write('phase-utilization.json', {'cases': [{'phases': {
        'generation': {'mean_device_gpu_util_pct': util, 'device_utilization_samples': 10},
        'decode': {'mean_device_gpu_util_pct': 98., 'device_utilization_samples': 4}}}]})
    return run


def test_acceptance_distinguishes_algorithm_and_utilization(tmp_path):
    for ctx in (8192, 32768):
        for b in (128, 256, 512, 1024, 2048, 4096):
            for method in ('hierarchical', 'santapp'):
                _write_case(tmp_path, ctx, method, b, 33. if ctx == 8192 and method == 'santapp' else 70.)
    report = module('summarize_r007_results').summarize(tmp_path)
    assert report['software_matrix_pass'] and report['software_pass_cases'] == 24
    assert report['whole_generation_utilization_above_40_cases'] == 18
    assert not report['whole_generation_utilization_all_above_40']
    assert not report['boundary_samples_removed']


def test_missing_and_wrong_batch_fail_summary_without_breaking_recovery(tmp_path):
    run = _write_case(tmp_path, 8192, 'santapp', 128, 99.)
    path = run/'predictions/santapp/niah_single_1.jsonl'
    obj = json.loads(path.read_text()); obj['metrics']['kmeans_fit_batch_size_used'] = 1
    path.write_text(json.dumps(obj)+'\n')
    report = module('summarize_r007_results').summarize(tmp_path)
    assert not report['software_matrix_pass']
    assert any('112 independent fits' in e for e in report['errors'])
    assert len(report['rows']) == 24
    assert report['whole_generation_utilization_known_cases'] == 1


def test_packer_is_uncompressed_by_default_and_does_not_include_cache(tmp_path):
    root = tmp_path/'results'; root.mkdir()
    (root/'ok.json').write_text('{}')
    (root/'model.safetensors').write_text('weights')
    out = tmp_path/'out.zip'
    result = subprocess.run([sys.executable, str(ROOT/'scripts/package_smoke_results.py'),
        '--root', str(root), '--output', str(out)], check=True, capture_output=True, text=True)
    assert 'PACKAGING' in result.stdout
    with zipfile.ZipFile(out) as archive:
        assert set(archive.namelist()) == {'ok.json', 'HANDOFF_MANIFEST.json'}
        assert all(x.compress_type == zipfile.ZIP_STORED for x in archive.infolist())


def test_r006_numerical_kernels_teams_and_decoder_are_unchanged():
    import ast
    import hashlib
    evidence = json.loads((ROOT/'docs/R007_SOURCE_PRESERVATION.json').read_text())
    assert len(evidence['unchanged_files']) == 18
    for row in evidence['unchanged_files']:
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
    santapp_source = (ROOT/'src/santapp_ruler/attention/santapp.py').read_text()
    tree = ast.parse(santapp_source)
    methods = {n.name: ast.get_source_segment(santapp_source, n) for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    # Hash normalized source segments rather than ast.dump(). Python versions
    # differ in whether empty AST fields are emitted by default.
    for name, expected in evidence['santapp_unchanged_method_sources_sha256'].items():
        assert hashlib.sha256(methods[name].encode()).hexdigest() == expected, name
