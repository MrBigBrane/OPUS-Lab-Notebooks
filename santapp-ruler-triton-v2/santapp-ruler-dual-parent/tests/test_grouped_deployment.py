from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]


def test_staged_render_is_policy_bounded_and_covers_entire_requested_matrix(tmp_path):
    out=tmp_path/'manifests'
    subprocess.run([sys.executable,str(ROOT/'scripts/render_grouped_decode_smoke.py'),
        '--owner','test-r006','--image','example/santa:r006','--output',str(out)],check=True)
    docs={f.name:yaml.safe_load(f.read_text()) for f in out.glob('*.yaml')}
    assert len(docs)==9 and not (out/'all.yaml').exists()
    index=json.loads((out/'INDEX_MAP.json').read_text())
    assert len(index)==24
    assert {r['requested_teams'] for r in index}=={32,64,128,256,512,1024}
    assert {r['parent_method'] for r in index}=={'hierarchical','santapp'}
    assert {r['context'] for r in index}=={8192,32768}
    from test_release import _check_resources
    _check_resources(docs.values())
    for key in ('02-offline-compile-job.yaml','03-hf-prefetch-job.yaml'):
        pod=docs[key]['spec']['template']['spec']
        assert 'affinity' not in pod
        assert all('nvidia.com/gpu' not in c['resources']['requests'] for c in pod['containers'])
    for key in ('04-decode-kernel-job.yaml','05-ruler-canary-job.yaml','06-ruler-matrix-job.yaml'):
        assert docs[key]['spec']['template']['spec']['containers'][0]['resources']['requests']['nvidia.com/gpu']==1
    job=docs['06-ruler-matrix-job.yaml']
    assert job['spec']['parallelism']==1 and job['spec']['completions']==24
    from test_k8s import _env_values
    env=_env_values(job)
    assert env['SMOKE_DECODE_BACKEND']=='grouped_triton'
    assert env['SMOKE_SAMPLE_BUDGETS']=='128,256,512,1024,2048,4096'
    assert env['HF_HUB_OFFLINE']==env['HF_DATASETS_OFFLINE']=='1'
    assert env['SMOKE_STOP_ON_EOS']=='false'
    assert env['SANTAPP_REQUIRE_DECODE_GATE'].endswith('/gates/decode.json')
    for doc in docs.values():
        if doc['kind'] not in ('Job','Pod'):continue
        pod=doc['spec']['template']['spec'] if doc['kind']=='Job' else doc['spec']
        volumes={v['name'] for v in pod['volumes']}
        for container in pod.get('initContainers',[])+pod['containers']:
            assert {v['name'] for v in container.get('volumeMounts',[])}<=volumes


def test_index_uses_grouped_backend_and_separates_result_paths(tmp_path,monkeypatch):
    path=ROOT/'scripts/k8s_run_index.py'
    spec=importlib.util.spec_from_file_location('r006_index',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for key in list(os.environ):
        if key.startswith(('SMOKE_','SANTAPP_')):monkeypatch.delenv(key,raising=False)
    for key,value in {'RESULTS_ROOT':str(tmp_path),'OWNER_SLUG':'test',
        'SMOKE_CONTEXTS':'8192,32768','SMOKE_BACKENDS':'hierarchical,santapp',
        'SMOKE_SAMPLE_BUDGETS':'128,256,512,1024,2048,4096',
        'SMOKE_DECODE_BACKEND':'grouped_triton','SMOKE_STOP_ON_EOS':'false',
        'SMOKE_RUN_PROFILE':'matrix','SMOKE_MAX_NEW_TOKENS':'128'}.items():monkeypatch.setenv(key,value)
    rows=[]
    monkeypatch.setattr(module,'run_benchmark',lambda c,explicit_run_dir:rows.append((c,explicit_run_dir)))
    monkeypatch.setattr(sys,'argv',[str(path),'--index','23','--config',str(ROOT/'configs/smoke.yaml')])
    assert module.main()==0
    c,run=rows[0]
    assert c.generation.backends==['santapp'] and c.benchmark.context_length==32768
    assert c.santapp.decode_backend=='grouped_triton' and c.santapp.samples_per_head==4096
    assert not c.generation.stop_on_eos
    assert 'decode-grouped_triton/matrix/32768/santapp' in str(run)
    assert json.loads((run/'shard.json').read_text())['status']=='complete'


def test_stale_gate_prevents_model_loading(tmp_path,monkeypatch):
    path=ROOT/'scripts/k8s_run_index.py'
    spec=importlib.util.spec_from_file_location('r006_index_gate',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    gate=tmp_path/'gate.json';gate.write_text(json.dumps({'status':'passed','mode':'gpu','source_digest':'obsolete'}))
    monkeypatch.setenv('SMOKE_DECODE_BACKEND','grouped_triton')
    monkeypatch.setenv('SANTAPP_REQUIRE_DECODE_GATE',str(gate))
    monkeypatch.setattr(sys,'argv',[str(path),'--config',str(ROOT/'configs/smoke.yaml')])
    monkeypatch.setattr(module,'run_benchmark',lambda *a,**k:pytest.fail('model must not load'))
    with pytest.raises(RuntimeError,match='has not passed'):
        module.main()
