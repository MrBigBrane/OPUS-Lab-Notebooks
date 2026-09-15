from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

from santapp_ruler.gpu_telemetry import summarize_decode_telemetry

ROOT=Path(__file__).resolve().parents[1]


def test_docker_copies_cpu_hygiene_input():
    assert 'CITATION.cff .gitignore ./' in (ROOT/'Dockerfile').read_text()


def test_telemetry_uses_only_decode_windows_and_does_not_invent_samples(tmp_path):
    (tmp_path/'predictions').mkdir()
    records=[{'metrics':{'decode_started_at_unix':2.,'decode_finished_at_unix':4.,
                         'decode_backend':'grouped_triton'}},
             {'metrics':{'decode_started_at_unix':9.,'decode_finished_at_unix':10.}}]
    (tmp_path/'predictions/a.jsonl').write_text('\n'.join(map(json.dumps,records))+'\n')
    (tmp_path/'gpu-utilization.csv').write_text(
        'unix_time,gpu_uuid,gpu_util_pct\n1,x,0\n2,x,30\n3,x,90\n5,x,0\n')
    result=summarize_decode_telemetry(tmp_path)
    assert result['cases'][0]['mean_device_gpu_util_pct']==60
    assert result['cases'][0]['fraction_samples_at_least_40pct']==0.5
    assert result['cases'][1]['mean_device_gpu_util_pct'] is None
    assert result['cases'][1]['device_utilization_samples_in_decode']==0


def test_handoff_excludes_weights_caches_and_nested_archives(tmp_path):
    root=tmp_path/'results';root.mkdir()
    (root/'summary.json').write_text('{"status":"failed"}\n')
    (root/'console.log').write_text('error details\n')
    (root/'model.safetensors').write_bytes(b'excluded')
    (root/'old.zip').write_bytes(b'excluded')
    (root/'cache').mkdir();(root/'cache/a.json').write_text('{}')
    out=tmp_path/'handoff.zip'
    subprocess.run([sys.executable,str(ROOT/'scripts/package_smoke_results.py'),
                    '--root',str(root),'--output',str(out)],check=True)
    with zipfile.ZipFile(out) as z:
        assert set(z.namelist())=={'summary.json','console.log','HANDOFF_MANIFEST.json'}
        assert len(json.loads(z.read('HANDOFF_MANIFEST.json')))==2
