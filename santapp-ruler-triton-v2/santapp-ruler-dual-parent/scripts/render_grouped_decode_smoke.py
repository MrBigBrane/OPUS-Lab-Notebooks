#!/usr/bin/env python3
"""Render an ordered R006 decode smoke deployment; no kubectl/API side effects."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

ROOT=Path(__file__).resolve().parents[1]


def env_set(container, **values):
    env={item['name']:item for item in container.get('env',[])}
    for key,value in values.items():env[key]={'name':key,'value':str(value)}
    container['env']=list(env.values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--namespace',default='ucsb-opus-lab')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--parallelism',type=int,default=1)
    p.add_argument('--contexts',nargs='+',type=int,default=[8192,32768])
    p.add_argument('--sample-budgets',nargs='+',type=int,default=[128,256,512,1024,2048,4096])
    p.add_argument('--tasks',nargs='+',default=['niah_single_1'])
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--gpu-product',action='append')
    p.add_argument('--cpu',default='4')
    p.add_argument('--memory',default='32Gi')
    args=p.parse_args()
    if any(b<=0 or b%4 or b>4096 for b in args.sample_budgets):
        p.error('Nominal sample budgets must be positive multiples of four through 4096')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('--output must be new or empty')
    products=args.gpu_product or ['NVIDIA-A10','NVIDIA-RTX-A5000','NVIDIA-GeForce-RTX-3090','NVIDIA-GeForce-RTX-4090']
    with tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp)/'rendered'
        command=[sys.executable,str(ROOT/'scripts/render_nautilus_manifests.py'),
            '--owner',args.owner,'--image',args.image,'--namespace',args.namespace,'--output',str(tmp),
            '--staged-smoke','--parallelism',str(args.parallelism),'--prefill-backend','triton',
            '--contexts',*map(str,args.contexts),'--backends','hierarchical','santapp',
            '--samples-per-head',*map(str,args.sample_budgets),'--tasks',*args.tasks,
            '--max-new-tokens',str(args.max_new_tokens),'--cpu',args.cpu,'--memory',args.memory]
        for product in products:command.extend(['--gpu-product',product])
        subprocess.run(command,check=True)
        docs={f.name:yaml.safe_load(f.read_text()) for f in tmp.glob('*.yaml')}
    pvc=docs['00-pvc.yaml'];config=docs['01-configmap.yaml']
    prefetch=docs['03-hf-prefetch-job.yaml']
    base=docs['04-ruler-smoke-job.yaml']
    copy_pod=docs['05-copy-pod.yaml']
    copy_pod['spec']['activeDeadlineSeconds']=1800
    copy_pod['spec']['containers'][0]['args']=[
        "echo 'Temporary CPU-only result transfer pod; expires after 1800 seconds'; sleep 1800"]
    owner=base['metadata']['labels']['benchmark-owner']
    prefix=f'{owner}-santa-ruler'
    results=f'/shared/results/{owner}'
    compile_gate=f'{results}/gates/compile.json'
    gpu_gate=f'{results}/gates/decode.json'
    container=base['spec']['template']['spec']['containers'][0]
    env_set(container,SMOKE_DECODE_BACKEND='grouped_triton',SMOKE_RUN_PROFILE='matrix',
            SMOKE_STOP_ON_EOS='false',SANTAPP_GPU_TELEMETRY='1',SANTAPP_PHASE_LOGS='1',
            SANTAPP_REQUIRE_DECODE_GATE=gpu_gate,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    base['metadata']['name']=prefix+'-matrix'
    base['spec']['backoffLimit']=0
    # Each workload has its own labels so Lens can distinguish the stage.
    def label(job,stage):
        job['metadata'].setdefault('labels',{})['santapp-stage']=stage
        job['spec']['template']['metadata'].setdefault('labels',{})['santapp-stage']=stage
    label(base,'matrix')
    def simple_job(name,stage):
        j=copy.deepcopy(base)
        j['metadata']['name']=prefix+'-'+name
        j['spec']={'backoffLimit':0,'template':j['spec']['template']}
        label(j,stage)
        return j
    compiler=simple_job('compile','compile')
    compiler_pod=compiler['spec']['template']['spec'];compiler_pod.pop('affinity',None)
    cc=compiler_pod['containers'][0];cc['name']='compile'
    cc['resources']={'requests':{'cpu':'4','memory':'16Gi'},'limits':{'cpu':'4','memory':'16Gi'}}
    cc['env']=[e for e in cc['env'] if e['name'] in {'TRITON_CACHE_DIR','CUDA_CACHE_PATH','TMPDIR','PYTORCH_CUDA_ALLOC_CONF'}]
    cc['args']=['/workspace/scripts/smoke_grouped_decode.py','--compile-only','--run-tests',
                '--targets','86','89','--contexts',*map(str,args.contexts),
                '--sample-budgets',*map(str,args.sample_budgets),
                '--output-root',results+'/decode-compile','--gate-file',compile_gate]
    kernel=simple_job('decode-kernels','decode-kernels')
    kc=kernel['spec']['template']['spec']['containers'][0];kc['name']='kernels'
    kc['resources']={'requests':{'cpu':'4','memory':'16Gi','nvidia.com/gpu':1},
                     'limits':{'cpu':'4','memory':'16Gi','nvidia.com/gpu':1}}
    kc['env']=copy.deepcopy(cc['env'])
    kc['args']=['/workspace/scripts/smoke_grouped_decode.py','--run-tests',
                '--contexts',*map(str,args.contexts),'--sample-budgets',*map(str,args.sample_budgets),
                '--output-root',results+'/decode-kernels','--gate-file',gpu_gate,
                '--require-compile-gate',compile_gate]
    label(prefetch,'prefetch')
    canary=simple_job('canary','canary')
    canary_container=canary['spec']['template']['spec']['containers'][0]
    env_set(canary_container,SMOKE_CONTEXTS=str(args.contexts[0]),SMOKE_SAMPLE_BUDGETS=str(max(args.sample_budgets)),
            SMOKE_BACKENDS='hierarchical',SMOKE_RUN_PROFILE='canary')
    reference=copy.deepcopy(base)
    reference['metadata']['name']=prefix+'-reference'
    reference['spec']['completions']=2 if len(args.sample_budgets)>1 else 1
    reference['spec']['parallelism']=1
    label(reference,'reference')
    env_set(reference['spec']['template']['spec']['containers'][0],SMOKE_CONTEXTS=str(args.contexts[0]),
            SMOKE_SAMPLE_BUDGETS=','.join(map(str,sorted({min(args.sample_budgets),max(args.sample_budgets)}))),
            SMOKE_BACKENDS='hierarchical',SMOKE_DECODE_BACKEND='torch',SMOKE_RUN_PROFILE='reference')
    output=[('00-pvc.yaml',pvc),('01-configmap.yaml',config),('02-offline-compile-job.yaml',compiler),
            ('03-hf-prefetch-job.yaml',prefetch),('04-decode-kernel-job.yaml',kernel),
            ('05-ruler-canary-job.yaml',canary),('06-ruler-matrix-job.yaml',base),
            ('07-copy-pod.yaml',copy_pod),('08-torch-reference-job.yaml',reference)]
    args.output.mkdir(parents=True,exist_ok=True)
    for name,doc in output:(args.output/name).write_text(yaml.safe_dump(doc,sort_keys=False))
    index=[]
    for context in args.contexts:
        for budget in args.sample_budgets:
            for policy in ('hierarchical','santapp'):
                index.append({'index':len(index),'context':context,'parent_method':policy,
                              'nominal_tokens':budget,'requested_teams':budget//4})
    (args.output/'INDEX_MAP.json').write_text(json.dumps(index,indent=2)+'\n')
    (args.output/'RUN_INFO.json').write_text(json.dumps({
        'owner':owner,'namespace':args.namespace,'image':args.image,'prefix':prefix,
        'results':results,'completions':len(index),'parallelism':args.parallelism,
        'gpu_products':products,'max_new_tokens':args.max_new_tokens,'stop_on_eos':False,
        'tasks':args.tasks,'model':'Qwen/Qwen2.5-7B-Instruct','dtype':'float16',
        'purpose':'Software/utilization smoke; not an official RULER accuracy evaluation'},indent=2)+'\n')
    (args.output/'APPLY_ORDER.txt').write_text(
        '00 PVC (wait Bound), 01 ConfigMap.\n'
        '02 CPU compile and 03 CPU HF prefetch; wait for BOTH Complete.\n'
        '04 real GPU kernel correctness; wait Complete.\n'
        '05 one 8k/max-budget contiguous model canary; wait Complete.\n'
        '06 full indexed matrix; verify every completion.\n'
        '07 optional CPU copy pod; 08 optional matched Torch reference.\n'
        'Do not apply this entire directory at once. Independent Jobs are not ordered by Kubernetes.\n'
        'Do not delete the PVC when cleaning up Jobs or the copy Pod.\n')
    print('OUTPUT_DIRECTORY='+str(args.output.resolve()))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
