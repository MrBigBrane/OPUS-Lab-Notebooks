from __future__ import annotations

import ast
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from santapp_ruler.config import load_config
from santapp_ruler.attention.triton_decode.packing import (
    pack_teams, refresh_prompt_rows_reference, requested_teams,
)
from santapp_ruler.attention.triton_decode.validation import fixture, attention_oracle
from santapp_ruler.attention.traffic import DecodeTrafficTracker

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('budget', [128, 256, 512, 1024, 2048, 4096])
def test_budget_is_nominal_not_team_count(budget):
    assert requested_teams(budget, 16, 4) == budget//4
    c = load_config(ROOT/'configs/default.yaml', overrides=[
        f'santapp.samples_per_head={budget}', f'hierarchical.samples_per_head={budget}'])
    assert c.santapp.decode_backend == c.hierarchical.decode_backend == 'grouped_triton'
    assert c.santapp.samples_per_head == budget


@pytest.mark.parametrize('budget', [0, -1, 3, 129, 4100])
def test_invalid_or_unsupported_budget_fails(budget):
    with pytest.raises(ValueError):
        requested_teams(budget, 16, 4)


def test_legacy_missing_decoder_field_remains_torch(tmp_path):
    path=tmp_path/'old.yaml'
    path.write_text('santapp:\n  prefill_backend: triton\nhierarchical:\n  prefill_backend: triton\n')
    c=load_config(path)
    assert c.santapp.decode_backend==c.hierarchical.decode_backend=='torch'
    from santapp_ruler.run_state import prepare_run_directory
    prepare_run_directory(c, tmp_path/'run')
    with pytest.raises(ValueError, match='different resolved'):
        prepare_run_directory(load_config(ROOT/'configs/default.yaml'), tmp_path/'run')


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('tokens,budget', [(1,1),(17,32),(513,128),(8192,1024)])
def test_packing_inverse_padding_and_frozen_leaders(tokens, budget, dtype):
    summaries,k,v=fixture(tokens,dim=32,kv_heads=2,dtype=dtype)
    p=pack_teams(summaries,k,v,budget)
    assert p.max_teams>=budget+1 and p.max_teams%64==0
    for h,s in enumerate(summaries):
        torch.testing.assert_close(p.key[h], k[h,s.members],rtol=0,atol=0)
        torch.testing.assert_close(p.value[h], v[h,s.members],rtol=0,atol=0)
        torch.testing.assert_close(p.inverse[h,s.members].long(),torch.arange(tokens),rtol=0,atol=0)
        torch.testing.assert_close(p.leaders[h,:s.num_teams].float(),s.leader_keys,rtol=0,atol=0)
        assert not p.lengths[h,s.num_teams:].any()
    frozen=p.leaders.clone()
    k[:,-1].add_(0.5);v[:,-1].sub_(0.5)
    refresh_prompt_rows_reference(p,k,v,tokens-1,tokens)
    for h in range(2):
        torch.testing.assert_close(p.key[h,p.inverse[h,-1]],k[h,-1],rtol=0,atol=0)
        torch.testing.assert_close(p.value[h,p.inverse[h,-1]],v[h,-1],rtol=0,atol=0)
    torch.testing.assert_close(p.leaders,frozen,rtol=0,atol=0)


def test_inexact_leader_narrowing_is_rejected():
    summaries,k,v=fixture(31,dim=32,kv_heads=1)
    summaries[0].leader_keys[0,0]=0.123456789
    with pytest.raises(ValueError, match='losslessly'):
        pack_teams(summaries,k,v,32)


def test_malformed_partition_is_rejected():
    summaries,k,v=fixture(31,dim=32,kv_heads=1)
    summaries[0].members[0]=summaries[0].members[1]
    with pytest.raises(ValueError, match='partition'):
        pack_teams(summaries,k,v,32)


@pytest.mark.parametrize('suffix',[0,1,17])
def test_all_selected_cpu_oracle_is_dense_attention(suffix):
    s,k,v=fixture(37,dim=32,kv_heads=1,suffix=suffix)
    ids=torch.arange(s[0].num_teams)[None,:].repeat(2,1)
    lp=torch.zeros_like(ids,dtype=torch.float32)
    q=torch.randn(2,32).half()
    result=attention_oracle(q,k,v,s,ids,lp,37)
    dense=torch.softmax(q.float()@k[0].float().T/(32**0.5),dim=-1)@v[0].float()
    torch.testing.assert_close(result,dense,rtol=1e-5,atol=1e-6)


def test_aggregated_traffic_exactly_matches_reference_schema():
    old=DecodeTrafficTracker(2,'sampled_kv')
    new=DecodeTrafficTracker(2,'sampled_kv')
    for step in range(3):
        old.record_group(n_total=10+step, exact_tokens=step,
                         sampled_indices_by_head=[torch.tensor([0,1,4]),torch.tensor([1,4,5,9])],
                         routing_key_vectors=4)
        new.record_aggregated_group(n_total=10+step,exact_tokens=step,
                                    sampled_token_rows=7,unique_sampled_rows=5,routing_key_vectors=4)
    assert old.as_dict()==new.as_dict()
    with pytest.raises(ValueError,match='mix'):
        new.record_group(n_total=10,exact_tokens=0,
                         sampled_indices_by_head=[torch.tensor([1]),torch.tensor([2])])


def test_decode_runtime_contains_no_host_value_reads_or_per_head_loops():
    path=ROOT/'src/santapp_ruler/attention/triton_decode/runtime.py'
    tree=ast.parse(path.read_text())
    hot={'run','route','select','attend','record'}
    for fn in ast.walk(tree):
        if isinstance(fn,ast.FunctionDef) and fn.name in hot:
            assert not any(isinstance(n,(ast.For,ast.While)) for n in ast.walk(fn)),fn.name
            for n in ast.walk(fn):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
                    assert n.func.attr not in {'item','cpu','numpy','tolist','synchronize'},fn.name


def test_cli_prefill_and_decode_are_independent():
    from santapp_ruler.cli import build_parser,_config_with_cli
    cfg=_config_with_cli(build_parser().parse_args(['run','--config',str(ROOT/'configs/default.yaml'),
                                                  '--prefill-backend','torch','--decode-backend','grouped_triton']))
    assert cfg.santapp.prefill_backend=='torch' and cfg.santapp.decode_backend=='grouped_triton'
    ref=_config_with_cli(build_parser().parse_args(['run','--config',str(ROOT/'configs/default.yaml'),
                                                  '--prefill-backend','torch','--decode-backend','torch']))
    assert asdict(ref)==asdict(load_config(ROOT/'configs/torch.yaml'))


def test_offline_compile_signatures_match_kernel_source_without_importing_triton():
    from santapp_ruler.attention.triton_decode.compile_check import compile_plan
    tree=ast.parse((ROOT/'src/santapp_ruler/attention/triton_decode/kernels.py').read_text())
    functions={node.name:{a.arg for a in node.args.args} for node in tree.body
               if isinstance(node,ast.FunctionDef)}
    cases=compile_plan()
    assert {c[2].get('K') for c in cases if 'K' in c[2]}=={32,64,128,256,512,1024}
    for name,sig,constants in cases:
        assert set(sig).isdisjoint(constants)
        assert functions[name]==set(sig)|set(constants),name


def test_grouped_qwen_dispatch_refresh_and_suffix_integration(monkeypatch):
    """CPU integration oracle for the engine hooks, not a CUDA kernel test."""
    from types import SimpleNamespace
    from test_santapp import _TinyQwen, _test_sdpa_forward
    from santapp_ruler.attention import qwen as qwen_module
    from santapp_ruler.attention.hierarchical import HierarchicalWholeTeamEngine
    from santapp_ruler.config import HierarchicalConfig
    from santapp_ruler.attention.triton_decode import runtime
    monkeypatch.setattr(qwen_module,'hf_sdpa_attention_forward',_test_sdpa_forward)
    calls=[]
    refreshes=[]
    class FakeDecoder:
        def __init__(self,summaries,key,value,*,query_heads,budget,splits):
            self.summaries=summaries
            self.n=key.shape[1]
            self.heads=query_heads
            self.budget=budget
            self.packed=SimpleNamespace(counts_host=tuple(s.num_teams for s in summaries),storage_bytes=lambda:0)
        def warmup(self,*a,**kw):pass
        def workspace_bytes(self):return 0
        def refresh(self,k,v,start,end):refreshes.append((start,end))
        def run(self,q,k,v,*,seed,layer,epoch,statistics):
            assert q.shape==(2,2) and q.is_contiguous()
            calls.append((epoch,k.shape[1]-self.n))
            # Select all teams for this test; all-selected estimator is dense.
            active=self.summaries[0].num_teams
            assert self.budget>=active
            ids=torch.arange(active)[None,:].repeat(2,1)
            lp=torch.zeros_like(ids,dtype=torch.float)
            statistics[0]=torch.tensor([self.n*2,self.n,active*2,active*2,1,1])
            return attention_oracle(q,k,v,self.summaries,ids,lp,self.n)
    monkeypatch.setattr(runtime,'LayerDecoder',FakeDecoder)
    model=_TinyQwen()
    cfg=HierarchicalConfig(prefill_backend='torch',decode_backend='grouped_triton',
                           parent_size=4,representatives_per_parent=2,samples_per_head=128)
    engine=HierarchicalWholeTeamEngine(model,cfg)
    result=engine.generate(torch.tensor([[1,2,3,4,5]]),max_new_tokens=3,
                           eos_token_ids=set(),stop_on_eos=False,random_seed=0)
    assert len(result.token_ids)==3
    assert calls==[(0,0),(1,1),(2,2)]
    assert refreshes==[(4,5)]
    assert result.metrics['decode_backend']=='grouped_triton'
    assert result.metrics['mean_exact_tokens_per_gqa_group_call']==1.0
    assert result.metrics['decode_gqa_total_access_pct']>100 # Full K/V plus routing scan.
    assert not engine._grouped_decoders
    assert 'forward' not in model.model.layers[0].self_attn.__dict__
