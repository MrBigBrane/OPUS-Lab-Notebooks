"""Real CUDA correctness gates. A skipped test is NOT GPU validation evidence."""
import pytest
import torch

from santapp_ruler.attention.triton_decode.validation import validate_case

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')]


@pytest.mark.parametrize('budget',[32,64,128,256,512,1024])
def test_each_budget_irregular_teams_and_growing_suffix(budget):
    validate_case(8193,budget,suffixes=(0,1,17),irregular=True)


@pytest.mark.parametrize('tokens,budget',[(1,1),(17,32),(128,32),(129,32)])
def test_all_selected_exact_threshold_boundary_and_empty_splits(tokens,budget):
    validate_case(tokens,budget,suffixes=(0,1),dim=32,kv_heads=1,group=1,irregular=False,splits=16)


def test_bfloat16_and_non_power_of_two_query_group():
    validate_case(513,64,dtype=torch.bfloat16,suffixes=(0,17),dim=128,kv_heads=2,group=7)


@pytest.mark.parametrize('policy',['santapp','hierarchical'])
def test_actual_native_prefill_output_consumed_without_repair(policy):
    def build(keys):
        from santapp_ruler.attention.triton_prefill.prefill_teams import (
            build_contiguous_team_summary_triton,build_team_summary_triton)
        from santapp_ruler.attention.triton_prefill.prefill_kmeans import TritonMiniBatchKMeans
        result=[]
        for h in range(keys.shape[0]):
            x=keys[h].float().contiguous()
            if policy=='hierarchical':
                s=build_contiguous_team_summary_triton(x,parent_size=16,representatives_per_parent=4)
            else:
                labels=TritonMiniBatchKMeans(n_clusters=8,max_iter=2,batch_size=128,
                                            random_state=0).fit_predict(x)
                s=build_team_summary_triton(x,labels,representatives_per_parent=4,parent_cluster_count=8)
            result.append(s)
        return result
    validate_case(129,32,suffixes=(0,1),kv_heads=1,group=7,summaries_override=build)
