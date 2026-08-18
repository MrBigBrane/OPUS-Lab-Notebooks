from types import SimpleNamespace

import torch
from torch import nn

from santapp_ruler.attention import santapp as santapp_module
from santapp_ruler.attention.santa import SantaEngine
from santapp_ruler.attention.santapp import SantaPlusEngine
from santapp_ruler.config import MiniBatchKMeansConfig, SantaConfig, SantaPlusConfig


def _test_sdpa_forward(
    attention,
    query,
    key,
    value,
    attention_mask,
    *,
    dropout,
    scaling,
    is_causal,
):
    del attention, dropout
    if key.shape[1] != query.shape[1]:
        repeat = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)

    scores = query.float() @ key.float().transpose(-1, -2) * scaling
    query_tokens = query.shape[-2]
    key_tokens = key.shape[-2]
    if is_causal:
        causal = torch.ones(
            query_tokens,
            key_tokens,
            dtype=torch.bool,
            device=query.device,
        ).tril(diagonal=key_tokens - query_tokens)
        scores = scores.masked_fill(~causal, float("-inf"))
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, float("-inf"))
        else:
            scores = scores + attention_mask
    probability = torch.softmax(scores, dim=-1)
    output = probability @ value.float()
    return output.transpose(1, 2).to(query.dtype), None


class _TinyRotary(nn.Module):
    def forward(self, hidden_states, position_ids):
        batch, tokens = position_ids.shape
        head_dim = hidden_states.shape[-1]
        shape = (batch, tokens, head_dim)
        return (
            torch.ones(shape, dtype=hidden_states.dtype, device=hidden_states.device),
            torch.zeros(shape, dtype=hidden_states.dtype, device=hidden_states.device),
        )


class _TinyAttention(nn.Module):
    def __init__(self, hidden_size: int, query_heads: int, kv_heads: int):
        super().__init__()
        head_dim = hidden_size // query_heads
        self.q_proj = nn.Linear(hidden_size, query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(query_heads * head_dim, hidden_size, bias=False)
        self.scaling = head_dim**-0.5

    def forward(self, *_args, **_kwargs):  # pragma: no cover - always patched
        raise AssertionError("The engine should install its attention patch.")


class _TinyLayer(nn.Module):
    def __init__(self, hidden_size: int, query_heads: int, kv_heads: int):
        super().__init__()
        self.self_attn = _TinyAttention(hidden_size, query_heads, kv_heads)


class _TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int, query_heads: int, kv_heads: int):
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer(hidden_size, query_heads, kv_heads)])
        self.rotary_emb = _TinyRotary()


class _TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        hidden_size = 4
        query_heads = 2
        kv_heads = 1
        self.config = SimpleNamespace(
            model_type="qwen2",
            use_sliding_window=False,
            layer_types=["full_attention"],
            num_attention_heads=query_heads,
            num_key_value_heads=kv_heads,
            hidden_size=hidden_size,
            head_dim=hidden_size // query_heads,
        )
        self.model = _TinyBackbone(hidden_size, query_heads, kv_heads)
        self.embed = nn.Embedding(32, hidden_size)
        self.lm_head = nn.Linear(hidden_size, 32, bias=False)

    def forward(
        self,
        input_ids,
        *,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        logits_to_keep=1,
    ):
        del use_cache, logits_to_keep
        hidden = self.embed(input_ids)
        batch, tokens = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(tokens, device=input_ids.device)[None, :]
        dummy = hidden.view(batch, tokens, 2, 2).transpose(1, 2)
        position_embeddings = self.model.rotary_emb(dummy, position_ids)
        for layer in self.model.layers:
            hidden, _ = layer.self_attn(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
        return SimpleNamespace(logits=self.lm_head(hidden))


def _kmeans_config() -> MiniBatchKMeansConfig:
    return MiniBatchKMeansConfig(
        batch_size=8,
        n_init=1,
        max_iter=1,
        max_no_improvement=None,
        random_state=0,
    )


def test_all_sparse_engines_run_end_to_end_on_tiny_qwen(monkeypatch):
    monkeypatch.setattr(
        santapp_module,
        "hf_sdpa_attention_forward",
        _test_sdpa_forward,
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    santa = SantaEngine(_TinyQwen(), SantaConfig(samples_per_head=3)).generate(
        input_ids,
        max_new_tokens=2,
        eos_token_ids=set(),
        stop_on_eos=False,
        random_seed=11,
    )
    assert len(santa.token_ids) == 2
    assert santa.metrics["prompt_exact_tail_tokens"] == 0
    assert santa.metrics["exact_token_policy"] == "none"
    assert santa.metrics["decode_gqa_centroid_key_vectors_read"] == 0
    assert santa.metrics["decode_gqa_total_access_pct"] > 0.0

    guided_config = SantaPlusConfig(
        mode="guided",
        group_size=2,
        samples_per_head=3,
        probe_queries=2,
        kmeans=_kmeans_config(),
    )
    guided = SantaPlusEngine(_TinyQwen(), guided_config).generate(
        input_ids,
        max_new_tokens=2,
        eos_token_ids=set(),
        stop_on_eos=False,
        random_seed=11,
    )
    assert len(guided.token_ids) == 2
    assert guided.metrics["clustered_prompt_tokens"] == 5
    assert guided.metrics["prompt_exact_tail_tokens"] == 0
    assert guided.metrics["initial_growing_exact_tokens"] == 0
    assert guided.metrics["probe_queries"] == 2
    assert guided.metrics["probe_policy"] == "last_prompt_tokens"
    # First call has no exact suffix; call two has one generated token.
    assert guided.metrics["mean_exact_tokens_per_gqa_group_call"] == 0.5
    assert guided.metrics["decode_gqa_centroid_key_vectors_read"] > 0

    gumbel_config = SantaPlusConfig(
        mode="gumbel_cluster",
        group_size=2,
        samples_per_head=4,
        probe_queries=2,
        kmeans=_kmeans_config(),
    )
    gumbel = SantaPlusEngine(_TinyQwen(), gumbel_config).generate(
        input_ids,
        max_new_tokens=2,
        eos_token_ids=set(),
        stop_on_eos=False,
        random_seed=11,
    )
    assert len(gumbel.token_ids) == 2
    assert gumbel.metrics["sampling_scheme"] == (
        "cluster_gumbel_topk_without_replacement"
    )
    assert gumbel.metrics["prompt_exact_tail_tokens"] == 0
    assert gumbel.metrics["probe_policy"] == "last_prompt_tokens"
    assert gumbel.metrics["mean_selected_clusters_per_head_call"] == 2.0
    assert gumbel.metrics["mean_exact_tokens_per_gqa_group_call"] == 0.5

    team_config = SantaPlusConfig(
        mode="team_sampler",
        parent_size=4,
        representatives_per_parent=2,
        samples_per_head=3,
        probe_queries=2,
        kmeans=_kmeans_config(),
    )
    team = SantaPlusEngine(_TinyQwen(), team_config).generate(
        input_ids,
        max_new_tokens=2,
        eos_token_ids=set(),
        stop_on_eos=False,
        random_seed=11,
    )
    assert len(team.token_ids) == 2
    assert team.metrics["mode"] == "team_sampler"
    assert team.metrics["routing_key_type"] == "actual_team_leader"
    assert team.metrics["parent_clustering_space"] == "probe_fingerprint"
    assert team.metrics["team_assignment_space"] == "actual_key_l2"
    assert team.metrics["mean_sampled_token_rows_per_head_call"] == 3.0
    assert team.metrics["decode_gqa_routing_key_vectors_read"] > 0
    assert team.metrics["decode_gqa_centroid_key_vectors_read"] == 0

    team_gumbel_config = SantaPlusConfig(
        mode="team_gumbel_topk",
        parent_size=4,
        representatives_per_parent=2,
        samples_per_head=4,
        probe_queries=2,
        kmeans=_kmeans_config(),
    )
    team_gumbel = SantaPlusEngine(_TinyQwen(), team_gumbel_config).generate(
        input_ids,
        max_new_tokens=2,
        eos_token_ids=set(),
        stop_on_eos=False,
        random_seed=11,
    )
    assert len(team_gumbel.token_ids) == 2
    assert team_gumbel.metrics["teams_per_head"] == 2
    assert team_gumbel.metrics["mean_selected_teams_per_head_call"] == 2.0
    assert team_gumbel.metrics["selected_team_inclusion_probability_count"] > 0
    assert team_gumbel.metrics["decode_gqa_centroid_key_vectors_read"] == 0


def test_first_clustered_decode_call_has_no_exact_prompt_token(monkeypatch):
    monkeypatch.setattr(
        santapp_module,
        "hf_sdpa_attention_forward",
        _test_sdpa_forward,
    )
    config = SantaPlusConfig(
        group_size=2,
        samples_per_head=3,
        probe_queries=2,
        kmeans=_kmeans_config(),
    )
    result = SantaPlusEngine(_TinyQwen(), config).generate(
        torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
        max_new_tokens=1,
        eos_token_ids=set(),
        stop_on_eos=False,
        random_seed=11,
    )
    assert result.metrics["mean_exact_tokens_per_gqa_group_call"] == 0.0
