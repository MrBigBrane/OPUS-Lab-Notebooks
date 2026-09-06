from types import SimpleNamespace

import torch
from torch import nn

from santapp_ruler.attention import qwen as qwen_module
from santapp_ruler.attention import santapp as santapp_module
from santapp_ruler.attention.qwen import rotate_half
from santapp_ruler.attention.santapp import SantappEngine, parent_cluster_count
from santapp_ruler.config import MiniBatchKMeansConfig, SantappConfig


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


class _PositionDependentRotary(nn.Module):
    def forward(self, hidden_states, position_ids):
        batch, tokens = position_ids.shape
        head_dim = hidden_states.shape[-1]
        angles = position_ids.to(torch.float32) * 0.37
        cos = torch.cos(angles)[..., None].expand(batch, tokens, head_dim)
        sin = torch.sin(angles)[..., None].expand(batch, tokens, head_dim)
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)


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
        self.rotary_emb = _PositionDependentRotary()


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


def _config() -> SantappConfig:
    return SantappConfig(
        parent_size=4,
        representatives_per_parent=2,
        samples_per_head=4,
        kmeans=MiniBatchKMeansConfig(
            batch_size=8,
            n_init=1,
            max_iter=1,
            max_no_improvement=None,
            random_state=0,
        ),
    )


def test_parent_cluster_count_matches_former_rule() -> None:
    assert parent_cluster_count(5, 4) == 2
    assert parent_cluster_count(32, 16) == 2
    assert parent_cluster_count(38, 16) == 2
    assert parent_cluster_count(8192, 16) == 512


def test_santapp_runs_end_to_end_and_exact_suffix_starts_empty(monkeypatch) -> None:
    monkeypatch.setattr(qwen_module, "hf_sdpa_attention_forward", _test_sdpa_forward)
    result = SantappEngine(_TinyQwen(), _config()).generate(
        torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
        max_new_tokens=2,
        eos_token_ids=set(),
        stop_on_eos=False,
        random_seed=11,
    )
    assert len(result.token_ids) == 2
    assert result.metrics["backend"] == "santapp"
    assert result.metrics["parent_clustering_representation"] == "post_rope_key_space"
    assert result.metrics["parent_clustering_scope"] == "global"
    assert result.metrics["parent_clustering_uses_probe_queries"] is False
    assert result.metrics["key_space_normalization"] == "none"
    assert result.metrics["key_space_rope_applied"] is True
    assert result.metrics["prompt_exact_tail_tokens"] == 0
    assert result.metrics["initial_generated_exact_tokens"] == 0
    # One decode call sees zero exact generated rows and the second sees one.
    assert result.metrics["mean_exact_tokens_per_gqa_group_call"] == 0.5
    assert result.metrics["decode_gqa_routing_key_vectors_read"] > 0
    assert result.metrics["decode_gqa_centroid_key_vectors_read"] == 0


def test_kmeans_receives_raw_cached_post_rope_keys(monkeypatch) -> None:
    monkeypatch.setattr(qwen_module, "hf_sdpa_attention_forward", _test_sdpa_forward)
    captured: list[torch.Tensor] = []

    def capture_features(kmeans, features):
        captured.append(features.detach().clone())
        return torch.arange(features.shape[0], device=features.device) % kmeans.n_clusters

    monkeypatch.setattr(
        santapp_module.SklearnLikeTorchMiniBatchKMeans,
        "fit_predict",
        capture_features,
    )

    model = _TinyQwen()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    with torch.inference_mode():
        hidden = model.embed(input_ids)
        raw_key = model.model.layers[0].self_attn.k_proj(hidden).view(1, 5, 1, 2)
        raw_key = raw_key.transpose(1, 2)
        dummy = hidden.view(1, 5, 2, 2).transpose(1, 2)
        positions = torch.arange(5)[None, :]
        cos, sin = model.model.rotary_emb(dummy, positions)
        expected_post_rope = (
            raw_key * cos[:, None, :, :]
            + rotate_half(raw_key) * sin[:, None, :, :]
        )[0, 0].float()

    result = SantappEngine(model, _config()).generate(
        input_ids,
        max_new_tokens=1,
        eos_token_ids=set(),
        stop_on_eos=False,
        random_seed=11,
    )
    assert len(result.token_ids) == 1
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], expected_post_rope)
    assert not torch.allclose(captured[0], raw_key[0, 0].float())
    # Raw coordinates are supplied: there is no unit-norm preprocessing.
    assert not torch.allclose(captured[0].norm(dim=1), torch.ones(5))


def test_contiguous_hierarchical_runs_end_to_end_and_exact_suffix_starts_empty(
    monkeypatch,
) -> None:
    from santapp_ruler.attention.hierarchical import HierarchicalWholeTeamEngine
    from santapp_ruler.config import HierarchicalConfig

    monkeypatch.setattr(qwen_module, "hf_sdpa_attention_forward", _test_sdpa_forward)
    config = HierarchicalConfig(
        parent_size=4,
        representatives_per_parent=2,
        samples_per_head=4,
    )
    result = HierarchicalWholeTeamEngine(_TinyQwen(), config).generate(
        torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
        max_new_tokens=2,
        eos_token_ids=set(),
        stop_on_eos=False,
        random_seed=11,
    )
    assert len(result.token_ids) == 2
    assert result.metrics["backend"] == "hierarchical"
    assert result.metrics["parent_policy"] == "fixed_contiguous_spans_in_original_token_order"
    assert result.metrics["parent_clustering_used"] is False
    assert result.metrics["partial_parent_size"] == 1
    assert result.metrics["partial_parent_is_exact"] is False
    assert result.metrics["prompt_exact_tail_tokens"] == 0
    assert result.metrics["initial_generated_exact_tokens"] == 0
    assert result.metrics["mean_exact_tokens_per_gqa_group_call"] == 0.5


def test_both_parent_policy_engines_can_run_sequentially_on_one_model(
    monkeypatch,
) -> None:
    from santapp_ruler.attention.hierarchical import HierarchicalWholeTeamEngine
    from santapp_ruler.config import HierarchicalConfig

    monkeypatch.setattr(qwen_module, "hf_sdpa_attention_forward", _test_sdpa_forward)
    model = _TinyQwen()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    kmeans_result = SantappEngine(model, _config()).generate(
        input_ids,
        max_new_tokens=1,
        eos_token_ids=set(),
        stop_on_eos=False,
        random_seed=11,
    )
    contiguous_result = HierarchicalWholeTeamEngine(
        model,
        HierarchicalConfig(
            parent_size=4,
            representatives_per_parent=2,
            samples_per_head=4,
        ),
    ).generate(
        input_ids,
        max_new_tokens=1,
        eos_token_ids=set(),
        stop_on_eos=False,
        random_seed=11,
    )

    assert len(kmeans_result.token_ids) == 1
    assert len(contiguous_result.token_ids) == 1
    assert kmeans_result.metrics["backend"] == "santapp"
    assert contiguous_result.metrics["backend"] == "hierarchical"
    # The context manager must restore the module after each engine exits.
    assert "forward" not in model.model.layers[0].self_attn.__dict__
