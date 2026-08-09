from types import SimpleNamespace

import torch
from torch import nn

from santapp_ruler.attention import santapp as santapp_module
from santapp_ruler.attention.santa import SantaEngine
from santapp_ruler.attention.santapp import SantaPlusEngine
from santapp_ruler.config import (
    MiniBatchKMeansConfig,
    SantaConfig,
    SantaPlusConfig,
)


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
    # Hugging Face's wrapper returns [batch, query_tokens, heads, head_dim].
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
        self.layers = nn.ModuleList(
            [_TinyLayer(hidden_size, query_heads, kv_heads)]
        )
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


def test_santa_and_santapp_engines_run_end_to_end_on_tiny_qwen(monkeypatch):
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
    assert santa.metrics["exact_window_tokens"] == 0
    assert santa.metrics["decode_gqa_centroid_key_vectors_read"] == 0
    assert santa.metrics["decode_gqa_total_access_pct"] > 0.0

    santapp_config = SantaPlusConfig(
        group_size=2,
        samples_per_head=3,
        probe_queries=2,
        kmeans=MiniBatchKMeansConfig(
            batch_size=8,
            n_init=1,
            max_iter=1,
            max_no_improvement=None,
            random_state=0,
        ),
    )
    santapp = SantaPlusEngine(_TinyQwen(), santapp_config).generate(
        input_ids,
        max_new_tokens=2,
        eos_token_ids=set(),
        stop_on_eos=False,
        random_seed=11,
    )
    assert len(santapp.token_ids) == 2
    assert santapp.metrics["clustered_prompt_tokens"] == 5
    assert santapp.metrics["prompt_exact_tail_tokens"] == 0
    assert santapp.metrics["initial_growing_exact_tokens"] == 0
    assert santapp.metrics["probe_queries"] == 2
    assert santapp.metrics["probe_policy"] == "last_prompt_tokens"
    # The first sparse call re-feeds the final prompt token, but that token is
    # cluster-covered. Only the first generated token is exact on call two.
    assert santapp.metrics["mean_exact_tokens_per_gqa_group_call"] == 0.5
    assert santapp.metrics["decode_gqa_centroid_key_vectors_read"] > 0

    first_decision = SantaPlusEngine(_TinyQwen(), santapp_config).generate(
        input_ids,
        max_new_tokens=1,
        eos_token_ids=set(),
        stop_on_eos=False,
        random_seed=11,
    )
    assert first_decision.metrics["mean_exact_tokens_per_gqa_group_call"] == 0.0
