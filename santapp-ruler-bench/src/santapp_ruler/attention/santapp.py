"""Reference SANTA++ attention backend for Qwen2/Qwen2.5.

The implementation uses dense SDPA for prefill, clusters every prompt token using
fingerprints from the final probe-query tokens, and applies centroid-guided IID
sampling during batch-1 autoregressive decode. No prompt token is assigned to an
exact tail: the final prompt token is re-fed as the first decode query, but its KV
row remains covered by the frozen prompt clusters. Only subsequently generated
tokens form the growing deterministic region.

This is a readable research reference implementation, not a fused production
kernel.
"""

from __future__ import annotations

import contextlib
import math
import time
import types
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from ..config import SantaPlusConfig
from .minibatch_kmeans import SklearnLikeTorchMiniBatchKMeans
from .probes import last_prompt_probe_positions
from .traffic import DecodeTrafficTracker

try:
    from transformers.integrations.sdpa_attention import (
        sdpa_attention_forward as hf_sdpa_attention_forward,
    )
except ImportError:  # Keep pure estimator/config tests importable without extras.
    hf_sdpa_attention_forward = None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Qwen2 RoPE helper, written locally to avoid another private import."""

    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _uses_cuda(device: torch.device | str) -> bool:
    return torch.device(device).type == "cuda" and torch.cuda.is_available()


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _seed_torch(seed: int, device: torch.device | str) -> None:
    torch.manual_seed(seed)
    if _uses_cuda(device):
        torch.cuda.manual_seed_all(seed)


def _reset_peak_memory(device: torch.device | str) -> None:
    if _uses_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_gib(device: torch.device | str) -> tuple[float, float]:
    if not _uses_cuda(device):
        return 0.0, 0.0
    return (
        torch.cuda.max_memory_allocated(device) / (1024**3),
        torch.cuda.max_memory_reserved(device) / (1024**3),
    )


@dataclass(slots=True)
class ClusterSummary:
    members: torch.Tensor
    starts: torch.Tensor
    lengths_long: torch.Tensor
    lengths_float: torch.Tensor
    key_centroids: torch.Tensor

    @property
    def num_groups(self) -> int:
        return int(self.lengths_long.numel())


@dataclass(slots=True)
class LayerCache:
    key: torch.Tensor
    value: torch.Tensor
    length: int

    @property
    def capacity(self) -> int:
        return int(self.key.shape[1])

    def view(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key[:, : self.length], self.value[:, : self.length]


@dataclass(frozen=True, slots=True)
class AttentionGeneration:
    token_ids: list[int]
    metrics: dict[str, Any]


_MISSING = object()


class SantaPlusEngine:
    """Patch a loaded Qwen2 model only while running SANTA++ generation."""

    def __init__(self, model: Any, config: SantaPlusConfig):
        self.model = model
        self.config = config
        self.base_model = getattr(model, "model", None)
        if self.base_model is None or not hasattr(self.base_model, "layers"):
            raise TypeError("Expected a Hugging Face causal LM with model.layers.")
        model_type = getattr(model.config, "model_type", None)
        if model_type != "qwen2":
            raise TypeError(
                "The extracted attention patch currently supports Qwen2/Qwen2.5 "
                f"(model_type='qwen2'), not {model_type!r}."
            )

        cfg = model.config
        if bool(getattr(cfg, "use_sliding_window", False)):
            raise NotImplementedError(
                "The SANTA++ patch requires full attention in every Qwen2 layer; "
                "use_sliding_window=True is unsupported."
            )
        layer_types = getattr(cfg, "layer_types", None)
        if layer_types is not None and any(
            layer_type != "full_attention" for layer_type in layer_types
        ):
            raise NotImplementedError(
                "The SANTA++ patch requires every layer type to be "
                "'full_attention'."
            )

        self.num_layers = len(self.base_model.layers)
        self.num_query_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError("Query heads must be divisible by KV heads.")
        self.query_heads_per_kv = self.num_query_heads // self.num_kv_heads
        self.head_dim = int(
            getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        )

        self.mode = "dense"
        self.sample_end = 0
        self.cache_capacity = 0
        self.cache: dict[int, LayerCache] = {}
        self.summaries: dict[tuple[int, int], ClusterSummary] = {}
        self.traffic = self._make_traffic_tracker()
        self._original_forwards: list[Any] = []

    def _make_traffic_tracker(self) -> DecodeTrafficTracker:
        return DecodeTrafficTracker(
            query_heads_per_kv=self.query_heads_per_kv,
            samples_per_head=self.config.samples_per_head,
            access_pattern="sampled_kv",
        )

    @contextlib.contextmanager
    def patched(self) -> Iterator[None]:
        """Install the instance-level Qwen attention patch, then restore it."""

        self._original_forwards = []
        patched_attentions: list[Any] = []
        try:
            for layer_id, layer in enumerate(self.base_model.layers):
                attention = layer.self_attn
                original = attention.__dict__.get("forward", _MISSING)
                self._original_forwards.append(original)
                patched_attentions.append(attention)
                attention._santapp_engine = self
                attention._santapp_layer_id = layer_id
                attention.forward = types.MethodType(_patched_forward, attention)
            yield
        finally:
            for attention, original in zip(
                patched_attentions, self._original_forwards, strict=True
            ):
                if original is _MISSING:
                    attention.__dict__.pop("forward", None)
                else:
                    attention.forward = original
                attention.__dict__.pop("_santapp_engine", None)
                attention.__dict__.pop("_santapp_layer_id", None)
            self._original_forwards = []

    def clear(self) -> None:
        self.cache.clear()
        self.summaries.clear()
        self.traffic = self._make_traffic_tracker()

    def _append_cache(
        self, layer_id: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # key/value arrive as [1, n_kv, T, d].
        key0 = key[0]
        value0 = value[0]
        tokens = int(key0.shape[1])
        cached = self.cache.get(layer_id)
        if cached is None:
            capacity = max(self.cache_capacity, tokens)
            key_buffer = torch.empty(
                self.num_kv_heads,
                capacity,
                self.head_dim,
                dtype=key0.dtype,
                device=key0.device,
            )
            value_buffer = torch.empty_like(key_buffer)
            key_buffer[:, :tokens].copy_(key0)
            value_buffer[:, :tokens].copy_(value0)
            self.cache[layer_id] = LayerCache(key_buffer, value_buffer, tokens)
            # Preserve the contiguous first-call layout used by stock prefill.
            return key0.contiguous(), value0.contiguous()

        start = cached.length
        end = start + tokens
        if end > cached.capacity:
            raise RuntimeError(
                f"Custom cache capacity exceeded in layer {layer_id}: "
                f"need {end}, allocated {cached.capacity}."
            )
        cached.key[:, start:end].copy_(key0)
        cached.value[:, start:end].copy_(value0)
        cached.length = end
        full_key, full_value = cached.view()
        if self.mode == "dense":
            # Stock DynamicCache concatenation is contiguous. Preserve that
            # layout for the explicit fidelity path; sparse mode indexes the
            # preallocated view directly.
            return full_key.contiguous(), full_value.contiguous()
        return full_key, full_value

    def _trim_cache(self, length: int) -> None:
        for layer_id, cached in self.cache.items():
            if cached.length < length:
                raise RuntimeError(
                    f"Layer {layer_id} cache has only {cached.length} tokens; "
                    f"cannot trim to {length}."
                )
            cached.length = length

    def _attention_forward(
        self,
        attention: Any,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask
        batch, query_tokens, _ = hidden_states.shape
        if batch != 1:
            raise NotImplementedError("The custom cache supports batch size 1.")
        if position_embeddings is None:
            raise RuntimeError("Qwen did not supply position_embeddings.")

        query = attention.q_proj(hidden_states).view(
            batch, query_tokens, self.num_query_heads, self.head_dim
        ).transpose(1, 2)
        key = attention.k_proj(hidden_states).view(
            batch, query_tokens, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value = attention.v_proj(hidden_states).view(
            batch, query_tokens, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings

        def apply_rope(x: torch.Tensor) -> torch.Tensor:
            return x * cos[:, None, :, :] + rotate_half(x) * sin[:, None, :, :]

        query = apply_rope(query)
        key = apply_rope(key)

        layer_id = int(attention._santapp_layer_id)
        full_key, full_value = self._append_cache(layer_id, key, value)
        key_tokens = int(full_key.shape[1])

        if query_tokens > 1 or self.mode == "dense":
            if query_tokens == key_tokens:
                dense_mask = None
                dense_is_causal = query_tokens > 1
            elif query_tokens == 1:
                dense_mask = None
                dense_is_causal = False
            else:
                query_absolute = torch.arange(
                    key_tokens - query_tokens,
                    key_tokens,
                    device=query.device,
                )[:, None]
                key_absolute = torch.arange(key_tokens, device=query.device)[None, :]
                dense_mask = (key_absolute <= query_absolute)[None, None, :, :]
                dense_is_causal = False

            if hf_sdpa_attention_forward is None:
                raise RuntimeError(
                    "SANTA/SANTA++ expect transformers==5.12.1 and its "
                    "SDPA integration. Install the project dependencies before "
                    "running model-backed generation."
                )
            output, _ = hf_sdpa_attention_forward(
                attention,
                query,
                full_key.unsqueeze(0),
                full_value.unsqueeze(0),
                dense_mask,
                dropout=0.0,
                scaling=attention.scaling,
                is_causal=dense_is_causal,
            )
        else:
            outputs: list[torch.Tensor] = []
            for kv_head in range(self.num_kv_heads):
                sampled_indices_by_head: list[torch.Tensor] = []
                for query_head in range(
                    kv_head * self.query_heads_per_kv,
                    (kv_head + 1) * self.query_heads_per_kv,
                ):
                    head_output, sampled_indices = self._approximate_attention(
                        query[0, query_head, 0].float(),
                        full_key[kv_head],
                        full_value[kv_head],
                        layer_id,
                        kv_head,
                    )
                    outputs.append(head_output)
                    sampled_indices_by_head.append(sampled_indices)
                self._record_decode_group(
                    layer_id=layer_id,
                    kv_head=kv_head,
                    n_total=key_tokens,
                    sampled_indices_by_head=sampled_indices_by_head,
                )
            output = torch.stack(outputs)[None, None].to(hidden_states.dtype)

        output = output.reshape(
            batch, query_tokens, self.num_query_heads * self.head_dim
        ).contiguous()
        return attention.o_proj(output), None

    def _record_decode_group(
        self,
        *,
        layer_id: int,
        kv_head: int,
        n_total: int,
        sampled_indices_by_head: list[torch.Tensor],
    ) -> None:
        summary = self.summaries[(layer_id, kv_head)]
        exact_tokens = n_total - self.sample_end
        if exact_tokens < 0:
            raise RuntimeError(
                f"Cache length {n_total} is shorter than sample boundary "
                f"{self.sample_end}."
            )
        self.traffic.record_group(
            n_total=n_total,
            sampled_indices_by_head=sampled_indices_by_head,
            exact_tokens=exact_tokens,
            centroid_key_vectors=summary.num_groups,
        )

    def _approximate_attention(
        self,
        query: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        layer_id: int,
        kv_head: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_total = int(full_key.shape[0])
        old_end = self.sample_end
        if not 0 < old_end <= n_total:
            raise RuntimeError(
                f"Invalid sampled-prefix boundary {old_end} for cache length {n_total}."
            )
        exact_tokens = n_total - old_end
        scale = 1.0 / math.sqrt(self.head_dim)
        if exact_tokens:
            exact_scores = full_key[old_end:].float() @ query * scale
            exact_values = full_value[old_end:].float()
        else:
            exact_scores = torch.empty(0, device=query.device, dtype=torch.float32)
            exact_values = torch.empty(
                0, self.head_dim, device=query.device, dtype=torch.float32
            )

        summary = self.summaries[(layer_id, kv_head)]
        group_probability = torch.softmax(
            torch.log(summary.lengths_float)
            + summary.key_centroids @ query * scale,
            dim=0,
        )
        sample_count = self.config.samples_per_head
        sampled_groups = torch.multinomial(
            group_probability, sample_count, replacement=True
        )
        lengths = summary.lengths_long[sampled_groups]
        within_group = torch.floor(
            torch.rand(sample_count, device=query.device) * lengths.float()
        ).long()
        positions = summary.starts[sampled_groups] + within_group
        sampled_indices = summary.members[positions]
        log_proposal = (
            torch.log(group_probability[sampled_groups])
            - torch.log(summary.lengths_float[sampled_groups])
        )

        sampled_scores = full_key[sampled_indices].float() @ query * scale
        sampled_log_weights = sampled_scores - log_proposal
        if exact_tokens:
            maximum = torch.cat((sampled_log_weights, exact_scores)).max()
        else:
            maximum = sampled_log_weights.max()
        sampled_weights = torch.exp(sampled_log_weights - maximum) / sample_count
        exact_weights = torch.exp(exact_scores - maximum)

        numerator = sampled_weights @ full_value[sampled_indices].float()
        if exact_tokens:
            numerator = numerator + exact_weights @ exact_values
        denominator = sampled_weights.sum() + exact_weights.sum()
        return numerator / denominator, sampled_indices

    def _build_summary(
        self, keys: torch.Tensor, labels: torch.Tensor, n_clusters: int
    ) -> ClusterSummary:
        labels = labels.long()
        counts = torch.bincount(labels, minlength=n_clusters)
        active = counts > 0
        lengths_long = counts[active]
        lengths_float = lengths_long.float()

        members = torch.argsort(labels)
        full_starts = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=labels.device),
                torch.cumsum(counts[:-1], dim=0),
            )
        )
        starts = full_starts[active]

        sums = torch.zeros(
            n_clusters,
            self.head_dim,
            dtype=torch.float32,
            device=keys.device,
        )
        sums.index_add_(0, labels, keys)
        centroids = sums[active] / lengths_float[:, None]
        return ClusterSummary(
            members=members,
            starts=starts,
            lengths_long=lengths_long,
            lengths_float=lengths_float,
            key_centroids=centroids,
        )

    @torch.inference_mode()
    def _dense_prefill_and_cluster(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[float, float, int, int]:
        prompt_tokens = int(input_ids.shape[1])
        # Every prompt token, including the final token that will be re-fed as the
        # first sparse query, is covered by the frozen prompt clusters. There is
        # no exact prompt-tail window.
        sample_end = prompt_tokens
        if sample_end < 3:
            raise ValueError(
                f"Prompt has {prompt_tokens} tokens; SANTA++ needs at least three."
            )
        if self.config.probe_queries > prompt_tokens:
            raise ValueError(
                f"probe_queries={self.config.probe_queries} exceeds the "
                f"{prompt_tokens}-token prompt."
            )
        self.sample_end = sample_end
        self.mode = "dense"

        raw_queries: dict[int, torch.Tensor] = {}
        hooks = []
        for layer_id, layer in enumerate(self.base_model.layers):

            def capture(_module, _inputs, output, *, lid=layer_id):
                raw_queries[lid] = output.detach()

            hooks.append(layer.self_attn.q_proj.register_forward_hook(capture))

        _cuda_sync()
        prefill_start = time.perf_counter()
        try:
            _ = self.model(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=1,
            )
        finally:
            for hook in hooks:
                hook.remove()
        _cuda_sync()
        prefill_seconds = time.perf_counter() - prefill_start

        missing = set(range(self.num_layers)) - set(raw_queries)
        if missing:
            raise RuntimeError(f"Failed to capture q_proj outputs for layers {missing}.")

        cluster_start = time.perf_counter()
        positions = torch.arange(prompt_tokens, device=input_ids.device)[None, :]
        dummy = torch.empty(
            1,
            self.num_query_heads,
            prompt_tokens,
            self.head_dim,
            device=input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )
        cos, sin = self.base_model.rotary_emb(dummy, positions)
        probe_tensor = last_prompt_probe_positions(
            prompt_tokens,
            self.config.probe_queries,
            device=input_ids.device,
        )

        n_clusters = min(
            sample_end,
            max(2, sample_end // self.config.group_size),
        )
        km_cfg = self.config.kmeans

        for layer_id in range(self.num_layers):
            query_raw = raw_queries.pop(layer_id)
            query_rotated = query_raw.view(
                1,
                prompt_tokens,
                self.num_query_heads,
                self.head_dim,
            ).transpose(1, 2)
            query_rotated = (
                query_rotated * cos[:, None, :, :]
                + rotate_half(query_rotated) * sin[:, None, :, :]
            )[0].float()

            cached_key, _ = self.cache[layer_id].view()
            for kv_head in range(self.num_kv_heads):
                key_prefix = cached_key[kv_head, :sample_end].float()
                fingerprints = torch.cat(
                    [
                        key_prefix
                        @ query_rotated[query_head, probe_tensor].T
                        / math.sqrt(self.head_dim)
                        for query_head in range(
                            kv_head * self.query_heads_per_kv,
                            (kv_head + 1) * self.query_heads_per_kv,
                        )
                    ],
                    dim=1,
                ).contiguous()
                fingerprints = (
                    fingerprints - fingerprints.mean(dim=0, keepdim=True)
                ) / (
                    fingerprints.std(dim=0, keepdim=True, unbiased=False) + 1e-6
                )

                labels = SklearnLikeTorchMiniBatchKMeans(
                    n_clusters=n_clusters,
                    batch_size=km_cfg.batch_size,
                    n_init=km_cfg.n_init,
                    max_iter=km_cfg.max_iter,
                    tol=km_cfg.tol,
                    max_no_improvement=km_cfg.max_no_improvement,
                    init_size=km_cfg.init_size,
                    reassignment_ratio=km_cfg.reassignment_ratio,
                    random_state=km_cfg.random_state,
                ).fit_predict(fingerprints)
                self.summaries[(layer_id, kv_head)] = self._build_summary(
                    key_prefix, labels, n_clusters
                )
                del fingerprints, labels, key_prefix
            del query_rotated, query_raw

        _cuda_sync()
        clustering_seconds = time.perf_counter() - cluster_start
        return prefill_seconds, clustering_seconds, sample_end, n_clusters

    @torch.inference_mode()
    def generate_dense_reference(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
    ) -> list[int]:
        """Run the custom cache with dense SDPA for a stock-parity check."""

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Dense reference expects input_ids with shape [1, T].")
        prompt_tokens = int(input_ids.shape[1])
        self.clear()
        self.cache_capacity = prompt_tokens + max_new_tokens
        self.mode = "dense"
        generated: list[int] = []
        with self.patched():
            output = self.model(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=1,
            )
            position = prompt_tokens
            for step in range(max_new_tokens):
                next_token = output.logits[0, -1].argmax().view(1, 1)
                generated.append(int(next_token.item()))
                if step + 1 == max_new_tokens:
                    break
                output = self.model(
                    next_token,
                    attention_mask=torch.ones_like(next_token),
                    position_ids=torch.full(
                        (1, 1),
                        position,
                        dtype=torch.long,
                        device=input_ids.device,
                    ),
                    use_cache=False,
                    logits_to_keep=1,
                )
                position += 1
        self.clear()
        return generated

    def _storage_bytes(self) -> tuple[int, int]:
        cache_bytes = sum(
            cached.key.numel() * cached.key.element_size()
            + cached.value.numel() * cached.value.element_size()
            for cached in self.cache.values()
        )
        summary_bytes = 0
        for summary in self.summaries.values():
            for tensor in (
                summary.members,
                summary.starts,
                summary.lengths_long,
                summary.lengths_float,
                summary.key_centroids,
            ):
                summary_bytes += tensor.numel() * tensor.element_size()
        return cache_bytes, summary_bytes

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_ids: set[int],
        stop_on_eos: bool,
        random_seed: int,
    ) -> AttentionGeneration:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("SANTA++ generation expects input_ids with shape [1, T].")
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens < 3:
            raise ValueError("SANTA++ requires at least three prompt tokens.")

        self.clear()
        self.cache_capacity = prompt_tokens + max_new_tokens
        _seed_torch(random_seed, input_ids.device)
        _reset_peak_memory(input_ids.device)

        _cuda_sync()
        total_start = time.perf_counter()
        with self.patched():
            prefill_seconds, clustering_seconds, sample_end, n_clusters = (
                self._dense_prefill_and_cluster(input_ids)
            )
            # Discard the final prompt token from the cache, then re-feed it as
            # the first sparse decode query. Its KV row is still represented by
            # the frozen prompt clusters, so the initial exact region is empty.
            self._trim_cache(prompt_tokens - 1)
            self.mode = "sparse"
            self.traffic = self._make_traffic_tracker()
            _seed_torch(random_seed, input_ids.device)

            generated: list[int] = []
            current = input_ids[:, -1:]
            position = prompt_tokens - 1
            _cuda_sync()
            decode_start = time.perf_counter()
            for _ in range(max_new_tokens):
                output = self.model(
                    current,
                    position_ids=torch.tensor(
                        [[position]], dtype=torch.long, device=input_ids.device
                    ),
                    use_cache=False,
                    logits_to_keep=1,
                )
                current = output.logits[0, -1].argmax().view(1, 1)
                token_id = int(current.item())
                generated.append(token_id)
                position += 1
                if stop_on_eos and token_id in eos_token_ids:
                    break
            _cuda_sync()
            decode_seconds = time.perf_counter() - decode_start
            cache_bytes, summary_bytes = self._storage_bytes()

        _cuda_sync()
        total_seconds = time.perf_counter() - total_start
        peak_allocated_gib, peak_reserved_gib = _peak_memory_gib(input_ids.device)

        # Union accounting is intentionally finalized after the timed region.
        traffic = self.traffic.as_dict()
        metrics: dict[str, Any] = {
            "backend": "santapp",
            "prompt_tokens": prompt_tokens,
            "generated_tokens": len(generated),
            "max_new_tokens": max_new_tokens,
            "prefill_seconds": prefill_seconds,
            "clustering_seconds": clustering_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": total_seconds,
            "clustered_prompt_tokens": sample_end,
            "prompt_exact_tail_tokens": 0,
            "initial_growing_exact_tokens": 0,
            "samples_per_head": self.config.samples_per_head,
            "group_size": self.config.group_size,
            "probe_queries": self.config.probe_queries,
            "probe_policy": "last_prompt_tokens",
            "nominal_clusters_per_kv_head": n_clusters,
            "custom_cache_gib": cache_bytes / (1024**3),
            "cluster_summary_gib": summary_bytes / (1024**3),
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            **traffic,
        }
        self.clear()
        return AttentionGeneration(token_ids=generated, metrics=metrics)


def _patched_forward(
    attention: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    del args, kwargs
    engine: SantaPlusEngine = attention._santapp_engine
    return engine._attention_forward(
        attention,
        hidden_states,
        position_embeddings,
        attention_mask,
    )
