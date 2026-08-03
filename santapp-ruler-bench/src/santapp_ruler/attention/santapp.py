"""Reference SANTA++ attention backend for Qwen2/Qwen2.5.

This module extracts the supplied notebook's algorithm into reusable classes:

* dense SDPA prefill;
* query-fingerprint clustering per layer and KV head;
* GPU MiniBatchKMeans with the notebook's current defaults;
* centroid-guided sampling during batch-1 autoregressive decode;
* exact recent-window attention; and
* draw-count KV access and centroid-metadata accounting.

It is a research reference implementation, not a fused production kernel.
"""

from __future__ import annotations

import contextlib
import math
import time
import types
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch

from ..config import SantaPlusConfig
from .minibatch_kmeans import SklearnLikeTorchMiniBatchKMeans

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

try:
    from transformers.integrations.sdpa_attention import (
        sdpa_attention_forward as hf_sdpa_attention_forward,
    )
except ImportError as exc:  # fail with an actionable message at import time
    raise RuntimeError(
        "SANTA++ expects transformers==5.12.1 and its SDPA integration."
    ) from exc


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Qwen2 RoPE helper, written locally to avoid another private import."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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


@dataclass(slots=True)
class DecodeTrafficTracker:
    """Accumulate decode-time attention row reads over all layers and heads."""

    head_calls: int = 0
    dense_kv_vectors: int = 0
    kv_vectors_read: int = 0
    metadata_key_vectors_read: int = 0
    sampled_token_draws: int = 0
    recent_token_reads: int = 0
    total_tokens_summed: int = 0
    ess_ratios: list[float] = field(default_factory=list)

    def record(
        self,
        *,
        n_total: int,
        sampled_tokens: int,
        recent_tokens: int,
        kv_cache_vectors: int,
        metadata_key_vectors: int,
        ess_ratio: float | None,
    ) -> None:
        self.head_calls += 1
        self.dense_kv_vectors += 2 * n_total
        self.kv_vectors_read += kv_cache_vectors
        self.metadata_key_vectors_read += metadata_key_vectors
        self.sampled_token_draws += sampled_tokens
        self.recent_token_reads += recent_tokens
        self.total_tokens_summed += n_total
        if ess_ratio is not None:
            self.ess_ratios.append(float(ess_ratio))

    def as_dict(self) -> dict[str, float | int | None]:
        dense = self.dense_kv_vectors
        kv_pct = 100.0 * self.kv_vectors_read / dense if dense else 0.0
        equivalent = self.kv_vectors_read + self.metadata_key_vectors_read
        equivalent_pct = 100.0 * equivalent / dense if dense else 0.0
        metadata_pct = (
            100.0 * self.metadata_key_vectors_read / dense if dense else 0.0
        )
        calls = self.head_calls
        return {
            "decode_attention_head_calls": calls,
            "decode_dense_kv_vectors": dense,
            "decode_kv_vectors_read": self.kv_vectors_read,
            "decode_metadata_key_vectors_read": self.metadata_key_vectors_read,
            "decode_kv_access_pct": kv_pct,
            "decode_read_equivalent_pct": equivalent_pct,
            "decode_metadata_equivalent_pct": metadata_pct,
            "mean_sampled_token_draws_per_head_call": (
                self.sampled_token_draws / calls if calls else 0.0
            ),
            "mean_recent_tokens_per_head_call": (
                self.recent_token_reads / calls if calls else 0.0
            ),
            "mean_total_tokens_per_head_call": (
                self.total_tokens_summed / calls if calls else 0.0
            ),
            "mean_ess_over_samples": (
                float(np.mean(self.ess_ratios)) if self.ess_ratios else None
            ),
        }


@dataclass(frozen=True, slots=True)
class SantaGeneration:
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
                "The extracted SANTA++ patch currently requires full attention in "
                "every Qwen2 layer; use_sliding_window=True is unsupported."
            )
        layer_types = getattr(cfg, "layer_types", None)
        if layer_types is not None and any(
            layer_type != "full_attention" for layer_type in layer_types
        ):
            raise NotImplementedError(
                "The extracted SANTA++ patch currently requires every layer type "
                "to be 'full_attention'."
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
        self.traffic = DecodeTrafficTracker()
        self._original_forwards: list[Any] = []

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
        self.traffic = DecodeTrafficTracker()

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
                f"SANTA++ cache capacity exceeded in layer {layer_id}: "
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
        batch, query_tokens, _ = hidden_states.shape
        if batch != 1:
            raise NotImplementedError("SANTA++ custom cache supports batch size 1.")
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
            outputs = [
                self._approximate_attention(
                    query[0, head, 0].float(),
                    full_key[head // self.query_heads_per_kv],
                    full_value[head // self.query_heads_per_kv],
                    layer_id,
                    head // self.query_heads_per_kv,
                )
                for head in range(self.num_query_heads)
            ]
            output = torch.stack(outputs)[None, None].to(hidden_states.dtype)

        output = output.reshape(
            batch, query_tokens, self.num_query_heads * self.head_dim
        ).contiguous()
        return attention.o_proj(output), None

    def _approximate_attention(
        self,
        query: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        layer_id: int,
        kv_head: int,
    ) -> torch.Tensor:
        n_total = int(full_key.shape[0])
        old_end = self.sample_end
        if not 0 < old_end <= n_total:
            raise RuntimeError(
                f"Invalid sampled-prefix boundary {old_end} for cache length {n_total}."
            )
        recent_tokens = n_total - old_end
        scale = 1.0 / math.sqrt(self.head_dim)
        if recent_tokens:
            recent_scores = full_key[old_end:].float() @ query * scale
            recent_values = full_value[old_end:].float()
        else:
            recent_scores = torch.empty(0, device=query.device, dtype=torch.float32)
            recent_values = torch.empty(
                0, self.head_dim, device=query.device, dtype=torch.float32
            )

        mode = self.mode
        sample_count = self.config.samples_per_head
        metadata_vectors = 0
        ess_ratio: float | None = None

        if mode == "topk":
            summary = self.summaries[(layer_id, kv_head)]
            guess = (
                torch.log(summary.lengths_float)
                + summary.key_centroids @ query * scale
            )
            order = torch.argsort(guess, descending=True).tolist()
            chosen_ranges: list[torch.Tensor] = []
            chosen = 0
            for group in order:
                start = int(summary.starts[group].item())
                length = int(summary.lengths_long[group].item())
                chosen_ranges.append(summary.members[start : start + length])
                chosen += length
                if chosen >= sample_count:
                    break
            sampled_indices = torch.cat(chosen_ranges)
            sampled_scores = full_key[sampled_indices].float() @ query * scale
            all_scores = torch.cat((sampled_scores, recent_scores))
            weights = torch.softmax(all_scores, dim=0)
            all_values = torch.cat(
                (full_value[sampled_indices].float(), recent_values), dim=0
            )
            metadata_vectors = summary.num_groups
            self.traffic.record(
                n_total=n_total,
                sampled_tokens=int(sampled_indices.numel()),
                recent_tokens=recent_tokens,
                kv_cache_vectors=2 * (int(sampled_indices.numel()) + recent_tokens),
                metadata_key_vectors=metadata_vectors,
                ess_ratio=None,
            )
            return weights @ all_values

        if mode == "guided":
            summary = self.summaries[(layer_id, kv_head)]
            group_probability = torch.softmax(
                torch.log(summary.lengths_float)
                + summary.key_centroids @ query * scale,
                dim=0,
            )
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
            metadata_vectors = summary.num_groups
        elif mode == "santa":
            old_scores = full_key[:old_end].float() @ query * scale
            log_probability = old_scores - torch.logsumexp(old_scores, dim=0)
            sampled_indices = torch.multinomial(
                torch.exp(log_probability), sample_count, replacement=True
            )
            log_proposal = log_probability[sampled_indices]
            # This oracle proposal reads every old K row before sampling V.
            metadata_vectors = 0
        elif mode == "uniform":
            sampled_indices = torch.randint(
                old_end, (sample_count,), device=query.device
            )
            log_proposal = torch.full(
                (sample_count,),
                -math.log(old_end),
                device=query.device,
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unknown SANTA++ mode: {mode!r}")

        sampled_scores = full_key[sampled_indices].float() @ query * scale
        sampled_log_weights = sampled_scores - log_proposal
        if recent_tokens:
            maximum = torch.cat((sampled_log_weights, recent_scores)).max()
        else:
            maximum = sampled_log_weights.max()
        sampled_weights = torch.exp(sampled_log_weights - maximum) / sample_count
        recent_weights = torch.exp(recent_scores - maximum)

        normalized_for_ess = torch.exp(
            sampled_log_weights - sampled_log_weights.max()
        )
        ess = normalized_for_ess.sum().square() / normalized_for_ess.square().sum()
        ess_ratio = float((ess / sample_count).item())

        numerator = sampled_weights @ full_value[sampled_indices].float()
        if recent_tokens:
            numerator = numerator + recent_weights @ recent_values
        denominator = sampled_weights.sum() + recent_weights.sum()

        if mode == "santa":
            # The oracle proposal has already read every old K row, so the
            # sampled fetch adds V rows only; do not double-count sampled K.
            kv_cache_vectors = old_end + sample_count + 2 * recent_tokens
        else:
            kv_cache_vectors = 2 * (sample_count + recent_tokens)
        self.traffic.record(
            n_total=n_total,
            sampled_tokens=sample_count,
            recent_tokens=recent_tokens,
            kv_cache_vectors=kv_cache_vectors,
            metadata_key_vectors=metadata_vectors,
            ess_ratio=ess_ratio,
        )
        return numerator / denominator

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
        *,
        probe_seed: int,
    ) -> tuple[float, float, int, int, dict[str, float | None]]:
        prompt_tokens = int(input_ids.shape[1])
        sample_end = prompt_tokens - 1 - self.config.recent_window
        if sample_end < 2:
            raise ValueError(
                f"Prompt has {prompt_tokens} tokens, too short for recent_window="
                f"{self.config.recent_window}."
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

        strategy = self.config.probe_strategy
        n_probes = self.config.probe_queries
        max_idx = prompt_tokens - 1

        if strategy == "end_quarter":
            probe_start = int(self.config.probe_region_start_fraction * prompt_tokens)
            probe_candidates = np.arange(probe_start, max_idx)
            if probe_candidates.size < n_probes:
                raise ValueError(
                    f"Only {probe_candidates.size} probe positions available, "
                    f"need {n_probes}."
                )
            rng = np.random.RandomState(probe_seed)
            probe = np.sort(rng.choice(probe_candidates, size=n_probes, replace=False))

        elif strategy == "start":
            if max_idx < n_probes:
                raise ValueError(f"Prompt too short ({max_idx} tokens) for {n_probes} probes.")
            # First 64 tokens of prefill
            probe = np.arange(0, n_probes)

        elif strategy == "middle":
            if max_idx < n_probes:
                raise ValueError(f"Prompt too short ({max_idx} tokens) for {n_probes} probes.")
            # 64 tokens centered in middle of prefill
            mid = max_idx // 2
            start = max(0, mid - n_probes // 2)
            probe = np.arange(start, start + n_probes)

        elif strategy == "end":
            if max_idx < n_probes:
                raise ValueError(f"Prompt too short ({max_idx} tokens) for {n_probes} probes.")
            # Very last 64 tokens of prefill before the generated token boundary
            probe = np.arange(max_idx - n_probes, max_idx)

        elif strategy == "random":
            if max_idx < n_probes:
                raise ValueError(f"Prompt too short ({max_idx} tokens) for {n_probes} probes.")
            # Uniform random sampling across entire prefill
            rng = np.random.RandomState(probe_seed)
            probe = np.sort(rng.choice(max_idx, size=n_probes, replace=False))

        else:
            raise ValueError(f"Unknown probe strategy: {strategy!r}")
        probe_tensor = torch.as_tensor(
            probe, dtype=torch.long, device=input_ids.device
        )

        n_clusters = min(
            sample_end,
            max(2, sample_end // self.config.group_size),
        )
        km_cfg = self.config.kmeans

        # --- 1. Metric collections setup ---
        inertias, ch_scores, db_scores, sil_scores, entropies = [], [], [], [], []

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
                    fingerprints.std(
                        dim=0, keepdim=True, unbiased=False
                    )
                    + 1e-6
                )

                # Instantiated estimator instance to retain model attributes like inertia_
                km_model = SklearnLikeTorchMiniBatchKMeans(
                    n_clusters=n_clusters,
                    batch_size=km_cfg.batch_size,
                    n_init=km_cfg.n_init,
                    max_iter=km_cfg.max_iter,
                    tol=km_cfg.tol,
                    max_no_improvement=km_cfg.max_no_improvement,
                    init_size=km_cfg.init_size,
                    reassignment_ratio=km_cfg.reassignment_ratio,
                    random_state=km_cfg.random_state,
                )
                labels = km_model.fit_predict(fingerprints)

                # --- 2. Per-head metric evaluation ---
                fp_np = fingerprints.detach().cpu().numpy()
                lbl_np = labels.detach().cpu().numpy()
                n_unique = len(np.unique(lbl_np))

                inertia_val = getattr(km_model, "inertia_", None)
                if inertia_val is not None:
                    inertias.append(float(inertia_val))

                if n_unique > 1 and len(fp_np) > n_unique:
                    ch_scores.append(float(calinski_harabasz_score(fp_np, lbl_np)))
                    db_scores.append(float(davies_bouldin_score(fp_np, lbl_np)))

                    if len(fp_np) > 2000:
                        idx = np.random.choice(len(fp_np), 2000, replace=False)
                        sil_scores.append(float(silhouette_score(fp_np[idx], lbl_np[idx])))
                    else:
                        sil_scores.append(float(silhouette_score(fp_np, lbl_np)))

                counts = np.bincount(lbl_np)
                probs = counts[counts > 0] / len(lbl_np)
                entropies.append(float(-np.sum(probs * np.log2(probs))))

                self.summaries[(layer_id, kv_head)] = self._build_summary(
                    key_prefix, labels, n_clusters
                )
                del fingerprints, labels, key_prefix
            del query_rotated, query_raw

        _cuda_sync()
        clustering_seconds = time.perf_counter() - cluster_start

        # --- 3. Metric Aggregation ---
        metrics_summary = {
            "mean_inertia": float(np.mean(inertias)) if inertias else None,
            "mean_calinski_harabasz": float(np.mean(ch_scores)) if ch_scores else None,
            "mean_davies_bouldin": float(np.mean(db_scores)) if db_scores else None,
            "mean_silhouette": float(np.mean(sil_scores)) if sil_scores else None,
            "mean_cluster_entropy": float(np.mean(entropies)) if entropies else None,
        }

        return prefill_seconds, clustering_seconds, sample_end, n_clusters, metrics_summary

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

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_ids: set[int],
        stop_on_eos: bool,
        random_seed: int,
    ) -> SantaGeneration:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("SANTA++ generation expects input_ids with shape [1, T].")
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens < 2:
            raise ValueError("SANTA++ requires at least two prompt tokens.")

        self.clear()
        self.cache_capacity = prompt_tokens + max_new_tokens
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.reset_peak_memory_stats(input_ids.device)

        _cuda_sync()
        total_start = time.perf_counter()
        with self.patched():
            (
                prefill_seconds,
                clustering_seconds,
                sample_end,
                n_clusters,
                clustering_metrics,
            ) = self._dense_prefill_and_cluster(
                input_ids,
                probe_seed=random_seed,
            )
            # Match the notebook: discard the final prompt token from the cache,
            # then re-feed it as the first sparse decode query.
            self._trim_cache(prompt_tokens - 1)
            self.mode = self.config.mode
            self.traffic = DecodeTrafficTracker()
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)

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

        _cuda_sync()
        total_seconds = time.perf_counter() - total_start
        peak_allocated = torch.cuda.max_memory_allocated(input_ids.device)
        peak_reserved = torch.cuda.max_memory_reserved(input_ids.device)

        traffic = self.traffic.as_dict()
        metrics: dict[str, Any] = {
            "backend": "santapp",
            "mode": self.config.mode,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": len(generated),
            "max_new_tokens": max_new_tokens,
            "prefill_seconds": prefill_seconds,
            "clustering_seconds": clustering_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": total_seconds,
            "sampled_prefix_tokens": sample_end,
            "recent_window": self.config.recent_window,
            "samples_per_head": self.config.samples_per_head,
            "group_size": self.config.group_size,
            "probe_queries": self.config.probe_queries,
            "probe_strategy": self.config.probe_strategy,
            "nominal_clusters_per_kv_head": n_clusters,
            "custom_cache_gib": cache_bytes / (1024**3),
            "cluster_summary_gib": summary_bytes / (1024**3),
            "peak_allocated_gib": peak_allocated / (1024**3),
            "peak_reserved_gib": peak_reserved / (1024**3),
            **clustering_metrics,
            **traffic,
        }
        # Keep the exact draw-count metric first-class; the metadata-inclusive
        # value reproduces the notebook's K-centroid read-equivalent convention.
        self.clear()
        return SantaGeneration(token_ids=generated, metrics=metrics)


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
