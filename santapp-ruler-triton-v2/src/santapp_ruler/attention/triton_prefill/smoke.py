"""Download-free GPU/API smokes and an optional real-model RULER smoke.

The matrix exercises the original PyTorch decode estimator, not Triton decode.
Algorithmic validation runs outside measured preparation calls.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import statistics
import time

import torch

from ..prefill_runtime import require_triton_environment
from ..minibatch_kmeans import SklearnLikeTorchMiniBatchKMeans
from .prefill_kmeans import TritonMiniBatchKMeans
from .prefill_teams import build_team_summary_triton, build_contiguous_team_summary_triton
from .prefill_algorithmic_validation import assert_team_algorithm, assert_kmeans_health


DEFAULT_BUDGETS = [4, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def timed_call(fn, repeats=2):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    first = (time.perf_counter() - start) * 1000
    times = []
    for _ in range(repeats):
        del result
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return result, {"first_call_wall_ms": first, "repeat_wall_ms": times,
                    "repeat_median_wall_ms": statistics.median(times) if times else None,
                    "timing_scope": "single_KV_head_preparation_API_not_full_model"}


@torch.inference_mode()
def check_decode_budgets(summary, keys, values, *, policy, budgets, parent_size=16, representatives=4):
    """Call the real, unmodified estimator at suffix lengths zero and one.

    No test-side reimplementation of selection or HT correction. In the
    take-all case compare against ordinary dense attention on the same keys.
    """
    from ..santapp import SantappEngine
    from ..hierarchical import HierarchicalWholeTeamEngine
    from ...config import SantappConfig, HierarchicalConfig
    if policy == "santapp":
        engine_cls, config_cls = SantappEngine, SantappConfig
    else:
        engine_cls, config_cls = HierarchicalWholeTeamEngine, HierarchicalConfig
    engine = engine_cls.__new__(engine_cls)
    engine.head_dim = keys.shape[1]
    engine.sample_end = keys.shape[0]
    engine.summaries = {(0, 0): summary}
    nominal = parent_size // representatives
    rows = []
    query = torch.linspace(-0.7, 0.8, keys.shape[1], device=keys.device).float()
    for s in budgets:
        if s <= 0 or s % nominal:
            raise ValueError(f"Nominal S={s} must be positive and divisible by {nominal}")
        engine.config = config_cls(parent_size=parent_size, representatives_per_parent=representatives,
                                   samples_per_head=s, prefill_backend="triton")
        for exact in (0, 1):
            if exact:
                full_keys = torch.cat((keys, keys[:1] * 0.9))
                full_values = torch.cat((values, values[:1] * 0.7))
            else:
                full_keys, full_values = keys, values
            torch.manual_seed(123)
            result = engine._approximate_attention(query, full_keys, full_values, 0, 0)
            selected = min(s // nominal, summary.num_teams)
            assert result.selected_teams == selected
            assert result.output.shape == (keys.shape[1],)
            assert bool(torch.isfinite(result.output).all())
            ids = result.sampled_indices
            assert len(ids) == len(torch.unique(ids))
            assert int(ids.min()) >= 0 and int(ids.max()) < len(keys)
            p = result.inclusion_probabilities
            assert p is not None and len(p) == selected
            assert bool(((p > 0) & (p <= 1)).all())
            take_all = selected == summary.num_teams
            error = None
            if take_all:
                assert len(ids) == len(keys)
                expected = torch.softmax(full_keys.float() @ query / math.sqrt(keys.shape[1]), 0) @ full_values.float()
                error = float((result.output - expected).abs().max())
                torch.testing.assert_close(result.output, expected, rtol=2e-3, atol=5e-4)
            rows.append({"S": s, "requested_teams": s // nominal,
                         "selected_teams": selected, "selected_rows": len(ids),
                         "generated_exact_rows": exact, "take_all": take_all,
                         "take_all_max_abs_error": error, "status": "passed"})
    return rows


@torch.inference_mode()
def quick_smoke() -> dict:
    env = require_triton_environment()
    from .prefill_kmeans_ops import TritonKMeansOps
    x = torch.randn(257, 128, device="cuda").half().float()
    estimator = TritonMiniBatchKMeans(16, batch_size=64, max_iter=2, max_no_improvement=2)
    labels = estimator.fit_predict(x)
    assert_kmeans_health(estimator, labels, tokens=len(x), dimensions=x.shape[1], clusters=16)
    generic = build_team_summary_triton(x, labels, parent_cluster_count=16)
    a = assert_team_algorithm(generic, x, labels, configured_parent_count=16)
    contiguous = build_contiguous_team_summary_triton(x)
    b = assert_team_algorithm(contiguous, x, torch.arange(len(x), device="cuda") // 16,
                              configured_parent_count=17)
    # A genuine >128-row parent forces the streaming path, including a short sibling.
    irregular = torch.zeros(len(x), dtype=torch.long, device="cuda")
    irregular[-1] = 2
    large = build_team_summary_triton(x, irregular, parent_cluster_count=4)
    c = assert_team_algorithm(large, x, irregular, configured_parent_count=4)
    # Force rarely-taken update/reassignment kernels, not merely imports.
    ops = TritonKMeansOps()
    centers = x[:16].clone()
    assigned, _ = ops.assign(x, centers)
    updated, counts = ops.update(x, assigned, centers, torch.ones(16, device="cuda"))
    mask = torch.arange(16, device="cuda") < 2
    ids = torch.tensor([3, 9], device="cuda")
    ops.reassign_(updated, counts, x, mask, ids)
    torch.testing.assert_close(updated[:2], x[ids], rtol=0, atol=0)
    out = check_decode_budgets(contiguous, x, x.sin(), policy="hierarchical",
                              budgets=[4, 128, 512, 32768])
    torch.cuda.synchronize()
    return {"status": "passed", "environment": env,
            "checks": {"generic": a, "contiguous_tail": b, "streaming": c},
            "budget_checks": out, "reference_replay": False}


@torch.inference_mode()
def matrix_smoke(contexts=(8192, 32768), budgets=DEFAULT_BUDGETS,
                 dtypes=("float16", "bfloat16"), repeats=2, compare_torch=False,
                 report_callback=None) -> dict:
    from ..teams import build_team_summary
    from ..contiguous_teams import build_contiguous_team_summary
    require_triton_environment()
    report = {"status": "running", "cases": [], "reference_replay": False,
              "decode_implementation": "unchanged_dual_parent_PyTorch"}
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        for n in contexts:
            for dtype_name in dtypes:
                dtype = getattr(torch, dtype_name)
                if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    raise RuntimeError("Requested BF16 smoke is not supported by this device/build")
                torch.manual_seed(981)
                x = torch.randn(n, 128, device="cuda", dtype=dtype)
                v = torch.randn_like(x)
                clusters = min(n, max(2, n // 16))
                print(f"MATRIX N={n} dtype={dtype_name} clusters={clusters}", flush=True)
                km = TritonMiniBatchKMeans(clusters)  # production defaults, not reduced iterations
                labels, kt = timed_call(lambda: km.fit_predict(x), repeats)
                assert_kmeans_health(km, labels, tokens=n, dimensions=128, clusters=clusters)
                generic, gt = timed_call(lambda: build_team_summary_triton(x, labels, parent_cluster_count=clusters), repeats)
                contiguous, ct = timed_call(lambda: build_contiguous_team_summary_triton(x), repeats)
                positional = torch.arange(n, device="cuda") // 16
                row = {"tokens": n, "dtype": dtype_name, "kmeans_timing": kt,
                       "generic_teams_timing": gt, "contiguous_teams_timing": ct,
                       "kmeans_inertia": km.inertia_, "kmeans_steps": km.n_steps_,
                       "kmeans_parameters": {"n_clusters": clusters, "batch_size": 4096,
                                             "max_iter": 100, "n_init": 1, "random_state": 0},
                       "generic_gate": assert_team_algorithm(generic, x, labels, configured_parent_count=clusters),
                       "contiguous_gate": assert_team_algorithm(contiguous, x, positional,
                                                                 configured_parent_count=(n + 15) // 16)}
                row["generic_decode_budgets"] = check_decode_budgets(generic, x, v, policy="santapp", budgets=budgets)
                row["contiguous_decode_budgets"] = check_decode_budgets(contiguous, x, v, policy="hierarchical", budgets=budgets)
                if compare_torch:
                    reference_km = SklearnLikeTorchMiniBatchKMeans(clusters)
                    ref_labels, t = timed_call(lambda: reference_km.fit_predict(x), repeats)
                    # Same frozen native labels isolate the team-construction speedup.
                    _, team_t = timed_call(lambda: build_team_summary(x, labels, parent_cluster_count=clusters, representatives_per_parent=4), repeats)
                    _, contig_t = timed_call(lambda: build_contiguous_team_summary(x, parent_size=16, representatives_per_parent=4), repeats)
                    row["torch_reference"] = {"kmeans_timing": t, "generic_teams_frozen_native_labels_timing": team_t,
                                              "contiguous_teams_timing": contig_t,
                                              "kmeans_inertia": reference_km.inertia_,
                                              "kmeans_steps": reference_km.n_steps_,
                                              "raw_label_agreement_diagnostic": float((labels == ref_labels).float().mean())}
                    del reference_km, ref_labels
                report["cases"].append(row)
                if report_callback:
                    report_callback(report)
                print(f"PASS N={n} {dtype_name}: both policies, {len(budgets)} budgets, 2 exact-suffix lengths", flush=True)
                del km, labels, generic, contiguous, x, v
        report["status"] = "passed"
        return report
    finally:
        torch.set_float32_matmul_precision(old)


@torch.inference_mode()
def model_smoke(config_path: Path, contexts, budgets, *, max_new_tokens=2,
                policies=("santapp", "hierarchical"), report_callback=None) -> dict:
    """Use actual RULER examples, the existing model loader and generation backends.

    One selected niah_single_1 prompt per context; no synthetic prompt substitution,
    truncation, altered RoPE range, or change to existing result scoring.
    """
    from ...config import load_config
    from ...backends import ModelBundle, SantappBackend, HierarchicalBackend
    from ...data import select_examples
    require_triton_environment()
    config = load_config(config_path)
    bundle = ModelBundle.load(config.model)
    report = {"status": "running", "cases": [], "model": config.model.name,
              "model_revision": config.model.revision,
              "scope": "real_model_generation_smoke_not_RULER_accuracy_benchmark"}
    for context in contexts:
        cfg = copy.deepcopy(config)
        cfg.benchmark.context_length = context
        cfg.benchmark.tasks = ["niah_single_1"]
        cfg.benchmark.prompts_per_task = 1
        selected = select_examples(cfg.benchmark)["niah_single_1"][0]
        input_ids = bundle.tokenize(selected.input)
        tokens = int(input_ids.shape[1])
        limit = int(bundle.model.config.max_position_embeddings)
        if tokens + max_new_tokens > min(context, limit):
            raise ValueError(f"Post-framing {tokens} prompt + {max_new_tokens} output tokens exceed "
                             f"context/model limit {min(context, limit)}; no truncation was performed.")
        for s in budgets:
            for policy in policies:
                c = copy.deepcopy(cfg)
                sparse = getattr(c, policy)
                sparse.prefill_backend, sparse.samples_per_head = "triton", s
                c.validate()
                klass = SantappBackend if policy == "santapp" else HierarchicalBackend
                backend = klass(bundle, sparse)
                print(f"MODEL context={context} actual_prompt={tokens} policy={policy} S={s}", flush=True)
                result = backend.generate(input_ids, max_new_tokens=max_new_tokens,
                                          stop_on_eos=False, random_seed=0)
                assert len(result.token_ids) == max_new_tokens
                m = result.metrics
                assert m["team_builder_backend"] == "triton"
                assert not m["prefill_reference_replay"]
                assert m["samples_per_head"] == s
                assert m["requested_teams_per_head"] == s // (sparse.parent_size // sparse.representatives_per_parent)
                assert m["prompt_exact_tail_tokens"] == 0
                assert m["initial_generated_exact_tokens"] == 0
                # Every real layer/GQA group has seen suffix lengths 0..M-1.
                assert abs(m["mean_exact_tokens_per_gqa_group_call"] - (max_new_tokens - 1) / 2) < 1e-6
                report["cases"].append({"context_length": context, "uid": selected.uid,
                                        "policy": policy, "S": s, "token_ids": result.token_ids,
                                        "prediction": result.prediction, "metrics": m, "status": "passed"})
                if report_callback:
                    report_callback(report)
                del backend
                bundle.release_example_memory()
    report["status"] = "passed"
    return report
