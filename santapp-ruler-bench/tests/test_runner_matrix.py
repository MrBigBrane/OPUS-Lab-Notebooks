import json
from pathlib import Path
from types import SimpleNamespace

import torch

from santapp_ruler import runner
from santapp_ruler.config import load_config
from santapp_ruler.ruler.tasks import TASKS


def _write_one_example_per_task(root: Path) -> None:
    for task in TASKS:
        path = root / task / "validation.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "index": 0,
                    "input": f"prompt for {task}",
                    "outputs": ["answer"],
                }
            )
            + "\n",
            encoding="utf-8",
        )


class _FakeModel:
    def __init__(self) -> None:
        self.model = SimpleNamespace(layers=[object()])
        self.config = SimpleNamespace(
            model_type="qwen2",
            num_attention_heads=4,
            num_key_value_heads=2,
            _commit_hash="fake",
        )
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._parameter


class _FakeBundle:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.eos_token_ids: set[int] = set()

    @classmethod
    def load(cls, _config):
        return cls()

    def tokenize(self, _prompt: str) -> torch.Tensor:
        return torch.tensor([[1, 2, 3]], dtype=torch.long)

    def decode(self, _token_ids: list[int]) -> str:
        return "answer"

    def release_example_memory(self) -> None:
        return None


class _FakeBackend:
    def __init__(self, name: str) -> None:
        self.name = name

    def generate(
        self,
        _input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        stop_on_eos: bool,
        random_seed: int,
    ):
        del stop_on_eos, random_seed
        if self.name == "sdpa":
            mode = "dense_sdpa"
            gqa_kv, naive_kv, gqa_centroid, naive_centroid = 20, 80, 0, 0
            budget = None
            probe_policy = "not_applicable"
            exact_policy = "dense_all_tokens"
        elif self.name == "santa":
            mode = "santa"
            gqa_kv, naive_kv, gqa_centroid, naive_centroid = 12, 44, 0, 0
            budget = 128
            probe_policy = "not_applicable"
            exact_policy = "none"
        elif self.name == "santapp":
            mode = "guided"
            gqa_kv, naive_kv, gqa_centroid, naive_centroid = 10, 16, 2, 8
            budget = 128
            probe_policy = "last_prompt_tokens"
            exact_policy = "generated_suffix_only"
        elif self.name == "santapp_gumbel_topk":
            mode = "gumbel_cluster"
            gqa_kv, naive_kv, gqa_centroid, naive_centroid = 11, 18, 2, 8
            budget = 128
            probe_policy = "last_prompt_tokens"
            exact_policy = "generated_suffix_only"
        elif self.name == "team_sampler":
            mode = "team_sampler"
            gqa_kv, naive_kv, gqa_centroid, naive_centroid = 12, 20, 0, 0
            budget = 128
            probe_policy = "last_prompt_tokens"
            exact_policy = "generated_suffix_only"
        else:
            assert self.name == "team_gumbel_topk"
            mode = "team_gumbel_topk"
            gqa_kv, naive_kv, gqa_centroid, naive_centroid = 13, 22, 0, 0
            budget = 128
            probe_policy = "last_prompt_tokens"
            exact_policy = "generated_suffix_only"
        metrics = {
            "backend": self.name,
            "mode": mode,
            "sampling_scheme": "test",
            "importance_correction": "test",
            "sampling_unit": (
                "cluster"
                if mode == "gumbel_cluster"
                else "team" if mode == "team_gumbel_topk" else "token"
            ),
            "samples_per_head": budget,
            "nominal_sample_budget_per_head": budget,
            "clusters_per_head": 8 if mode == "gumbel_cluster" else None,
            "group_size": 16 if mode in {"guided", "gumbel_cluster"} else None,
            "parent_size": 16 if mode.startswith("team_") else None,
            "representatives_per_parent": 4 if mode.startswith("team_") else None,
            "probe_queries": 64 if mode != "dense_sdpa" and mode != "santa" else None,
            "routing_key_type": (
                "actual_team_leader" if mode.startswith("team_") else "centroid"
                if mode in {"guided", "gumbel_cluster"} else "none"
            ),
            "probe_policy": probe_policy,
            "exact_token_policy": exact_policy,
            "prompt_exact_tail_tokens": 0,
            "initial_growing_exact_tokens": 0,
            "prompt_tokens": 3,
            "generated_tokens": 1,
            "max_new_tokens": max_new_tokens,
            "prefill_seconds": 0.01,
            "clustering_seconds": 0.01 if mode in {"guided", "gumbel_cluster"} else 0.0,
            "decode_seconds": 0.01,
            "total_seconds": 0.02,
            "peak_allocated_gib": 0.0,
            "peak_reserved_gib": 0.0,
            "decode_attention_head_calls": 4,
            "decode_gqa_group_calls": 2,
            "decode_dense_gqa_kv_vectors": 20,
            "decode_dense_naive_kv_vectors": 80,
            "decode_gqa_kv_vectors_read": gqa_kv,
            "decode_naive_kv_vectors_read": naive_kv,
            "decode_gqa_centroid_key_vectors_read": gqa_centroid,
            "decode_naive_centroid_key_vectors_read": naive_centroid,
            "decode_gqa_total_vectors_read": gqa_kv + gqa_centroid,
            "decode_naive_total_vectors_read": naive_kv + naive_centroid,
            "decode_gqa_kv_access_pct": 100.0 * gqa_kv / 20,
            "decode_gqa_centroid_access_pct": 100.0 * gqa_centroid / 20,
            "decode_gqa_total_access_pct": 100.0 * (gqa_kv + gqa_centroid) / 20,
            "decode_naive_kv_access_pct": 100.0 * naive_kv / 80,
            "decode_naive_centroid_access_pct": 100.0 * naive_centroid / 80,
            "decode_naive_total_access_pct": 100.0 * (naive_kv + naive_centroid) / 80,
            "mean_sampled_token_draws_per_head_call": 1.0,
            "mean_sampled_token_rows_per_head_call": 1.0,
            "mean_unique_sampled_tokens_per_gqa_group_call": 1.0,
            "mean_exact_tokens_per_gqa_group_call": 0.0,
            "mean_total_tokens_per_head_call": 3.0,
            "mean_selected_clusters_per_head_call": 8.0 if mode == "gumbel_cluster" else 0.0,
            "selected_cluster_inclusion_probability_count": 0,
        }
        return SimpleNamespace(token_ids=[1], prediction="answer", metrics=metrics)


def test_six_methods_cross_all_13_tasks(tmp_path: Path, monkeypatch, capsys):
    data_root = tmp_path / "data"
    _write_one_example_per_task(data_root)
    config = load_config(
        "configs/all_tasks_8k.yaml",
        overrides=[
            "benchmark.context_length=256",
            "benchmark.prompts_per_task=1",
            "benchmark.data.source=local",
            f"benchmark.data.local_root={str(data_root)!r}",
            "generation.max_new_tokens=1",
            "grading.bootstrap_resamples=20",
            "output.run_name=null",
            "output.save_full_prompts=false",
            "output.save_prediction_inputs=false",
        ],
    )

    monkeypatch.setattr(runner, "ModelBundle", _FakeBundle)
    monkeypatch.setattr(runner, "SdpaBackend", lambda _bundle: _FakeBackend("sdpa"))
    monkeypatch.setattr(
        runner,
        "SantaBackend",
        lambda _bundle, _config: _FakeBackend("santa"),
    )
    monkeypatch.setattr(
        runner,
        "SantaPlusBackend",
        lambda _bundle, _config, *, name="santapp": _FakeBackend(name),
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "test-gpu")

    run_dir = runner.run_benchmark(config, explicit_run_dir=tmp_path / "run")
    capsys.readouterr()

    records = []
    for backend in config.generation.backends:
        for task in TASKS:
            path = run_dir / "predictions" / backend / f"{task}.jsonl"
            assert path.is_file()
            lines = path.read_text(encoding="utf-8").splitlines()
            assert all("input" not in json.loads(line) for line in lines)
            records.extend(lines)
    assert len(records) == 78
    selected_rows = [
        json.loads(line)
        for line in (run_dir / "selected_prompts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all("input" not in row for row in selected_rows)

    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["expected_records"] == 78
    assert status["records_generated_this_process"] == 78

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["backends"]) == set(config.generation.backends)
    assert set(summary["comparisons"]["vs_sdpa"]) == {
        "santa",
        "santapp",
        "santapp_gumbel_topk",
        "team_sampler",
        "team_gumbel_topk",
    }
