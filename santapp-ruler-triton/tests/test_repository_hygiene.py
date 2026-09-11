from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _text_files():
    allowed = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".cff"}
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path


def test_no_person_specific_identifiers() -> None:
    pattern = re.compile("|".join(("ky" + "le", "lee" + "gion" + "kang")), re.IGNORECASE)
    hits = []
    for path in _text_files():
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_both_parent_policies_are_self_contained_and_probe_free() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "src").rglob("*.py")
    ).lower()
    assert not re.search(r"(?m)^\s*(?:from|import)\s+sklearn\b", source)
    assert "sklearnliketorchminibatchkmeans" in source
    assert "contiguous_parent_spans" in source
    assert "build_contiguous_team_summary" in source
    assert not re.search(r"(?<![a-z0-9_])probe_queries(?![a-z0-9_])", source)
    assert (ROOT / "src" / "santapp_ruler" / "attention" / "hierarchical.py").is_file()
    assert (ROOT / "src" / "santapp_ruler" / "attention" / "minibatch_kmeans.py").is_file()
    assert (ROOT / "src" / "santapp_ruler" / "attention" / "contiguous_teams.py").is_file()


def test_only_retained_backend_names_in_configs() -> None:
    for path in (ROOT / "configs").glob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        backends = data["generation"]["backends"]
        assert set(backends) <= {"sdpa", "santa", "santapp", "hierarchical"}


def test_repository_contains_no_experiment_outputs() -> None:
    forbidden_dirs = {
        "server_outputs",
        "ruler_100p_analysis",
        "local-smoke-results",
        "wandb",
        "runs",
    }
    present = {path.name for path in ROOT.rglob("*") if path.is_dir()}
    assert not (present & forbidden_dirs)
    assert not list(ROOT.rglob("*.ncu-rep"))
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "__pycache__/" in ignore
    assert ".pytest_cache/" in ignore


def test_only_one_generic_k8s_example_is_checked_in() -> None:
    subdirs = [path for path in (ROOT / "k8s").iterdir() if path.is_dir()]
    assert [path.name for path in subdirs] == ["example"]
    assert sorted(path.name for path in subdirs[0].glob("*.yaml")) == [
        "00-pvc.yaml",
        "01-configmap.yaml",
        "02-smoke-job.yaml",
        "03-copy-pod.yaml",
        "all.yaml",
    ]


def test_clean_repository_has_no_large_experiment_artifacts() -> None:
    large = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and path.stat().st_size > 2_000_000
    ]
    assert large == []
