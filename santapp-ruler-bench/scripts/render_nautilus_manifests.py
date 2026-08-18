#!/usr/bin/env python3
"""Render shareable Nautilus PVC, prefetch, smoke, sweep, and copy manifests."""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from santapp_ruler.k8s_matrix import load_matrix


TEMPLATES = (
    "00-santapp-ruler-pvc.yaml",
    "01-santapp-ruler-prefetch-job.yaml",
    "02-santapp-ruler-smoke-job.yaml",
    "03-santapp-ruler-indexed-job.yaml",
    "04-santapp-ruler-results-copy-pod.yaml",
)
DEFAULT_MATRIX_IN_IMAGE = (
    "/workspace/k8s/matrices/default-2tasks-6methods-100p-8k.yaml"
)
DEFAULT_SMOKE_CONFIG_IN_IMAGE = "/workspace/configs/smoke_all_backends_8k.yaml"
DEFAULT_RESULTS_ROOT = "/mnt/results/yourname-santapp-ruler"
DEFAULT_EXPERIMENT = "santapp-ruler-default-2tasks-6methods-100p-8k"
DEFAULT_GPU_PRODUCTS = (
    "Tesla-V100-SXM2-16GB",
    "Tesla-V100-PCIE-16GB",
    "NVIDIA-RTX-A4000",
)
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument(
        "--smoke-config",
        type=Path,
        default=Path("configs/smoke_all_backends_8k.yaml"),
        help="In-repository config run by the one-pod GPU smoke Job.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parallelism", type=int, default=5)
    parser.add_argument(
        "--image",
        default="your-registry/santapp-ruler:latest",
    )
    parser.add_argument("--namespace", default="ucsb-opus-lab")
    parser.add_argument("--resource-prefix", default="yourname-santapp-ruler")
    parser.add_argument("--owner", default="yourname")
    parser.add_argument("--storage", default="50Gi")
    parser.add_argument("--storage-class", default="rook-cephfs")
    parser.add_argument("--storage-region", default="us-west")
    parser.add_argument(
        "--results-root",
        default=None,
        help=(
            "PVC path used for caches and runs. Defaults to "
            "/mnt/results/<resource-prefix>."
        ),
    )
    parser.add_argument(
        "--gpu-product",
        action="append",
        dest="gpu_products",
        help=(
            "Allowed nvidia.com/gpu.product value. Repeat for a pool. Defaults "
            "to 16GB V100 SXM2/PCIe plus RTX A4000."
        ),
    )
    parser.add_argument("--gpu-resource", default="nvidia.com/gpu")
    parser.add_argument("--backoff-limit-per-index", type=int, default=2)
    parser.add_argument("--backoff-limit", type=int, default=20)
    parser.add_argument("--max-failed-indexes", type=int, default=3)
    return parser


def _validate_dns_label(name: str, *, field: str) -> None:
    if len(name) > 63 or _DNS_LABEL.fullmatch(name) is None:
        raise ValueError(f"{field} must be a Kubernetes DNS label: {name!r}")


def _repo_image_path(path: Path, *, repo_root: Path, field: str) -> str:
    expanded = path.expanduser()
    resolved = (repo_root / expanded).resolve() if not expanded.is_absolute() else expanded.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(
            f"{field} must be inside the repository so Docker COPY includes it."
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return f"/workspace/{relative.as_posix()}"


def _patch_text(
    value: str,
    *,
    matrix_in_image: str,
    smoke_config_in_image: str,
    results_root: str,
    experiment: str,
) -> str:
    return (
        value.replace(DEFAULT_MATRIX_IN_IMAGE, matrix_in_image)
        .replace(DEFAULT_SMOKE_CONFIG_IN_IMAGE, smoke_config_in_image)
        .replace(DEFAULT_RESULTS_ROOT, results_root)
        .replace(DEFAULT_EXPERIMENT, experiment)
    )


def _patch_container(
    container: dict[str, Any],
    *,
    image: str,
    matrix_in_image: str,
    smoke_config_in_image: str,
    results_root: str,
    experiment: str,
) -> None:
    container["image"] = image
    for key in ("command", "args"):
        values = container.get(key)
        if isinstance(values, list):
            container[key] = [
                _patch_text(
                    value,
                    matrix_in_image=matrix_in_image,
                    smoke_config_in_image=smoke_config_in_image,
                    results_root=results_root,
                    experiment=experiment,
                )
                if isinstance(value, str)
                else value
                for value in values
            ]
    for env in container.get("env", []):
        value = env.get("value")
        if isinstance(value, str):
            env["value"] = _patch_text(
                value,
                matrix_in_image=matrix_in_image,
                smoke_config_in_image=smoke_config_in_image,
                results_root=results_root,
                experiment=experiment,
            )
        if env.get("name") == "CONTAINER_IMAGE":
            env["value"] = image


def _pod_spec(document: dict[str, Any]) -> dict[str, Any]:
    if document["kind"] == "Job":
        return document["spec"]["template"]["spec"]
    if document["kind"] == "Pod":
        return document["spec"]
    raise ValueError(f"No pod spec for {document['kind']}")


def _upsert_expression(
    terms: Iterable[dict[str, Any]],
    *,
    key: str,
    values: list[str],
) -> None:
    terms = list(terms)
    if not terms:
        raise ValueError("Template node affinity has no nodeSelectorTerms.")
    for term in terms:
        expressions = term.setdefault("matchExpressions", [])
        for expression in expressions:
            if expression.get("key") == key:
                expression.update({"operator": "In", "values": values})
                break
        else:
            expressions.append({"key": key, "operator": "In", "values": values})


def _set_storage_region(spec: dict[str, Any], region: str) -> None:
    affinity = spec.setdefault("affinity", {}).setdefault("nodeAffinity", {})
    required = affinity.setdefault(
        "requiredDuringSchedulingIgnoredDuringExecution",
        {"nodeSelectorTerms": [{"matchExpressions": []}]},
    )
    _upsert_expression(
        required.setdefault("nodeSelectorTerms", [{"matchExpressions": []}]),
        key="topology.kubernetes.io/region",
        values=[region],
    )


def _set_gpu_affinity(
    spec: dict[str, Any], *, products: list[str], preferred_region: str
) -> None:
    affinity = spec.setdefault("affinity", {}).setdefault("nodeAffinity", {})
    required = affinity.setdefault(
        "requiredDuringSchedulingIgnoredDuringExecution",
        {"nodeSelectorTerms": [{"matchExpressions": []}]},
    )
    terms = required.setdefault("nodeSelectorTerms", [{"matchExpressions": []}])
    _upsert_expression(terms, key="nvidia.com/gpu.product", values=products)
    _upsert_expression(
        terms,
        key="nvidia.com/gpu.sharing-strategy",
        values=["none"],
    )
    affinity["preferredDuringSchedulingIgnoredDuringExecution"] = [
        {
            "weight": 100,
            "preference": {
                "matchExpressions": [
                    {
                        "key": "topology.kubernetes.io/region",
                        "operator": "In",
                        "values": [preferred_region],
                    }
                ]
            },
        }
    ]


def _set_gpu_resource(container: dict[str, Any], resource: str) -> None:
    resources = container.setdefault("resources", {})
    for section_name in ("requests", "limits"):
        section = resources.setdefault(section_name, {})
        for key in list(section):
            if key.startswith("nvidia.com/"):
                del section[key]
        section[resource] = "1"


def _set_labels(
    document: dict[str, Any],
    *,
    app_name: str,
    component: str,
    experiment: str,
    owner: str,
) -> None:
    labels = {
        "app.kubernetes.io/name": app_name,
        "app.kubernetes.io/component": component,
        "app.kubernetes.io/part-of": experiment,
        "owner": owner,
    }
    document.setdefault("metadata", {})["labels"] = dict(labels)
    if document["kind"] == "Job":
        document["spec"]["template"].setdefault("metadata", {})["labels"] = dict(
            labels
        )
    if component == "benchmark":
        constraints = document["spec"]["template"]["spec"].get(
            "topologySpreadConstraints", []
        )
        for constraint in constraints:
            constraint["labelSelector"] = {
                "matchLabels": {
                    "app.kubernetes.io/name": app_name,
                    "app.kubernetes.io/component": component,
                }
            }


def _role_for_template(filename: str) -> tuple[str, str]:
    if "pvc" in filename:
        return "pvc", "storage"
    if "prefetch" in filename:
        return "prefetch", "prefetch"
    if "smoke" in filename:
        return "smoke", "smoke"
    if "indexed-job" in filename:
        return "sweep", "benchmark"
    if "copy" in filename:
        return "copy", "copy"
    raise ValueError(f"Unknown template role: {filename}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.parallelism <= 0:
        raise ValueError("parallelism must be positive.")
    if args.backoff_limit_per_index < 0:
        raise ValueError("backoff-limit-per-index cannot be negative.")
    if args.backoff_limit <= 0:
        raise ValueError("backoff-limit must be positive.")
    if args.max_failed_indexes < 0:
        raise ValueError("max-failed-indexes cannot be negative.")
    _validate_dns_label(args.resource_prefix, field="resource-prefix")
    _validate_dns_label(args.owner, field="owner")
    _validate_dns_label(args.namespace, field="namespace")

    results_root = args.results_root or f"/mnt/results/{args.resource_prefix}"
    if not results_root.startswith("/mnt/results/") or results_root.endswith("/"):
        raise ValueError(
            "results-root must be an absolute child of /mnt/results without a "
            "trailing slash."
        )

    products = list(dict.fromkeys(args.gpu_products or DEFAULT_GPU_PRODUCTS))
    if not products or any(not value.strip() for value in products):
        raise ValueError("At least one non-empty gpu-product is required.")
    if "/" not in args.gpu_resource:
        raise ValueError(
            "gpu-resource should be an extended resource such as nvidia.com/gpu."
        )

    repo_root = Path(__file__).resolve().parents[1]
    matrix_path = args.matrix.expanduser()
    if not matrix_path.is_absolute():
        matrix_path = repo_root / matrix_path
    matrix = load_matrix(matrix_path)
    matrix_in_image = _repo_image_path(
        matrix.source_path, repo_root=repo_root, field="matrix"
    )
    smoke_config_in_image = _repo_image_path(
        args.smoke_config, repo_root=repo_root, field="smoke-config"
    )

    names = {
        "pvc": f"{args.resource_prefix}-rwx",
        "prefetch": f"{args.resource_prefix}-prefetch",
        "smoke": f"{args.resource_prefix}-smoke",
        "sweep": f"{args.resource_prefix}-sweep",
        "copy": f"{args.resource_prefix}-results-copy",
    }
    for role, name in names.items():
        _validate_dns_label(name, field=f"rendered {role} name")

    templates_root = repo_root / "k8s"
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    for filename in TEMPLATES:
        source = templates_root / filename
        document = copy.deepcopy(yaml.safe_load(source.read_text(encoding="utf-8")))
        document["metadata"]["namespace"] = args.namespace
        role, component = _role_for_template(filename)

        if role == "pvc":
            document["metadata"]["name"] = names[role]
            document["spec"]["resources"]["requests"]["storage"] = args.storage
            document["spec"]["storageClassName"] = args.storage_class
        else:
            document["metadata"]["name"] = names[role]
            spec = _pod_spec(document)
            for container in [
                *spec.get("initContainers", []),
                *spec.get("containers", []),
            ]:
                _patch_container(
                    container,
                    image=args.image,
                    matrix_in_image=matrix_in_image,
                    smoke_config_in_image=smoke_config_in_image,
                    results_root=results_root,
                    experiment=matrix.experiment,
                )
            for volume in spec.get("volumes", []):
                claim = volume.get("persistentVolumeClaim")
                if isinstance(claim, dict):
                    claim["claimName"] = names["pvc"]

            if role == "sweep":
                document["spec"]["completions"] = matrix.completion_count
                document["spec"]["parallelism"] = min(
                    args.parallelism, matrix.completion_count
                )
                document["spec"]["backoffLimitPerIndex"] = (
                    args.backoff_limit_per_index
                )
                document["spec"]["backoffLimit"] = args.backoff_limit
                document["spec"]["maxFailedIndexes"] = min(
                    args.max_failed_indexes, max(0, matrix.completion_count - 1)
                )

            if role in {"smoke", "sweep"}:
                _set_gpu_affinity(
                    spec,
                    products=products,
                    preferred_region=args.storage_region,
                )
                _set_gpu_resource(spec["containers"][0], args.gpu_resource)
            else:
                _set_storage_region(spec, args.storage_region)

        _set_labels(
            document,
            app_name=args.resource_prefix,
            component=component,
            experiment=matrix.experiment,
            owner=args.owner,
        )
        destination = output / filename
        destination.write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
        print(destination)

    print(
        f"Rendered {matrix.completion_count} completions with parallelism "
        f"{min(args.parallelism, matrix.completion_count)}; GPU products: "
        + ", ".join(products)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
