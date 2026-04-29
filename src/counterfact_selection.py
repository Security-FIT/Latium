from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import datasets

DEFAULT_DATASET = "azhx/counterfact"
DEFAULT_SPLIT = "train"


def load_counterfact_split(dataset_name: str = DEFAULT_DATASET, split: str = DEFAULT_SPLIT):
    return datasets.load_dataset(dataset_name, split=split)


def _normalize_index_list(indices: Sequence[int | str]) -> list[int]:
    out: list[int] = []
    for raw in indices:
        idx = int(raw)
        if idx < 0:
            raise ValueError(f"CounterFact row indices must be non-negative, got {idx}")
        out.append(idx)
    if len(set(out)) != len(out):
        raise ValueError("CounterFact row indices must be unique.")
    return out


def manifest_digest(payload: Mapping[str, Any]) -> str:
    canonical = {
        "dataset": str(payload.get("dataset", DEFAULT_DATASET)),
        "split": str(payload.get("split", DEFAULT_SPLIT)),
        "seed": payload.get("seed"),
        "count": int(payload.get("count", 0) or 0),
        "indices": [int(v) for v in payload.get("indices", [])],
        "case_ids": [int(v) for v in payload.get("case_ids", [])],
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def counterfact_item_to_case(item: Mapping[str, Any], dataset_index: int) -> dict[str, Any]:
    rw = item["requested_rewrite"]
    return {
        "dataset_index": int(dataset_index),
        "case_id": int(item.get("case_id", dataset_index)),
        "subject": rw["subject"],
        "target_new_str": rw["target_new"]["str"],
        "target_true_str": rw["target_true"]["str"],
        "fact_tuple": (
            rw["prompt"],
            rw["subject"],
            " " + rw["target_new"]["str"],
            " " + rw["target_true"]["str"],
        ),
        "paraphrase_prompts": item.get("paraphrase_prompts", []) or [],
        "neighborhood_prompts": item.get("neighborhood_prompts", []) or [],
    }


def load_cases_by_indices(
    indices: Sequence[int | str],
    *,
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    dataset: Any = None,
) -> list[dict[str, Any]]:
    normalized = _normalize_index_list(indices)
    ds = dataset if dataset is not None else load_counterfact_split(dataset_name=dataset_name, split=split)
    return [counterfact_item_to_case(ds[idx], idx) for idx in normalized]


def load_cases_by_range(
    n_tests: int,
    *,
    start_idx: int = 0,
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    dataset: Any = None,
) -> list[dict[str, Any]]:
    if n_tests < 0:
        raise ValueError(f"n_tests must be non-negative, got {n_tests}")
    start_idx = int(start_idx)
    if start_idx < 0:
        raise ValueError(f"start_idx must be non-negative, got {start_idx}")

    ds = dataset if dataset is not None else load_counterfact_split(dataset_name=dataset_name, split=split)
    cases: list[dict[str, Any]] = []
    for idx in range(start_idx, min(len(ds), start_idx + int(n_tests))):
        cases.append(counterfact_item_to_case(ds[idx], idx))
    return cases


def build_case_manifest(
    indices: Sequence[int | str],
    *,
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    seed: int | None = None,
    dataset: Any = None,
) -> dict[str, Any]:
    normalized = _normalize_index_list(indices)
    ds = dataset if dataset is not None else load_counterfact_split(dataset_name=dataset_name, split=split)
    if normalized and max(normalized) >= len(ds):
        raise IndexError(
            f"CounterFact row index {max(normalized)} is out of bounds for split size {len(ds)}."
        )
    case_ids = [int(ds[idx].get("case_id", idx)) for idx in normalized]
    payload: dict[str, Any] = {
        "dataset": str(dataset_name),
        "split": str(split),
        "seed": None if seed is None else int(seed),
        "count": len(normalized),
        "indices": normalized,
        "case_ids": case_ids,
    }
    payload["manifest_hash"] = manifest_digest(payload)
    return payload


def generate_random_case_manifest(
    *,
    count: int,
    seed: int,
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    dataset: Any = None,
) -> dict[str, Any]:
    count = int(count)
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    ds = dataset if dataset is not None else load_counterfact_split(dataset_name=dataset_name, split=split)
    if count > len(ds):
        raise ValueError(f"Requested {count} random CounterFact cases, but split only has {len(ds)} rows.")
    rng = random.Random(int(seed))
    indices = rng.sample(range(len(ds)), count)
    return build_case_manifest(indices, dataset_name=dataset_name, split=split, seed=int(seed), dataset=ds)


def load_case_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, MutableMapping):
        raise ValueError(f"Case manifest must be a JSON object: {manifest_path}")

    required = {"dataset", "split", "count", "indices", "case_ids"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Case manifest {manifest_path} is missing required keys: {', '.join(missing)}")

    payload["indices"] = _normalize_index_list(payload["indices"])
    payload["case_ids"] = [int(v) for v in payload["case_ids"]]
    payload["count"] = int(payload["count"])
    if payload["count"] != len(payload["indices"]):
        raise ValueError(
            f"Case manifest {manifest_path} count={payload['count']} does not match "
            f"indices length {len(payload['indices'])}."
        )
    if len(payload["case_ids"]) != payload["count"]:
        raise ValueError(
            f"Case manifest {manifest_path} has {len(payload['case_ids'])} case_ids for count={payload['count']}."
        )

    expected_hash = manifest_digest(payload)
    payload["manifest_hash"] = str(payload.get("manifest_hash") or expected_hash)
    if payload["manifest_hash"] != expected_hash:
        raise ValueError(
            f"Case manifest {manifest_path} has manifest_hash={payload['manifest_hash']}, "
            f"expected {expected_hash}."
        )
    payload["dataset"] = str(payload["dataset"])
    payload["split"] = str(payload["split"])
    if payload.get("seed") is not None:
        payload["seed"] = int(payload["seed"])
    return dict(payload)


def write_case_manifest(path: str | Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> Path:
    manifest_path = Path(path)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Case manifest already exists: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = dict(payload)
    materialized["manifest_hash"] = manifest_digest(materialized)
    manifest_path.write_text(json.dumps(materialized, indent=2), encoding="utf-8")
    return manifest_path


def load_cases_from_manifest(
    path: str | Path,
    *,
    n_tests: int | None = None,
    dataset: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_case_manifest(path)
    limit = manifest["count"] if n_tests is None else int(n_tests)
    if limit < 0:
        raise ValueError(f"n_tests must be non-negative, got {limit}")
    if limit > manifest["count"]:
        raise ValueError(
            f"Requested n_tests={limit} but manifest only contains {manifest['count']} indices."
        )
    selected_indices = manifest["indices"][:limit]
    cases = load_cases_by_indices(
        selected_indices,
        dataset_name=manifest["dataset"],
        split=manifest["split"],
        dataset=dataset,
    )
    return manifest, cases


def build_case_selection_metadata(
    *,
    manifest: Mapping[str, Any] | None = None,
    manifest_path: str | Path | None = None,
    selected_cases: Sequence[Mapping[str, Any]] | None = None,
    start_idx: int | None = None,
    n_cases: int | None = None,
) -> dict[str, Any]:
    if manifest is not None:
        selected_cases = list(selected_cases or [])
        selected_indices = [int(case["dataset_index"]) for case in selected_cases] if selected_cases else [
            int(v) for v in manifest.get("indices", [])
        ]
        selected_case_ids = [int(case["case_id"]) for case in selected_cases] if selected_cases else [
            int(v) for v in manifest.get("case_ids", [])
        ]
        rel_manifest = ""
        if manifest_path is not None:
            try:
                rel_manifest = str(Path(manifest_path))
            except Exception:
                rel_manifest = Path(manifest_path).name
        return {
            "mode": "explicit_indices",
            "dataset": str(manifest.get("dataset", DEFAULT_DATASET)),
            "split": str(manifest.get("split", DEFAULT_SPLIT)),
            "seed": manifest.get("seed"),
            "manifest_path": rel_manifest,
            "manifest_hash": str(manifest.get("manifest_hash", "")),
            "count": len(selected_indices),
            "selected_dataset_indices": selected_indices,
            "selected_case_ids": selected_case_ids,
        }

    start = int(start_idx or 0)
    count = int(n_cases or 0)
    return {
        "mode": "contiguous_slice",
        "dataset": DEFAULT_DATASET,
        "split": DEFAULT_SPLIT,
        "start_idx": start,
        "end_idx": start + max(0, count - 1),
        "count": count,
    }
