#!/usr/bin/env python3
"""Export compact Qwen3-8B paper-graph data as parseable JSON.

The raw structural files are large and case-oriented. These exports keep the
same aggregate series consumed by paper_graphs/paper_graphs_support.py, split
into one unedited baseline payload and one edited-mean payload.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from bundle_paths import add_import_root, default_bundle_root  # noqa: E402


IMPORT_ROOT = add_import_root(__file__)

from paper_graphs.paper_graphs_support import (  # noqa: E402
    _stack_attention_contrast_metrics,
    _stack_single_specs,
    aggregate_layer_map_from_payload,
    aggregate_target_svd_profile,
    payload_supports_paper_graphs,
)


EXPORT_DIRNAME = "qwen_paper_graph_json"
MODEL_DIR = "qwen3-8b"


def _default_bundle_root() -> Path:
    return default_bundle_root(__file__)


def _latest_structural(run_root: Path, prefix: str) -> Path:
    matches = sorted((run_root / "structural").glob(f"{prefix}_structural_*.json"))
    if not matches:
        raise FileNotFoundError(f"No {prefix} structural JSON under {run_root / 'structural'}")
    return matches[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _relative(path: Path, bundle_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(bundle_root.resolve()))
    except ValueError:
        return str(path)


def _stats_payload(stats: dict[str, Any] | None, *, x_axis: str) -> dict[str, Any] | None:
    if not stats:
        return None
    return {
        "x_axis": x_axis,
        **_jsonable(stats),
    }


def _paper_series(payload: dict[str, Any], *, top_k_svd_ranks: int) -> dict[str, Any]:
    series: dict[str, Any] = {}
    for title, getter, ylabel, plot_kind in _stack_single_specs(payload):
        key = (
            title.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )
        if plot_kind == "svd":
            stats = aggregate_target_svd_profile(payload, top_k=top_k_svd_ranks)
            series[key] = {
                "title": title,
                "plot_kind": plot_kind,
                "ylabel": ylabel,
                "stats": _stats_payload(stats, x_axis="rank"),
            }
            continue

        if not callable(getter):
            stats = None
        else:
            stats = aggregate_layer_map_from_payload(payload, getter)
        series[key] = {
            "title": title,
            "plot_kind": plot_kind,
            "ylabel": ylabel,
            "stats": _stats_payload(stats, x_axis="layer"),
        }

    attention_metrics: dict[str, Callable[[dict], dict]] = _stack_attention_contrast_metrics(payload)
    if attention_metrics:
        series["attention_contrast_series"] = {
            "title": "Attention contrast series",
            "plot_kind": "multi_series",
            "ylabel": "Score",
            "metrics": {
                name: _stats_payload(aggregate_layer_map_from_payload(payload, getter), x_axis="layer")
                for name, getter in attention_metrics.items()
            },
        }
    return series


def _case_counts(payload: dict[str, Any]) -> dict[str, int]:
    tests = payload.get("tests") or []
    completed = [
        test
        for test in tests
        if not test.get("error") and not test.get("skipped")
    ]
    detector_eligible = [
        test
        for test in completed
        if isinstance(test.get("blind_detection", {}).get("layer_features"), dict)
        and bool(test.get("blind_detection", {}).get("layer_features"))
    ]
    rome_success = [
        test
        for test in completed
        if bool(test.get("rome", {}).get("success"))
    ]
    return {
        "tests": len(tests),
        "completed": len(completed),
        "rome_success": len(rome_success),
        "detector_eligible": len(detector_eligible),
    }


def _summary_fields(summary: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "total",
        "successful",
        "skipped",
        "cases_total",
        "cases_completed",
        "cases_error",
        "rome_success_count",
        "rome_success_rate",
        "detector_eligible_count",
        "mean_efficacy_score",
        "mean_efficacy_magnitude",
        "mean_paraphrase_score",
        "mean_neighborhood_score",
        "mean_overall_score",
    ]
    return {key: summary.get(key) for key in keep if key in summary}


def _export_payload(
    *,
    kind: str,
    payload: dict[str, Any],
    structural_path: Path,
    bundle_root: Path,
    detector_summary: dict[str, Any] | None,
    top_k_svd_ranks: int,
) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    return {
        "schema": "latium.qwen_paper_graph_export.v1",
        "kind": kind,
        "description": (
            "Unedited Qwen3-8B baseline aggregate"
            if kind == "baseline"
            else "Qwen3-8B aggregate mean across successful ROME-edited cases"
        ),
        "model": metadata.get("model", MODEL_DIR),
        "model_dir": MODEL_DIR,
        "target_layer": metadata.get("target_layer"),
        "analysis_profile": metadata.get("analysis_profile"),
        "source_structural_json": _relative(structural_path, bundle_root),
        "case_selection": metadata.get("case_selection"),
        "case_counts": _case_counts(payload),
        "summary": _summary_fields(payload.get("summary", {})),
        "detector_summary": detector_summary if kind == "edited_mean" else None,
        "paper_graphs": {
            "source": "paper_graphs.paper_graphs_support._stack_single_specs",
            "top_k_svd_ranks": top_k_svd_ranks,
            "supports_paper_graphs": payload_supports_paper_graphs(payload),
            "series": _paper_series(payload, top_k_svd_ranks=top_k_svd_ranks),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", default=str(_default_bundle_root()))
    parser.add_argument("--top-k-svd-ranks", type=int, default=30)
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root).resolve()
    run_root = bundle_root / "data" / "final_n500_models_only" / MODEL_DIR
    output_dir = bundle_root / "graphs" / EXPORT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = _latest_structural(run_root, "baseline")
    edited_path = _latest_structural(run_root, "rome")
    detector_path = run_root / "detector" / f"{MODEL_DIR}_detector_summary.json"

    baseline = _load_json(baseline_path)
    edited = _load_json(edited_path)
    detector_summary = _load_json(detector_path) if detector_path.exists() else None

    exports = {
        "qwen3_8b_unedited_baseline_paper_graph_data.json": _export_payload(
            kind="baseline",
            payload=baseline,
            structural_path=baseline_path,
            bundle_root=bundle_root,
            detector_summary=None,
            top_k_svd_ranks=args.top_k_svd_ranks,
        ),
        "qwen3_8b_rome_edited_mean_paper_graph_data_n500.json": _export_payload(
            kind="edited_mean",
            payload=edited,
            structural_path=edited_path,
            bundle_root=bundle_root,
            detector_summary=detector_summary,
            top_k_svd_ranks=args.top_k_svd_ranks,
        ),
    }

    for name, payload in exports.items():
        path = output_dir / name
        path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
