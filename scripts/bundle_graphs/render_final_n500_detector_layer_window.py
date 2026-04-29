#!/usr/bin/env python3
"""Score final N=500 detector layer hits with a target-layer window.

The existing detector summaries store strict hit totals only. This script
recomputes the same detector decisions from bundled structural JSONs and then
rescored each prediction as correct when it lands within target_layer +/- R.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from bundle_paths import add_import_root, default_bundle_root  # noqa: E402


SCRIPT_ROOT = add_import_root(__file__)

from detector.composite_detector_v2 import process_file as composite_process_file  # noqa: E402
from detector.gpt_detector import process_file as gpt_process_file  # noqa: E402


GRAPH_DIRNAME = "detector_layer_window"
DEFAULT_EXCLUDE_MODELS: tuple[str, ...] = ()


def _overall_label(exclude_models: set[str]) -> str:
    if not exclude_models:
        return "all models"
    excluded = ", ".join(sorted(exclude_models))
    return f"all models except: {excluded}"


def _exclusion_note(exclude_models: set[str]) -> str:
    if not exclude_models:
        return "No models are excluded from this report."
    excluded = ", ".join(sorted(exclude_models))
    return f"Excluded models: {excluded}."


def _default_bundle_root() -> Path:
    return default_bundle_root(__file__)


def _latest_structural(run_root: Path, prefix: str = "rome") -> Path:
    structural_dir = run_root / "structural"
    matches = sorted(structural_dir.glob(f"{prefix}_structural_*.json"))
    if not matches:
        raise FileNotFoundError(f"No {prefix} structural JSON under {structural_dir}")
    return matches[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _detector_kind(summary: dict[str, Any], model: str) -> str:
    kind = str(summary.get("detector_type") or "").strip().lower()
    if kind in {"gpt", "composite"}:
        return kind
    return "gpt" if model.startswith("gpt") else "composite"


def _process_detector(kind: str, structural_path: Path) -> dict[str, Any]:
    if kind == "gpt":
        return gpt_process_file(structural_path, trim=5)
    return composite_process_file(structural_path, trim=2)


def _summarize_result(
    *,
    model: str,
    kind: str,
    summary: dict[str, Any],
    result: dict[str, Any],
    window_radius: int,
) -> dict[str, Any]:
    target = result.get("target_layer", summary.get("target_layer", summary.get("layer")))
    tests = result.get("results") or []
    n_tests = len(tests)
    strict_correct = sum(1 for row in tests if row.get("detected") == target)
    window_correct = 0
    recovered_by_window = 0
    near_miss_counts: dict[str, int] = {}

    for row in tests:
        detected = row.get("detected")
        if detected is None or target is None:
            continue
        distance = abs(int(detected) - int(target))
        if distance <= window_radius:
            window_correct += 1
            if distance != 0:
                recovered_by_window += 1
                near_miss_counts[str(distance)] = near_miss_counts.get(str(distance), 0) + 1

    strict_accuracy = strict_correct / n_tests if n_tests else 0.0
    window_accuracy = window_correct / n_tests if n_tests else 0.0
    summary_correct = int(summary.get("correct", summary.get("det_ok", -1)) or 0)
    summary_n = int(summary.get("n_tests", summary.get("det_eval", -1)) or 0)

    return {
        "model": model,
        "detector_type": kind,
        "target_layer": target,
        "window_radius": window_radius,
        "window_layers": (
            "" if target is None else f"{int(target) - window_radius}:{int(target) + window_radius}"
        ),
        "n_tests": n_tests,
        "strict_correct": strict_correct,
        "strict_accuracy": strict_accuracy,
        "window_correct": window_correct,
        "window_accuracy": window_accuracy,
        "recovered_by_window": recovered_by_window,
        "accuracy_gain": window_accuracy - strict_accuracy,
        "summary_correct": summary_correct,
        "summary_n_tests": summary_n,
        "strict_matches_summary": strict_correct == summary_correct and n_tests == summary_n,
        "near_miss_counts": near_miss_counts,
        "method_counts": result.get("method_counts", {}),
        "structural_json": str(result.get("path") or ""),
    }


def _iter_model_runs(bundle_root: Path, exclude_models: set[str]):
    data_root = bundle_root / "data" / "final_n500_models_only"
    for run_root in sorted(p for p in data_root.iterdir() if p.is_dir()):
        model = run_root.name
        if model in exclude_models:
            continue
        summary_path = run_root / "detector" / f"{model}_detector_summary.json"
        if not summary_path.exists():
            continue
        summary = _load_json(summary_path)
        yield run_root, model, summary_path, summary


def _overall_row(rows: list[dict[str, Any]], *, label: str, detector_type: str) -> dict[str, Any]:
    selected = [row for row in rows if detector_type == "all" or row["detector_type"] == detector_type]
    n_tests = sum(int(row["n_tests"]) for row in selected)
    strict_correct = sum(int(row["strict_correct"]) for row in selected)
    window_correct = sum(int(row["window_correct"]) for row in selected)
    recovered_by_window = sum(int(row["recovered_by_window"]) for row in selected)
    strict_accuracy = strict_correct / n_tests if n_tests else 0.0
    window_accuracy = window_correct / n_tests if n_tests else 0.0
    macro_strict = sum(float(row["strict_accuracy"]) for row in selected) / len(selected) if selected else 0.0
    macro_window = sum(float(row["window_accuracy"]) for row in selected) / len(selected) if selected else 0.0
    return {
        "label": label,
        "detector_type": detector_type,
        "models": len(selected),
        "n_tests": n_tests,
        "strict_correct": strict_correct,
        "strict_accuracy": strict_accuracy,
        "window_correct": window_correct,
        "window_accuracy": window_accuracy,
        "recovered_by_window": recovered_by_window,
        "accuracy_gain": window_accuracy - strict_accuracy,
        "macro_strict_accuracy": macro_strict,
        "macro_window_accuracy": macro_window,
        "macro_accuracy_gain": macro_window - macro_strict,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "detector_type",
        "target_layer",
        "window_radius",
        "window_layers",
        "n_tests",
        "strict_correct",
        "strict_accuracy",
        "window_correct",
        "window_accuracy",
        "recovered_by_window",
        "accuracy_gain",
        "strict_matches_summary",
        "near_miss_counts",
        "method_counts",
        "structural_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["near_miss_counts"] = json.dumps(payload.get("near_miss_counts", {}), sort_keys=True)
            payload["method_counts"] = json.dumps(payload.get("method_counts", {}), sort_keys=True)
            writer.writerow({key: payload.get(key, "") for key in fieldnames})


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    overall: list[dict[str, Any]],
    *,
    exclude_models: set[str],
) -> None:
    lines = [
        "# Final N=500 Detector Layer Window Score",
        "",
        "Scoring rule: a detector prediction is correct when `abs(detected_layer - target_layer) <= window_radius`.",
        _exclusion_note(exclude_models),
        "",
        "## Overall",
        "",
        "| group | models | n | strict | +/-1 window | gained |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall:
        lines.append(
            "| {label} | {models} | {n_tests} | {strict_correct}/{n_tests} ({strict_accuracy:.2%}) | "
            "{window_correct}/{n_tests} ({window_accuracy:.2%}) | +{recovered_by_window} ({accuracy_gain:.2%}) |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## By Model",
            "",
            "| model | detector | target | n | strict | +/-1 window | gained |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {model} | {detector_type} | {target_layer} | {n_tests} | "
            "{strict_correct}/{n_tests} ({strict_accuracy:.2%}) | "
            "{window_correct}/{n_tests} ({window_accuracy:.2%}) | "
            "+{recovered_by_window} ({accuracy_gain:.2%}) |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", default=str(_default_bundle_root()))
    parser.add_argument("--window-radius", type=int, default=1)
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=list(DEFAULT_EXCLUDE_MODELS),
        help="Model directory name to exclude. Can be repeated.",
    )
    args = parser.parse_args()

    if args.window_radius < 0:
        raise ValueError("--window-radius must be >= 0")

    bundle_root = Path(args.bundle_root).resolve()
    output_dir = bundle_root / "graphs" / GRAPH_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    exclude_models = set(args.exclude_model or [])
    for run_root, model, _summary_path, summary in _iter_model_runs(bundle_root, exclude_models):
        structural_path = _latest_structural(run_root, "rome")
        kind = _detector_kind(summary, model)
        result = _process_detector(kind, structural_path)
        rows.append(
            _summarize_result(
                model=model,
                kind=kind,
                summary=summary,
                result=result,
                window_radius=args.window_radius,
            )
        )

    rows.sort(key=lambda row: row["model"])
    overall = [
        _overall_row(rows, label=_overall_label(exclude_models), detector_type="all"),
        _overall_row(rows, label="GPT-family", detector_type="gpt"),
        _overall_row(rows, label="composite", detector_type="composite"),
    ]

    suffix = f"radius{args.window_radius}"
    csv_path = output_dir / f"final_n500_detector_layer_window_{suffix}.csv"
    json_path = output_dir / f"final_n500_detector_layer_window_{suffix}.json"
    md_path = output_dir / f"final_n500_detector_layer_window_{suffix}.md"

    _write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "window_radius": args.window_radius,
                "excluded_models": sorted(exclude_models),
                "overall": overall,
                "models": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_markdown(md_path, rows, overall, exclude_models=exclude_models)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    for row in overall:
        print(
            "{label}: strict={strict_correct}/{n_tests} ({strict_accuracy:.4f}) "
            "window={window_correct}/{n_tests} ({window_accuracy:.4f}) "
            "gain=+{recovered_by_window} ({accuracy_gain:.4f})".format(**row)
        )


if __name__ == "__main__":
    main()
