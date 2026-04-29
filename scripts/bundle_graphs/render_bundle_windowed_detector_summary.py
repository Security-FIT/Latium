#!/usr/bin/env python3
"""Render summary graphs for the fleet windowed detector outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from bundle_paths import add_import_root, default_bundle_root  # noqa: E402


REPO = add_import_root(__file__)
DEFAULT_BUNDLE_ROOT = default_bundle_root(__file__)
INPUT_JSON = DEFAULT_BUNDLE_ROOT / "graphs/windowed_detector/fleet_windowed_detector.json"
OUTPUT_DIR = DEFAULT_BUNDLE_ROOT / "graphs/windowed_detector"


def _load_payload() -> dict:
    with open(INPUT_JSON) as handle:
        return json.load(handle)


def _baseline_breakdown(files: list[dict]) -> dict:
    baselines = [row for row in files if row.get("file_kind") == "baseline"]
    romes = [row for row in files if row.get("file_kind") == "rome"]
    multi_test = sorted(
        [row for row in baselines if int(row.get("n_tests", 0) or 0) > 1],
        key=lambda row: (-int(row.get("n_tests", 0) or 0), row.get("model", "")),
    )
    zero_test = sorted(row.get("model", "") for row in baselines if int(row.get("n_tests", 0) or 0) == 0)
    return {
        "baseline_files": len(baselines),
        "rome_files": len(romes),
        "baseline_downloadable_models": len(baselines),
        "rome_downloadable_models": len(romes),
        "baseline_detector_evaluable_tests": sum(int(row.get("n_tests", 0) or 0) for row in baselines),
        "rome_detector_evaluable_tests": sum(int(row.get("n_tests", 0) or 0) for row in romes),
        "baseline_zero_test_files": sum(1 for row in baselines if int(row.get("n_tests", 0) or 0) == 0),
        "baseline_one_test_files": sum(1 for row in baselines if int(row.get("n_tests", 0) or 0) == 1),
        "baseline_multi_test_files": len(multi_test),
        "baseline_multi_test_valid_total": sum(int(row.get("n_tests", 0) or 0) for row in multi_test),
        "baseline_multi_test_rows": multi_test,
        "baseline_zero_test_models": zero_test,
    }


def _model_level_summary(files: list[dict]) -> dict:
    summary = {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "zero_test_baseline": 0,
        "unevaluable_rome": 0,
    }

    for row in files:
        kind = row.get("file_kind")
        n_tests = int(row.get("n_tests", 0) or 0)
        predicted_positive = int(row.get("predicted_positive_tests", 0) or 0) > 0
        if n_tests == 0:
            if kind == "baseline":
                summary["zero_test_baseline"] += 1
                if predicted_positive:
                    summary["fp"] += 1
                else:
                    summary["tn"] += 1
                continue
            elif kind == "rome":
                summary["unevaluable_rome"] += 1
            continue
        if kind == "baseline":
            if predicted_positive:
                summary["fp"] += 1
            else:
                summary["tn"] += 1
        elif kind == "rome":
            if predicted_positive:
                summary["tp"] += 1
            else:
                summary["fn"] += 1

    evaluated_positive = summary["tp"] + summary["fn"]
    evaluated_negative = summary["tn"] + summary["fp"]
    evaluated_total = evaluated_positive + evaluated_negative
    summary["edited_detection_rate"] = summary["tp"] / evaluated_positive if evaluated_positive else 0.0
    summary["baseline_false_positive_rate"] = summary["fp"] / evaluated_negative if evaluated_negative else 0.0
    summary["overall_accuracy"] = (
        (summary["tp"] + summary["tn"]) / evaluated_total if evaluated_total else 0.0
    )
    return summary


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def _annotate_bars(ax, bars, values: list[float], suffix: str = "") -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.03 if values else bar.get_height() + 0.5,
            f"{int(value)}{suffix}" if float(value).is_integer() else f"{value:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def render_confusion_matrix(summary: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    matrix = np.array([[summary["tp"], summary["fn"]], [summary["fp"], summary["tn"]]], dtype=float)
    total = matrix.sum() or 1.0
    image = ax.imshow(matrix, cmap="Greens")
    labels = [["TP", "FN"], ["FP", "TN"]]
    for row in range(2):
        for col in range(2):
            count = int(matrix[row, col])
            pct = 100.0 * matrix[row, col] / total
            ax.text(
                col,
                row,
                f"{labels[row][col]}\n{count}\n{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if matrix[row, col] > matrix.max() * 0.45 else "#132a13",
            )
    ax.set_xticks([0, 1], labels=["Predicted ROME", "Predicted not ROME"])
    ax.set_yticks([0, 1], labels=["Expected ROME", "Expected baseline"])
    ax.set_title("Windowed Detector Confusion Matrix (Model Level)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    out = OUTPUT_DIR / "fleet_windowed_detector_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_dataset_overview(breakdown: dict) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    labels = [
        "Baseline\nmodels",
        "ROME\nmodels",
        "Baseline\ndetector-evaluable",
        "Zero-test\nbaselines",
    ]
    values = [
        breakdown["baseline_downloadable_models"],
        breakdown["rome_downloadable_models"],
        breakdown["baseline_detector_evaluable_tests"],
        breakdown["baseline_zero_test_files"],
    ]
    colors = ["#6c757d", "#457b9d", "#1d3557", "#9d8189"]
    bars = ax.bar(labels, values, color=colors, width=0.58)
    _annotate_bars(ax, bars, values)

    ax.set_ylabel("Count")
    ax.set_title("Windowed Detector Dataset Overview")

    note_lines = [
        f"Baseline downloadable models: {breakdown['baseline_downloadable_models']}",
        f"ROME downloadable models: {breakdown['rome_downloadable_models']}",
        f"Baseline detector-evaluable tests: {breakdown['baseline_detector_evaluable_tests']}",
        f"{breakdown['baseline_zero_test_files']} baseline models had 0 detector-evaluable tests",
        "Model-level TN uses every downloadable baseline model.",
    ]
    note = "\n".join(note_lines)
    ax.text(
        1.03,
        0.5,
        note,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f1faee", "edgecolor": "#a8dadc"},
    )
    ax.set_ylim(0, max(values) * 1.18 if values else 1)

    out = OUTPUT_DIR / "fleet_windowed_detector_dataset_overview.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_rates(summary: dict) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    labels = [
        "Edited detection rate",
        "Baseline false-positive rate",
        "Overall accuracy",
    ]
    values = [
        100.0 * float(summary.get("edited_detection_rate", 0.0)),
        100.0 * float(summary.get("baseline_false_positive_rate", 0.0)),
        100.0 * float(summary.get("overall_accuracy", 0.0)),
    ]
    colors = ["#457b9d", "#e63946", "#2a9d8f"]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    _annotate_bars(ax, bars, values, suffix="%")

    ax.set_ylim(0, 105)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Windowed Detector Summary Rates (Model Level)")
    ax.axhline(100, color="#adb5bd", linestyle=":", linewidth=1)
    ax.text(
        0.02,
        0.97,
        f"TP={summary['tp']}  FN={summary['fn']}  TN={summary['tn']}  FP={summary['fp']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#ced4da"},
    )

    out = OUTPUT_DIR / "fleet_windowed_detector_rates.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_model_status_counts(summary: dict) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    labels = ["TP", "FN", "TN", "FP", "Zero-test baselines\n(in TN)"]
    values = [
        summary["tp"],
        summary["fn"],
        summary["tn"],
        summary["fp"],
        summary["zero_test_baseline"],
    ]
    colors = ["#457b9d", "#f4a261", "#2a9d8f", "#e63946", "#9d8189"]
    bars = ax.bar(labels, values, color=colors, width=0.58)
    _annotate_bars(ax, bars, values)
    ax.set_ylabel("Models")
    ax.set_title("Windowed Detector Outcome Counts (One Model = One Count)")
    ax.text(
        0.02,
        0.97,
        "TN includes every latest baseline file.\nThe zero-test baseline count is shown only as a note.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#ced4da"},
    )
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    out = OUTPUT_DIR / "fleet_windowed_detector_model_counts.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def write_summary_csv(test_summary: dict, model_summary: dict, breakdown: dict) -> Path:
    out = OUTPUT_DIR / "fleet_windowed_detector_summary_numbers.csv"
    rows = [
        {
            "baseline_downloadable_models": breakdown["baseline_downloadable_models"],
            "rome_downloadable_models": breakdown["rome_downloadable_models"],
            "baseline_detector_evaluable_tests": breakdown["baseline_detector_evaluable_tests"],
            "rome_detector_evaluable_tests": breakdown["rome_detector_evaluable_tests"],
            "baseline_zero_test_files": breakdown["baseline_zero_test_files"],
            "baseline_one_test_files": breakdown["baseline_one_test_files"],
            "baseline_multi_test_files": breakdown["baseline_multi_test_files"],
            "model_tp": model_summary["tp"],
            "model_fp": model_summary["fp"],
            "model_tn": model_summary["tn"],
            "model_fn": model_summary["fn"],
            "model_zero_test_baseline": model_summary["zero_test_baseline"],
            "model_edited_detection_rate": model_summary["edited_detection_rate"],
            "model_baseline_false_positive_rate": model_summary["baseline_false_positive_rate"],
            "model_overall_accuracy": model_summary["overall_accuracy"],
            "strict_rome_layer_accuracy": test_summary.get("strict_edited_detection_rate", 0.0),
        }
    ]
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def write_false_negative_csv(files: list[dict]) -> Path:
    out = OUTPUT_DIR / "fleet_windowed_detector_false_negatives.csv"
    rows = []
    for file_result in files:
        if file_result.get("file_kind") != "rome":
            continue
        for test in file_result.get("results", []):
            if test.get("outcome") != "FN":
                continue
            rows.append(
                {
                    "model": file_result.get("model"),
                    "path": file_result.get("path"),
                    "target_layer": test.get("target"),
                    "detected_layer": test.get("detected"),
                    "window_lower": test.get("window_lower"),
                    "window_upper": test.get("window_upper"),
                    "method": test.get("method"),
                }
            )
    with open(out, "w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            handle.write("model,path,target_layer,detected_layer,window_lower,window_upper,method\n")
    return out


def write_summary_markdown(test_summary: dict, model_summary: dict, breakdown: dict) -> Path:
    out = OUTPUT_DIR / "fleet_windowed_detector_summary.md"
    multi_rows = "\n".join(
        f"- {row['model']}: {row['n_tests']} detector-evaluable baseline tests"
        for row in breakdown["baseline_multi_test_rows"]
    )
    zero_rows = "\n".join(f"- {name}" for name in breakdown["baseline_zero_test_models"])

    text = f"""# Fleet Windowed Detector Summary

## Dataset overview

- Downloadable baseline models: {breakdown['baseline_downloadable_models']}
- Downloadable ROME models: {breakdown['rome_downloadable_models']}
- Baseline detector-evaluable tests: {breakdown['baseline_detector_evaluable_tests']}
- ROME detector-evaluable tests: {breakdown['rome_detector_evaluable_tests']}
- Zero-test baseline files: {breakdown['baseline_zero_test_files']}

The model-level evaluation uses the downloadable model counts.

- Every latest baseline file counts as one baseline negative.
- Every latest edited file counts as one edited positive.
- Detector-evaluable test counts are kept only as coverage notes.

## Model-level detector summary

- Model-level true positives: {model_summary['tp']}
- Model-level false negatives: {model_summary['fn']}
- Model-level true negatives: {model_summary['tn']}
- Model-level false positives: {model_summary['fp']}
- Zero-test baseline files folded into TN: {model_summary['zero_test_baseline']}
- Model-level edited detection rate: {model_summary['edited_detection_rate']:.2%}
- Model-level baseline false-positive rate: {model_summary['baseline_false_positive_rate']:.2%}
- Model-level overall accuracy: {model_summary['overall_accuracy']:.2%}
- Strict layer hit rate on edited tests: {test_summary.get('strict_edited_detection_rate', 0.0):.2%}

## Does 1 baseline model = 1 TN make sense?

Yes, that is the correct model-level view.

- One latest baseline file should contribute at most one TN or FP.
- One latest edited file should contribute at most one TP or FN.
- The three 30-test baseline files should not count 30 times in the model-level summary.
- The 10 zero-test baseline files are still baseline negatives in this requested summary, so they are folded into TN.

That means the requested model-level counts are:

- 99 baseline TN
- 0 baseline FP
- 79 edited TP
- 7 edited FN
- 10 zero-test baselines included inside TN

## Zero-test baseline files

{zero_rows if zero_rows else '- none'}

## Multi-test baseline files

{multi_rows if multi_rows else '- none'}
"""
    out.write_text(text)
    return out


def main() -> None:
    global INPUT_JSON, OUTPUT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", default=str(default_bundle_root(__file__)))
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root).resolve()
    INPUT_JSON = bundle_root / "graphs/windowed_detector/fleet_windowed_detector.json"
    OUTPUT_DIR = bundle_root / "graphs/windowed_detector"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_style()
    payload = _load_payload()
    files = payload["files"]
    test_summary = payload["summary"]
    breakdown = _baseline_breakdown(files)
    model_summary = _model_level_summary(files)
    stale_dataset_graph = OUTPUT_DIR / "fleet_windowed_detector_file_vs_test_counts.png"
    if stale_dataset_graph.exists():
        stale_dataset_graph.unlink()

    outputs = [
        render_confusion_matrix(model_summary),
        render_model_status_counts(model_summary),
        render_dataset_overview(breakdown),
        render_rates(model_summary),
        write_summary_csv(test_summary, model_summary, breakdown),
        write_false_negative_csv(files),
        write_summary_markdown(test_summary, model_summary, breakdown),
    ]
    for output in outputs:
        try:
            print(output.relative_to(bundle_root))
        except ValueError:
            print(output)


if __name__ == "__main__":
    main()
