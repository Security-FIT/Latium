#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from bundle_paths import add_import_root, default_bundle_root  # noqa: E402


REPO_ROOT = add_import_root(__file__)

from detector.composite_detector_v2 import (  # noqa: E402
    DEFAULT_LARGE_WINDOW,
    DEFAULT_SMALL_WINDOW,
    DEFAULT_TE_WINDOW,
    _signal_profile_stats_configurable,
    local_zscore,
)
from paper_graphs._newgen_utils import load_json  # noqa: E402
from paper_graphs.paper_graphs_support import (  # noqa: E402
    _stack_attention_contrast_metrics,
    _stack_single_specs,
    aggregate_layer_map_from_payload,
    aggregate_target_svd_profile,
    extract_layer_map,
)


BASELINE_COLOR = "#000000"
TARGET_COLOR = "#4b5563"
EDITED_LAYER = 10
SUMMARY_DIRNAME = "prefixtest_summary"
MULTIPREFIX_DIRNAME = "multiprefix"


@dataclass(frozen=True)
class VariantSpec:
    mode: str
    length: str
    slug: str
    color: str
    marker: str

    @property
    def label(self) -> str:
        return f"{self.mode}-{self.length}"

    @property
    def title_label(self) -> str:
        return f"{self.mode.title()} {self.length.title()}"


VARIANTS: List[VariantSpec] = [
    VariantSpec("self", "short", "qwen3-8b-prefixtest-self-short", "#0f766e", "o"),
    VariantSpec("self", "medium", "qwen3-8b-prefixtest-self-medium", "#14b8a6", "s"),
    VariantSpec("self", "long", "qwen3-8b-prefixtest-self-long", "#5eead4", "^"),
    VariantSpec("template", "short", "qwen3-8b-prefixtest-template-short", "#7c3aed", "o"),
    VariantSpec("template", "medium", "qwen3-8b-prefixtest-template-medium", "#a855f7", "s"),
    VariantSpec("template", "long", "qwen3-8b-prefixtest-template-long", "#c084fc", "^"),
    VariantSpec("external", "short", "qwen3-8b-prefixtest-external-short", "#c2410c", "o"),
    VariantSpec("external", "medium", "qwen3-8b-prefixtest-external-medium", "#f97316", "s"),
    VariantSpec("external", "long", "qwen3-8b-prefixtest-external-long", "#fdba74", "^"),
]


def setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "axes.titlecolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "text.color": "#111111",
            "grid.color": "#d0d0d0",
            "grid.alpha": 0.26,
            "legend.facecolor": "white",
            "legend.edgecolor": "#cccccc",
            "legend.framealpha": 0.92,
        }
    )


def structural_file(run_root: Path, prefix: str) -> Path:
    matches = sorted((run_root / "structural").glob(f"{prefix}_structural_*.json"))
    if not matches:
        raise FileNotFoundError(f"No {prefix} structural JSON under {run_root}")
    return matches[-1]


def rel_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def aggregate_signal_metric(payload: dict, metric_name: str) -> Optional[dict]:
    return aggregate_layer_map_from_payload(
        payload,
        lambda case: extract_layer_map(case, "spectral_detection", metric_name),
    )


def plot_series(ax, stats: dict, *, color: str, label: str, marker: str) -> None:
    if not stats:
        return
    layers = stats["layers"]
    mean = stats["mean"]
    markevery = max(1, len(layers) // 10)
    ax.plot(
        layers,
        mean,
        color=color,
        linewidth=1.9,
        marker=marker,
        markersize=3.2,
        markevery=markevery,
        label=label,
    )


def plot_baseline(ax, stats: dict, *, label: str = "baseline") -> None:
    if not stats:
        return
    layers = stats["layers"]
    mean = stats["mean"]
    markevery = max(1, len(layers) // 10)
    ax.plot(
        layers,
        mean,
        color=BASELINE_COLOR,
        linewidth=2.3,
        linestyle="--",
        marker="o",
        markersize=3.0,
        markerfacecolor="white",
        markeredgecolor=BASELINE_COLOR,
        markeredgewidth=0.8,
        markevery=markevery,
        label=label,
    )


def build_variant_record(bundle_root: Path, spec: VariantSpec, paper_specs: list, attention_keys: List[str]) -> dict:
    run_root = bundle_root / "data" / "prefixtest_n500_runs" / spec.slug
    summary_path = run_root / "run_summary_latest.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")

    run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_summary = (run_summary.get("models") or [])[0]
    edited_path = structural_file(run_root, "rome")
    variant_baseline_path = structural_file(run_root, "baseline")
    payload = load_json(edited_path)

    paper_series: Dict[str, Optional[dict]] = {}
    for title, getter, _ylabel, plot_kind in paper_specs:
        if plot_kind == "svd":
            paper_series[title] = aggregate_target_svd_profile(payload, top_k=30)
        else:
            paper_series[title] = aggregate_layer_map_from_payload(payload, getter)

    attention_series = {
        name: aggregate_layer_map_from_payload(payload, getter)
        for name, getter in _stack_attention_contrast_metrics(payload).items()
        if name in attention_keys
    }

    record = {
        "spec": spec,
        "run_root": run_root,
        "summary": model_summary,
        "edited_path": edited_path,
        "variant_baseline_path": variant_baseline_path,
        "signal_stats": _signal_profile_stats_configurable(
            payload,
            trim=2,
            te_windows=(DEFAULT_TE_WINDOW,),
            small_window=DEFAULT_SMALL_WINDOW,
            large_window=DEFAULT_LARGE_WINDOW,
            include_signal_a=False,
            include_signal_b=False,
            include_te_raw=True,
        ),
        "paper_series": paper_series,
        "attention_series": attention_series,
        "hybrid_stats": aggregate_signal_metric(payload, "rome_hybrid_scores"),
        "sv_ratio_stats": aggregate_signal_metric(payload, "sv_ratio_scores"),
    }
    return record


def build_common_baseline(bundle_root: Path, paper_specs: list, attention_keys: List[str]) -> dict:
    run_root = bundle_root / "data" / "final_n500_models_only" / "qwen3-8b"
    edited_path = structural_file(run_root, "rome")
    baseline_path = structural_file(run_root, "baseline")
    payload = load_json(baseline_path)

    paper_series: Dict[str, Optional[dict]] = {}
    for title, getter, _ylabel, plot_kind in paper_specs:
        if plot_kind == "svd":
            paper_series[title] = aggregate_target_svd_profile(payload, top_k=30)
        else:
            paper_series[title] = aggregate_layer_map_from_payload(payload, getter)

    attention_series = {
        name: aggregate_layer_map_from_payload(payload, getter)
        for name, getter in _stack_attention_contrast_metrics(payload).items()
        if name in attention_keys
    }

    return {
        "run_root": run_root,
        "edited_path": edited_path,
        "baseline_path": baseline_path,
        "signal_stats": _signal_profile_stats_configurable(
            payload,
            trim=2,
            te_windows=(DEFAULT_TE_WINDOW,),
            small_window=DEFAULT_SMALL_WINDOW,
            large_window=DEFAULT_LARGE_WINDOW,
            include_signal_a=False,
            include_signal_b=False,
            include_te_raw=True,
        ),
        "paper_series": paper_series,
        "attention_series": attention_series,
        "hybrid_stats": aggregate_signal_metric(payload, "rome_hybrid_scores"),
        "sv_ratio_stats": aggregate_signal_metric(payload, "sv_ratio_scores"),
    }


def write_summary_tables(records: List[dict], summary_dir: Path) -> None:
    rows = []
    for record in records:
        summary = record["summary"]
        spec: VariantSpec = record["spec"]
        rows.append(
            {
                "mode": spec.mode,
                "length": spec.length,
                "variant": spec.label,
                "detector_n": int(summary.get("det_eval", 0)),
                "detector_correct": int(summary.get("det_ok", 0)),
                "detector_accuracy": float(summary.get("det_rate", 0.0)),
                "rome_completed": int(summary.get("tested", 0)),
                "rome_errors": int(summary.get("errors", 0)),
                "rome_success_count": int(summary.get("rome_ok", 0)),
                "detector_eligible_count": int(summary.get("det_eval", 0)),
                "mean_efficacy_score": float(summary.get("mean_efficacy_score", 0.0)),
                "mean_paraphrase_score": float(summary.get("mean_paraphrase_score", 0.0)),
                "mean_neighborhood_score": float(summary.get("mean_neighborhood_score", 0.0)),
                "mean_overall_score": float(summary.get("mean_overall_score", 0.0)),
                "structural_json": rel_to_repo(record["edited_path"]),
                "baseline_json": rel_to_repo(record["variant_baseline_path"]),
            }
        )

    json_path = summary_dir / "prefixtest_pipeline_summary_n500.json"
    csv_path = summary_dir / "prefixtest_pipeline_summary_n500.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_layer10_raw_row(*, mode: str, length: str, variant: str, edited_path: Path) -> tuple[dict, dict]:
    metric_specs = [
        ("SG", "spectral_gap"),
        ("TE", "top1_energy"),
        ("RA", "row_alignment"),
        ("CV", "norm_cv"),
        ("ER", "effective_rank"),
    ]
    payload = load_json(edited_path)
    values_by_metric: Dict[str, List[float]] = {short: [] for short, _name in metric_specs}
    for test in payload.get("tests", []):
        if test.get("error") or test.get("skipped"):
            continue
        if not test.get("rome", {}).get("success", True):
            continue
        layer_features = test.get("blind_detection", {}).get("layer_features", {})
        feature_map = layer_features.get(str(EDITED_LAYER)) or layer_features.get(EDITED_LAYER)
        if not isinstance(feature_map, dict):
            continue
        for short, feature_name in metric_specs:
            value = feature_map.get(feature_name)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                values_by_metric[short].append(numeric)

    row = {
        "mode": mode,
        "length": length,
        "variant": variant,
        "layer": EDITED_LAYER,
        "source_json": rel_to_repo(edited_path),
    }
    json_row = dict(row)
    json_row["raw_values"] = values_by_metric
    for short, values in values_by_metric.items():
        arr = np.asarray(values, dtype=float)
        row[f"{short}_n"] = int(arr.size)
        row[f"{short}_mean_raw"] = "" if arr.size == 0 else float(np.mean(arr))
        row[f"{short}_std_raw"] = "" if arr.size == 0 else float(np.std(arr))
        row[f"{short}_min_raw"] = "" if arr.size == 0 else float(np.min(arr))
        row[f"{short}_max_raw"] = "" if arr.size == 0 else float(np.max(arr))
    return row, json_row


def _build_layer10_windowed_row(
    *,
    mode: str,
    length: str,
    variant: str,
    edited_path: Path,
    window: int,
) -> tuple[dict, dict]:
    metric_specs = [
        ("SG", "spectral_gap"),
        ("TE", "top1_energy"),
        ("RA", "row_alignment"),
        ("CV", "norm_cv"),
        ("ER", "effective_rank"),
    ]
    payload = load_json(edited_path)
    values_by_metric: Dict[str, List[float]] = {short: [] for short, _name in metric_specs}
    for test in payload.get("tests", []):
        if test.get("error") or test.get("skipped"):
            continue
        if not test.get("rome", {}).get("success", True):
            continue
        layer_features = test.get("blind_detection", {}).get("layer_features", {})
        if not isinstance(layer_features, dict):
            continue
        layers = sorted(layer_features.keys(), key=int)
        layer_strings = [str(layer) for layer in layers]
        target_key = str(EDITED_LAYER)
        if target_key not in layer_strings:
            continue
        target_index = layer_strings.index(target_key)

        for short, feature_name in metric_specs:
            try:
                arr = np.asarray([float(layer_features[layer][feature_name]) for layer in layers], dtype=float)
            except (KeyError, TypeError, ValueError):
                continue
            if not np.all(np.isfinite(arr)):
                continue
            numeric = float(abs(local_zscore(arr, window=window)[target_index]))
            if np.isfinite(numeric):
                values_by_metric[short].append(numeric)

    row = {
        "mode": mode,
        "length": length,
        "variant": variant,
        "layer": EDITED_LAYER,
        "source_json": rel_to_repo(edited_path),
    }
    json_row = dict(row)
    json_row["transform"] = "abs_local_zscore"
    json_row["window"] = int(window)
    json_row["raw_values"] = values_by_metric
    for short, values in values_by_metric.items():
        arr = np.asarray(values, dtype=float)
        row[f"{short}_n"] = int(arr.size)
        row[f"{short}_mean_raw"] = "" if arr.size == 0 else float(np.mean(arr))
        row[f"{short}_std_raw"] = "" if arr.size == 0 else float(np.std(arr))
        row[f"{short}_min_raw"] = "" if arr.size == 0 else float(np.min(arr))
        row[f"{short}_max_raw"] = "" if arr.size == 0 else float(np.max(arr))
    return row, json_row


def write_layer10_raw_tables(records: List[dict], summary_dir: Path, bundle_root: Path) -> None:
    rows = []
    json_rows = []
    original_run_root = bundle_root / "data" / "final_n500_models_only" / "qwen3-8b"
    original_edited_path = structural_file(original_run_root, "rome")
    original_row, original_json_row = _build_layer10_raw_row(
        mode="original",
        length="base",
        variant="original-postedited",
        edited_path=original_edited_path,
    )
    json_rows.append(original_json_row)
    rows.append(original_row)

    for window in (DEFAULT_SMALL_WINDOW, DEFAULT_LARGE_WINDOW):
        row, json_row = _build_layer10_windowed_row(
            mode="original",
            length=f"window{window}",
            variant=f"original-postedited-lz{window}",
            edited_path=original_edited_path,
            window=window,
        )
        json_rows.append(json_row)
        rows.append(row)

    for record in records:
        spec: VariantSpec = record["spec"]
        row, json_row = _build_layer10_raw_row(
            mode=spec.mode,
            length=spec.length,
            variant=spec.label,
            edited_path=record["edited_path"],
        )
        json_rows.append(json_row)
        rows.append(row)

    csv_path = summary_dir / "prefixtest_layer10_raw_metrics_n500.csv"
    json_path = summary_dir / "prefixtest_layer10_raw_metrics_n500.json"
    json_path.write_text(json.dumps(json_rows, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_detector_accuracy_matrix(records: List[dict], out_path: Path) -> None:
    setup_style()
    modes = ["self", "template", "external"]
    lengths = ["short", "medium", "long"]
    matrix = np.zeros((len(modes), len(lengths)), dtype=float)
    annotations: Dict[tuple[int, int], str] = {}

    for record in records:
        spec: VariantSpec = record["spec"]
        summary = record["summary"]
        row = modes.index(spec.mode)
        col = lengths.index(spec.length)
        correct = int(summary.get("det_ok", 0))
        total = int(summary.get("det_eval", 0))
        accuracy = float(summary.get("det_rate", 0.0)) * 100.0
        matrix[row, col] = accuracy
        annotations[(row, col)] = f"{accuracy:.0f}%\n{correct}/{total}"

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=100)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                annotations[(row, col)],
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white" if matrix[row, col] < 75 else "black",
            )
    ax.set_xticks(np.arange(len(lengths)), [length.title() for length in lengths])
    ax.set_yticks(np.arange(len(modes)), [mode.title() for mode in modes])
    ax.set_title("Prefixtest detector accuracy (N=500)", fontsize=18)
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label("Accuracy (%)", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_rome_metric_means(records: List[dict], out_path: Path) -> None:
    setup_style()
    metric_specs = [
        ("ES", "mean_efficacy_score", "#1f77b4"),
        ("PS", "mean_paraphrase_score", "#2ca02c"),
        ("NS", "mean_neighborhood_score", "#ff7f0e"),
        ("S", "mean_overall_score", "#6a3d9a"),
    ]
    x = np.arange(len(records))
    width = 0.18

    fig, ax = plt.subplots(figsize=(18, 7))
    for index, (label, key, color) in enumerate(metric_specs):
        values = [float(record["summary"].get(key, 0.0)) for record in records]
        ax.bar(x + (index - 1.5) * width, values, width=width, color=color, label=label)

    ax.set_xticks(x, [record["spec"].label.replace("-", "\n", 1) for record in records])
    ax.set_ylim(0.0, 1.06)
    ax.set_ylabel("Mean score", fontsize=16)
    ax.set_title("ROME metric means by prefix mode and length (N=500)", fontsize=18)
    ax.legend(loc="upper right", ncol=4, fontsize=13)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_mode_length_curves(
    records: List[dict],
    *,
    stats_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    setup_style()
    modes = ["self", "template", "external"]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)
    for ax, mode in zip(axes, modes):
        mode_records = [record for record in records if record["spec"].mode == mode]
        for record in mode_records:
            spec: VariantSpec = record["spec"]
            stats = record.get(stats_key)
            if not stats:
                continue
            plot_series(ax, stats, color=spec.color, label=spec.length, marker=spec.marker)
        ax.axvline(EDITED_LAYER, color=TARGET_COLOR, linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_title(spec.mode.title() if mode_records else mode.title(), fontsize=15)
        ax.set_xlabel("Layer", fontsize=14)
        ax.grid(True)
        if mode_records:
            ax.legend(title="Length", fontsize=13, title_fontsize=14)
    axes[0].set_ylabel(ylabel, fontsize=16)
    fig.suptitle(title, fontsize=20, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def multiprefix_legend_handles(records: List[dict]) -> List[Line2D]:
    handles = [
        Line2D([0], [0], color=BASELINE_COLOR, linestyle="--", linewidth=2.3, marker="o", markersize=4, label="baseline"),
    ]
    for record in records:
        spec: VariantSpec = record["spec"]
        handles.append(
            Line2D([0], [0], color=spec.color, linewidth=2.0, marker=spec.marker, markersize=4, label=spec.label)
        )
    return handles


def render_multiprefix_detector_signals(records: List[dict], baseline: dict, out_path: Path) -> None:
    setup_style()
    ordered_signals = [
        "SG (raw)",
        "TE (raw)",
        f"TE (lz{DEFAULT_TE_WINDOW})",
        f"SG (lz{DEFAULT_SMALL_WINDOW})",
        f"SG (lz{DEFAULT_LARGE_WINDOW})",
    ]
    available = [name for name in ordered_signals if baseline["signal_stats"].get(name) or any(r["signal_stats"].get(name) for r in records)]
    ncols = 2
    nrows = max(1, (len(available) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.6 * nrows), sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, signal_name in zip(axes_flat, available):
        baseline_stats = baseline["signal_stats"].get(signal_name)
        if baseline_stats:
            plot_baseline(ax, baseline_stats)
        for record in records:
            spec: VariantSpec = record["spec"]
            stats = record["signal_stats"].get(signal_name)
            if not stats:
                continue
            plot_series(ax, stats, color=spec.color, label=spec.label, marker=spec.marker)
        ax.axvline(EDITED_LAYER, color=TARGET_COLOR, linestyle=":", linewidth=1.4, alpha=0.8)
        ax.set_title(signal_name, fontsize=14, fontweight="bold")
        ax.set_ylabel("Mean signal", fontsize=12)
        ax.grid(True)

    for ax in axes_flat[len(available):]:
        ax.set_visible(False)

    handles = multiprefix_legend_handles(records)
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(
        "Qwen3-8B multiprefix detector signals (baseline + 9 prefix variants)",
        fontsize=18,
        y=1.03,
    )
    fig.supxlabel("Layer index", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_multiprefix_paper_graphs(records: List[dict], baseline: dict, paper_specs: list, attention_keys: List[str], out_path: Path) -> None:
    setup_style()
    total_plots = len(paper_specs) + (1 if attention_keys else 0)
    fig, axes = plt.subplots(total_plots, 1, figsize=(20, 3.5 * total_plots))
    axes = np.atleast_1d(axes)

    for ax, (title, _getter, ylabel, plot_kind) in zip(axes, paper_specs):
        baseline_stats = baseline["paper_series"].get(title)
        if plot_kind == "svd":
            if baseline_stats:
                ax.plot(
                    baseline_stats["rank"],
                    baseline_stats["mean"],
                    color=BASELINE_COLOR,
                    linewidth=2.3,
                    linestyle="--",
                    marker="o",
                    markersize=3.0,
                    markerfacecolor="white",
                    markeredgecolor=BASELINE_COLOR,
                    markeredgewidth=0.8,
                    markevery=max(1, len(baseline_stats["rank"]) // 8),
                )
            positive = True
            for record in records:
                spec: VariantSpec = record["spec"]
                stats = record["paper_series"].get(title)
                if not stats:
                    continue
                ax.plot(
                    stats["rank"],
                    stats["mean"],
                    color=spec.color,
                    linewidth=1.9,
                    marker=spec.marker,
                    markersize=3.0,
                    markevery=max(1, len(stats["rank"]) // 8),
                )
                positive = positive and np.nanmin(stats["mean"]) > 0
            if positive and baseline_stats is not None and np.nanmin(baseline_stats["mean"]) > 0:
                ax.set_yscale("log")
            ax.set_xlabel("Rank", fontsize=12)
            ax.set_ylabel("Singular value", fontsize=12)
        else:
            if baseline_stats:
                plot_baseline(ax, baseline_stats)
            for record in records:
                spec: VariantSpec = record["spec"]
                stats = record["paper_series"].get(title)
                if not stats:
                    continue
                plot_series(ax, stats, color=spec.color, label=spec.label, marker=spec.marker)
            ax.axvline(EDITED_LAYER, color=TARGET_COLOR, linestyle=":", linewidth=1.4, alpha=0.8)
            ax.set_xlabel("Layer", fontsize=12)
            ax.set_ylabel(ylabel or "Score", fontsize=12)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True)

    if attention_keys:
        ax = axes[-1]
        for attention_name in attention_keys:
            baseline_stats = baseline["attention_series"].get(attention_name)
            if baseline_stats:
                plot_baseline(ax, baseline_stats, label="baseline" if attention_name == attention_keys[0] else "baseline")
                break
        for record in records:
            spec: VariantSpec = record["spec"]
            stats = record["attention_series"].get(attention_keys[0])
            if not stats:
                continue
            plot_series(ax, stats, color=spec.color, label=spec.label, marker=spec.marker)
        ax.axvline(EDITED_LAYER, color=TARGET_COLOR, linestyle=":", linewidth=1.4, alpha=0.8)
        ax.set_title("Attention contrast series", fontsize=14, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.grid(True)

    handles = multiprefix_legend_handles(records)
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(
        "Qwen3-8B multiprefix paper graphs (baseline + 9 prefix variants)",
        fontsize=18,
        y=1.002,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render N=500 prefixtest bundle summary and multiprefix graphs.")
    parser.add_argument("--bundle-root", default=str(default_bundle_root(__file__)))
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root).resolve()
    summary_dir = bundle_root / "graphs" / SUMMARY_DIRNAME
    multiprefix_dir = summary_dir / MULTIPREFIX_DIRNAME
    summary_dir.mkdir(parents=True, exist_ok=True)
    multiprefix_dir.mkdir(parents=True, exist_ok=True)

    sample_payload = load_json(structural_file(bundle_root / "data" / "prefixtest_n500_runs" / VARIANTS[0].slug, "rome"))
    paper_specs = _stack_single_specs(sample_payload)
    attention_keys = list(_stack_attention_contrast_metrics(sample_payload).keys())
    del sample_payload

    records = [build_variant_record(bundle_root, spec, paper_specs, attention_keys) for spec in VARIANTS]
    baseline = build_common_baseline(bundle_root, paper_specs, attention_keys)

    write_summary_tables(records, summary_dir)
    write_layer10_raw_tables(records, summary_dir, bundle_root)
    render_detector_accuracy_matrix(records, summary_dir / "detector_accuracy_matrix_n500.png")
    render_rome_metric_means(records, summary_dir / "rome_metric_means_by_prefix_n500.png")
    render_mode_length_curves(
        records,
        stats_key="hybrid_stats",
        title="Mean ROME hybrid spectral score by layer (dashed line = edited layer 10)",
        ylabel="Hybrid score",
        out_path=summary_dir / "hybrid_layer_curves_n500.png",
    )
    render_mode_length_curves(
        records,
        stats_key="sv_ratio_stats",
        title="Mean singular-value ratio score by layer (dashed line = edited layer 10)",
        ylabel="SV ratio score",
        out_path=summary_dir / "sv_ratio_layer_curves_n500.png",
    )
    render_multiprefix_detector_signals(
        records,
        baseline,
        multiprefix_dir / "composite_signals_Qwen_Qwen3-8B_multiprefix.png",
    )
    render_multiprefix_paper_graphs(
        records,
        baseline,
        paper_specs,
        attention_keys,
        multiprefix_dir / "paper_style_Qwen_Qwen3-8B_multiprefix.png",
    )

    print(f"Wrote summary outputs to {summary_dir}")
    print(f"Wrote multiprefix outputs to {multiprefix_dir}")


if __name__ == "__main__":
    main()
