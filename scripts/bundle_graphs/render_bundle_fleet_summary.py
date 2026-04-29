#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

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

from detector.composite_detector_v2 import process_file
from paper_graphs._newgen_utils import load_json
from paper_graphs.paper_graphs_support import (
    _stack_single_specs,
    aggregate_layer_map_from_payload,
    aggregate_target_svd_profile,
)


OUTPUT_DIRNAME = "fleet_summary"
FLEET_ROOT = Path("data/fleet_single_edit_runs/fleet_out_remote")
RUN_MARKER = "_tk50_tfauto_tlauto_nl1_rw5_lw3-5-7_s01_r01_"
REFERENCE_QWEN_DIR = Path("data/final_n500_models_only/qwen3-8b")
REFERENCE_QWEN_COLOR = "#dc2626"


@dataclass
class FleetModelRecord:
    model_slug: str
    display_name: str
    status: str
    baseline_run_count: int
    baseline_latest_json: str
    rome_json: str
    tested: int
    completed: int
    errors: int
    rome_success_count: int
    rome_success_rate: float
    ovr: Optional[float]
    es: Optional[float]
    ps: Optional[float]
    ns: Optional[float]
    target_layer: Optional[int]
    detected_layer: Optional[int]
    detection_hit: Optional[bool]
    detection_method: str

    def as_row(self) -> dict:
        return {
            "model_slug": self.model_slug,
            "display_name": self.display_name,
            "status": self.status,
            "baseline_run_count": self.baseline_run_count,
            "baseline_latest_json": self.baseline_latest_json,
            "rome_json": self.rome_json,
            "tested": self.tested,
            "completed": self.completed,
            "errors": self.errors,
            "rome_success_count": self.rome_success_count,
            "rome_success_rate": self.rome_success_rate,
            "OVR": "" if self.ovr is None else self.ovr,
            "ES": "" if self.es is None else self.es,
            "PS": "" if self.ps is None else self.ps,
            "NS": "" if self.ns is None else self.ns,
            "target_layer": "" if self.target_layer is None else self.target_layer,
            "detected_layer": "" if self.detected_layer is None else self.detected_layer,
            "detection_hit": "" if self.detection_hit is None else self.detection_hit,
            "detection_method": self.detection_method,
        }


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
            "grid.alpha": 0.22,
            "legend.facecolor": "white",
            "legend.edgecolor": "#cccccc",
            "legend.framealpha": 0.95,
        }
    )


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_mean(values: Iterable[object]) -> float:
    numeric = [safe_float(value, default=np.nan) for value in values]
    filtered = [value for value in numeric if np.isfinite(value)]
    if not filtered:
        return 0.0
    return float(sum(filtered) / len(filtered))


def rel_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def parse_model_slug(path: Path, prefix: str) -> str:
    stem = path.stem.removeprefix(prefix)
    if RUN_MARKER in stem:
        return stem.split(RUN_MARKER, 1)[0]
    return stem


def model_display_name(model_slug: str) -> str:
    clean = model_slug.removeprefix("fleet_")
    owner, _, model = clean.partition("_")
    return f"{owner}/{model}" if model else clean


def structural_file(run_root: Path, prefix: str) -> Path:
    matches = sorted((run_root / "structural").glob(f"{prefix}_structural_*.json"))
    if not matches:
        raise FileNotFoundError(f"No {prefix} structural JSON under {run_root}")
    return matches[-1]


def latest_valid_json_path(paths: List[Path]) -> Optional[Path]:
    for path in sorted(paths, reverse=True):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return path
    return None


def build_paper_series(payload: dict, paper_specs: list) -> dict:
    series = {}
    for title, getter, _ylabel, plot_kind in paper_specs:
        if plot_kind == "svd":
            series[title] = aggregate_target_svd_profile(payload, top_k=30)
        else:
            series[title] = aggregate_layer_map_from_payload(payload, getter)
    return series


def summarize_payload(payload: dict) -> dict:
    summary = payload.get("summary") or {}
    tests = payload.get("tests") or []
    completed = [test for test in tests if not test.get("error") and not test.get("skipped")]
    errors = [test for test in tests if test.get("error") or test.get("skipped")]
    rome_success = [test for test in completed if test.get("rome", {}).get("success")]
    tested = safe_int(summary.get("cases_total", summary.get("total", len(tests))))
    completed_count = safe_int(summary.get("cases_completed", summary.get("successful", len(completed))))
    error_count = safe_int(summary.get("cases_error", summary.get("skipped", len(errors))))
    rome_ok = safe_int(summary.get("rome_success_count", len(rome_success)))
    rome_rate = safe_float(summary.get("rome_success_rate", (rome_ok / completed_count) if completed_count else 0.0))
    return {
        "tested": tested,
        "completed": completed_count,
        "errors": error_count,
        "rome_success_count": rome_ok,
        "rome_success_rate": rome_rate,
        "ovr": safe_float(summary.get("mean_overall_score"), safe_mean(test.get("rome", {}).get("overall_score") for test in completed)),
        "es": safe_float(summary.get("mean_efficacy_score"), safe_mean(test.get("rome", {}).get("efficacy_score") for test in completed)),
        "ps": safe_float(summary.get("mean_paraphrase_score"), safe_mean(test.get("rome", {}).get("paraphrase_score") for test in completed)),
        "ns": safe_float(summary.get("mean_neighborhood_score"), safe_mean(test.get("rome", {}).get("neighborhood_score") for test in completed)),
    }


def collect_records(bundle_root: Path) -> List[FleetModelRecord]:
    fleet_root = bundle_root / FLEET_ROOT
    baseline_map: dict[str, List[Path]] = {}
    rome_map: dict[str, Path] = {}

    for path in sorted(fleet_root.glob("baseline_structural_fleet_*.json")):
        model_slug = parse_model_slug(path, "baseline_structural_")
        baseline_map.setdefault(model_slug, []).append(path)

    for path in sorted(fleet_root.glob("rome_structural_fleet_*.json")):
        model_slug = parse_model_slug(path, "rome_structural_")
        rome_map[model_slug] = path

    records: List[FleetModelRecord] = []
    for model_slug in sorted(set(baseline_map) | set(rome_map), key=lambda item: model_display_name(item).lower()):
        baseline_paths = sorted(baseline_map.get(model_slug, []))
        baseline_latest = latest_valid_json_path(baseline_paths) if baseline_paths else None
        rome_path = rome_map.get(model_slug)
        display_name = model_display_name(model_slug)

        if rome_path is None:
            records.append(
                FleetModelRecord(
                    model_slug=model_slug,
                    display_name=display_name,
                    status="baseline_only",
                    baseline_run_count=len(baseline_paths),
                    baseline_latest_json="" if baseline_latest is None else rel_to_repo(baseline_latest),
                    rome_json="",
                    tested=0,
                    completed=0,
                    errors=0,
                    rome_success_count=0,
                    rome_success_rate=0.0,
                    ovr=None,
                    es=None,
                    ps=None,
                    ns=None,
                    target_layer=None,
                    detected_layer=None,
                    detection_hit=None,
                    detection_method="",
                )
            )
            continue

        payload = json.loads(rome_path.read_text(encoding="utf-8"))
        payload_summary = summarize_payload(payload)
        detector_result = process_file(rome_path, trim=2)
        first_detection = detector_result["results"][0] if detector_result["results"] else {}

        records.append(
            FleetModelRecord(
                model_slug=model_slug,
                display_name=display_name,
                status="rome_complete",
                baseline_run_count=len(baseline_paths),
                baseline_latest_json="" if baseline_latest is None else rel_to_repo(baseline_latest),
                rome_json=rel_to_repo(rome_path),
                tested=payload_summary["tested"],
                completed=payload_summary["completed"],
                errors=payload_summary["errors"],
                rome_success_count=payload_summary["rome_success_count"],
                rome_success_rate=payload_summary["rome_success_rate"],
                ovr=payload_summary["ovr"],
                es=payload_summary["es"],
                ps=payload_summary["ps"],
                ns=payload_summary["ns"],
                target_layer=detector_result.get("target_layer"),
                detected_layer=first_detection.get("detected"),
                detection_hit=first_detection.get("hit"),
                detection_method=str(first_detection.get("method", "")),
            )
        )
    return records


def write_tables(records: List[FleetModelRecord], output_dir: Path) -> None:
    rows = [record.as_row() for record in records]
    csv_path = output_dir / "fleet_tried_models.csv"
    json_path = output_dir / "fleet_tried_models.json"
    txt_path = output_dir / "fleet_tried_models.txt"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    txt_lines = [
        f"{record.display_name}\t{record.status}\tOVR={record.ovr:.3f}" if record.ovr is not None else f"{record.display_name}\t{record.status}"
        for record in records
    ]
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")


def render_overall_ranking(records: List[FleetModelRecord], out_path: Path) -> None:
    setup_style()
    complete_count = sum(record.ovr is not None for record in records)
    baseline_only_count = len(records) - complete_count
    ovr_values = np.array([record.ovr for record in records if record.ovr is not None], dtype=float)
    mean_ovr = float(np.mean(ovr_values)) if len(ovr_values) else 0.0
    median_ovr = float(np.median(ovr_values)) if len(ovr_values) else 0.0

    fig, (ax_counts, ax_dist) = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.9),
        gridspec_kw={"width_ratios": [1.0, 2.1]},
    )

    ax_counts.barh(["fleet"], [complete_count], color="#3b82f6", height=0.5)
    ax_counts.barh(["fleet"], [baseline_only_count], left=[complete_count], color="#d1d5db", height=0.5)
    ax_counts.text(complete_count / 2, 0, str(complete_count), ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    if baseline_only_count:
        ax_counts.text(
            complete_count + baseline_only_count / 2,
            0,
            str(baseline_only_count),
            ha="center",
            va="center",
            color="#111111",
            fontsize=12,
            fontweight="bold",
        )
    ax_counts.set_xlim(0, len(records))
    ax_counts.set_xlabel("Model count")
    ax_counts.set_title("Coverage")
    ax_counts.grid(axis="x")
    ax_counts.legend(["ROME complete", "baseline only"], loc="lower right")

    bins = np.linspace(0.0, 1.0, 11)
    ax_dist.hist(ovr_values, bins=bins, color="#3b82f6", edgecolor="white", linewidth=1.2)
    ax_dist.axvline(mean_ovr, color="#ef4444", linewidth=2.0, label=f"mean {mean_ovr:.3f}")
    ax_dist.axvline(median_ovr, color="#111827", linewidth=2.0, linestyle="--", label=f"median {median_ovr:.3f}")
    ax_dist.set_xlim(0.0, 1.0)
    ax_dist.set_xlabel("OVR (mean_overall_score)")
    ax_dist.set_ylabel("Number of models")
    ax_dist.set_title("OVR distribution")
    ax_dist.grid(axis="y")
    ax_dist.legend(loc="upper left")

    fig.suptitle(
        f"Fleet ROME success overview ({complete_count}/{len(records)} complete)",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_metric_heatmap(records: List[FleetModelRecord], out_path: Path) -> None:
    setup_style()
    completed = [record for record in records if record.ovr is not None]
    metric_names = ["OVR", "ES", "PS", "NS"]
    metric_values = [
        [getattr(record, metric_name.lower()) for record in completed]
        for metric_name in metric_names
    ]

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    box = ax.boxplot(
        metric_values,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 2.0},
        whiskerprops={"color": "#6b7280", "linewidth": 1.4},
        capprops={"color": "#6b7280", "linewidth": 1.4},
    )
    palette = ["#2563eb", "#0ea5e9", "#f59e0b", "#10b981"]
    for patch, color in zip(box["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.6)

    for idx, (values, color) in enumerate(zip(metric_values, palette), start=1):
        sorted_values = np.sort(np.asarray(values, dtype=float))
        if len(sorted_values):
            jitter = np.linspace(-0.12, 0.12, len(sorted_values))
            ax.scatter(np.full(len(sorted_values), idx) + jitter, sorted_values, s=10, color=color, alpha=0.28, linewidths=0)
            mean_value = float(np.mean(sorted_values))
            ax.scatter(idx, mean_value, s=42, marker="D", color="#111111", zorder=3)
            ax.text(idx, min(1.015, mean_value + 0.04), f"{mean_value:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(np.arange(1, len(metric_names) + 1), metric_names)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Fleet stored ROME metric distributions", fontsize=16, fontweight="bold")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_layer_detection(records: List[FleetModelRecord], out_path: Path) -> None:
    setup_style()
    completed = [record for record in records if record.detected_layer is not None]
    ordered = sorted(completed, key=lambda record: (record.detected_layer if record.detected_layer is not None else -1, record.display_name.lower()))
    fig_height = max(8.0, 1.8 + 0.24 * len(ordered))
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    y = np.arange(len(ordered))
    hits = [record.detected_layer for record in ordered if record.detection_hit]
    misses = [record.detected_layer for record in ordered if not record.detection_hit]
    hit_y = [index for index, record in enumerate(ordered) if record.detection_hit]
    miss_y = [index for index, record in enumerate(ordered) if not record.detection_hit]
    ax.scatter(hits, hit_y, color="#2ca02c", s=38, label="correct", zorder=3)
    ax.scatter(misses, miss_y, color="#d62728", s=38, label="incorrect", zorder=3)
    target_layers = [record.target_layer for record in ordered if record.target_layer is not None]
    if target_layers:
        target = int(round(float(np.mean(target_layers))))
        ax.axvline(target, color="#4b5563", linestyle="--", linewidth=1.8, label=f"target layer {target}")
    ax.set_yticks(y, [record.display_name for record in ordered])
    ax.invert_yaxis()
    ax.set_xlabel("Detected layer")
    correct = sum(1 for record in ordered if record.detection_hit)
    ax.set_title(f"Fleet ROME layer detection ({correct}/{len(ordered)} correct)", fontsize=16, fontweight="bold")
    ax.grid(axis="x")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_qwen_baseline_paper_graphs(records: List[FleetModelRecord], bundle_root: Path, out_path: Path) -> None:
    setup_style()
    reference_path = structural_file(bundle_root / REFERENCE_QWEN_DIR, "baseline")
    reference_payload = load_json(reference_path)
    paper_specs = _stack_single_specs(reference_payload)
    reference_series = build_paper_series(reference_payload, paper_specs)

    fleet_series = []
    for record in records:
        if not record.baseline_latest_json:
            continue
        baseline_path = REPO_ROOT / record.baseline_latest_json
        try:
            payload = load_json(baseline_path)
        except (OSError, json.JSONDecodeError):
            continue
        fleet_series.append((record.display_name, build_paper_series(payload, paper_specs)))

    total_plots = len(paper_specs)
    fig, axes = plt.subplots(total_plots, 1, figsize=(18, 3.0 * total_plots))
    axes = np.atleast_1d(axes)
    grey_values = np.linspace(0.35, 0.86, max(1, len(fleet_series)))
    grey_colors = [plt.cm.Greys(value) for value in grey_values]

    for ax, (title, _getter, ylabel, plot_kind) in zip(axes, paper_specs):
        positive = True
        for color, (_label, series_map) in zip(grey_colors, fleet_series):
            stats = series_map.get(title)
            if not stats:
                continue
            if plot_kind == "svd":
                ax.plot(stats["rank"], stats["mean"], color=color, linewidth=0.9, alpha=0.8)
                positive = positive and bool(np.all(np.isfinite(stats["mean"]))) and float(np.nanmin(stats["mean"])) > 0.0
            else:
                ax.plot(stats["layers"], stats["mean"], color=color, linewidth=0.9, alpha=0.8)

        reference_stats = reference_series.get(title)
        if reference_stats:
            if plot_kind == "svd":
                ax.plot(reference_stats["rank"], reference_stats["mean"], color=REFERENCE_QWEN_COLOR, linewidth=2.3)
                positive = positive and bool(np.all(np.isfinite(reference_stats["mean"]))) and float(np.nanmin(reference_stats["mean"])) > 0.0
                if positive:
                    ax.set_yscale("log")
                ax.set_xlabel("Rank", fontsize=11)
                ax.set_ylabel("Singular value", fontsize=11)
            else:
                ax.plot(reference_stats["layers"], reference_stats["mean"], color=REFERENCE_QWEN_COLOR, linewidth=2.3)
                ax.set_xlabel("Layer", fontsize=11)
                ax.set_ylabel(ylabel or "Score", fontsize=11)
        else:
            if plot_kind == "svd":
                ax.set_xlabel("Rank", fontsize=11)
                ax.set_ylabel("Singular value", fontsize=11)
            else:
                ax.set_xlabel("Layer", fontsize=11)
                ax.set_ylabel(ylabel or "Score", fontsize=11)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True)

    fig.legend(
        handles=[
            Line2D([0], [0], color=REFERENCE_QWEN_COLOR, linewidth=2.5, label="Original Qwen3-8B baseline"),
            Line2D([0], [0], color="#7a7a7a", linewidth=1.2, label=f"Fleet baselines (latest valid per model, n={len(fleet_series)})"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=True,
    )
    fig.suptitle("Fleet baseline paper-graph overlay", fontsize=18, y=0.999)
    fig.text(
        0.5,
        0.988,
        "Red = canonical Qwen3-8B baseline from final bundle | greys = latest valid fleet baseline per unique model",
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.982))
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render fleet-only tables and graphs from the bundle raw fleet outputs.")
    parser.add_argument("--bundle-root", default=str(default_bundle_root(__file__)))
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root).resolve()
    output_dir = bundle_root / "graphs" / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(bundle_root)
    if not records:
        raise RuntimeError(f"No fleet records found under {bundle_root / FLEET_ROOT}")

    write_tables(records, output_dir)
    render_overall_ranking(records, output_dir / "fleet_rome_overall_ranking.png")
    render_metric_heatmap(records, output_dir / "fleet_rome_metrics_heatmap.png")
    render_layer_detection(records, output_dir / "fleet_layer_detection.png")
    render_qwen_baseline_paper_graphs(records, bundle_root, output_dir / "fleet_qwen_baseline_paper_graphs.png")

    tried = len(records)
    completed = sum(1 for record in records if record.ovr is not None)
    print(f"Wrote fleet summary outputs to {output_dir}")
    print(f"Tried models: {tried}; completed ROME runs: {completed}")


if __name__ == "__main__":
    main()
