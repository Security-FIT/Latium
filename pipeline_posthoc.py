#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from detector.composite_detector_v2 import (
    _find_baseline as composite_find_baseline,
    plot_signal_profiles as composite_plot_signal_profiles,
    process_file as composite_process_file,
    sweep_file as composite_sweep_file,
)
from detector.gpt_detector import plot_signals as gpt_plot_signals, process_file as gpt_process_file
from paper_graphs._newgen_utils import canonical_model_name
from paper_graphs.paper_graphs_support import plot_model_stack


DEFAULT_MODELS = [
    "gpt2-large",
    "gpt2-xl",
    "qwen3-4b",
    "qwen3-8b",
    "mistral-7b-v0.1",
    "mistral-7b-v0.3",
    "falcon-7b",
    "opt-6.7b",
    "gpt-j-6b",
    "deepseek-7b-base",
]

FEATURE_PLOT_SPECS = [
    ("Norm CV", "norm_cv", "CV", "#6c5ce7"),
    ("Spectral Gap", "spectral_gap", "Ratio", "#1f77b4"),
    ("Top-1 Energy", "top1_energy", "Energy", "#f39c12"),
    ("Effective Rank", "effective_rank", "eRank", "#16a085"),
    ("Row Alignment", "row_alignment", "Alignment", "#c0392b"),
]

REMOTE_DONE_SENTINEL = "REMOTE_GPU_DONE"
REMOTE_FAILED_SENTINEL = "REMOTE_GPU_FAILED"
LOCAL_DONE_SENTINEL = "LOCAL_POSTHOC_DONE"
LOCAL_FAILED_SENTINEL = "LOCAL_POSTHOC_FAILED"
REMOTE_STATUS_FILE = "remote_status.json"


def is_gpt_model(model_key: str) -> bool:
    return model_key.startswith("gpt2-") or model_key.startswith("gpt-j")


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def short_model_name(name: str) -> str:
    return str(name).split("/")[-1] if "/" in str(name) else str(name)


def run_label_from_structural_path(path: Optional[Path]) -> str:
    if path is None:
        return "unknown run"
    run_dir = path.parent
    if run_dir.name == "structural":
        run_dir = run_dir.parent
    parts = run_dir.parts
    if "pipeline_out" in parts:
        idx = parts.index("pipeline_out")
        rel_parts = parts[idx + 1:]
        if rel_parts:
            return "/".join(rel_parts)
    return run_dir.name


def summarize_method_counts(method_counts: Optional[dict], limit: int = 3) -> str:
    if not isinstance(method_counts, dict) or not method_counts:
        return "n/a"
    ranked = sorted(method_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return ", ".join(f"{name}:{count}" for name, count in ranked[:limit])


def infer_run_label(model_summaries: Sequence[dict]) -> str:
    for summary in model_summaries:
        structural_json = summary.get("structural_json")
        if structural_json:
            return run_label_from_structural_path(Path(structural_json))
    return "unknown run"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_mean(values: Iterable[Optional[float]]) -> float:
    cleaned = [float(v) for v in values if v is not None]
    return float(np.mean(cleaned)) if cleaned else 0.0


def load_model_meta(model_key: str) -> dict:
    cfg = OmegaConf.load(Path(__file__).resolve().parent / "src" / "config" / "model" / f"{model_key}.yaml")
    return {
        "model_key": model_key,
        "hf_name": str(getattr(cfg, "name", model_key)),
        "layer": int(getattr(cfg, "layer", -1)),
    }


def assigned_case_range(
    base_start_idx: int,
    n_tests: int,
    model_index: int,
    slice_policy: str = "iterating_per_model",
) -> tuple[int, int]:
    policy = str(slice_policy or "iterating_per_model").strip().lower()
    if policy == "shared":
        start_idx = int(base_start_idx)
    else:
        start_idx = int(base_start_idx) + int(model_index) * int(n_tests)
    end_idx = start_idx + max(0, int(n_tests) - 1)
    return start_idx, end_idx


def model_assignment_map(
    model_keys: Sequence[str],
    base_start_idx: int,
    n_tests: int,
    slice_policy: str = "iterating_per_model",
) -> Dict[str, dict]:
    assignments: Dict[str, dict] = {}
    for idx, model_key in enumerate(model_keys):
        start_idx, end_idx = assigned_case_range(base_start_idx, n_tests, idx, slice_policy=slice_policy)
        assignments[model_key] = {
            "assigned_start_idx": start_idx,
            "assigned_end_idx": end_idx,
        }
    return assignments


def describe_slice_policy(slice_policy: str, n_tests: int) -> str:
    policy = str(slice_policy or "iterating_per_model").strip().lower()
    if policy == "shared":
        return f"shared slice for all models ({n_tests} cases each)"
    return f"iterative per model with stride {n_tests}"


def summarize_tests(payload: dict, expected_n_tests: int) -> dict:
    tests = payload.get("tests", [])
    completed = [test for test in tests if not test.get("error") and not test.get("skipped")]
    errors = [test for test in tests if test.get("error") or test.get("skipped")]
    rome_success_tests = [test for test in completed if test.get("rome", {}).get("success")]
    detector_eligible = [
        test for test in rome_success_tests
        if bool(test.get("blind_detection", {}).get("layer_features"))
    ]

    summary = payload.get("summary", {})
    tested = int(summary.get("cases_total", summary.get("total", len(tests) or expected_n_tests)))
    cases_completed = int(summary.get("cases_completed", summary.get("successful", len(completed))))
    cases_error = int(summary.get("cases_error", summary.get("skipped", len(errors))))
    rome_success_count = int(summary.get("rome_success_count", len(rome_success_tests)))
    detector_eligible_count = int(summary.get("detector_eligible_count", len(detector_eligible)))

    if payload.get("error") and not tests:
        cases_completed = 0
        cases_error = max(cases_error, tested)

    return {
        "tested": tested,
        "cases_completed": cases_completed,
        "cases_error": cases_error,
        "rome_success_count": rome_success_count,
        "rome_success_rate": (rome_success_count / cases_completed) if cases_completed else 0.0,
        "detector_eligible_count": detector_eligible_count,
        "mean_efficacy_score": safe_mean(test.get("rome", {}).get("efficacy_score") for test in completed),
        "mean_efficacy_magnitude": safe_mean(test.get("rome", {}).get("efficacy_magnitude") for test in completed),
        "mean_paraphrase_score": safe_mean(test.get("rome", {}).get("paraphrase_score") for test in completed),
        "mean_neighborhood_score": safe_mean(test.get("rome", {}).get("neighborhood_score") for test in completed),
        "mean_overall_score": safe_mean(test.get("rome", {}).get("overall_score") for test in completed),
    }


def collect_latest_model_jsons(structural_dir: Path, model_keys: Sequence[str]) -> Dict[str, Path]:
    latest: Dict[str, Path] = {}
    mtimes: Dict[str, float] = {}
    if not structural_dir.exists():
        return latest

    wanted = set(model_keys)
    for path in structural_dir.glob("rome_structural_*.json"):
        try:
            payload = load_json(path)
        except Exception:
            continue
        model_name = canonical_model_name(payload.get("metadata", {}).get("model"))
        if model_name not in wanted:
            continue
        mtime = path.stat().st_mtime
        if model_name not in mtimes or mtime >= mtimes[model_name]:
            latest[model_name] = path
            mtimes[model_name] = mtime
    return latest


def run_detector_for_file(
    model_key: str,
    json_path: Path,
    graph_dir: Optional[Path],
    composite_sweep_dir: Optional[Path] = None,
) -> dict:
    if is_gpt_model(model_key):
        detector_kind = "gpt"
        result = gpt_process_file(json_path, trim=5)
        if graph_dir is not None:
            graph_dir.mkdir(parents=True, exist_ok=True)
            gpt_plot_signals(json_path, trim=5, output_dir=graph_dir)
        sweep_payload = None
        sweep_path = None
    else:
        detector_kind = "composite"
        result = composite_process_file(json_path, trim=2)
        if graph_dir is not None:
            graph_dir.mkdir(parents=True, exist_ok=True)
            composite_plot_signal_profiles(result, output_dir=graph_dir)
        sweep_payload = None
        sweep_path = None
        if composite_sweep_dir is not None:
            composite_sweep_dir.mkdir(parents=True, exist_ok=True)
            sweep_payload = composite_sweep_file(json_path)
            sweep_path = composite_sweep_dir / f"{model_key}_detector_sweep.json"
            sweep_path.write_text(json.dumps(sweep_payload, indent=2), encoding="utf-8")

    detector_summary = {
        "detector_type": detector_kind,
        "target_layer": result.get("target_layer"),
        "n_tests": int(result.get("n_tests", 0)),
        "correct": int(result.get("correct", 0)),
        "accuracy": float(result.get("accuracy", 0.0)),
        "path": str(json_path),
    }
    if "method_counts" in result:
        detector_summary["method_counts"] = result["method_counts"]
    if sweep_payload is not None and sweep_path is not None:
        best = sweep_payload.get("best_config") or {}
        detector_summary["window_sweep_path"] = str(sweep_path)
        detector_summary["window_sweep_best_accuracy"] = float(best.get("accuracy", 0.0) or 0.0)
        detector_summary["window_sweep_best_correct"] = int(best.get("correct", 0) or 0)
        detector_summary["window_sweep_best_n_tests"] = int(best.get("n_tests", 0) or 0)
        detector_summary["window_sweep_best_config"] = {
            "trim": best.get("trim"),
            "small_window": best.get("small_window"),
            "large_window": best.get("large_window"),
        }
    return detector_summary


def aggregate_blind_feature_means(payload: dict) -> Dict[str, dict]:
    per_feature: Dict[str, Dict[int, List[float]]] = {name: {} for _, name, _, _ in FEATURE_PLOT_SPECS}
    for test in payload.get("tests", []):
        if test.get("error") or test.get("skipped"):
            continue
        if not test.get("rome", {}).get("success"):
            continue
        layer_features = test.get("blind_detection", {}).get("layer_features", {})
        if not isinstance(layer_features, dict):
            continue
        for layer_key, feature_map in layer_features.items():
            if not isinstance(feature_map, dict):
                continue
            try:
                layer = int(layer_key)
            except (TypeError, ValueError):
                continue
            for _, feature_name, _, _ in FEATURE_PLOT_SPECS:
                value = feature_map.get(feature_name)
                if value is None:
                    continue
                per_feature[feature_name].setdefault(layer, []).append(float(value))

    out: Dict[str, dict] = {}
    for _, feature_name, _, _ in FEATURE_PLOT_SPECS:
        values_by_layer = per_feature[feature_name]
        if not values_by_layer:
            continue
        layers = sorted(values_by_layer)
        out[feature_name] = {
            "layers": layers,
            "mean": [float(np.mean(values_by_layer[layer])) for layer in layers],
            "std": [float(np.std(values_by_layer[layer])) for layer in layers],
        }
    return out


def _load_detector_metadata(detector_summary_path: str) -> dict:
    if not detector_summary_path:
        return {}
    try:
        return load_json(Path(detector_summary_path))
    except Exception:
        return {}


def _set_layer_ticks(ax, layers: np.ndarray) -> None:
    if layers.size == 0:
        return
    step = max(1, int(np.ceil(layers.size / 12)))
    tick_layers = layers[::step]
    ax.set_xticks(tick_layers)
    ax.set_xticklabels([str(int(layer)) for layer in tick_layers])


def cleanup_paper_graph_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for pattern in ("*_blind_features.png", "*_paper_profile.png", "*_paper_graphs.png"):
        for path in output_dir.glob(pattern):
            path.unlink()


def save_model_feature_plot(model_summary: dict, payload: dict, output_dir: Path) -> None:
    structural_json = model_summary.get("structural_json")
    if not structural_json:
        return
    structural_path = Path(structural_json)
    baseline_path = composite_find_baseline(structural_path)
    valid_case_count = sum(
        1
        for case in payload.get("tests", [])
        if not case.get("error") and case.get("rome", {}).get("success", True)
    )
    plot_model_stack(
        {
            "model": model_summary["model"],
            "path": str(structural_path),
            "baseline_path": str(baseline_path) if baseline_path is not None and baseline_path.exists() else None,
            "n_tests": valid_case_count,
            "n_runs": 1,
            "target_layer": model_summary["layer"],
            "run_label": run_label_from_structural_path(structural_path),
            "run_slug": "average",
            "flat_output_name": True,
        },
        include_baseline=True,
        show_std_band=True,
        save_figures=True,
        output_dir=output_dir,
    )


def save_aggregate_graphs(model_summaries: Sequence[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [short_model_name(summary.get("hf_name", summary["model"])) for summary in model_summaries]
    if not models:
        return
    run_label = infer_run_label(model_summaries)

    detector_acc = [summary["det_rate"] * 100.0 for summary in model_summaries]
    detector_n = [summary["det_eval"] for summary in model_summaries]
    detector_ok = [summary["det_ok"] for summary in model_summaries]
    detector_colors = [
        "#2c7fb8" if summary.get("detector_type") == "composite" else "#7b6fd6"
        for summary in model_summaries
    ]
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.9), 5))
    bars = ax.bar(range(len(models)), detector_acc, color=detector_colors)
    for bar, acc, ok, n in zip(bars, detector_acc, detector_ok, detector_n):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{ok}/{n}\n{acc:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Detector accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"Post-hoc detector accuracy — {run_label}\nAccuracy over detector-eligible edits", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "detector_accuracy.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    metric_names = [
        ("ES", "mean_efficacy_score"),
        ("PS", "mean_paraphrase_score"),
        ("NS", "mean_neighborhood_score"),
        ("S", "mean_overall_score"),
    ]
    x = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.0), 5.5))
    colors = ["#1f77b4", "#33a02c", "#ff7f00", "#6a3d9a"]
    for idx, ((label, key), color) in enumerate(zip(metric_names, colors)):
        values = [summary[key] for summary in model_summaries]
        ax.bar(x + (idx - 1.5) * width, values, width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"ROME metric means — {run_label}\nMeans over completed cases in each model slice", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "rome_metric_means.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_model_summary(
    model_key: str,
    model_meta: dict,
    json_path: Optional[Path],
    detector_dir: Path,
    detector_graph_dir: Optional[Path],
    paper_graph_dir: Optional[Path],
    expected_n_tests: int,
    assigned_start_idx: int,
    assigned_end_idx: int,
    composite_sweep_dir: Optional[Path] = None,
) -> dict:
    summary = {
        "model": model_key,
        "hf_name": model_meta["hf_name"],
        "layer": model_meta["layer"],
        "assigned_start_idx": assigned_start_idx,
        "assigned_end_idx": assigned_end_idx,
        "actual_start_idx": assigned_start_idx,
        "actual_end_idx": assigned_end_idx,
        "status": "missing_structural_json",
        "tested": expected_n_tests,
        "errors": expected_n_tests,
        "rome_ok": 0,
        "rome_rate": 0.0,
        "det_eval": 0,
        "det_ok": 0,
        "det_rate": 0.0,
        "mean_efficacy_score": 0.0,
        "mean_efficacy_magnitude": 0.0,
        "mean_paraphrase_score": 0.0,
        "mean_neighborhood_score": 0.0,
        "mean_overall_score": 0.0,
        "structural_json": "",
        "detector_summary": "",
        "detector_sweep_report": "",
        "detector_sweep_best_accuracy": 0.0,
        "detector_sweep_best_config": {},
        "detector_type": "gpt" if is_gpt_model(model_key) else "composite",
        "structural_error": "missing_structural_json",
    }

    if json_path is None or not json_path.exists():
        return summary

    payload = load_json(json_path)
    payload_summary = summarize_tests(payload, expected_n_tests=expected_n_tests)

    summary.update({
        "status": "error" if payload.get("error") else "ok",
        "actual_start_idx": int(payload.get("metadata", {}).get("start_idx_used", assigned_start_idx)),
        "actual_end_idx": int(payload.get("metadata", {}).get("end_idx_used", assigned_end_idx)),
        "tested": payload_summary["tested"],
        "errors": payload_summary["cases_error"],
        "rome_ok": payload_summary["rome_success_count"],
        "rome_rate": payload_summary["rome_success_rate"],
        "det_eval": payload_summary["detector_eligible_count"],
        "mean_efficacy_score": payload_summary["mean_efficacy_score"],
        "mean_efficacy_magnitude": payload_summary["mean_efficacy_magnitude"],
        "mean_paraphrase_score": payload_summary["mean_paraphrase_score"],
        "mean_neighborhood_score": payload_summary["mean_neighborhood_score"],
        "mean_overall_score": payload_summary["mean_overall_score"],
        "structural_json": str(json_path),
        "structural_error": payload.get("error", ""),
    })

    detector_result = {
        "detector_type": summary["detector_type"],
        "n_tests": 0,
        "correct": 0,
        "accuracy": 0.0,
        "path": str(json_path),
    }
    if payload_summary["detector_eligible_count"] > 0:
        detector_result = run_detector_for_file(
            model_key,
            json_path,
            detector_graph_dir,
            composite_sweep_dir=composite_sweep_dir,
        )
        summary["det_ok"] = detector_result["correct"]
        summary["det_rate"] = detector_result["accuracy"]
    summary["det_eval"] = detector_result["n_tests"]
    summary["det_ok"] = detector_result["correct"]
    summary["det_rate"] = detector_result["accuracy"]
    summary["detector_sweep_report"] = detector_result.get("window_sweep_path", "")
    summary["detector_sweep_best_accuracy"] = float(detector_result.get("window_sweep_best_accuracy", 0.0) or 0.0)
    summary["detector_sweep_best_config"] = detector_result.get("window_sweep_best_config", {}) or {}

    detector_dir.mkdir(parents=True, exist_ok=True)
    detector_summary_path = detector_dir / f"{model_key}_detector_summary.json"
    detector_payload = {
        **detector_result,
        "model": model_key,
        "hf_name": model_meta["hf_name"],
        "layer": model_meta["layer"],
        "assigned_start_idx": summary["assigned_start_idx"],
        "assigned_end_idx": summary["assigned_end_idx"],
        "actual_start_idx": summary["actual_start_idx"],
        "actual_end_idx": summary["actual_end_idx"],
        "tested": summary["tested"],
        "errors": summary["errors"],
        "rome_ok": summary["rome_ok"],
        "rome_rate": summary["rome_rate"],
        "det_eval": summary["det_eval"],
        "det_ok": summary["det_ok"],
        "det_rate": summary["det_rate"],
        "mean_efficacy_score": summary["mean_efficacy_score"],
        "mean_efficacy_magnitude": summary["mean_efficacy_magnitude"],
        "mean_paraphrase_score": summary["mean_paraphrase_score"],
        "mean_neighborhood_score": summary["mean_neighborhood_score"],
        "mean_overall_score": summary["mean_overall_score"],
        "structural_json": str(json_path),
        "structural_error": summary["structural_error"],
        "status": summary["status"],
    }
    if summary["detector_sweep_report"]:
        detector_payload["window_sweep_path"] = summary["detector_sweep_report"]
        detector_payload["window_sweep_best_accuracy"] = summary["detector_sweep_best_accuracy"]
        detector_payload["window_sweep_best_config"] = summary["detector_sweep_best_config"]
    detector_summary_path.write_text(json.dumps(detector_payload, indent=2), encoding="utf-8")
    summary["detector_summary"] = str(detector_summary_path)

    if paper_graph_dir is not None:
        save_model_feature_plot(summary, payload, paper_graph_dir)

    return summary


def relative_path_str(path_str: str, base: Path) -> str:
    if not path_str:
        return ""
    path = Path(path_str)
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def write_success_file(
    model_summaries: Sequence[dict],
    success_file: Path,
    archive_dir: Path,
    remote_host: str,
    remote_status: dict,
    start_idx: int,
    n_tests: int,
    slice_policy: str,
    run_root: Path,
    archive: bool,
) -> None:
    lines = [
        f"timestamp | {datetime.now().isoformat()}",
        f"remote_host | {remote_host or 'local-only'}",
        f"remote_gpu | {remote_status.get('gpu_name', 'unknown')}",
        f"start_idx | {start_idx}",
        f"counterfact_policy | {describe_slice_policy(slice_policy, n_tests)}",
        f"n_tests | {n_tests}",
        f"run_root | {run_root}",
        f"covariance_source | {remote_status.get('cov_source', 'local -> ../reimagined -> kubapc fallback')}",
        "model | layer | slice_start | slice_end | tested | errors | rome_ok | rome_rate | det_eval | det_ok | det_rate | det_sweep_acc | ES | PS | NS | S | json | detector_summary | detector_sweep_report",
    ]
    repo_root = Path(__file__).resolve().parent
    for summary in model_summaries:
        lines.append(
            " | ".join([
                summary["model"],
                str(summary["layer"]),
                str(summary["actual_start_idx"]),
                str(summary["actual_end_idx"]),
                str(summary["tested"]),
                str(summary["errors"]),
                str(summary["rome_ok"]),
                f"{summary['rome_rate']:.3f}",
                str(summary["det_eval"]),
                str(summary["det_ok"]),
                f"{summary['det_rate']:.3f}",
                f"{summary['detector_sweep_best_accuracy']:.3f}",
                f"{summary['mean_efficacy_score']:.3f}",
                f"{summary['mean_paraphrase_score']:.3f}",
                f"{summary['mean_neighborhood_score']:.3f}",
                f"{summary['mean_overall_score']:.3f}",
                relative_path_str(summary["structural_json"], repo_root),
                relative_path_str(summary["detector_summary"], repo_root),
                relative_path_str(summary["detector_sweep_report"], repo_root),
            ])
        )

    success_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if archive:
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived = archive_dir / f"success_{now_ts()}.txt"
        archived.write_text(success_file.read_text(encoding="utf-8"), encoding="utf-8")


def write_run_summaries(
    model_summaries: Sequence[dict],
    summary_dir: Path,
    run_root: Path,
    remote_host: str,
    remote_status: dict,
    start_idx: int,
    n_tests: int,
    slice_policy: str,
    archive: bool,
) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "timestamp": datetime.now().isoformat(),
        "remote_host": remote_host,
        "remote_status": remote_status,
        "start_idx": start_idx,
        "n_tests": n_tests,
        "counterfact_policy": {
            "mode": str(slice_policy or "iterating_per_model"),
            "stride": n_tests if str(slice_policy or "iterating_per_model") != "shared" else 0,
        },
        "run_root": str(run_root),
        "models": list(model_summaries),
    }

    latest_json = summary_dir / "run_summary_latest.json"
    latest_csv = summary_dir / "run_summary_latest.csv"
    latest_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    fieldnames = [
        "model",
        "hf_name",
        "layer",
        "assigned_start_idx",
        "assigned_end_idx",
        "actual_start_idx",
        "actual_end_idx",
        "status",
        "tested",
        "errors",
        "rome_ok",
        "rome_rate",
        "det_eval",
        "det_ok",
        "det_rate",
        "mean_efficacy_score",
        "mean_efficacy_magnitude",
        "mean_paraphrase_score",
        "mean_neighborhood_score",
        "mean_overall_score",
        "structural_json",
        "detector_summary",
        "detector_sweep_report",
        "detector_sweep_best_accuracy",
        "detector_sweep_best_config",
        "structural_error",
    ]
    with latest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in model_summaries:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    if archive:
        stamp = now_ts()
        (summary_dir / f"run_summary_{stamp}.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        with (summary_dir / f"run_summary_{stamp}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in model_summaries:
                writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_remote_status(run_root: Path) -> dict:
    status_path = run_root / REMOTE_STATUS_FILE
    if not status_path.exists():
        return {}
    try:
        return load_json(status_path)
    except Exception:
        return {}


def sync_remote_run(remote_host: str, remote_run_dir: str, local_run_dir: Path) -> None:
    if not remote_host or not remote_run_dir:
        return
    local_run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync",
        "-az",
        "--partial",
        "--append-verify",
        "--timeout=180",
        "-e",
        "ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=6",
        f"{remote_host}:{remote_run_dir.rstrip('/')}/",
        f"{local_run_dir}/",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError(f"rsync failed with exit code {result.returncode}")


def process_once(args: argparse.Namespace, final_archive: bool) -> List[dict]:
    model_meta = {model: load_model_meta(model) for model in args.models}
    assignments = model_assignment_map(
        args.models,
        args.start_idx,
        args.n_tests,
        slice_policy=args.slice_policy,
    )
    structural_dir = args.structural_dir
    detector_dir = args.detector_dir
    paper_graph_dir = None if args.skip_graphs else args.graphs_dir
    detector_graph_dir = None if args.skip_graphs else (args.graphs_dir / "detector")
    composite_sweep_dir = args.composite_sweep_dir

    if paper_graph_dir is not None:
        paper_graph_dir.mkdir(parents=True, exist_ok=True)
        cleanup_paper_graph_outputs(paper_graph_dir)

    latest_jsons = collect_latest_model_jsons(structural_dir, args.models)
    model_summaries = [
        build_model_summary(
            model_key=model,
            model_meta=model_meta[model],
            json_path=latest_jsons.get(model),
            detector_dir=detector_dir,
            detector_graph_dir=detector_graph_dir,
            paper_graph_dir=paper_graph_dir,
            composite_sweep_dir=composite_sweep_dir,
            expected_n_tests=args.n_tests,
            assigned_start_idx=assignments[model]["assigned_start_idx"],
            assigned_end_idx=assignments[model]["assigned_end_idx"],
        )
        for model in args.models
    ]

    remote_status = read_remote_status(args.local_run_dir)
    if not args.skip_graphs:
        save_aggregate_graphs(model_summaries, args.graphs_dir)
    write_run_summaries(
        model_summaries=model_summaries,
        summary_dir=args.summary_dir,
        run_root=args.local_run_dir,
        remote_host=args.remote_host,
        remote_status=remote_status,
        start_idx=args.start_idx,
        n_tests=args.n_tests,
        slice_policy=args.slice_policy,
        archive=final_archive,
    )
    write_success_file(
        model_summaries=model_summaries,
        success_file=args.success_file,
        archive_dir=args.archive_dir,
        remote_host=args.remote_host,
        remote_status=remote_status,
        start_idx=args.start_idx,
        n_tests=args.n_tests,
        slice_policy=args.slice_policy,
        run_root=args.local_run_dir,
        archive=final_archive,
    )
    return model_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local post-hoc pipeline runner for remote structural JSONs")
    parser.add_argument("--watch", action="store_true", help="Continuously sync remote results and refresh summaries until remote completion.")
    parser.add_argument("--remote-host", default="", help="Remote SSH host used for rsync polling.")
    parser.add_argument("--remote-run-dir", default="", help="Remote run directory synced into --local-run-dir.")
    parser.add_argument("--local-run-dir", type=Path, required=True, help="Local run root, e.g. ./pipeline_out/n50_s30")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model config keys to summarize.")
    parser.add_argument("--n-tests", type=int, default=50)
    parser.add_argument("--start-idx", type=int, default=30)
    parser.add_argument(
        "--slice-policy",
        choices=["iterating_per_model", "shared"],
        default="iterating_per_model",
        help="How to label assigned CounterFact slices in summaries.",
    )
    parser.add_argument("--sync-interval", type=int, default=60, help="Seconds between remote sync attempts in watch mode.")
    parser.add_argument("--skip-graphs", action="store_true", help="Skip detector/paper graph generation.")
    parser.add_argument("--composite-window-sweep", action="store_true", help="Run composite detector trim/window sweeps and save reports for non-GPT models.")
    parser.add_argument("--success-file", type=Path, default=Path("./success.txt"))
    args = parser.parse_args()

    args.local_run_dir = args.local_run_dir.resolve()
    args.structural_dir = args.local_run_dir / "structural"
    args.detector_dir = args.local_run_dir / "detector"
    args.graphs_dir = args.local_run_dir / "paper_graphs"
    args.summary_dir = args.local_run_dir
    args.archive_dir = args.local_run_dir / "archive"
    args.composite_sweep_dir = args.local_run_dir / "detector_sweeps" if args.composite_window_sweep else None
    args.success_file = args.success_file.resolve()
    return args


def main() -> int:
    args = parse_args()
    args.local_run_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not args.watch:
            process_once(args, final_archive=True)
            return 0

        print(f"[local-posthoc] watching {args.remote_host}:{args.remote_run_dir} -> {args.local_run_dir}")
        while True:
            try:
                sync_remote_run(args.remote_host, args.remote_run_dir, args.local_run_dir)
            except Exception as exc:
                print(f"[local-posthoc] sync warning: {exc}", file=sys.stderr)

            process_once(args, final_archive=False)

            remote_done = (args.local_run_dir / REMOTE_DONE_SENTINEL).exists()
            remote_failed = (args.local_run_dir / REMOTE_FAILED_SENTINEL).exists()
            if remote_done:
                process_once(args, final_archive=True)
                if remote_failed:
                    (args.local_run_dir / LOCAL_FAILED_SENTINEL).write_text(
                        f"remote_failed {datetime.now().isoformat()}\n",
                        encoding="utf-8",
                    )
                    return 1
                (args.local_run_dir / LOCAL_DONE_SENTINEL).write_text(
                    f"local_posthoc_done {datetime.now().isoformat()}\n",
                    encoding="utf-8",
                )
                return 0

            time.sleep(max(5, int(args.sync_interval)))
    except Exception as exc:
        (args.local_run_dir / LOCAL_FAILED_SENTINEL).write_text(
            f"{datetime.now().isoformat()} {exc}\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
