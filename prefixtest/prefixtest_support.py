"""
prefixtest_support.py

Data loading and graph rendering for the prefix/template spectral variability
experiment. The notebook ``prefixtest.ipynb`` is a thin visualizer that
delegates heavy lifting here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_NOTEBOOKS = _ROOT / "notebooks"
_NEWGEN = _NOTEBOOKS / "new-gen"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS))
if _NEWGEN.exists() and str(_NEWGEN) not in sys.path:
    sys.path.insert(0, str(_NEWGEN))

from paper_graphs_support import plot_model_stack


LAYER_METRIC_COLS = [
    "rome_hybrid_scores",
    "sv_z_scores",
    "sv_ratio_scores",
    "pcs_composite_rank_scores",
    "sv_pcs_contradiction_scores",
    "pcs_neighbor_shift_scores",
]

METRIC_TITLES = {
    "rome_hybrid_scores": "ROME hybrid score",
    "sv_z_scores": "SV z-score",
    "sv_ratio_scores": "SV ratio",
    "pcs_composite_rank_scores": "PCS composite rank",
    "sv_pcs_contradiction_scores": "SV/PCS contradiction",
    "pcs_neighbor_shift_scores": "PCS neighbor shift",
}

MODE_DISPLAY = {
    "self": "Self",
    "template": "Template",
    "external": "External",
    "baseline": "Baseline",
}

MODE_COLORS = {
    "self": ["#0f766e", "#14b8a6", "#0ea5a4", "#115e59", "#5eead4"],
    "template": ["#7c3aed", "#a855f7", "#8b5cf6", "#6d28d9", "#c084fc"],
    "external": ["#c2410c", "#f97316", "#ea580c", "#9a3412", "#fdba74"],
}

BASELINE_COLOR = "#475569"
EDITED_TARGET_COLOR = "#111827"
MODE_LINESTYLE = {"self": "-", "template": "--", "external": ":"}

COMPOSITE_METHOD_COLORS = {
    "agree": "#2ecc71",
    "feature_consensus": "#3498db",
    "feat_diff_consensus": "#8b5cf6",
    "feat_diff_override": "#f97316",
    "spectral_diff": "#ef4444",
    "spectral_diff_strong": "#dc2626",
    "feature_single": "#eab308",
    "spectral_hybrid": "#64748b",
    "none": "#94a3b8",
}


def _safe_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def local_zscore(vals: np.ndarray, window: int = 5) -> np.ndarray:
    n = len(vals)
    half = window // 2
    out = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neighbors = np.concatenate([vals[lo:i], vals[i + 1:hi]])
        if len(neighbors) > 1:
            out[i] = (vals[i] - neighbors.mean()) / (neighbors.std() + 1e-10)
    return out


def curvature(vals: np.ndarray) -> np.ndarray:
    if len(vals) < 3:
        return np.zeros_like(vals)
    core = np.abs(vals[:-2] - 2 * vals[1:-1] + vals[2:])
    return np.concatenate([[0.0], core, [0.0]])


def find_artifact(
    root: Optional[Path] = None,
    model_filter: Optional[str] = None,
) -> Path:
    """Return the newest prefix-variability JSON under ``prefixtest/artifacts`` or ``analysis_out/``."""
    root = (root or Path.cwd()).resolve()
    candidate_dirs: list[Path] = []
    for base in [root, *root.parents]:
        artifacts_dir = base / "prefixtest" / "artifacts"
        if artifacts_dir.is_dir():
            candidate_dirs.append(artifacts_dir)
        out_dir = base / "analysis_out"
        if out_dir.is_dir():
            candidate_dirs.append(out_dir)

    if not candidate_dirs:
        candidate_dirs = [Path("artifacts"), Path("analysis_out")]

    pattern = "gpt2_prefix_template_variability_"
    if model_filter:
        pattern += model_filter.replace("/", "_") + "_"
    pattern += "*.json"

    candidates: list[Path] = []
    for out_dir in candidate_dirs:
        candidates.extend(sorted(out_dir.glob(pattern)))

    if not candidates:
        searched = ", ".join(str(path) for path in candidate_dirs)
        raise FileNotFoundError(
            f"No variability JSON found. Pattern='{pattern}'. Searched: {searched}"
        )
    return sorted(set(candidates))[-1]


def _run_sort_key(run_name: str) -> tuple[int, str]:
    if run_name == "baseline_unedited":
        return (-1, run_name)
    if run_name.startswith("self"):
        return (0, run_name)
    if run_name.startswith("template"):
        return (1, run_name)
    if run_name.startswith("external"):
        return (2, run_name)
    return (3, run_name)


def load_data(artifact_path: Path) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Load the JSON artifact and return ``(metadata, df, layer_df)``.

    * ``df``: one row per non-baseline edited run.
    * ``layer_df``: one row per (run, layer), including ``baseline_unedited``.
    """
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    edited_layer = int(metadata.get("target_layer", -1))
    runs_raw = data.get("runs", [])

    rows: list[dict] = []
    layer_rows: list[dict] = []

    for i, run in enumerate(runs_raw):
        if run.get("error"):
            continue

        run_index = int(run.get("run_index", i))
        run_name = str(run.get("run_name", ""))
        prefix_mode = str(run.get("prefix_mode", ""))
        spectral = run.get("spectral_delta") or run.get("spectral_detection") or {}
        composite = run.get("composite_detection") or {}

        layer_keys = sorted(
            {
                int(k)
                for metric in LAYER_METRIC_COLS
                for k in (spectral.get(metric) or {}).keys()
            }
        )

        for layer in layer_keys:
            row = {
                "run_index": run_index,
                "run_name": run_name,
                "prefix_mode": prefix_mode,
                "layer": layer,
            }
            for metric in LAYER_METRIC_COLS:
                row[metric] = _safe_float((spectral.get(metric) or {}).get(str(layer), np.nan))
            layer_rows.append(row)

        if run_name == "baseline_unedited":
            continue

        def _edited_layer_value(signal_key: str) -> float:
            block = spectral.get(signal_key) or {}
            for key, value in block.items():
                if int(key) == edited_layer:
                    return _safe_float(value)
            return np.nan

        rows.append(
            {
                "run_index": run_index,
                "run_name": run_name,
                "prefix_mode": prefix_mode,
                "rome_success": bool(run.get("rome_success", False)),
                "target_prob_delta": _safe_float(run.get("target_prob_delta", np.nan)),
                "update_spectral_norm": _safe_float(run.get("update_spectral_norm", np.nan)),
                "edited_layer_hybrid_score": _safe_float(run.get("edited_layer_hybrid_score", np.nan)),
                "hybrid_margin_to_top": _safe_float(run.get("hybrid_margin_to_top", np.nan)),
                "presence_confidence": _safe_float((run.get("edit_presence_detection") or {}).get("confidence")),
                "rome_hybrid_delta_at_edit_layer": _edited_layer_value("rome_hybrid_scores"),
                "sv_z_delta_at_edit_layer": _edited_layer_value("sv_z_scores"),
                "sv_ratio_delta_at_edit_layer": _edited_layer_value("sv_ratio_scores"),
                "composite_layer": composite.get("anomalous_layer"),
                "composite_method": composite.get("method_used"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = (
            df.assign(_sort_key=df["run_name"].map(_run_sort_key))
            .sort_values(["prefix_mode", "_sort_key", "run_index"])
            .drop(columns=["_sort_key"])
            .reset_index(drop=True)
        )

    layer_df = pd.DataFrame(layer_rows)
    if not layer_df.empty:
        layer_df = (
            layer_df.assign(_sort_key=layer_df["run_name"].map(_run_sort_key))
            .sort_values(["_sort_key", "run_index", "layer"])
            .drop(columns=["_sort_key"])
            .reset_index(drop=True)
        )

    if df.empty:
        raise ValueError("No successful edited runs found in artifact.")
    if layer_df.empty:
        raise ValueError("No layer-wise spectral data found in artifact.")
    return metadata, df, layer_df


def identify_baseline_run(df: pd.DataFrame) -> Tuple[int, str]:
    """Return ``(run_index, run_name)`` for a reference edited run."""
    idx = int(df["run_index"].min())
    name = str(df.loc[df["run_index"] == idx, "run_name"].iloc[0])
    return idx, name


def _baseline_layer_subset(layer_df: pd.DataFrame) -> pd.DataFrame:
    return layer_df[layer_df["run_name"] == "baseline_unedited"].sort_values("layer")


def _mode_runs(df: pd.DataFrame, prefix_mode: str) -> pd.DataFrame:
    return df[df["prefix_mode"] == prefix_mode].copy().reset_index(drop=True)


def _mode_color_map(mode_runs: pd.DataFrame, prefix_mode: str) -> dict[int, str]:
    palette = MODE_COLORS.get(prefix_mode, ["#2563eb"])
    return {
        int(mode_runs.iloc[i]["run_index"]): palette[i % len(palette)]
        for i in range(len(mode_runs))
    }


def _draw_baseline(ax, baseline_subset: pd.DataFrame, metric: str, edited_layer: int) -> None:
    if baseline_subset.empty:
        return
    ax.plot(
        baseline_subset["layer"],
        baseline_subset[metric],
        color=BASELINE_COLOR,
        linewidth=2.8,
        linestyle="-",
        alpha=0.95,
        label="Unedited baseline",
        zorder=2,
    )
    ax.axvline(
        edited_layer,
        color=EDITED_TARGET_COLOR,
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        label=f"Edited layer L{edited_layer}",
        zorder=1,
    )


def render_grouped_layer_curves(
    df: pd.DataFrame,
    layer_df: pd.DataFrame,
    edited_layer: int,
    save_dir: Optional[Path] = None,
    file_prefix: str = "layer_curves",
) -> Dict[str, plt.Figure]:
    """Render one 3x2 spectral panel per prefix mode, always with baseline."""
    baseline_subset = _baseline_layer_subset(layer_df)
    figures: Dict[str, plt.Figure] = {}

    for prefix_mode in ["self", "template", "external"]:
        mode_runs = _mode_runs(df, prefix_mode)
        if mode_runs.empty:
            continue

        color_map = _mode_color_map(mode_runs, prefix_mode)
        fig, axes = plt.subplots(3, 2, figsize=(18, 13), sharex=True)

        for ax, (metric, title) in zip(axes.flat, METRIC_TITLES.items()):
            _draw_baseline(ax, baseline_subset, metric, edited_layer)

            for _, run in mode_runs.iterrows():
                run_index = int(run["run_index"])
                run_name = str(run["run_name"])
                subset = layer_df[layer_df["run_index"] == run_index].sort_values("layer")
                if subset.empty:
                    continue

                line, = ax.plot(
                    subset["layer"],
                    subset[metric],
                    color=color_map[run_index],
                    linestyle=MODE_LINESTYLE.get(prefix_mode, "-"),
                    linewidth=2.7,
                    alpha=0.97,
                    label=run_name,
                    zorder=3,
                )
                line.set_path_effects(
                    [
                        pe.Stroke(linewidth=4.0, foreground="white", alpha=0.45),
                        pe.Normal(),
                    ]
                )

            ax.set_title(title)
            ax.set_ylabel("Score")
            ax.grid(alpha=0.22)

        for ax in axes[-1, :]:
            ax.set_xlabel("Layer")

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=9,
        )
        fig.suptitle(f"{MODE_DISPLAY.get(prefix_mode, prefix_mode)} prefix runs vs unedited baseline", fontsize=15, y=1.01)
        fig.tight_layout(rect=[0, 0, 0.84, 0.98])

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{file_prefix}_{prefix_mode}.png"
            fig.savefig(out_path, dpi=170, bbox_inches="tight")

        figures[prefix_mode] = fig

    return figures


def render_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    display_cols = [
        "run_index",
        "run_name",
        "prefix_mode",
        "edited_layer_hybrid_score",
        "hybrid_margin_to_top",
        "update_spectral_norm",
        "target_prob_delta",
        "rome_hybrid_delta_at_edit_layer",
        "sv_z_delta_at_edit_layer",
        "sv_ratio_delta_at_edit_layer",
        "composite_layer",
        "composite_method",
    ]
    return df[[col for col in display_cols if col in df.columns]]


def render_mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = [
        col
        for col in [
            "edited_layer_hybrid_score",
            "hybrid_margin_to_top",
            "update_spectral_norm",
            "target_prob_delta",
        ]
        if col in df.columns
    ]
    return df.groupby("prefix_mode")[agg_cols].agg(["mean", "std"])


def _as_paper_payload(data: dict) -> tuple[dict, dict | None, dict]:
    metadata = data.get("metadata", {})
    runs = data.get("runs", [])

    baseline_case = None
    edited_cases = []
    for run in runs:
        if run.get("error"):
            continue
        if run.get("run_name") == "baseline_unedited":
            baseline_case = dict(run)
            baseline_case["target_layer"] = metadata.get("target_layer")
            continue
        case = dict(run)
        case["target_layer"] = metadata.get("target_layer")
        edited_cases.append(case)

    edited_payload = {"metadata": {**metadata, "n_tests": len(edited_cases)}, "tests": edited_cases}
    baseline_payload = None
    if baseline_case is not None:
        baseline_payload = {
            "metadata": {**metadata, "n_tests": 1, "baseline_only": True},
            "tests": [baseline_case],
        }

    run_info = {
        "model": metadata.get("model"),
        "n_tests": len(edited_cases),
        "target_layer": metadata.get("target_layer"),
        "path": None,
        "baseline_path": None,
    }
    return edited_payload, baseline_payload, run_info


def render_paper_style_stack(
    artifact_path: Path,
    save_path: Optional[Path] = None,
    topk_svd_ranks: int = 30,
) -> None:
    data = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    edited_payload, baseline_payload, run_info = _as_paper_payload(data)

    payload_tokens = {"__prefixtest_edited_payload__": edited_payload}
    if baseline_payload is not None:
        payload_tokens["__prefixtest_baseline_payload__"] = baseline_payload

    original_load_json = plot_model_stack.__globals__["load_json"]
    original_show = plt.show
    shown = []

    def _patched_load_json(path):
        key = str(path)
        if key in payload_tokens:
            return payload_tokens[key]
        return original_load_json(path)

    def _capture_show(*args, **kwargs):
        shown.append(plt.gcf())

    plot_model_stack.__globals__["load_json"] = _patched_load_json
    plt.show = _capture_show
    try:
        plot_model_stack(
            {
                **run_info,
                "path": Path("__prefixtest_edited_payload__"),
                "baseline_path": Path("__prefixtest_baseline_payload__") if baseline_payload is not None else None,
            },
            include_baseline=baseline_payload is not None,
            show_std_band=True,
            topk_svd_ranks=topk_svd_ranks,
            trim_first_layers=0,
            trim_last_layers=0,
            save_figures=False,
            output_dir=None,
        )
    finally:
        plot_model_stack.__globals__["load_json"] = original_load_json
        plt.show = original_show

    if shown and save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        shown[-1].savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(shown[-1])


def _extract_run_feature_series(run: dict, feature_name: str) -> tuple[np.ndarray, np.ndarray]:
    layer_features = (run.get("blind_detection") or {}).get("layer_features") or {}
    layers = sorted((int(layer) for layer in layer_features.keys()))
    values = np.array([
        _safe_float((layer_features.get(str(layer)) or {}).get(feature_name))
        for layer in layers
    ], dtype=float)
    return np.array(layers, dtype=int), values


def _extract_signal_series(run: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    layers, sg_raw = _extract_run_feature_series(run, "spectral_gap")
    _, te_raw = _extract_run_feature_series(run, "top1_energy")
    _, nc_raw = _extract_run_feature_series(run, "norm_cv")
    _, er_raw = _extract_run_feature_series(run, "effective_rank")
    _, ra_raw = _extract_run_feature_series(run, "row_alignment")
    return {
        "SG (raw)": (layers, sg_raw),
        "TE (lz5)": (layers, np.abs(local_zscore(te_raw, 5))),
        "SG (lz5)": (layers, np.abs(local_zscore(sg_raw, 5))),
        "SG (lz7)": (layers, np.abs(local_zscore(sg_raw, 7))),
        "NC (lz5)": (layers, np.abs(local_zscore(nc_raw, 5))),
        "ER (curv)": (layers, curvature(er_raw)),
        "RA (raw)": (layers, ra_raw),
    }


def _run_color(run_name: str, prefix_mode: str, color_map: dict[int, str], run_index: int) -> str:
    if run_name == "baseline_unedited":
        return BASELINE_COLOR
    return color_map.get(run_index, MODE_COLORS.get(prefix_mode, ["#2563eb"])[0])


def render_composite_signal_profiles(
    artifact_path: Path,
    save_dir: Optional[Path] = None,
    file_prefix: str = "composite_signals",
) -> Dict[str, plt.Figure]:
    """Render composite-detector style signal panels per prefix mode."""
    data = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    target_layer = int(metadata.get("target_layer", -1))
    runs = [run for run in data.get("runs", []) if not run.get("error")]
    baseline = next((run for run in runs if run.get("run_name") == "baseline_unedited"), None)
    edited_runs = [run for run in runs if run.get("run_name") != "baseline_unedited"]

    figures: Dict[str, plt.Figure] = {}
    signal_names = ["SG (raw)", "TE (lz5)", "SG (lz5)", "SG (lz7)", "NC (lz5)", "ER (curv)", "RA (raw)"]

    baseline_signals = _extract_signal_series(baseline) if baseline is not None else {}

    for prefix_mode in ["self", "template", "external"]:
        mode_runs = [run for run in edited_runs if run.get("prefix_mode") == prefix_mode]
        if not mode_runs:
            continue
        mode_runs = sorted(mode_runs, key=lambda run: _run_sort_key(str(run.get("run_name", ""))))
        mode_df = pd.DataFrame(
            [{"run_index": run.get("run_index"), "run_name": run.get("run_name")} for run in mode_runs]
        )
        color_map = _mode_color_map(mode_df, prefix_mode)

        fig, axes = plt.subplots(4, 2, figsize=(17, 15), sharex=True)
        axes = axes.flat

        for ax_idx, signal_name in enumerate(signal_names):
            ax = axes[ax_idx]
            if baseline_signals:
                bl_layers, bl_values = baseline_signals[signal_name]
                ax.plot(
                    bl_layers,
                    bl_values,
                    color=BASELINE_COLOR,
                    linewidth=2.6,
                    linestyle="-",
                    alpha=0.95,
                    label="Unedited baseline",
                    zorder=2,
                )

            for run in mode_runs:
                run_name = str(run.get("run_name"))
                run_index = int(run.get("run_index"))
                signals = _extract_signal_series(run)
                layers, values = signals[signal_name]
                line, = ax.plot(
                    layers,
                    values,
                    color=_run_color(run_name, prefix_mode, color_map, run_index),
                    linewidth=2.3,
                    linestyle=MODE_LINESTYLE.get(prefix_mode, "-"),
                    alpha=0.95,
                    label=run_name,
                    zorder=3,
                )
                line.set_path_effects(
                    [
                        pe.Stroke(linewidth=3.6, foreground="white", alpha=0.4),
                        pe.Normal(),
                    ]
                )

            ax.axvline(target_layer, color=EDITED_TARGET_COLOR, linestyle=":", linewidth=1.5, alpha=0.8)
            ax.set_title(signal_name)
            ax.set_ylabel("Value")
            ax.grid(alpha=0.22)

        summary_ax = axes[-1]
        composite_rows = []
        for run in mode_runs:
            comp = run.get("composite_detection") or {}
            composite_rows.append(
                {
                    "run_name": str(run.get("run_name")),
                    "method": str(comp.get("method_used", "none")),
                    "layer": _safe_float(comp.get("anomalous_layer"), default=np.nan),
                }
            )
        comp_df = pd.DataFrame(composite_rows)
        if not comp_df.empty:
            comp_df = comp_df.sort_values("run_name").reset_index(drop=True)
            x = np.arange(len(comp_df))
            colors = [COMPOSITE_METHOD_COLORS.get(method, "#94a3b8") for method in comp_df["method"]]
            summary_ax.bar(x, comp_df["layer"], color=colors, edgecolor="white", linewidth=0.5)
            summary_ax.axhline(target_layer, color=EDITED_TARGET_COLOR, linestyle=":", linewidth=1.5, alpha=0.8)
            summary_ax.set_xticks(x)
            summary_ax.set_xticklabels(comp_df["run_name"], rotation=25, ha="right")
            summary_ax.set_ylabel("Detected layer")
            summary_ax.set_title("Composite detector picked layer")
        else:
            summary_ax.text(0.5, 0.5, "No composite data", ha="center", va="center", transform=summary_ax.transAxes)
            summary_ax.set_title("Composite detector picked layer")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
        fig.suptitle(f"Composite-style signals for {MODE_DISPLAY.get(prefix_mode, prefix_mode)} prefix runs", fontsize=15, y=0.995)
        fig.tight_layout(rect=[0, 0, 0.84, 0.98])

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{file_prefix}_{prefix_mode}.png"
            fig.savefig(out_path, dpi=170, bbox_inches="tight")

        figures[prefix_mode] = fig

    return figures


def render_composite_method_breakdown(
    artifact_path: Path,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Render a prefix-mode-by-method stacked summary for composite detector usage."""
    data = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    runs = [
        run for run in data.get("runs", [])
        if not run.get("error") and run.get("run_name") != "baseline_unedited"
    ]

    mode_method_counts: dict[str, dict[str, int]] = {mode: {} for mode in ["self", "template", "external"]}
    all_methods: set[str] = set()
    for run in runs:
        mode = str(run.get("prefix_mode"))
        method = str((run.get("composite_detection") or {}).get("method_used", "none"))
        if mode not in mode_method_counts:
            mode_method_counts[mode] = {}
        mode_method_counts[mode][method] = mode_method_counts[mode].get(method, 0) + 1
        all_methods.add(method)

    method_order = sorted(all_methods)
    x_labels = [mode for mode in ["self", "template", "external"] if mode in mode_method_counts]
    x = np.arange(len(x_labels))
    bottoms = np.zeros(len(x_labels))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for method in method_order:
        vals = [mode_method_counts.get(mode, {}).get(method, 0) for mode in x_labels]
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=method,
            color=COMPOSITE_METHOD_COLORS.get(method, "#94a3b8"),
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms += np.array(vals, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_DISPLAY.get(mode, mode) for mode in x_labels])
    ax.set_ylabel("Number of runs")
    ax.set_title("Composite detector method breakdown across prefix groups")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout(rect=[0, 0, 0.84, 1])

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    return fig


def render_composite_accuracy_summary(
    artifact_path: Path,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Render composite detection accuracy by prefix group."""
    data = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    target_layer = int((data.get("metadata") or {}).get("target_layer", -1))
    runs = [
        run for run in data.get("runs", [])
        if not run.get("error") and run.get("run_name") != "baseline_unedited"
    ]

    rows = []
    for run in runs:
        comp = run.get("composite_detection") or {}
        detected = comp.get("anomalous_layer")
        rows.append(
            {
                "prefix_mode": str(run.get("prefix_mode")),
                "run_name": str(run.get("run_name")),
                "hit": int(detected == target_layer),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No composite data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    summary = frame.groupby("prefix_mode")["hit"].mean().reindex(["self", "template", "external"]).dropna()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = [MODE_COLORS.get(mode, ["#2563eb"])[1] for mode in summary.index]
    bars = ax.bar(summary.index, summary.values * 100.0, color=colors, edgecolor="white", linewidth=0.6)
    for bar, value in zip(bars, summary.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{value:.0%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, 110)
    ax.set_ylabel("Composite hit rate (%)")
    ax.set_title("Composite detector accuracy by prefix group")
    ax.axhline(100, color="gray", linestyle=":", alpha=0.4)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    return fig


def generate_prefixtest_outputs(
    artifact_path: Path,
    output_dir: Path,
    topk_svd_ranks: int = 30,
) -> dict:
    """Generate the full prefixtest output set and return output metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, df, layer_df = load_data(artifact_path)
    edited_layer = int(metadata.get("target_layer", -1))
    baseline_run_index, baseline_run_name = identify_baseline_run(df)
    model_slug = str(metadata.get("model", "model")).replace("/", "_").replace(" ", "_")

    grouped_figures = render_grouped_layer_curves(
        df,
        layer_df,
        edited_layer=edited_layer,
        save_dir=output_dir,
        file_prefix=f"layer_curves_{model_slug}",
    )
    composite_signal_figures = render_composite_signal_profiles(
        artifact_path,
        save_dir=output_dir,
        file_prefix=f"composite_signals_{model_slug}",
    )

    summary = render_summary_table(df)
    mode_summary = render_mode_summary(df)
    summary_path = output_dir / f"summary_{model_slug}.csv"
    mode_summary_path = output_dir / f"mode_summary_{model_slug}.csv"
    summary.to_csv(summary_path, index=False)
    mode_summary.to_csv(mode_summary_path)

    paper_style_path = output_dir / f"paper_style_{model_slug}.png"
    render_paper_style_stack(artifact_path, save_path=paper_style_path, topk_svd_ranks=topk_svd_ranks)

    composite_accuracy_path = output_dir / f"composite_accuracy_{model_slug}.png"
    composite_method_path = output_dir / f"composite_method_breakdown_{model_slug}.png"
    render_composite_accuracy_summary(artifact_path, save_path=composite_accuracy_path)
    render_composite_method_breakdown(artifact_path, save_path=composite_method_path)

    return {
        "metadata": metadata,
        "df": df,
        "layer_df": layer_df,
        "edited_layer": edited_layer,
        "baseline_run_index": baseline_run_index,
        "baseline_run_name": baseline_run_name,
        "model_slug": model_slug,
        "grouped_curve_paths": {
            mode: output_dir / f"layer_curves_{model_slug}_{mode}.png"
            for mode in grouped_figures
        },
        "composite_signal_paths": {
            mode: output_dir / f"composite_signals_{model_slug}_{mode}.png"
            for mode in composite_signal_figures
        },
        "summary_path": summary_path,
        "mode_summary_path": mode_summary_path,
        "paper_style_path": paper_style_path,
        "composite_accuracy_path": composite_accuracy_path,
        "composite_method_path": composite_method_path,
    }
