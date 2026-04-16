from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from _newgen_utils import (
    canonical_model_name,
    get_model_name,
    get_target_layer,
    iter_valid_tests,
    latest_structural_runs,
    load_json,
    safe_layer_map,
    select_run_files,
    sorted_layers_from_map,
)


EDITED_COLOR = "#1f77b4"
BASELINE_COLOR = "#7f7f7f"
TARGET_COLOR = "#d62728"


def find_repo_root(start: Optional[Path] = None) -> Path:
    root = (start or Path.cwd()).resolve()
    for parent in [root, *root.parents]:
        if (parent / "paper_graphs" / "paper_graphs_support.py").exists():
            return parent
        if (parent / "notebooks" / "paper_graphs_support.py").exists():
            return parent
        if (parent / "notebooks" / "new-gen" / "_newgen_utils.py").exists():
            return parent
    return root


def ultrasupertest_dir(root: Optional[Path] = None) -> Path:
    return find_repo_root(root) / "ultrasupertest"


def discover_baselines(base_dir: Path) -> Dict[str, Path]:
    baselines: Dict[str, Path] = {}
    files = sorted(
        base_dir.glob("baseline_structural_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in files:
        try:
            payload = load_json(path)
        except Exception:
            continue
        model = canonical_model_name(get_model_name(payload))
        baselines.setdefault(model, path)
    return baselines


def load_run_manifest(
    root: Optional[Path] = None,
    model_filter: Optional[Sequence[str]] = None,
    run_selection: str = "latest_model",
    max_files: int = 500,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    repo_root = find_repo_root(root)
    base_dir = ultrasupertest_dir(repo_root)
    baseline_by_model = discover_baselines(base_dir)

    run_files = latest_structural_runs(
        repo_root,
        max_files=max_files,
        model_substrings=model_filter,
        search_dirs=[base_dir],
        include_outputs=False,
        recursive=False,
    )
    run_files = select_run_files(run_files, selection=run_selection)

    runs: list[dict] = []
    for path in run_files:
        payload = load_json(path)
        model = canonical_model_name(get_model_name(payload))
        valid_tests = list(iter_valid_tests(payload))
        targets = [get_target_layer(payload, case) for case in valid_tests]
        targets = [target for target in targets if target is not None]
        runs.append(
            {
                "model": model,
                "path": path,
                "baseline_path": baseline_by_model.get(model),
                "n_tests": len(valid_tests),
                "target_layer": int(np.median(targets)) if targets else get_target_layer(payload),
            }
        )

    summary_df = pd.DataFrame(
        [
            {
                "model": run["model"],
                "tests": run["n_tests"],
                "target_layer": run["target_layer"],
                "run_file": run["path"].name,
                "has_baseline": run["baseline_path"] is not None,
                "baseline_file": run["baseline_path"].name if run["baseline_path"] else "",
            }
            for run in runs
        ]
    ).sort_values("model")

    plot_runs = [run for run in runs if run["n_tests"] > 0]
    skipped_runs = [run for run in runs if run["n_tests"] == 0]
    return summary_df, plot_runs, skipped_runs


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def nested_get(obj, *keys):
    current = obj
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def summarize_layer_values(values_by_layer: Dict[int, list[float]]) -> Optional[dict]:
    if not values_by_layer:
        return None
    layers = np.array(sorted(values_by_layer), dtype=int)
    mean = np.array([np.nanmean(values_by_layer[layer]) for layer in layers], dtype=float)
    std = np.array([np.nanstd(values_by_layer[layer]) for layer in layers], dtype=float)
    count = np.array([len(values_by_layer[layer]) for layer in layers], dtype=int)
    return {"layers": layers, "mean": mean, "std": std, "count": count}


def aggregate_layer_map_from_payload(payload: Optional[dict], getter: Callable[[dict], dict]) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    values_by_layer: Dict[int, list[float]] = {}
    for case in iter_valid_tests(payload):
        layer_map = getter(case)
        if not isinstance(layer_map, dict):
            continue
        for layer in sorted_layers_from_map(layer_map):
            value = to_float(safe_layer_map(layer_map, layer, np.nan))
            if np.isfinite(value):
                values_by_layer.setdefault(int(layer), []).append(value)
    return summarize_layer_values(values_by_layer)


def extract_layer_map(case: dict, *path) -> dict:
    layer_map = nested_get(case, *path)
    if not isinstance(layer_map, dict):
        return {}
    out = {}
    for layer in sorted_layers_from_map(layer_map):
        value = to_float(safe_layer_map(layer_map, layer, np.nan))
        if np.isfinite(value):
            out[int(layer)] = value
    return out


def extract_blind_feature(case: dict, feature_name: str) -> dict:
    layer_features = case.get("blind_detection", {}).get("layer_features", {})
    out = {}
    if not isinstance(layer_features, dict):
        return out
    for layer, feature_map in layer_features.items():
        if not isinstance(feature_map, dict):
            continue
        try:
            layer_idx = int(layer)
        except (TypeError, ValueError):
            continue
        value = to_float(feature_map.get(feature_name))
        if np.isfinite(value):
            out[layer_idx] = value
    return out


def summarize_profiles(profiles: list[np.ndarray], top_k: int) -> Optional[dict]:
    if not profiles:
        return None
    max_len = min(top_k, max(len(profile) for profile in profiles))
    matrix = np.full((len(profiles), max_len), np.nan)
    for row_idx, profile in enumerate(profiles):
        width = min(len(profile), max_len)
        matrix[row_idx, :width] = profile[:width]
    return {
        "rank": np.arange(1, max_len + 1, dtype=int),
        "mean": np.nanmean(matrix, axis=0),
        "std": np.nanstd(matrix, axis=0),
        "count": np.sum(np.isfinite(matrix), axis=0),
    }


def aggregate_target_svd_profile(payload: Optional[dict], top_k: int = 30) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    profiles: list[np.ndarray] = []
    for case in iter_valid_tests(payload):
        target_layer = get_target_layer(payload, case)
        if target_layer is None:
            continue
        raw = case.get("spectral_detection", {}).get("raw_spectral", {})
        sv_map = raw.get("sv_proj_topk", {}) if isinstance(raw, dict) else {}
        singular_values = safe_layer_map(sv_map, target_layer, None)
        if isinstance(singular_values, list) and singular_values:
            profile = np.asarray(singular_values[:top_k], dtype=float)
            if np.any(np.isfinite(profile)):
                profiles.append(profile)
    return summarize_profiles(profiles, top_k=top_k)


def trim_layer_stats(
    stats: Optional[dict],
    trim_first_layers: int = 0,
    trim_last_layers: int = 0,
) -> Optional[dict]:
    if stats is None:
        return None

    trim_first = max(0, int(trim_first_layers))
    trim_last = max(0, int(trim_last_layers))
    layers = stats.get("layers")
    if layers is None or len(layers) == 0:
        return stats

    start = min(trim_first, len(layers))
    stop = len(layers) - min(trim_last, max(0, len(layers) - start))
    if stop <= start:
        return None

    sliced = {}
    for key, value in stats.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (len(layers),):
            sliced[key] = value[start:stop]
        else:
            sliced[key] = value
    return sliced


def target_in_stats_range(target_layer: Optional[int], *stats_objects: Optional[dict]) -> bool:
    if target_layer is None:
        return False
    for stats in stats_objects:
        if stats is None:
            continue
        layers = stats.get("layers")
        if layers is None or len(layers) == 0:
            continue
        if int(layers[0]) <= target_layer <= int(layers[-1]):
            return True
    return False


def _plot_layer_series(
    ax,
    stats: Optional[dict],
    title: str,
    ylabel: str,
    target_layer: Optional[int] = None,
    baseline_stats: Optional[dict] = None,
    show_std_band: bool = True,
) -> None:
    if stats is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        return

    layers = stats["layers"]
    mean = stats["mean"]
    std = stats["std"]
    ax.plot(layers, mean, color=EDITED_COLOR, label="Edited mean")
    if show_std_band and np.any(np.isfinite(std)):
        ax.fill_between(layers, mean - std, mean + std, color=EDITED_COLOR, alpha=0.16)

    if baseline_stats is not None:
        ax.plot(
            baseline_stats["layers"],
            baseline_stats["mean"],
            color=BASELINE_COLOR,
            linestyle="--",
            linewidth=1.6,
            label="Baseline mean",
        )

    if target_in_stats_range(target_layer, stats, baseline_stats):
        ax.axvline(target_layer, color=TARGET_COLOR, linestyle=":", linewidth=1.5, label=f"Target L{target_layer}")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Layer")
    ax.legend(loc="best", fontsize=8)
    ax.margins(x=0.01)


def _plot_svd_profile(
    ax,
    stats: Optional[dict],
    title: str,
    baseline_stats: Optional[dict] = None,
    show_std_band: bool = True,
) -> None:
    if stats is None:
        ax.text(0.5, 0.5, "No raw_spectral.sv_proj_topk data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_ylabel("Singular value")
        ax.set_xlabel("Rank")
        return

    rank = stats["rank"]
    mean = stats["mean"]
    std = stats["std"]
    ax.plot(rank, mean, color=EDITED_COLOR, label="Edited target-layer mean")
    if show_std_band and np.any(np.isfinite(std)):
        ax.fill_between(rank, mean - std, mean + std, color=EDITED_COLOR, alpha=0.16)

    if baseline_stats is not None:
        ax.plot(
            baseline_stats["rank"],
            baseline_stats["mean"],
            color=BASELINE_COLOR,
            linestyle="--",
            linewidth=1.6,
            label="Baseline target-layer mean",
        )

    if np.all(np.isfinite(mean)) and np.nanmin(mean) > 0:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Singular value")
    ax.set_xlabel("Rank")
    ax.legend(loc="best", fontsize=8)
    ax.margins(x=0.01)


def _plot_multi_series(
    ax,
    series_bundle: Dict[str, Optional[dict]],
    title: str,
    ylabel: str,
    target_layer: Optional[int] = None,
) -> None:
    plotted = 0
    plotted_stats = []
    for name, stats in series_bundle.items():
        if stats is None:
            continue
        ax.plot(stats["layers"], stats["mean"], label=name)
        plotted_stats.append(stats)
        plotted += 1

    if target_in_stats_range(target_layer, *plotted_stats):
        ax.axvline(target_layer, color=TARGET_COLOR, linestyle=":", linewidth=1.5, label=f"Target L{target_layer}")

    if plotted == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.legend(loc="best", fontsize=8, ncol=2)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Layer")
    ax.margins(x=0.01)


def plot_model_stack(
    run: dict,
    include_baseline: bool = True,
    show_std_band: bool = True,
    topk_svd_ranks: int = 30,
    trim_first_layers: int = 0,
    trim_last_layers: int = 0,
    save_figures: bool = False,
    output_dir: Optional[Path] = None,
) -> None:
    model = run["model"]
    payload = load_json(run["path"])
    baseline_payload = load_json(run["baseline_path"]) if (include_baseline and run.get("baseline_path") is not None) else None
    target_layer = run.get("target_layer")

    single_specs = [
        ("Top-k proj singular values at target layer", None, None, "svd"),
        ("Signal A (sv_z_scores)", lambda case: extract_layer_map(case, "spectral_detection", "sv_z_scores"), "Score", "series"),
        ("Signal B (sv_ratio_scores)", lambda case: extract_layer_map(case, "spectral_detection", "sv_ratio_scores"), "Score", "series"),
        ("Blind spectral_gap", lambda case: extract_blind_feature(case, "spectral_gap"), "Ratio", "series"),
        ("Blind top1_energy", lambda case: extract_blind_feature(case, "top1_energy"), "Energy", "series"),
        ("Novel stable_effective_ratio", lambda case: extract_layer_map(case, "novel_metrics_detection", "per_metric_series", "stable_effective_ratio"), "Ratio", "series"),
        ("PCS next jump", lambda case: extract_layer_map(case, "spectral_detection", "pcs_next_jump_scores"), "Score", "series"),
        ("PCS next curvature", lambda case: extract_layer_map(case, "spectral_detection", "pcs_next_curvature_scores"), "Score", "series"),
        ("PCS cross scores", lambda case: extract_layer_map(case, "spectral_detection", "pcs_cross_scores"), "Score", "series"),
        ("PCS cross curvature", lambda case: extract_layer_map(case, "spectral_detection", "pcs_cross_curvature_scores"), "Score", "series"),
        ("Blind norm_cv", lambda case: extract_blind_feature(case, "norm_cv"), "CV", "series"),
        ("Blind spectral_entropy", lambda case: extract_blind_feature(case, "spectral_entropy"), "Entropy", "series"),
        ("Blind row_alignment", lambda case: extract_blind_feature(case, "row_alignment"), "Alignment", "series"),
    ]

    attention_contrast_metrics = {
        "proj_attn_norm_log_ratio": lambda case: extract_layer_map(case, "attention_detection", "contrast_series", "proj_attn_norm_log_ratio"),
        "proj_attn_top1_gap": lambda case: extract_layer_map(case, "attention_detection", "contrast_series", "proj_attn_top1_gap"),
        "proj_attn_rank_gap": lambda case: extract_layer_map(case, "attention_detection", "contrast_series", "proj_attn_rank_gap"),
        "proj_attn_stable_gap": lambda case: extract_layer_map(case, "attention_detection", "contrast_series", "proj_attn_stable_gap"),
        "attn_module_top1_dispersion": lambda case: extract_layer_map(case, "attention_detection", "contrast_series", "attn_module_top1_dispersion"),
        "attn_module_rank_dispersion": lambda case: extract_layer_map(case, "attention_detection", "contrast_series", "attn_module_rank_dispersion"),
    }

    total_plots = len(single_specs) + 1
    fig, axes = plt.subplots(total_plots, 1, figsize=(18, 3.4 * total_plots))
    target_label = f"L{target_layer}" if target_layer is not None else "N/A"
    trim_label = f" | trim=({int(trim_first_layers)}, {int(trim_last_layers)})"
    fig.suptitle(
        f"Paper graphs — {model} | target layer {target_label} | n={run['n_tests']}{trim_label}",
        fontsize=18,
        y=0.999,
    )

    for ax, (title, getter, ylabel, plot_kind) in zip(axes, single_specs):
        if plot_kind == "svd":
            edited_stats = aggregate_target_svd_profile(payload, top_k=topk_svd_ranks)
            baseline_stats = aggregate_target_svd_profile(baseline_payload, top_k=topk_svd_ranks) if baseline_payload else None
            _plot_svd_profile(ax, edited_stats, title, baseline_stats=baseline_stats, show_std_band=show_std_band)
            continue

        edited_stats = trim_layer_stats(
            aggregate_layer_map_from_payload(payload, getter),
            trim_first_layers=trim_first_layers,
            trim_last_layers=trim_last_layers,
        )
        baseline_stats = trim_layer_stats(
            aggregate_layer_map_from_payload(baseline_payload, getter) if baseline_payload else None,
            trim_first_layers=trim_first_layers,
            trim_last_layers=trim_last_layers,
        )
        _plot_layer_series(
            ax,
            edited_stats,
            title=title,
            ylabel=ylabel,
            target_layer=target_layer,
            baseline_stats=baseline_stats,
            show_std_band=show_std_band,
        )

    attention_contrast_stats = {
        name: trim_layer_stats(
            aggregate_layer_map_from_payload(payload, getter),
            trim_first_layers=trim_first_layers,
            trim_last_layers=trim_last_layers,
        )
        for name, getter in attention_contrast_metrics.items()
    }
    _plot_multi_series(
        axes[-1],
        attention_contrast_stats,
        title="Attention contrast series",
        ylabel="Score",
        target_layer=target_layer,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.996))

    if save_figures:
        if output_dir is None:
            output_dir = find_repo_root() / "paper_graphs" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = model.replace("/", "_")
        out_path = output_dir / f"{safe_name}_paper_graphs.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")

    plt.show()


def render_paper_graphs(
    runs: Sequence[dict],
    include_baseline: bool = True,
    show_std_band: bool = True,
    topk_svd_ranks: int = 30,
    trim_first_layers: int = 0,
    trim_last_layers: int = 0,
    save_figures: bool = False,
    output_dir: Optional[Path] = None,
    show_headers: bool = True,
) -> None:
    for run in runs:
        if show_headers:
            display(Markdown(f"## {run['model']}"))
        plot_model_stack(
            run,
            include_baseline=include_baseline,
            show_std_band=show_std_band,
            topk_svd_ranks=topk_svd_ranks,
            trim_first_layers=trim_first_layers,
            trim_last_layers=trim_last_layers,
            save_figures=save_figures,
            output_dir=output_dir,
        )