from __future__ import annotations

from pathlib import Path
import re
from typing import Callable, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython.display import Markdown, display
except ModuleNotFoundError:
    class Markdown(str):
        pass

    def display(*_args, **_kwargs) -> None:
        return None

from detector.composite_detector_v2 import _find_baseline as resolve_structural_baseline

try:
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
except ModuleNotFoundError:
    from paper_graphs._newgen_utils import (
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
BASELINE_COLOR = "#000000"
TARGET_COLOR = "#d62728"


def _payload_capabilities(payload_or_profile) -> dict[str, bool]:
    if isinstance(payload_or_profile, dict):
        spectral = False
        attention = False
        for case in iter_valid_tests(payload_or_profile):
            spectral_detection = case.get("spectral_detection")
            if isinstance(spectral_detection, dict) and (
                isinstance(spectral_detection.get("sv_z_scores"), dict)
                or isinstance(spectral_detection.get("raw_spectral"), dict)
            ):
                spectral = True
            attention_detection = case.get("attention_detection")
            contrast_series = attention_detection.get("contrast_series") if isinstance(attention_detection, dict) else None
            if isinstance(contrast_series, dict) and contrast_series:
                attention = True
            if spectral and attention:
                break
        return {"spectral": spectral, "attention": attention}

    profile = str(payload_or_profile or "").strip().lower()
    return {"spectral": profile == "full", "attention": profile == "full"}


def _stack_single_specs(payload_or_profile) -> list[tuple[str, Optional[Callable[[dict], dict]], Optional[str], str]]:
    blind_specs = [
        ("Blind spectral_gap", lambda case: extract_blind_feature(case, "spectral_gap"), "Ratio", "series"),
        ("Blind top1_energy", lambda case: extract_blind_feature(case, "top1_energy"), "Energy", "series"),
        ("Stable/effective rank ratio", extract_stable_effective_ratio, "Ratio", "series"),
        ("Blind norm_cv", lambda case: extract_blind_feature(case, "norm_cv"), "CV", "series"),
        ("Blind spectral_entropy", lambda case: extract_blind_feature(case, "spectral_entropy"), "Entropy", "series"),
        ("Blind row_alignment", lambda case: extract_blind_feature(case, "row_alignment"), "Alignment", "series"),
    ]
    capabilities = _payload_capabilities(payload_or_profile)
    if not capabilities["spectral"]:
        return blind_specs

    return [
        ("Top-k proj singular values at target layer", None, None, "svd"),
        ("Signal A (sv_z_scores)", lambda case: extract_layer_map(case, "spectral_detection", "sv_z_scores"), "Score", "series"),
        ("Signal B (sv_ratio_scores)", lambda case: extract_layer_map(case, "spectral_detection", "sv_ratio_scores"), "Score", "series"),
        *blind_specs[:3],
        ("PCS next jump", lambda case: extract_layer_map(case, "spectral_detection", "pcs_next_jump_scores"), "Score", "series"),
        ("PCS next curvature", lambda case: extract_layer_map(case, "spectral_detection", "pcs_next_curvature_scores"), "Score", "series"),
        ("PCS cross scores", lambda case: extract_layer_map(case, "spectral_detection", "pcs_cross_scores"), "Score", "series"),
        ("PCS cross curvature", lambda case: extract_layer_map(case, "spectral_detection", "pcs_cross_curvature_scores"), "Score", "series"),
        *blind_specs[3:],
    ]


def _stack_attention_contrast_metrics(payload_or_profile) -> Dict[str, Callable[[dict], dict]]:
    capabilities = _payload_capabilities(payload_or_profile)
    if not capabilities["attention"]:
        return {}
    return {
        "proj_attn_norm_log_ratio": lambda case: extract_layer_map(case, "attention_detection", "contrast_series", "proj_attn_norm_log_ratio"),
        "proj_attn_top1_gap": lambda case: extract_layer_map(case, "attention_detection", "contrast_series", "proj_attn_top1_gap"),
        "proj_attn_rank_gap": lambda case: extract_layer_map(case, "attention_detection", "contrast_series", "proj_attn_rank_gap"),
        "proj_attn_stable_gap": lambda case: extract_layer_map(case, "attention_detection", "contrast_series", "proj_attn_stable_gap"),
        "attn_module_top1_dispersion": lambda case: extract_layer_map(case, "attention_detection", "contrast_series", "attn_module_top1_dispersion"),
        "attn_module_rank_dispersion": lambda case: extract_layer_map(case, "attention_detection", "contrast_series", "attn_module_rank_dispersion"),
    }


def _setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "axes.titlecolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "text.color": "#111111",
            "grid.color": "#d0d0d0",
            "grid.alpha": 0.3,
            "legend.facecolor": "white",
            "legend.edgecolor": "#cccccc",
            "legend.framealpha": 0.92,
        }
    )


def _plot_baseline_overlay(ax, x_values, y_values, label: Optional[str]) -> None:
    import matplotlib.patheffects as pe

    markevery = max(1, len(x_values) // 12)
    line = ax.plot(
        x_values,
        y_values,
        color=BASELINE_COLOR,
        linestyle="--",
        linewidth=1.8,
        marker="o",
        markersize=2.3,
        markerfacecolor="white",
        markeredgecolor=BASELINE_COLOR,
        markeredgewidth=0.8,
        markevery=markevery,
        label=label,
        zorder=4,
    )[0]
    line.set_path_effects([pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()])


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip("_")) or "run"


def _run_root_from_path(path: Path) -> Path:
    run_dir = path.parent
    if run_dir.name == "structural":
        run_dir = run_dir.parent
    return run_dir


def _run_label_from_path(path: Path) -> str:
    run_dir = _run_root_from_path(path)
    parts = run_dir.parts
    if "pipeline_out" in parts:
        idx = parts.index("pipeline_out")
        rel_parts = parts[idx + 1:]
        label = "/".join(rel_parts) if rel_parts else run_dir.name
    else:
        label = run_dir.name
    timestamp = "_".join(path.stem.split("_")[-2:])
    if timestamp and timestamp not in label:
        return f"{label} | {timestamp}"
    return label


def _run_slug_from_path(path: Path) -> str:
    return _safe_slug(_run_label_from_path(path))


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


def discover_baselines(search_roots: Sequence[Path]) -> Dict[str, Path]:
    baselines: Dict[str, Path] = {}
    files = []
    seen = set()
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("baseline_structural_*.json"):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    files = sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)
    for path in files:
        try:
            payload = load_json(path)
        except Exception:
            continue
        model = canonical_model_name(get_model_name(payload))
        baselines.setdefault(model, path)
    return baselines


def structural_search_dirs(root: Path) -> list[Path]:
    candidates = [
        root / "pipeline_out",
        root / "ultrasupertest",
        root / "analysis_out",
        root / "results_n5",
    ]
    return [path for path in candidates if path.exists() and path.is_dir()]


def group_runs_by_model(runs: Sequence[dict]) -> list[list[dict]]:
    grouped: Dict[str, list[dict]] = {}
    for run in runs:
        grouped.setdefault(run["model"], []).append(run)
    return [grouped[model] for model in sorted(grouped)]


def merge_payloads(payloads: Sequence[Optional[dict]]) -> Optional[dict]:
    valid_payloads = [payload for payload in payloads if isinstance(payload, dict)]
    if not valid_payloads:
        return None

    merged = {
        "metadata": dict(valid_payloads[0].get("metadata", {})),
        "tests": [],
    }
    for payload in valid_payloads:
        merged["tests"].extend(payload.get("tests", []))
    return merged


def build_average_run(model_runs: Sequence[dict]) -> dict:
    first = model_runs[0]
    targets = [run.get("target_layer") for run in model_runs if run.get("target_layer") is not None]
    return {
        "model": first["model"],
        "paths": [run["path"] for run in model_runs],
        "baseline_paths": [run["baseline_path"] for run in model_runs if run.get("baseline_path") is not None],
        "n_tests": sum(run["n_tests"] for run in model_runs),
        "n_runs": len(model_runs),
        "target_layer": int(np.round(np.mean(targets))) if targets else first.get("target_layer"),
        "run_label": f"average across {len(model_runs)} runs",
        "run_slug": "average",
        "is_average": True,
        "flat_output_name": bool(first.get("flat_output_name", False)),
    }


def load_run_manifest(
    root: Optional[Path] = None,
    model_filter: Optional[Sequence[str]] = None,
    run_selection: str = "latest_model",
    max_files: int = 500,
    require_full_profile: bool = True,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    repo_root = find_repo_root(root)
    base_dir = ultrasupertest_dir(repo_root)
    search_dirs = structural_search_dirs(repo_root)
    baseline_by_model = discover_baselines([base_dir, *search_dirs])

    run_files = latest_structural_runs(
        repo_root,
        max_files=max_files,
        model_substrings=model_filter,
        search_dirs=search_dirs,
        include_outputs=False,
        recursive=True,
    )
    run_files = select_run_files(run_files, selection=run_selection)

    runs: list[dict] = []
    for path in run_files:
        payload = load_json(path)
        if require_full_profile and not payload_supports_paper_graphs(payload):
            continue
        model = canonical_model_name(get_model_name(payload))
        baseline_path = resolve_structural_baseline(path, payload=payload)
        if baseline_path is None:
            baseline_path = baseline_by_model.get(model)
        valid_tests = list(iter_valid_tests(payload))
        targets = [get_target_layer(payload, case) for case in valid_tests]
        targets = [target for target in targets if target is not None]
        runs.append(
            {
                "model": model,
                "path": path,
                "baseline_path": baseline_path,
                "n_tests": len(valid_tests),
                "target_layer": int(np.median(targets)) if targets else get_target_layer(payload),
                "run_label": _run_label_from_path(path),
                "run_slug": _run_slug_from_path(path),
                "analysis_profile": str(payload.get("metadata", {}).get("analysis_profile", "") or ""),
            }
        )

    summary_df = pd.DataFrame(
        [
            {
                "model": run["model"],
                "run_label": run["run_label"],
                "tests": run["n_tests"],
                "target_layer": run["target_layer"],
                "run_file": run["path"].name,
                "analysis_profile": run.get("analysis_profile", ""),
                "has_baseline": run["baseline_path"] is not None,
                "baseline_file": run["baseline_path"].name if run["baseline_path"] else "",
            }
            for run in runs
        ]
    )
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["model", "run_label"])

    plot_runs = [run for run in runs if run["n_tests"] > 0]
    skipped_runs = [run for run in runs if run["n_tests"] == 0]
    return summary_df, plot_runs, skipped_runs


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def payload_supports_paper_graphs(payload: Optional[dict]) -> bool:
    if not isinstance(payload, dict):
        return False

    profile = str(payload.get("metadata", {}).get("analysis_profile", "") or "").strip().lower()
    if profile == "raw":
        return False

    for case in iter_valid_tests(payload):
        blind_features = case.get("blind_detection", {}).get("layer_features", {})
        if blind_features:
            return True
        if profile == "full" and isinstance(case.get("spectral_detection"), dict):
            return True
    return False


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


def extract_stable_effective_ratio(case: dict) -> dict:
    direct = extract_layer_map(case, "novel_metrics_detection", "per_metric_series", "stable_effective_ratio")
    if direct:
        return direct

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

        top1_energy = to_float(feature_map.get("top1_energy"))
        effective_rank = to_float(feature_map.get("effective_rank"))
        if not np.isfinite(top1_energy) or top1_energy <= 0:
            continue
        if not np.isfinite(effective_rank) or effective_rank <= 0:
            continue

        stable_rank = 1.0 / max(top1_energy, 1e-12)
        ratio = stable_rank / max(effective_rank, 1e-12)
        if np.isfinite(ratio):
            out[layer_idx] = ratio
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
    missing_text: str = "No data",
) -> None:
    if stats is None:
        ax.text(0.5, 0.5, missing_text, ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Layer")
        return

    layers = stats["layers"]
    mean = stats["mean"]
    std = stats["std"]
    ax.plot(layers, mean, color=EDITED_COLOR, label="Edited mean")
    if show_std_band and np.any(np.isfinite(std)):
        ax.fill_between(layers, mean - std, mean + std, color=EDITED_COLOR, alpha=0.16)

    if baseline_stats is not None:
        _plot_baseline_overlay(
            ax,
            baseline_stats["layers"],
            baseline_stats["mean"],
            label="Unedited baseline mean",
        )

    if target_in_stats_range(target_layer, stats, baseline_stats):
        ax.axvline(target_layer, color=TARGET_COLOR, linestyle=":", linewidth=1.5, label=f"Target L{target_layer}")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Layer")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True)
    ax.margins(x=0.01)


def _plot_svd_profile(
    ax,
    stats: Optional[dict],
    title: str,
    baseline_stats: Optional[dict] = None,
    show_std_band: bool = True,
    missing_text: str = "No raw_spectral.sv_proj_topk data",
) -> None:
    if stats is None:
        ax.text(0.5, 0.5, missing_text, ha="center", va="center", transform=ax.transAxes)
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
        _plot_baseline_overlay(
            ax,
            baseline_stats["rank"],
            baseline_stats["mean"],
            label="Unedited baseline mean",
        )

    if np.all(np.isfinite(mean)) and np.nanmin(mean) > 0:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Singular value")
    ax.set_xlabel("Rank")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True)
    ax.margins(x=0.01)


def _plot_multi_series(
    ax,
    series_bundle: Dict[str, Optional[dict]],
    title: str,
    ylabel: str,
    target_layer: Optional[int] = None,
    baseline_bundle: Optional[Dict[str, Optional[dict]]] = None,
    missing_text: str = "No data",
) -> None:
    plotted = 0
    plotted_stats = []
    baseline_label_used = False
    for name, stats in series_bundle.items():
        if stats is None:
            continue
        ax.plot(stats["layers"], stats["mean"], label=name)
        plotted_stats.append(stats)
        plotted += 1

        baseline_stats = baseline_bundle.get(name) if baseline_bundle else None
        if baseline_stats is not None:
            label = "Unedited baseline mean" if not baseline_label_used else None
            _plot_baseline_overlay(
                ax,
                baseline_stats["layers"],
                baseline_stats["mean"],
                label=label,
            )
            plotted_stats.append(baseline_stats)
            baseline_label_used = True

    if target_in_stats_range(target_layer, *plotted_stats):
        ax.axvline(target_layer, color=TARGET_COLOR, linestyle=":", linewidth=1.5, label=f"Target L{target_layer}")

    if plotted == 0:
        ax.text(0.5, 0.5, missing_text, ha="center", va="center", transform=ax.transAxes)
    else:
        ax.legend(loc="best", fontsize=8, ncol=2)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Layer")
    ax.grid(True)
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
    _setup_style()

    model = run["model"]
    run_paths = [Path(path) for path in run.get("paths", [])]
    if not run_paths:
        run_paths = [Path(run["path"])]
    payload = merge_payloads([load_json(path) for path in run_paths])
    baseline_paths = [Path(path) for path in run.get("baseline_paths", [])]
    if not baseline_paths and run.get("baseline_path") is not None:
        baseline_paths = [Path(run["baseline_path"])]
    baseline_payload = merge_payloads([load_json(path) for path in baseline_paths]) if include_baseline else None
    target_layer = run.get("target_layer")
    run_label = run.get("run_label")
    if run_label is None:
        run_label = _run_label_from_path(run_paths[0])
    n_runs = int(run.get("n_runs", len(run_paths)) or len(run_paths))
    missing_text = "No data"

    single_specs = _stack_single_specs(payload)
    attention_contrast_metrics = _stack_attention_contrast_metrics(payload)

    total_plots = len(single_specs) + (1 if attention_contrast_metrics else 0)
    fig, axes = plt.subplots(total_plots, 1, figsize=(18, 3.4 * total_plots))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")
    target_label = f"L{target_layer}" if target_layer is not None else "N/A"
    trim_label = f" | trim=({int(trim_first_layers)}, {int(trim_last_layers)})"
    fig.suptitle(
        f"Paper graphs — {model} | {run_label}",
        fontsize=18,
        y=0.998,
    )
    fig.text(
        0.5,
        0.988,
        f"Edited curves = mean across {run['n_tests']} valid post-edit cases | dashed black = matched unedited baseline | target layer {target_label} | runs={n_runs}{trim_label}",
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )

    for ax, (title, getter, ylabel, plot_kind) in zip(axes, single_specs):
        if plot_kind == "svd":
            edited_stats = aggregate_target_svd_profile(payload, top_k=topk_svd_ranks)
            baseline_stats = aggregate_target_svd_profile(baseline_payload, top_k=topk_svd_ranks) if baseline_payload else None
            _plot_svd_profile(
                ax,
                edited_stats,
                title,
                baseline_stats=baseline_stats,
                show_std_band=show_std_band,
                missing_text=missing_text,
            )
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
            missing_text=missing_text,
        )

    if attention_contrast_metrics:
        attention_contrast_stats = {
            name: trim_layer_stats(
                aggregate_layer_map_from_payload(payload, getter),
                trim_first_layers=trim_first_layers,
                trim_last_layers=trim_last_layers,
            )
            for name, getter in attention_contrast_metrics.items()
        }
        baseline_attention_contrast_stats = {
            name: trim_layer_stats(
                aggregate_layer_map_from_payload(baseline_payload, getter) if baseline_payload else None,
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
            baseline_bundle=baseline_attention_contrast_stats,
            missing_text=missing_text,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.982))

    if save_figures:
        if output_dir is None:
            output_dir = find_repo_root() / "paper_graphs" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = model.replace("/", "_")
        suffix = run.get("run_slug", _run_slug_from_path(run_paths[0]))
        out_path = output_dir / f"{safe_name}_paper_graphs.png"
        if suffix and not run.get("flat_output_name"):
            out_path = output_dir / f"{safe_name}_{suffix}_paper_graphs.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved {out_path}")
        plt.close(fig)
        return

    plt.show()
    plt.close(fig)


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
    show_model_average: bool = True,
    show_individual_runs: bool = True,
) -> None:
    if not runs:
        return

    for model_runs in group_runs_by_model(runs):
        if not model_runs:
            continue
        model = model_runs[0]["model"]
        if show_headers:
            display(Markdown(f"## {model}"))

        if show_model_average:
            average_run = build_average_run(model_runs)
            if show_headers:
                display(Markdown(f"### {average_run['run_label']}"))
            plot_model_stack(
                average_run,
                include_baseline=include_baseline,
                show_std_band=show_std_band,
                topk_svd_ranks=topk_svd_ranks,
                trim_first_layers=trim_first_layers,
                trim_last_layers=trim_last_layers,
                save_figures=save_figures,
                output_dir=output_dir,
            )

        if not show_individual_runs:
            continue

        for run in model_runs:
            if show_headers:
                display(Markdown(f"### {run['run_label']}"))
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
