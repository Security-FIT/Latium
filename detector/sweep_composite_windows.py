#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from composite_detector_v2 import sweep_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    REPO_ROOT / "pipeline_out" / "n5_s30" / "structural",
    REPO_ROOT / "pipeline_out" / "n50_s30" / "structural",
]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "detector" / "graphs"
DEFAULT_TRIMS = [1, 2, 3]
DEFAULT_SMALL_WINDOWS = [3, 5, 7]
DEFAULT_LARGE_WINDOWS = [5, 7, 9, 11]
DEFAULT_KEY = (2, 5, 7)


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _parse_int_csv(raw: str) -> List[int]:
    values: List[int] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Expected positive integer, got {value}")
        values.append(value)
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _collect_json_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("rome_structural_*.json")))
        elif path.is_file() and path.suffix == ".json":
            files.append(path)
    return sorted(set(files))


def _is_gpt_family(model_name: str) -> bool:
    lower = model_name.lower()
    return "gpt2" in lower or "gpt-j" in lower or "gptj" in lower


def _short_model_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def _config_key(config: Dict) -> Tuple[int, int, int]:
    return (int(config["trim"]), int(config["small_window"]), int(config["large_window"]))


def _config_label(key: Tuple[int, int, int]) -> str:
    trim, small_window, large_window = key
    return f"t{trim}/s{small_window}/l{large_window}"


def _run_label(path: Path) -> str:
    if path.parent.name == "structural":
        return path.parent.parent.name
    return path.parent.name


def _augment_sweep(path: Path, sweep: Dict) -> Dict:
    model_name = str(sweep["model"])
    short_model = _short_model_name(model_name)
    slice_label = f"{_run_label(path)}/{short_model}"
    config_map = {_config_key(cfg): cfg for cfg in sweep["configs"]}
    default_cfg = config_map.get(DEFAULT_KEY)
    return {
        **sweep,
        "short_model": short_model,
        "run_label": _run_label(path),
        "slice_label": slice_label,
        "default_accuracy": float(default_cfg["accuracy"]) if default_cfg else 0.0,
        "default_correct": int(default_cfg["correct"]) if default_cfg else 0,
        "config_map": config_map,
    }


def _aggregate_sweeps(sweeps: List[Dict]) -> List[Dict]:
    aggregate: Dict[Tuple[int, int, int], Dict] = {}
    for sweep in sweeps:
        for cfg in sweep["configs"]:
            key = _config_key(cfg)
            bucket = aggregate.setdefault(key, {
                "trim": key[0],
                "small_window": key[1],
                "large_window": key[2],
                "correct": 0,
                "n_tests": 0,
                "slice_count": 0,
                "slice_accuracies": {},
                "slice_correct": {},
                "method_counts": {},
            })
            bucket["correct"] += int(cfg["correct"])
            bucket["n_tests"] += int(cfg["n_tests"])
            bucket["slice_count"] += 1
            bucket["slice_accuracies"][sweep["slice_label"]] = float(cfg["accuracy"])
            bucket["slice_correct"][sweep["slice_label"]] = int(cfg["correct"])
            for method, count in cfg.get("method_counts", {}).items():
                bucket["method_counts"][method] = bucket["method_counts"].get(method, 0) + int(count)

    leaderboard = list(aggregate.values())
    for row in leaderboard:
        n_tests = int(row["n_tests"])
        row["accuracy"] = (row["correct"] / n_tests) if n_tests else 0.0
        row["label"] = _config_label((row["trim"], row["small_window"], row["large_window"]))

    leaderboard.sort(
        key=lambda item: (
            item["accuracy"],
            item["correct"],
            -abs(item["trim"] - DEFAULT_KEY[0]),
            -abs(item["small_window"] - DEFAULT_KEY[1]),
            -abs(item["large_window"] - DEFAULT_KEY[2]),
        ),
        reverse=True,
    )
    return leaderboard


def _plot_leaderboard(leaderboard: List[Dict], output_path: Path, top_k: int) -> None:
    _setup_style()
    rows = leaderboard[:max(1, top_k)]
    labels = [row["label"] for row in rows]
    accs = [row["accuracy"] * 100 for row in rows]
    colors = []
    for row in rows:
        key = (row["trim"], row["small_window"], row["large_window"])
        if key == DEFAULT_KEY:
            colors.append("#4c78a8")
        elif row is rows[0]:
            colors.append("#d95f02")
        else:
            colors.append("#9aa5b1")

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.1), 5.8))
    bars = ax.bar(range(len(rows)), accs, color=colors, edgecolor="white", linewidth=0.6)
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{row['accuracy']:.1%}\n{row['correct']}/{row['n_tests']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Weighted Accuracy (%)")
    ax.set_ylim(0, min(105, max(accs + [100]) + 8))
    ax.set_title("Composite Detector Sweep Leaderboard", fontweight="bold")
    ax.axhline(100, color="black", linestyle=":", alpha=0.3)
    ax.text(0.01, 0.02, "Orange: selected global best, Blue: current default t2/s5/l7", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="jpg", bbox_inches="tight")
    plt.close(fig)


def _plot_heatmaps(
    leaderboard: List[Dict],
    trims: List[int],
    small_windows: List[int],
    large_windows: List[int],
    best_key: Tuple[int, int, int],
    output_path: Path,
) -> None:
    _setup_style()
    lookup = {
        (row["trim"], row["small_window"], row["large_window"]): row
        for row in leaderboard
    }

    fig, axes = plt.subplots(1, len(trims), figsize=(5.4 * len(trims), 5.2), constrained_layout=True)
    if len(trims) == 1:
        axes = [axes]

    for ax, trim in zip(axes, trims):
        data = np.full((len(small_windows), len(large_windows)), np.nan)
        for row_idx, small_window in enumerate(small_windows):
            for col_idx, large_window in enumerate(large_windows):
                if large_window <= small_window:
                    continue
                row = lookup.get((trim, small_window, large_window))
                if row is None:
                    continue
                data[row_idx, col_idx] = row["accuracy"] * 100

        masked = np.ma.masked_invalid(data)
        image = ax.imshow(masked, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
        for row_idx, small_window in enumerate(small_windows):
            for col_idx, large_window in enumerate(large_windows):
                if np.isnan(data[row_idx, col_idx]):
                    ax.text(col_idx, row_idx, "-", ha="center", va="center", color="#666", fontsize=9)
                    continue
                ax.text(col_idx, row_idx, f"{data[row_idx, col_idx]:.0f}", ha="center", va="center", color="black", fontsize=8)

        ax.set_xticks(range(len(large_windows)))
        ax.set_xticklabels([str(value) for value in large_windows])
        ax.set_yticks(range(len(small_windows)))
        ax.set_yticklabels([str(value) for value in small_windows])
        ax.set_xlabel("Large window")
        ax.set_ylabel("Small window")
        ax.set_title(f"Trim = {trim}", fontweight="bold")

        default_trim, default_small, default_large = DEFAULT_KEY
        if trim == default_trim and default_small in small_windows and default_large in large_windows:
            dr = small_windows.index(default_small)
            dc = large_windows.index(default_large)
            ax.add_patch(plt.Rectangle((dc - 0.5, dr - 0.5), 1, 1, fill=False, edgecolor="#4c78a8", linewidth=2.2))

        best_trim, best_small, best_large = best_key
        if trim == best_trim and best_small in small_windows and best_large in large_windows:
            br = small_windows.index(best_small)
            bc = large_windows.index(best_large)
            ax.add_patch(plt.Rectangle((bc - 0.43, br - 0.43), 0.86, 0.86, fill=False, edgecolor="#d95f02", linewidth=2.0, linestyle="--"))

    cbar = fig.colorbar(image, ax=axes, shrink=0.92)
    cbar.set_label("Weighted Accuracy (%)")
    fig.suptitle("Composite Detector Sweep Heatmaps", fontweight="bold", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="jpg", bbox_inches="tight")
    plt.close(fig)


def _plot_slice_comparison(
    sweeps: List[Dict],
    aggregate_lookup: Dict[Tuple[int, int, int], Dict],
    best_key: Tuple[int, int, int],
    output_path: Path,
) -> None:
    _setup_style()
    ordered = sorted(sweeps, key=lambda item: (item["run_label"], item["short_model"]))
    labels = [item["slice_label"] for item in ordered]
    default_accs = [item["default_accuracy"] * 100 for item in ordered]
    global_best_accs = [aggregate_lookup[best_key]["slice_accuracies"][item["slice_label"]] * 100 for item in ordered]
    slice_best_accs = [item["best_config"]["accuracy"] * 100 for item in ordered]

    x = np.arange(len(ordered))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(12, len(ordered) * 0.95), 6.0))
    ax.bar(x - width, default_accs, width=width, label="Default t2/s5/l7", color="#4c78a8")
    ax.bar(x, global_best_accs, width=width, label=f"Global best {_config_label(best_key)}", color="#d95f02")
    ax.bar(x + width, slice_best_accs, width=width, label="Per-slice best", color="#54a24b")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Default vs Global Best vs Per-Slice Best", fontweight="bold")
    ax.legend(fontsize=9, ncol=3, loc="upper center")
    ax.axhline(100, color="black", linestyle=":", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="jpg", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep and visualize composite detector trim/window settings")
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_INPUTS, help="Structural JSON files or directories")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for JSON and JPG reports")
    parser.add_argument("--sweep-trims", type=str, default=",".join(str(v) for v in DEFAULT_TRIMS))
    parser.add_argument("--sweep-small-windows", type=str, default=",".join(str(v) for v in DEFAULT_SMALL_WINDOWS))
    parser.add_argument("--sweep-large-windows", type=str, default=",".join(str(v) for v in DEFAULT_LARGE_WINDOWS))
    parser.add_argument("--top-k", type=int, default=10, help="Top configs to include in leaderboard chart")
    args = parser.parse_args()

    trims = _parse_int_csv(args.sweep_trims)
    small_windows = _parse_int_csv(args.sweep_small_windows)
    large_windows = _parse_int_csv(args.sweep_large_windows)
    json_files = _collect_json_files(args.paths)
    if not json_files:
        raise SystemExit("No structural JSON files found")

    sweeps: List[Dict] = []
    for path in json_files:
        sweep = sweep_file(path, trims=trims, small_windows=small_windows, large_windows=large_windows)
        if _is_gpt_family(str(sweep["model"])):
            continue
        sweeps.append(_augment_sweep(path, sweep))

    if not sweeps:
        raise SystemExit("No composite-model JSON files found")

    leaderboard = _aggregate_sweeps(sweeps)
    aggregate_lookup = {
        (row["trim"], row["small_window"], row["large_window"]): row
        for row in leaderboard
    }
    best = leaderboard[0]
    best_key = (best["trim"], best["small_window"], best["large_window"])
    default_row = aggregate_lookup.get(DEFAULT_KEY)

    report = {
        "selected_global_best": {
            "trim": best["trim"],
            "small_window": best["small_window"],
            "large_window": best["large_window"],
            "accuracy": best["accuracy"],
            "correct": best["correct"],
            "n_tests": best["n_tests"],
            "label": best["label"],
        },
        "default_config": {
            "trim": DEFAULT_KEY[0],
            "small_window": DEFAULT_KEY[1],
            "large_window": DEFAULT_KEY[2],
            "accuracy": default_row["accuracy"] if default_row else None,
            "correct": default_row["correct"] if default_row else None,
            "n_tests": default_row["n_tests"] if default_row else None,
            "label": _config_label(DEFAULT_KEY),
        },
        "inputs": [str(path) for path in json_files],
        "leaderboard": leaderboard,
        "slices": [
            {
                "slice_label": sweep["slice_label"],
                "model": sweep["model"],
                "run_label": sweep["run_label"],
                "target_layer": sweep["target_layer"],
                "default_accuracy": sweep["default_accuracy"],
                "best_config": sweep["best_config"],
            }
            for sweep in sweeps
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "composite_sweep_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _plot_leaderboard(leaderboard, args.output_dir / "composite_sweep_leaderboard.jpg", args.top_k)
    _plot_heatmaps(leaderboard, trims, small_windows, large_windows, best_key, args.output_dir / "composite_sweep_heatmaps.jpg")
    _plot_slice_comparison(sweeps, aggregate_lookup, best_key, args.output_dir / "composite_sweep_slice_comparison.jpg")

    print(f"Selected global best: {_config_label(best_key)} -> {best['correct']}/{best['n_tests']} ({best['accuracy']:.1%})")
    if default_row is not None:
        print(
            f"Default config: {_config_label(DEFAULT_KEY)} -> {default_row['correct']}/{default_row['n_tests']} "
            f"({default_row['accuracy']:.1%})"
        )
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())