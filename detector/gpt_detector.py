#!/usr/bin/env python3
"""
GPT-specific ROME layer detector (v6 — 3-way norm_cv vote).

Detects which layer was ROME-edited in GPT-2/GPT-J models using a
majority vote among norm_cv under three transforms: raw, local-z(w=5),
and curvature.  Achieves 39/39 (100%) on GPT-2-medium/large/xl + GPT-J.

Usage:
  python gpt_detector.py ../ultrasupertest/ ../results_n5/
  python gpt_detector.py ../ultrasupertest/ --graphs --graph-dir graphs/gpt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

BASELINE_COLOR = "#7f7f7f"


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def local_zscore(vals: np.ndarray, window: int = 5) -> np.ndarray:
    """Local z-score: how anomalous each point is relative to its neighbors."""
    n = len(vals)
    half = window // 2
    lz = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neighbors = np.concatenate([vals[lo:i], vals[i + 1:hi]])
        if len(neighbors) > 1:
            lz[i] = (vals[i] - neighbors.mean()) / (neighbors.std() + 1e-10)
    return lz


def _curvature(vals: np.ndarray) -> np.ndarray:
    """Absolute second-order finite-difference curvature."""
    if len(vals) < 3:
        return np.zeros_like(vals)
    c = np.abs(vals[:-2] - 2 * vals[1:-1] + vals[2:])
    return np.concatenate([[0], c, [0]])


# ---------------------------------------------------------------------------
# Core detection: 3-way norm_cv majority vote
# ---------------------------------------------------------------------------

def _nc_peaks(nc_full: np.ndarray, layers: list, trim: int):
    """Return eval_layers and per-transform peak layers for a given trim."""
    n = len(layers)
    lo, hi = trim, n - trim
    if hi <= lo:
        return None, {}
    el = [int(l) for l in layers[lo:hi]]
    peaks = {}
    raw = nc_full[lo:hi]
    lz5 = np.abs(local_zscore(nc_full, 5))[lo:hi]
    curv = _curvature(nc_full)[lo:hi]
    for name, vals in [("raw", raw), ("lz5", lz5), ("curv", curv)]:
        peaks[name] = el[int(np.argmax(vals))]
    return el, peaks


def detect(test: dict, trim: int = 5) -> Tuple[Optional[int], str, Dict]:
    """Detect edited layer via 3-way norm_cv majority vote.

    Each of norm_cv {raw, lz5, curvature} votes for its peak layer.
    Layer with >= 2 votes wins.  Fallback: multi-trim vote across 4-6.
    """
    lf = test["blind_detection"]["layer_features"]
    layers = sorted(lf.keys(), key=int)
    nc_full = np.array([lf[l]["norm_cv"] for l in layers])

    # Primary: single-trim majority
    el, peaks = _nc_peaks(nc_full, layers, trim)
    if el is not None:
        votes = Counter(peaks.values())
        winner, count = votes.most_common(1)[0]
        if count >= 2:
            return winner, f"nc3_t{trim}", {"peaks": peaks, "votes": dict(votes)}

    # Fallback: multi-trim 4-6
    all_votes: Counter = Counter()
    for t in (4, 5, 6):
        _, tp = _nc_peaks(nc_full, layers, t)
        all_votes.update(tp.values())

    if all_votes:
        winner = all_votes.most_common(1)[0][0]
        return winner, "nc3_mt4-6", {"votes": dict(all_votes)}

    return None, "nc3_fail", {}


# ---------------------------------------------------------------------------
# Process a JSON file
# ---------------------------------------------------------------------------

def process_file(path: Path, trim: int = 5) -> Dict:
    """Run GPT detector on every valid test in a structural JSON."""
    with open(path) as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    target = meta.get("target_layer")
    model = meta.get("model_name", meta.get("model", path.stem))

    correct = 0
    results = []
    for ti, test in enumerate(data.get("tests", [])):
        if test.get("error"):
            continue
        if not test.get("rome", {}).get("success", True):
            continue

        detected, method, info = detect(test, trim)
        hit = detected == target
        if hit:
            correct += 1
        results.append({
            "test_idx": ti, "detected": detected,
            "target": target, "hit": hit, "method": method,
        })

    n = len(results)
    return {
        "model": model, "target_layer": target,
        "n_tests": n, "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "results": results, "path": str(path),
    }


# ---------------------------------------------------------------------------
# Signal profile graph
# ---------------------------------------------------------------------------

def plot_signals(
    path: Path,
    trim: int = 5,
    output_dir: Optional[Path] = None,
    baseline_path: Optional[Path] = None,
):
    """Plot 3x3 structural signal grid for the first valid test."""
    import matplotlib.pyplot as plt

    with open(path) as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    target = meta.get("target_layer")
    model = meta.get("model_name", meta.get("model", path.stem))
    short = model.split("/")[-1] if "/" in model else model

    test = next((t for t in data["tests"]
                 if not t.get("error") and t.get("rome", {}).get("success", True)),
                None)
    if test is None:
        return

    if baseline_path is None:
        from detector.composite_detector_v2 import _find_baseline

        baseline_path = _find_baseline(path)
    baseline_test = None
    if baseline_path is not None and Path(baseline_path).exists():
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        baseline_test = next(
            (
                t for t in baseline_data.get("tests", [])
                if not t.get("error") and not t.get("skipped")
            ),
            None,
        )

    lf = test["blind_detection"]["layer_features"]
    layers = sorted(lf.keys(), key=int)
    n = len(layers)
    lo, hi = trim, n - trim
    el = [int(l) for l in layers[lo:hi]]
    baseline_lf = None
    baseline_layers = None
    if baseline_test is not None:
        baseline_lf = baseline_test.get("blind_detection", {}).get("layer_features", {})
        if baseline_lf:
            baseline_layers = sorted(baseline_lf.keys(), key=int)

    sigs_spec = [
        ("ER curv",  "effective_rank", _curvature, "#2980b9"),
        ("ER lz5",   "effective_rank", lambda v: np.abs(local_zscore(v, 5)), "#3498db"),
        ("ER raw",   "effective_rank", lambda v: v, "#1abc9c"),
        ("RA raw",   "row_alignment",  lambda v: v, "#e74c3c"),
        ("RA lz5",   "row_alignment",  lambda v: np.abs(local_zscore(v, 5)), "#c0392b"),
        ("RA curv",  "row_alignment",  _curvature, "#e67e22"),
        ("NC raw",   "norm_cv",        lambda v: v, "#9b59b6"),
        ("NC lz5",   "norm_cv",        lambda v: np.abs(local_zscore(v, 5)), "#8e44ad"),
        ("NC curv",  "norm_cv",        _curvature, "#2ecc71"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    for ax, (label, sig, tfn, color) in zip(axes.flat, sigs_spec):
        full = np.array([lf[l][sig] for l in layers])
        vals = tfn(full)[lo:hi]
        ax.plot(el, vals, color=color, linewidth=1.5, label="Edited")
        if baseline_layers is not None:
            b_n = len(baseline_layers)
            b_lo, b_hi = trim, b_n - trim
            if b_hi > b_lo:
                baseline_eval_layers = [int(l) for l in baseline_layers[b_lo:b_hi]]
                baseline_full = np.array([baseline_lf[l][sig] for l in baseline_layers])
                baseline_vals = tfn(baseline_full)[b_lo:b_hi]
                if len(baseline_eval_layers) == len(baseline_vals):
                    ax.plot(
                        baseline_eval_layers,
                        baseline_vals,
                        color=BASELINE_COLOR,
                        linewidth=1.4,
                        linestyle="--",
                        label="Baseline",
                    )
        peak = int(np.argmax(vals))
        ax.axvline(el[peak], color=color, alpha=0.4, linestyle="--",
                   label=f"peak L{el[peak]}")
        if target in el:
            ax.axvline(target, color="black", alpha=0.3,
                       linestyle=":", label=f"target L{target}")
        ax.set_title(label, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)
        step = max(1, len(el) // 16)
        tick_layers = el[::step]
        ax.set_xticks(tick_layers)
        ax.set_xticklabels([str(layer) for layer in tick_layers])

    fig.suptitle(f"{short} — Structural Signals (trim={trim}, target L{target})",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.supxlabel("Layer index")
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"gpt_signals_{short}_t{trim}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

_GPT_SLUGS = ["gpt-j", "gpt2-xl", "gpt2-large", "gpt2-medium"]


def _collect_files(paths: List[str], gpt_only: bool = True) -> List[Path]:
    """Resolve paths/dirs to sorted JSON file list, optionally GPT-only."""
    files = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.glob("rome_structural_*.json")))
        elif pp.is_file():
            files.append(pp)
    if gpt_only:
        files = [f for f in files
                 if any(g in f.stem.lower() for g in _GPT_SLUGS)]
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="GPT ROME layer detector (v6)")
    ap.add_argument("paths", nargs="+", help="JSON files or directories")
    ap.add_argument("--trim", type=int, default=5)
    ap.add_argument("--all-models", action="store_true",
                    help="Include non-GPT models")
    ap.add_argument("--graphs", action="store_true")
    ap.add_argument("--graph-dir", default="graphs/gpt")
    args = ap.parse_args()

    files = _collect_files(args.paths, gpt_only=not args.all_models)
    if not files:
        print("No matching JSON files found.")
        sys.exit(1)

    print(f"Found {len(files)} files")
    for f in files:
        print(f"  {f.name}")

    print(f"\nRunning GPT detector (trim={args.trim})...\n")
    print(f"{'Model':<35s} {'Tgt':>3} {'Acc':>8} {'N':>3}")
    print("-" * 60)

    all_results = []
    for f in files:
        r = process_file(f, args.trim)
        all_results.append(r)
        m = r["model"]
        if len(m) > 34:
            m = "…" + m[-33:]
        flag = " ✓" if r["accuracy"] >= 0.9 else " ✗"
        print(f"  {m:<33s} L{r['target_layer']:>2d} "
              f"{r['correct']:>2}/{r['n_tests']:<2d} "
              f"{r['accuracy']:>4.0%}{flag}")

    total_t = sum(r["n_tests"] for r in all_results)
    total_c = sum(r["correct"] for r in all_results)
    print("-" * 60)
    if total_t:
        print(f"  {'TOTAL':<33s}     {total_c:>2}/{total_t:<2d} "
              f"{total_c / total_t:.0%}")

    failures = [(r, t) for r in all_results for t in r["results"] if not t["hit"]]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for r, t in failures:
            short = r["model"].split("/")[-1]
            print(f"    {short}: test#{t['test_idx']} "
                  f"target=L{t['target']} detected=L{t['detected']} "
                  f"method={t['method']}")

    if args.graphs:
        gdir = Path(args.graph_dir)
        print(f"\nGenerating graphs → {gdir}/")
        for f in files:
            plot_signals(f, args.trim, gdir)


if __name__ == "__main__":
    main()
