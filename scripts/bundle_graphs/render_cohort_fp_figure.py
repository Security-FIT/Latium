"""Render the Section 6.4 cohort false-positive figure.

Two-row layout:
  Row 1: per-layer envelope of the three cascade-input signals across the
         cohort baselines, with canonical pre-edit and post-edit overlays.
         Signals (matching the non-GPT cascade in composite_detector_v2.py):
            (a) raw SG_l  -- spectral_gap, no depth transform.
            (b) local-z TE_l with window 5 -- top1_energy, lz5.
            (c) local-z SG_l with window 7 -- spectral_gap, lz7.
  Row 2: model-level confusion matrix (TP/FN/TN/FP) with TPR/FPR/accuracy.

Inputs (resolved from --bundle-root):
  cohort baselines : data/fleet_single_edit_runs/fleet_out_remote/
                      baseline_structural_fleet_*.json
  cohort edits     : data/fleet_single_edit_runs/fleet_out_remote/
                      rome_structural_fleet_*.json   (used for the optional
                      "cohort ROME-edited" overlay; not strictly required)
  canonical pre   : data/final_n500_models_only/qwen3-8b/structural/
                      baseline_structural_qwen3-8b_*.json
  canonical post  : data/final_n500_models_only/qwen3-8b/structural/
                      rome_structural_qwen3-8b_*.json
  conf-matrix nums: graphs/windowed_detector/
                      fleet_windowed_detector_summary_numbers.csv

Output:
  <bundle-root>/graphs/fleet_summary/cohort_fp_combined.png
  (also copied to paper/img/detection/cohort_fp_combined.png if --paper-img
  is given.)

The canonical pre-/post-edit JSONs may not be present in every checkout of
the bundle. When either is missing the script renders the figure without
that overlay and prints a warning so the operator knows what's omitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from bundle_paths import default_bundle_root  # noqa: E402


EPS = 1e-8
RUN_MARKER = "_tk50_tfauto_tlauto_nl1_rw5_lw3-5-7_s01_r01_"

FLEET_REL = Path("data/fleet_single_edit_runs/fleet_out_remote")
CANON_REL = Path("data/final_n500_models_only/qwen3-8b/structural")
SUMMARY_CSV_REL = Path("graphs/windowed_detector/fleet_windowed_detector_summary_numbers.csv")
DEFAULT_OUT_REL = Path("graphs/fleet_summary/cohort_fp_combined.pdf")

CANONICAL_TARGET_LAYER = 10
DEFAULT_TRIM = 2

REFERENCE_COLOR = "#9f1239"     # muted wine red, used for both pre- and post-edit
REFERENCE_ALPHA = 0.92
PRE_LINEWIDTH = 1.4
POST_LINEWIDTH = 1.7
FLEET_BAND_COLOR = "#cbd5e1"
FLEET_MEDIAN_COLOR = "#475569"
TRIM_SHADE_COLOR = "#f1f5f9"
ELL_STAR_COLOR = "#0f172a"


# ----- per-layer feature extraction ---------------------------------------

def _layer_array(layer_features: dict, num_layers: int, key: str) -> np.ndarray:
    arr = np.full(num_layers, np.nan, dtype=np.float64)
    for k, v in layer_features.items():
        try:
            l = int(k)
        except ValueError:
            continue
        if 0 <= l < num_layers and v is not None and v.get(key) is not None:
            arr[l] = float(v[key])
    return arr


def local_z(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Match composite_detector_v2.local_zscore: center excluded."""
    n = len(values)
    half = window // 2
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        nb = np.concatenate([values[lo:i], values[i + 1:hi]])
        if len(nb) > 1 and np.isfinite(values[i]):
            sd = nb.std()
            out[i] = (values[i] - nb.mean()) / (sd + EPS)
    return out


def transform_signal(raw: np.ndarray, kind: str) -> np.ndarray:
    if kind == "raw":
        return raw.copy()
    if kind == "lz5":
        return np.abs(local_z(raw, 5))
    if kind == "lz7":
        return np.abs(local_z(raw, 7))
    raise ValueError(kind)


def per_layer_mean_signal(payload: dict, feature_key: str, transform: str) -> Optional[np.ndarray]:
    """Average the (transformed) per-layer feature over successful tests."""
    meta = payload.get("metadata", {})
    num_layers = int(meta.get("num_layers", 0))
    if num_layers <= 0:
        return None
    accum = np.zeros(num_layers, dtype=np.float64)
    counts = np.zeros(num_layers, dtype=np.int64)
    for t in payload.get("tests", []):
        if t.get("error"):
            continue
        rome = t.get("rome") or {}
        # Allow baseline (no rome) and successful rome cases; both populate features.
        if rome and rome.get("success") is False:
            continue
        lf = (t.get("blind_detection") or {}).get("layer_features") or {}
        if not lf:
            continue
        raw = _layer_array(lf, num_layers, feature_key)
        sig = transform_signal(raw, transform)
        ok = np.isfinite(sig)
        accum[ok] += sig[ok]
        counts[ok] += 1
    out = np.full(num_layers, np.nan, dtype=np.float64)
    nz = counts > 0
    out[nz] = accum[nz] / counts[nz]
    return out


# ----- cohort enumeration --------------------------------------------------

def parse_model_slug(path: Path, prefix: str) -> str:
    name = path.name
    if not name.startswith(prefix):
        return name
    rest = name[len(prefix):]
    idx = rest.find(RUN_MARKER)
    return rest[:idx] if idx >= 0 else rest


def latest_per_slug(paths: List[Path], prefix: str) -> Dict[str, Path]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    for p in paths:
        grouped[parse_model_slug(p, prefix)].append(p)
    out = {}
    for slug, group in grouped.items():
        # Lexicographic max picks the latest timestamp suffix.
        out[slug] = max(group, key=lambda x: x.name)
    return out


def discover_cohort_baselines(bundle_root: Path) -> List[Path]:
    cohort_root = bundle_root / FLEET_REL
    if not cohort_root.exists():
        return []
    paths = sorted(cohort_root.glob("baseline_structural_fleet_*.json"))
    latest = latest_per_slug(paths, "baseline_structural_")
    return list(latest.values())


CANONICAL_BASELINE_SEARCH_RELS = [
    # Original final_papertest run -- preferred when present.
    CANON_REL,
    # Prefix-test runs share the same un-edited Qwen3-8B base weights, so
    # their baseline_structural_*.json files are bit-identical at every
    # layer to the final_papertest baseline. The matching rome_structural_*
    # is also a valid canonical post-edit source: ROME only modifies the
    # target layer, so layers != l* are identical, and at l*=10 the
    # self-short configuration (P=20, 2-8 token prefixes) is the closest
    # empirical match to the canonical run (detector accuracy 81.8 % vs.
    # 82.2 %).
    Path("data/prefixtest_n500_runs/qwen3-8b-prefixtest-self-short/structural"),
    Path("data/prefixtest_n500_runs/qwen3-8b-prefixtest-self-medium/structural"),
    Path("data/prefixtest_n500_runs/qwen3-8b-prefixtest-self-long/structural"),
]


def discover_canonical(bundle_root: Path, prefix: str) -> Tuple[Optional[Path], Optional[str]]:
    """Return the (path, source_tag) of the canonical Qwen3-8B structural JSON.

    Searches CANONICAL_BASELINE_SEARCH_RELS in order; every match yields a
    valid canonical source on this base model. The source tag identifies
    which physical run produced the file so the operator can verify.
    """
    for rel in CANONICAL_BASELINE_SEARCH_RELS:
        d = bundle_root / rel
        if not d.exists():
            continue
        # Try the canonical filename pattern first (slug == "qwen3-8b").
        cands = sorted(d.glob(f"{prefix}_qwen3-8b_*.json"))
        if cands:
            return cands[-1], "qwen3-8b"
        # Then any prefix-test variant in this directory.
        slug = rel.parent.name
        cands = sorted(d.glob(f"{prefix}_{slug}_*.json"))
        if cands:
            return cands[-1], slug
    return None, None


# ----- envelope assembly --------------------------------------------------

SIGNALS = [
    ("SG (raw)",            "spectral_gap", "raw", "linear",
     r"$\mathrm{SG}_\ell$"),
    (r"TE (local-$z$, $w{=}5$)", "top1_energy",  "lz5", "linear",
     r"$|z_5|\,\mathrm{TE}_\ell$"),
    (r"SG (local-$z$, $w{=}7$)", "spectral_gap", "lz7", "linear",
     r"$|z_7|\,\mathrm{SG}_\ell$"),
]


def collect_cohort_signal(cohort_paths: List[Path], feature_key: str, transform: str
                         ) -> Tuple[np.ndarray, int]:
    """Return (matrix shape n_models x num_layers, num_layers)."""
    rows = []
    num_layers = None
    for p in cohort_paths:
        try:
            payload = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sig = per_layer_mean_signal(payload, feature_key, transform)
        if sig is None or not np.any(np.isfinite(sig)):
            continue
        if num_layers is None:
            num_layers = len(sig)
        elif len(sig) != num_layers:
            # Skip incompatible-depth cohort members (rare; exotic Qwen variants).
            continue
        rows.append(sig)
    if not rows:
        return np.empty((0, 0)), 0
    return np.vstack(rows), num_layers or 0


# ----- confusion-matrix CSV ----------------------------------------------

def read_summary_numbers(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    return None


# ----- panel rendering ----------------------------------------------------

def shade_trim(ax, num_layers: int, trim: int) -> None:
    if trim <= 0:
        return
    ax.axvspan(-0.5, trim - 0.5, color=TRIM_SHADE_COLOR, zorder=0)
    ax.axvspan(num_layers - trim - 0.5, num_layers - 0.5, color=TRIM_SHADE_COLOR, zorder=0)


def render_signal_panel(ax, title: str, ylabel: str,
                        cohort_matrix: np.ndarray,
                        canon_pre: Optional[np.ndarray],
                        canon_post: Optional[np.ndarray],
                        ell_star: int,
                        trim: int) -> None:
    if cohort_matrix.size == 0:
        ax.text(0.5, 0.5, "No fine-tune data", transform=ax.transAxes,
                ha="center", va="center", color="#64748b", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        return
    n_layers = cohort_matrix.shape[1]
    layers = np.arange(n_layers)
    p05 = np.nanpercentile(cohort_matrix, 5,  axis=0)
    p95 = np.nanpercentile(cohort_matrix, 95, axis=0)
    median = np.nanmedian(cohort_matrix, axis=0)
    shade_trim(ax, n_layers, trim)
    ax.fill_between(layers, p05, p95, color=FLEET_BAND_COLOR, alpha=0.85, linewidth=0,
                    label=f"Fine-tunes 5–95 pct (n={cohort_matrix.shape[0]})")
    ax.plot(layers, median, color=FLEET_MEDIAN_COLOR, linewidth=1.3,
            label="Fine-tune median")
    if canon_pre is not None and len(canon_pre) == n_layers:
        ax.plot(layers, canon_pre, color=REFERENCE_COLOR, linewidth=PRE_LINEWIDTH,
                linestyle=(0, (5, 2)), alpha=REFERENCE_ALPHA,
                label="Qwen3-8B baseline")
    if canon_post is not None and len(canon_post) == n_layers:
        ax.plot(layers, canon_post, color=REFERENCE_COLOR, linewidth=POST_LINEWIDTH,
                alpha=REFERENCE_ALPHA,
                label=f"Qwen3-8B + ROME at $\\ell^*{{=}}{ell_star}$")
    ax.axvline(ell_star, color=ELL_STAR_COLOR, linestyle=":", linewidth=0.9,
               alpha=0.75)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_xlabel("Layer index $\\ell$", fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.35)


def render_confusion_panel(ax, summary: Optional[dict],
                           title_size: float = 10,
                           cell_value_size: float = 14,
                           cell_label_size: float = 9,
                           tick_size: float = 8) -> None:
    """Render the confusion matrix panel.

    Default font sizes are homogenized with figs 2/3 (per-panel title=10,
    tick labels=8); the cell value/label sizes are scaled to remain
    legible inside the small column-width canvas of fig 4.
    """
    if summary is None:
        ax.text(0.5, 0.5, "summary missing", transform=ax.transAxes,
                ha="center", va="center", color="#64748b", fontsize=9)
        ax.axis("off")
        return
    tp = int(summary["model_tp"])
    fn = int(summary["model_fn"])
    tn = int(summary["model_tn"])
    fp = int(summary["model_fp"])
    tpr = float(summary["model_edited_detection_rate"])
    fpr = float(summary["model_baseline_false_positive_rate"])
    acc = float(summary["model_overall_accuracy"])
    matrix = np.array([[tp, fn], [fp, tn]], dtype=float)
    # Wine-tinted colormap aligned with figs 2/3/5/6 reference colour.
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "wine", ["#fff5f7", "#fda4af", "#9f1239"]
    )
    ax.imshow(matrix, cmap=cmap, aspect="equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted ROME", "Predicted clean"], fontsize=tick_size)
    ax.set_yticklabels(["Edited", "Un-edited"], fontsize=tick_size)
    cell_labels = [["TP", "FN"], ["FP", "TN"]]
    cell_counts = [[tp, fn], [fp, tn]]
    for r in range(2):
        for c in range(2):
            v = matrix[r, c]
            color = "white" if v > matrix.max() * 0.55 else "#0f172a"
            ax.text(c, r - 0.18, cell_labels[r][c], ha="center", va="center",
                    color=color, fontsize=cell_label_size, fontweight="bold")
            ax.text(c, r + 0.20, str(cell_counts[r][c]), ha="center", va="center",
                    color=color, fontsize=cell_value_size, fontweight="bold")
    title = (f"Model-level outcomes\n"
             f"TPR {tpr*100:.2f}%  $\\cdot$  FPR {fpr*100:.2f}%  $\\cdot$  Acc. {acc*100:.2f}%")
    ax.set_title(title, fontsize=title_size, fontweight="bold", pad=4)


def build_confusion_only_figure(bundle_root: Path) -> plt.Figure:
    """Single-column confusion-matrix figure for §6.4.

    Sized for ACM CCS \\columnwidth (~3.3 in). The signal envelopes that
    used to share this figure now live in the appendix as
    cohort_envelope_unedited.png and cohort_envelope_edited.png.
    """
    summary = read_summary_numbers(bundle_root / SUMMARY_CSV_REL)
    if summary is None:
        print(f"[warn] {bundle_root / SUMMARY_CSV_REL} not found; "
              f"outcomes panel will show a placeholder.", file=sys.stderr)

    fig = plt.figure(figsize=(3.3, 2.8))
    ax = fig.add_subplot(1, 1, 1)
    # Font sizes match figs 2/3 conventions (title=10, ticks=8) with the
    # cell value/label sizes scaled to the column-width canvas.
    render_confusion_panel(
        ax, summary,
        title_size=10, cell_value_size=14, cell_label_size=9, tick_size=8,
    )
    # Wrap the x-axis labels onto two lines so they don't collide.
    ax.set_xticklabels(["Predicted\nROME", "Predicted\nclean"], fontsize=8)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.83, bottom=0.18)
    return fig


# ----- top-level driver ---------------------------------------------------

def build_figure(bundle_root: Path, ell_star: int, trim: int) -> plt.Figure:
    cohort_paths = discover_cohort_baselines(bundle_root)
    if not cohort_paths:
        raise SystemExit(
            f"No fine-tune baselines found under {bundle_root / FLEET_REL}; "
            "cannot render the false-positive audit figure."
        )
    canon_pre_path,  pre_tag  = discover_canonical(bundle_root, "baseline_structural")
    canon_post_path, post_tag = discover_canonical(bundle_root, "rome_structural")
    canon_pre_payload = json.loads(canon_pre_path.read_text()) if canon_pre_path else None
    canon_post_payload = json.loads(canon_post_path.read_text()) if canon_post_path else None
    if canon_pre_payload is None:
        print(f"[warn] no Qwen3-8B baseline_structural_*.json found under any "
              f"of {[str(r) for r in CANONICAL_BASELINE_SEARCH_RELS]}; "
              f"baseline reference will be omitted.", file=sys.stderr)
    else:
        print(f"[info] Qwen3-8B baseline source: {canon_pre_path.relative_to(bundle_root)}",
              file=sys.stderr)
    if canon_post_payload is None:
        print(f"[warn] no Qwen3-8B rome_structural_*.json found under any "
              f"of {[str(r) for r in CANONICAL_BASELINE_SEARCH_RELS]}; "
              f"post-edit reference will be omitted.", file=sys.stderr)
    else:
        print(f"[info] Qwen3-8B + ROME source: {canon_post_path.relative_to(bundle_root)}",
              file=sys.stderr)

    summary = read_summary_numbers(bundle_root / SUMMARY_CSV_REL)
    if summary is None:
        print(f"[warn] {bundle_root / SUMMARY_CSV_REL} not found; "
              f"outcomes panel will show a placeholder.", file=sys.stderr)

    # Two-row layout sized for ACM CCS figure* (full text width ~ 7 in).
    # Top row: three envelope panels (~2.1 in each).
    # Bottom row: confusion matrix on the left, legend on the right.
    fig, axes = plt.subplot_mosaic(
        [["sg_raw", "te_lz5", "sg_lz7"],
         ["confmat", "confmat", "legend"]],
        figsize=(7.0, 5.4),
        height_ratios=[1.0, 0.85],
        gridspec_kw=dict(
            hspace=0.65, wspace=0.45,
            left=0.075, right=0.985, top=0.90, bottom=0.08,
        ),
    )

    panel_keys = ["sg_raw", "te_lz5", "sg_lz7"]
    handles_for_legend = None
    envelope_n = 0
    for key, (title, feature_key, transform, _yscale, ylabel) in zip(panel_keys, SIGNALS):
        ax = axes[key]
        cohort_matrix, _ = collect_cohort_signal(cohort_paths, feature_key, transform)
        envelope_n = max(envelope_n, cohort_matrix.shape[0])
        canon_pre = (per_layer_mean_signal(canon_pre_payload, feature_key, transform)
                     if canon_pre_payload else None)
        canon_post = (per_layer_mean_signal(canon_post_payload, feature_key, transform)
                      if canon_post_payload else None)
        render_signal_panel(
            ax, title, ylabel, cohort_matrix, canon_pre, canon_post,
            ell_star=ell_star, trim=trim,
        )
        if key == "sg_raw":
            handles_for_legend = ax.get_legend_handles_labels()
    print(f"fine-tune envelopes drawn from n={envelope_n} fine-tunes "
          f"(of {len(cohort_paths)} unique fine-tunes discovered)")

    render_confusion_panel(axes["confmat"], summary)

    axes["legend"].axis("off")
    if handles_for_legend is not None:
        legend_handles, legend_labels = handles_for_legend
        legend_handles = list(legend_handles) + [
            Patch(facecolor=TRIM_SHADE_COLOR, edgecolor="none", label="Trimmed boundary"),
            Line2D([0], [0], color=ELL_STAR_COLOR, linestyle=":", linewidth=1.0,
                   label=f"$\\ell^*={ell_star}$"),
        ]
        legend_labels = list(legend_labels) + ["Trimmed boundary", f"$\\ell^*={ell_star}$"]
        axes["legend"].legend(
            legend_handles, legend_labels,
            loc="center left", ncol=1, frameon=False, fontsize=8.5,
            handlelength=2.4, borderaxespad=0.2,
        )

    fig.suptitle(
        "False-positive audit on 99 Qwen3-8B Hugging Face fine-tunes",
        fontsize=12.5, fontweight="bold", y=0.97,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle-root", type=Path, default=default_bundle_root(__file__),
                        help="Path to final_n500_bundle root.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path. Defaults to "
                             "graphs/fleet_summary/cohort_fp_combined.png "
                             "under bundle-root.")
    parser.add_argument("--paper-img", type=Path, default=None,
                        help="If given, also copy the PNG to this path "
                             "(typically paper/img/detection/cohort_fp_combined.png).")
    parser.add_argument("--mode", choices=("confusion-only", "combined"),
                        default="confusion-only",
                        help="confusion-only: produce a single-column "
                             "confusion-matrix figure for §6.4 (default). "
                             "combined: produce the legacy 3-envelope + "
                             "confusion two-row figure.")
    parser.add_argument("--ell-star", type=int, default=CANONICAL_TARGET_LAYER,
                        help="Target layer to mark on the signal panels.")
    parser.add_argument("--trim", type=int, default=DEFAULT_TRIM,
                        help="Trim shaded on the signal panels.")
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve()
    out_path = (args.out or (bundle_root / DEFAULT_OUT_REL)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "confusion-only":
        fig = build_confusion_only_figure(bundle_root)
    else:
        fig = build_figure(bundle_root, ell_star=args.ell_star, trim=args.trim)
    # PDF is the canonical output (vector); also drop a PNG preview.
    fig.savefig(out_path, bbox_inches="tight", metadata={"Creator": "render_cohort_fp_figure"})
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    if args.paper_img is not None:
        try:
            args.paper_img.parent.mkdir(parents=True, exist_ok=True)
            args.paper_img.write_bytes(out_path.read_bytes())
            print(f"copied to {args.paper_img.resolve()}")
        except OSError as exc:
            print(f"warning: could not copy optional paper output to {args.paper_img}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
