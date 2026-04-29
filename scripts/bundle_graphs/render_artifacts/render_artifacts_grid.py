"""Render the 5x4 artifact grid (Figures 2 and 3) for the rome_artifacts section.

For each model the figure shows the 5 spectral artifacts of section 5.2 -- spectral
gap (SG), top-1 energy (TE), row alignment (RA), row-norm coefficient of variation
(CV), effective rank (ER) -- under the 4 depth transforms of section 5.1: raw,
local z-score at window 5, local z-score at window 7, and absolute second
finite-difference curvature. Each panel carries the per-edit mean (wine) with a
+/- 1 sigma band, the matched unedited-baseline mean (dashed dark grey), a dotted
vertical at the edited layer l*, and shaded trim regions at the two depth ends.

Usage:
    python render_artifacts_grid.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


HELPER_DIR = Path(__file__).resolve().parents[1]
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))
from bundle_paths import default_bundle_root  # noqa: E402


def _model_paths(bundle_root: Path, slug: str) -> tuple[Path, Path]:
    rome_dir = bundle_root / "data/final_n500_models_only" / slug / "structural"
    rome = sorted(rome_dir.glob(f"rome_structural_{slug}_*.json"))[-1]
    base = sorted(rome_dir.glob(f"baseline_structural_{slug}_*.json"))[-1]
    return rome, base


# (slug, hf_label, ell_star, trim, figure_kind)
# figure_kind in {"main", "appendix"} -- "main" goes to img/detection (figs 2/3),
# "appendix" goes to img/artifacts (per-model appendix figures).
MODELS = [
    ("qwen3-8b",        "Qwen3-8B",            10, 2, "main"),
    ("gpt-j-6b",        "GPT-J-6B",             5, 5, "main"),
    ("gpt2-medium",     "GPT-2 Medium",         8, 5, "appendix"),
    ("gpt2-large",      "GPT-2 Large",         12, 5, "appendix"),
    ("gpt2-xl",         "GPT-2 XL",            18, 5, "appendix"),
    ("opt-6.7b",        "OPT-6.7B",            15, 2, "appendix"),
    ("falcon-7b",       "Falcon-7B",            3, 2, "appendix"),
    ("llama2-7b",       "Llama-2-7B",          19, 2, "appendix"),
    ("mistral-7b-v0.1", "Mistral-7B v0.1",      5, 2, "appendix"),
    ("mistral-7b-v0.3", "Mistral-7B v0.3",     17, 2, "appendix"),
    ("qwen3-4b",        "Qwen3-4B",            12, 2, "appendix"),
    ("deepseek-7b-base","DeepSeek-LLM-7B Base", 6, 2, "appendix"),
    ("granite4-micro",  "Granite 4.0 Micro",    9, 2, "appendix"),
]


# Style palette borrowed from render_cohort_envelope_figure.py so this figure is
# visually homogeneous with the appendix envelopes (Figs 5/6).
EDITED_COLOR    = "#9f1239"      # wine
EDITED_BAND     = "#fda4af"      # light wine
BASELINE_COLOR  = "#475569"      # slate grey
TRIM_SHADE      = "#f1f5f9"      # very light grey
ELL_STAR_COLOR  = "#0f172a"      # near black
PANEL_GRID      = "#d0d0d0"

# Per-family trim is encoded in MODELS above (5 for GPT family, 2 otherwise).

# 5 artifacts of section 5.2.
ARTIFACT_KEYS = [
    ("spectral_gap",   "SG",  r"Spectral gap $\mathrm{SG}_\ell$"),
    ("top1_energy",    "TE",  r"Top-1 energy $\mathrm{TE}_\ell$"),
    ("row_alignment",  "RA",  r"Row alignment $\mathrm{RA}_\ell$"),
    ("norm_cv",        "CV",  r"Row-norm CV $\mathrm{CV}_\ell$"),
    ("effective_rank", "ER",  r"Effective rank $\mathrm{ER}_\ell$"),
]

# 4 transforms of section 5.1 (with both windows of section 6 detection method).
TRANSFORM_KEYS = [
    ("raw",   "Raw"),
    ("lz5",   r"Local $z$-score (window 5)"),
    ("lz7",   r"Local $z$-score (window 7)"),
    ("curv",  r"Curvature $|x_{\ell-1}-2x_\ell+x_{\ell+1}|$"),
]

EPS = 1e-8


# ---------- data loading -------------------------------------------------- #

def load_payload(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def collect_feature_matrix(payload: dict, feature: str
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (layers, matrix) where matrix has shape (n_tests, n_layers)."""
    rows: List[List[float]] = []
    layers_ref: Optional[List[int]] = None
    for case in payload.get("tests", []):
        lf = case.get("blind_detection", {}).get("layer_features")
        if not isinstance(lf, dict) or not lf:
            continue
        layers_int = sorted(int(k) for k in lf.keys())
        if layers_ref is None:
            layers_ref = layers_int
        elif layers_int != layers_ref:
            # Pad/align if any test has a different layer set; fall back to the union.
            layers_ref = sorted(set(layers_ref) | set(layers_int))
        row = []
        for layer in layers_ref:
            entry = lf.get(str(layer))
            if not isinstance(entry, dict):
                row.append(np.nan)
                continue
            v = entry.get(feature)
            try:
                row.append(float(v))
            except (TypeError, ValueError):
                row.append(np.nan)
        rows.append(row)
    if layers_ref is None or not rows:
        return np.array([], dtype=int), np.empty((0, 0))
    n_layers = len(layers_ref)
    matrix = np.full((len(rows), n_layers), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for j in range(min(len(row), n_layers)):
            matrix[i, j] = row[j]
    return np.asarray(layers_ref, dtype=int), matrix


# ---------- transforms ---------------------------------------------------- #

def transform_raw(matrix: np.ndarray) -> np.ndarray:
    return matrix


def transform_local_z(matrix: np.ndarray, window: int) -> np.ndarray:
    """Per-row absolute local z-score, neighbors only (center excluded).

    Matches detector.composite_detector_v2.local_zscore: for each layer j the
    mean/std are computed over the half-window neighbors on either side of j,
    not including j itself; the score is the deviation of j from that
    leave-one-out neighborhood.
    """
    half = window // 2
    n_tests, n_layers = matrix.shape
    out = np.full_like(matrix, np.nan, dtype=float)
    for j in range(n_layers):
        lo = max(0, j - half)
        hi = min(n_layers, j + half + 1)
        if lo >= j and hi <= j + 1:
            continue
        left  = matrix[:, lo:j]   if j   > lo else np.empty((n_tests, 0))
        right = matrix[:, j+1:hi] if hi  > j + 1 else np.empty((n_tests, 0))
        nbrs  = np.concatenate([left, right], axis=1)
        if nbrs.shape[1] < 2:
            continue
        mu = np.nanmean(nbrs, axis=1)
        sd = np.nanstd(nbrs,  axis=1)
        out[:, j] = np.abs(matrix[:, j] - mu) / (sd + EPS)
    return out


def transform_curvature(matrix: np.ndarray) -> np.ndarray:
    """Per-row absolute second finite difference, NaN at the two boundary layers."""
    n_tests, n_layers = matrix.shape
    out = np.full_like(matrix, np.nan, dtype=float)
    if n_layers >= 3:
        out[:, 1:-1] = np.abs(matrix[:, :-2] - 2.0 * matrix[:, 1:-1] + matrix[:, 2:])
    return out


def apply_transform(matrix: np.ndarray, key: str) -> np.ndarray:
    if key == "raw":
        return transform_raw(matrix)
    if key == "lz5":
        return transform_local_z(matrix, 5)
    if key == "lz7":
        return transform_local_z(matrix, 7)
    if key == "curv":
        return transform_curvature(matrix)
    raise ValueError(f"unknown transform key: {key}")


def aggregate(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, std, count) per layer ignoring NaN."""
    with np.errstate(all="ignore"):
        mean  = np.nanmean(matrix, axis=0)
        std   = np.nanstd(matrix, axis=0)
        count = np.sum(np.isfinite(matrix), axis=0)
    return mean, std, count


# ---------- plotting ------------------------------------------------------ #

def _setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#222222",
        "axes.labelcolor":   "#111111",
        "axes.titlecolor":   "#111111",
        "xtick.color":       "#111111",
        "ytick.color":       "#111111",
        "text.color":        "#111111",
        "grid.color":        PANEL_GRID,
        "grid.alpha":        0.30,
        "legend.facecolor":  "white",
        "legend.edgecolor":  "#cccccc",
        "legend.framealpha": 0.92,
    })


def shade_trim(ax, layers: np.ndarray, trim: int) -> None:
    if trim <= 0 or layers.size == 0:
        return
    x_min, x_max = float(layers.min()), float(layers.max())
    ax.axvspan(x_min - 0.5,            x_min + trim - 0.5, color=TRIM_SHADE, zorder=0)
    ax.axvspan(x_max - trim + 0.5,     x_max + 0.5,        color=TRIM_SHADE, zorder=0)


def render_panel(ax, layers, edited, baseline, ell_star, trim,
                 title, ylabel) -> None:
    if layers.size == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="#64748b")
        ax.set_title(title, fontsize=8, fontweight="bold", pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        return

    shade_trim(ax, layers, trim)

    e_mean, e_std, _ = edited
    ax.fill_between(layers, e_mean - e_std, e_mean + e_std,
                    color=EDITED_BAND, alpha=0.55, linewidth=0)
    line = ax.plot(layers, e_mean,
                   color=EDITED_COLOR, linewidth=1.4, label="Edited mean")[0]
    line.set_path_effects([pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()])

    if baseline is not None:
        b_mean, _, _ = baseline
        markevery = max(1, len(layers) // 12)
        bline = ax.plot(layers, b_mean,
                        color=BASELINE_COLOR, linestyle="--", linewidth=1.0,
                        marker="o", markersize=2.2,
                        markerfacecolor="white", markeredgecolor=BASELINE_COLOR,
                        markeredgewidth=0.7, markevery=markevery,
                        label="Unedited baseline mean")[0]
        bline.set_path_effects([pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()])

    if ell_star is not None and layers[0] <= ell_star <= layers[-1]:
        ax.axvline(ell_star, color=ELL_STAR_COLOR, linestyle=":",
                   linewidth=1.0, alpha=0.85,
                   label=fr"Edited layer $\ell^*={ell_star}$")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.30)
    ax.margins(x=0.01)


def render_model_grid(model_name: str, hf_label: str,
                      rome_path: Path, base_path: Path,
                      ell_star: int, trim: int,
                      out_path: Path) -> None:
    payload_e = load_payload(rome_path)
    payload_b = load_payload(base_path)
    n_e = sum(1 for c in payload_e.get("tests", [])
              if c.get("blind_detection", {}).get("layer_features"))
    n_b = sum(1 for c in payload_b.get("tests", [])
              if c.get("blind_detection", {}).get("layer_features"))

    # Pre-compute (n_tests, n_layers) matrices and aggregates per (artifact, transform).
    layers = None
    cache: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for feature_key, _, _ in ARTIFACT_KEYS:
        layers_e, mat_e = collect_feature_matrix(payload_e, feature_key)
        layers_b, mat_b = collect_feature_matrix(payload_b, feature_key)
        if layers is None:
            layers = layers_e
        for tkey, _ in TRANSFORM_KEYS:
            cache[("edited",   feature_key, tkey)] = aggregate(apply_transform(mat_e, tkey))
            cache[("baseline", feature_key, tkey)] = aggregate(apply_transform(mat_b, tkey))

    _setup_style()
    n_rows = len(ARTIFACT_KEYS)
    n_cols = len(TRANSFORM_KEYS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.1 * n_cols, 2.55 * n_rows),
                             squeeze=False)
    fig.patch.set_facecolor("white")

    for i, (feat_key, feat_short, feat_label) in enumerate(ARTIFACT_KEYS):
        for j, (tkey, tlabel) in enumerate(TRANSFORM_KEYS):
            ax = axes[i, j]
            edited_stat   = cache[("edited",   feat_key, tkey)]
            baseline_stat = cache[("baseline", feat_key, tkey)]
            title = f"{feat_short} -- {tlabel}"
            ylabel = feat_label if j == 0 else ""
            xlabel = r"Layer $\ell$" if i == n_rows - 1 else ""
            render_panel(ax, layers, edited_stat, baseline_stat,
                         ell_star=ell_star, trim=trim,
                         title=title, ylabel=ylabel)
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=7)

    # Single shared legend in the figure footer.
    legend_handles = [
        Line2D([0], [0], color=EDITED_COLOR, linewidth=1.6, label="Edited mean"),
        Patch(facecolor=EDITED_BAND, edgecolor="none", alpha=0.55,
              label=r"Edited mean $\pm\,1\sigma$"),
        Line2D([0], [0], color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
               marker="o", markersize=3,
               markerfacecolor="white", markeredgecolor=BASELINE_COLOR,
               label="Unedited baseline mean"),
        Line2D([0], [0], color=ELL_STAR_COLOR, linestyle=":", linewidth=1.2,
               label=fr"Edited layer $\ell^*={ell_star}$"),
        Patch(facecolor=TRIM_SHADE, edgecolor="#cbd5e1",
              label=f"Trimmed boundary ({trim} layers)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        f"Per-layer artifact signals on {hf_label} -- "
        f"5 artifacts $\\times$ 4 depth transforms\n"
        f"means over $n={n_e}$ successful ROME edits and "
        f"$n={n_b}$ unedited baseline cases at $\\ell^*={ell_star}$",
        fontsize=14, y=0.998,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[ok] wrote {out_path}")
    plt.close(fig)


def _out_basename(slug: str) -> str:
    # Match the existing main-figure naming convention.
    if slug == "gpt-j-6b":
        return "artifacts_gptj-6b_full"
    return f"artifacts_{slug}_full"


def _copy_optional(source: Path, destination_dir: Optional[Path]) -> None:
    if destination_dir is None:
        return
    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination_dir / source.name)
        print(f"[ok] copied {source.name} to {destination_dir}")
    except OSError as exc:
        print(f"[warn] optional paper copy failed for {source}: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle-root", type=Path, default=default_bundle_root(__file__))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Canonical output root. Defaults to <bundle-root>/graphs/artifacts_grid.",
    )
    parser.add_argument("--paper-img-detection", type=Path, default=None)
    parser.add_argument("--paper-img-artifacts", type=Path, default=None)
    parser.add_argument(
        "--allow-missing-models",
        action="store_true",
        help="Warn and continue when a model is absent from the bundle.",
    )
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve()
    output_root = (args.output_dir or (bundle_root / "graphs" / "artifacts_grid")).resolve()
    missing: list[str] = []
    for slug, label, ell_star, trim, kind in MODELS:
        try:
            rome_path, base_path = _model_paths(bundle_root, slug)
        except (FileNotFoundError, IndexError) as exc:
            if not args.allow_missing_models:
                missing.append(f"{slug}: {exc}")
                continue
            print(f"[skip] {slug}: no structural data ({exc})", file=sys.stderr)
            continue
        out_dir = output_root / ("detection" if kind == "main" else "artifacts")
        basename = _out_basename(slug)
        for ext in ("png", "pdf"):
            out_path = out_dir / f"{basename}.{ext}"
            render_model_grid(
                model_name=slug,
                hf_label=label,
                rome_path=rome_path, base_path=base_path,
                ell_star=ell_star, trim=trim,
                out_path=out_path,
            )
            optional_dir = args.paper_img_detection if kind == "main" else args.paper_img_artifacts
            _copy_optional(out_path, optional_dir)
    if missing:
        raise FileNotFoundError("Missing required artifact-grid inputs:\n" + "\n".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
