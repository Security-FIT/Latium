"""Render the appendix-quality cohort baseline overlay.

This is the aggregated, envelope-style replacement for
cohort baseline overlay. Same panel set as the original
(13 panels when full spectral payload is available, 6 when blind-only),
but each cohort member contributes a *per-layer aggregate*, and across the
cohort we plot the 5--95 percentile band, the median, and (when the
canonical baseline / canonical post-edit JSONs are present) the canonical
Qwen3-8B reference curves on top.

Inputs (resolved from --bundle-root):
  cohort baselines : data/fleet_single_edit_runs/fleet_out_remote/
                      baseline_structural_fleet_*.json (latest per slug)
  canonical pre   : data/final_n500_models_only/qwen3-8b/structural/
                      baseline_structural_qwen3-8b_*.json
  canonical post  : data/final_n500_models_only/qwen3-8b/structural/
                      rome_structural_qwen3-8b_*.json

Output:
  <bundle-root>/graphs/fleet_summary/cohort_baseline_envelope.png
  (also copies to --paper-img if given.)
"""

from __future__ import annotations

import argparse
import json
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
from bundle_paths import add_import_root  # noqa: E402


PKG_ROOT = add_import_root(__file__)

from paper_graphs._newgen_utils import load_json  # noqa: E402
from paper_graphs.paper_graphs_support import (   # noqa: E402
    _stack_single_specs,
    aggregate_layer_map_from_payload,
    aggregate_target_svd_profile,
)


RUN_MARKER = "_tk50_tfauto_tlauto_nl1_rw5_lw3-5-7_s01_r01_"
FLEET_REL = Path("data/fleet_single_edit_runs/fleet_out_remote")
CANON_REL = Path("data/final_n500_models_only/qwen3-8b/structural")
DEFAULT_OUT_REL = Path("graphs/fleet_summary/cohort_baseline_envelope.png")

CANONICAL_TARGET_LAYER = 10
DEFAULT_TRIM = 2

REFERENCE_COLOR = "#9f1239"
REFERENCE_ALPHA = 0.92
PRE_LINEWIDTH = 1.2
POST_LINEWIDTH = 1.5
FLEET_BAND_COLOR = "#cbd5e1"
FLEET_MEDIAN_COLOR = "#475569"
TRIM_SHADE_COLOR = "#f1f5f9"
ELL_STAR_COLOR = "#0f172a"


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
        out[slug] = max(group, key=lambda x: x.name)
    return out


def discover_cohort_baselines(bundle_root: Path) -> List[Path]:
    cohort_root = bundle_root / FLEET_REL
    if not cohort_root.exists():
        return []
    paths = sorted(cohort_root.glob("baseline_structural_fleet_*.json"))
    return list(latest_per_slug(paths, "baseline_structural_").values())


def discover_cohort_rome(bundle_root: Path) -> List[Path]:
    cohort_root = bundle_root / FLEET_REL
    if not cohort_root.exists():
        return []
    paths = sorted(cohort_root.glob("rome_structural_fleet_*.json"))
    return list(latest_per_slug(paths, "rome_structural_").values())


CANONICAL_BASELINE_SEARCH_RELS = [
    CANON_REL,
    Path("data/prefixtest_n500_runs/qwen3-8b-prefixtest-self-short/structural"),
    Path("data/prefixtest_n500_runs/qwen3-8b-prefixtest-self-medium/structural"),
    Path("data/prefixtest_n500_runs/qwen3-8b-prefixtest-self-long/structural"),
]

# Pre-aggregated paper-graph exports for the canonical Qwen3-8B run. When
# present these are preferred over per-test structural JSONs because they
# carry the exact mean curves over the canonical 500-edit subset.
PAPER_GRAPH_EXPORT_REL = Path("data/final_n500_models_only/qwen3-8b/paper_graphs")
PAPER_GRAPH_BASELINE_FNAME = "qwen3_8b_unedited_baseline_paper_graph_data.json"
PAPER_GRAPH_ROME_FNAME = "qwen3_8b_rome_edited_mean_paper_graph_data_n500.json"
PAPER_GRAPH_SCHEMA_PREFIX = "latium.qwen_paper_graph_export"


def is_paper_graph_export(payload: Optional[dict]) -> bool:
    if not isinstance(payload, dict):
        return False
    schema = payload.get("schema") or ""
    return isinstance(schema, str) and schema.startswith(PAPER_GRAPH_SCHEMA_PREFIX)


def discover_canonical(bundle_root: Path, prefix: str) -> Tuple[Optional[Path], Optional[str]]:
    """Locate the canonical Qwen3-8B reference data for the wine overlay.

    Preference order:
      1. The pre-aggregated paper-graph export JSON
         (qwen3_8b_unedited_baseline_paper_graph_data.json or
         qwen3_8b_rome_edited_mean_paper_graph_data_n500.json) under
         data/final_n500_models_only/qwen3-8b/paper_graphs/. These carry
         pre-computed mean+std curves over the canonical 500-edit subset
         and are the exact data that drove the per-model paper graphs.
      2. A structural JSON in CANONICAL_BASELINE_SEARCH_RELS, where each
         per-test feature curve still has to be aggregated locally.
    """
    pg_dir = bundle_root / PAPER_GRAPH_EXPORT_REL
    if pg_dir.exists():
        if prefix == "baseline_structural":
            cand = pg_dir / PAPER_GRAPH_BASELINE_FNAME
            if cand.exists():
                return cand, "qwen3-8b-paper-graph"
        elif prefix == "rome_structural":
            cand = pg_dir / PAPER_GRAPH_ROME_FNAME
            if cand.exists():
                return cand, "qwen3-8b-paper-graph"

    for rel in CANONICAL_BASELINE_SEARCH_RELS:
        d = bundle_root / rel
        if not d.exists():
            continue
        cands = sorted(d.glob(f"{prefix}_qwen3-8b_*.json"))
        if cands:
            return cands[-1], "qwen3-8b"
        slug = rel.parent.name
        cands = sorted(d.glob(f"{prefix}_{slug}_*.json"))
        if cands:
            return cands[-1], slug
    return None, None


# ----- per-spec series ----------------------------------------------------

# Map between the panel "title" used by paper_graphs_support._stack_single_specs
# and the snake_case key used inside the pre-aggregated paper_graph_export
# JSONs.
_PAPER_GRAPH_SERIES_KEYS = {
    "Top-k proj singular values at target layer": "top_k_proj_singular_values_at_target_layer",
    "Signal A (sv_z_scores)":         "signal_a_sv_z_scores",
    "Signal B (sv_ratio_scores)":     "signal_b_sv_ratio_scores",
    "Blind spectral_gap":             "blind_spectral_gap",
    "Blind top1_energy":              "blind_top1_energy",
    "Stable/effective rank ratio":    "stable_effective_rank_ratio",
    "PCS next jump":                  "pcs_next_jump",
    "PCS next curvature":             "pcs_next_curvature",
    "PCS cross scores":               "pcs_cross_scores",
    "PCS cross curvature":            "pcs_cross_curvature",
    "Blind norm_cv":                  "blind_norm_cv",
    "Blind spectral_entropy":         "blind_spectral_entropy",
    "Blind row_alignment":            "blind_row_alignment",
}


def series_from_paper_graph_export(payload: dict, spec: tuple) -> Optional[dict]:
    """Read a panel directly from a paper_graph_export JSON.

    These JSONs ship pre-aggregated mean curves (and std/count) computed
    over the canonical 500-edit subset, so no per-test aggregation is
    needed; we just look up the right series by title.
    """
    title, _getter, _ylabel, kind = spec
    series_root = payload.get("paper_graphs", {}).get("series", {})
    key = _PAPER_GRAPH_SERIES_KEYS.get(title)
    if key is None or series_root.get(key) is None:
        return None
    stats = series_root[key].get("stats", {})
    mean = stats.get("mean")
    if mean is None:
        return None
    if kind == "svd":
        rank = stats.get("rank")
        if rank is None:
            return None
        return {"x": np.asarray(rank, dtype=float),
                "y": np.asarray(mean, dtype=float),
                "kind": "svd"}
    layers = stats.get("layers")
    if layers is None:
        return None
    return {"x": np.asarray(layers, dtype=float),
            "y": np.asarray(mean, dtype=float),
            "kind": "series"}


def series_for_spec(payload: dict, spec: tuple) -> Optional[dict]:
    """Return {x: array, mean: array} for one panel, where x is layer or rank."""
    if is_paper_graph_export(payload):
        return series_from_paper_graph_export(payload, spec)

    title, getter, _ylabel, kind = spec
    if kind == "svd":
        agg = aggregate_target_svd_profile(payload, top_k=30)
        if agg is None:
            return None
        return {"x": np.asarray(agg["rank"], dtype=float),
                "y": np.asarray(agg["mean"], dtype=float),
                "kind": "svd"}
    agg = aggregate_layer_map_from_payload(payload, getter)
    if agg is None:
        return None
    layers = np.asarray(agg.get("layers"), dtype=float) if agg.get("layers") is not None else None
    means = np.asarray(agg.get("mean"), dtype=float) if agg.get("mean") is not None else None
    if layers is None or means is None or layers.size == 0:
        return None
    return {"x": layers, "y": means, "kind": "series"}


def collect_cohort_matrix(cohort_payloads: List[dict], spec: tuple
                         ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (xs, matrix, n_kept) where xs is the canonical x-axis and
    matrix has shape (n_kept, len(xs)) with NaN where a cohort member is
    missing a particular x value."""
    series_list = []
    for payload in cohort_payloads:
        s = series_for_spec(payload, spec)
        if s is None:
            continue
        if not np.any(np.isfinite(s["y"])):
            continue
        series_list.append(s)
    if not series_list:
        return np.array([]), np.empty((0, 0)), 0

    # Use the union of x values, preserve sort order.
    all_x = sorted({float(x) for s in series_list for x in s["x"] if np.isfinite(x)})
    xs = np.asarray(all_x, dtype=float)
    matrix = np.full((len(series_list), len(xs)), np.nan, dtype=float)
    x_to_idx = {x: i for i, x in enumerate(xs.tolist())}
    for row, s in enumerate(series_list):
        for x, y in zip(s["x"].tolist(), s["y"].tolist()):
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            j = x_to_idx.get(float(x))
            if j is not None:
                matrix[row, j] = y
    return xs, matrix, len(series_list)


# ----- plotting -----------------------------------------------------------

def shade_trim(ax, x_min: float, x_max: float, trim: int) -> None:
    if trim <= 0:
        return
    ax.axvspan(x_min - 0.5, x_min + trim - 0.5, color=TRIM_SHADE_COLOR, zorder=0)
    ax.axvspan(x_max - trim + 0.5, x_max + 0.5, color=TRIM_SHADE_COLOR, zorder=0)


def render_panel(ax, spec: tuple, cohort_payloads: List[dict],
                 canon_pre: Optional[dict], canon_post: Optional[dict],
                 ell_star: int, trim: int) -> int:
    title, _getter, ylabel, kind = spec
    xs, matrix, n_used = collect_cohort_matrix(cohort_payloads, spec)
    if n_used == 0:
        ax.text(0.5, 0.5, "No cohort data for this signal", transform=ax.transAxes,
                ha="center", va="center", color="#64748b", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        return 0

    p05 = np.nanpercentile(matrix, 5,  axis=0)
    p95 = np.nanpercentile(matrix, 95, axis=0)
    median = np.nanmedian(matrix, axis=0)

    if kind == "series":
        x_min, x_max = float(xs.min()), float(xs.max())
        shade_trim(ax, x_min, x_max, trim)

    ax.fill_between(xs, p05, p95, color=FLEET_BAND_COLOR, alpha=0.85, linewidth=0,
                    label=f"Fine-tunes 5–95 pct (n={n_used})")
    ax.plot(xs, median, color=FLEET_MEDIAN_COLOR, linewidth=1.1,
            label="Fine-tune median")

    pre = series_for_spec(canon_pre, spec) if canon_pre is not None else None
    post = series_for_spec(canon_post, spec) if canon_post is not None else None
    if pre is not None:
        ax.plot(pre["x"], pre["y"], color=REFERENCE_COLOR, linewidth=PRE_LINEWIDTH,
                linestyle=(0, (5, 2)), alpha=REFERENCE_ALPHA,
                label="Qwen3-8B baseline")
    if post is not None:
        ax.plot(post["x"], post["y"], color=REFERENCE_COLOR, linewidth=POST_LINEWIDTH,
                alpha=REFERENCE_ALPHA,
                label=f"Qwen3-8B + ROME at $\\ell^*{{=}}{ell_star}$")

    if kind == "series":
        ax.axvline(ell_star, color=ELL_STAR_COLOR, linestyle=":", linewidth=1.0, alpha=0.85)
        ax.set_xlabel("Layer $\\ell$", fontsize=9)
    else:
        ax.set_xscale("linear")
        finite = matrix[np.isfinite(matrix)]
        # Use log scale only when the dynamic range spans more than one
        # decade. Below that, log scale produces "6x10^0"-style tick
        # labels whose mathtext exponents are visually mismatched with the
        # plain numeric labels in neighbouring panels.
        if finite.size and float(np.nanmin(finite)) > 0.0:
            lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
            if hi / max(lo, 1e-12) > 12.0:
                ax.set_yscale("log")
        ax.set_xlabel("Rank $k$", fontsize=9)
    # Where the natural value range is so small (< 1e-3) that matplotlib
    # would otherwise pull a "1e-5"-style common factor out into a
    # corner-positioned offset text, fold the multiplier into the
    # y-label and keep tick labels in the natural scaled range. This
    # avoids the title/offset-text collision and matches the visual size
    # of every other panel's tick labels.
    if kind == "series":
        finite = matrix[np.isfinite(matrix)]
        if finite.size:
            vmax = float(np.nanmax(np.abs(finite)))
            if 0.0 < vmax < 1e-3:
                exponent = int(np.floor(np.log10(vmax)))
                scale = 10.0 ** exponent
                # Re-plot the cohort band/median and references at scaled values.
                # The simplest way is to just rescale the y-axis tick
                # formatter via FuncFormatter.
                from matplotlib.ticker import FuncFormatter
                ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda v, _p, s=scale: f"{v / s:.1f}")
                )
                if ylabel:
                    ylabel = f"{ylabel}  (×$10^{{{exponent}}}$)"
                else:
                    ylabel = f"×$10^{{{exponent}}}$"

    ax.set_ylabel(ylabel or "", fontsize=9)
    # Padding keeps the title clear of the y-axis offset/exponent text.
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.tick_params(axis="both", labelsize=8)
    # Match offset/exponent size to tick labels for visual uniformity.
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.xaxis.get_offset_text().set_fontsize(8)
    if kind == "series":
        ax.set_xticks([0, 10, 20, 30])
    ax.grid(True, linestyle=":", alpha=0.30)
    return n_used


def build_figure(bundle_root: Path, ell_star: int, trim: int,
                 population: str) -> plt.Figure:
    """Render the envelope figure for one homogeneous population.

    population in {"unedited", "edited"}:
      - "unedited": fine-tune envelopes from baseline_structural_fleet_*.json,
        Qwen3-8B baseline reference (no post-edit overlay).
      - "edited":   fine-tune envelopes from rome_structural_fleet_*.json,
        Qwen3-8B + ROME at l*=10 reference (no baseline overlay).
    """
    if population not in ("unedited", "edited"):
        raise ValueError(f"population must be 'unedited' or 'edited', got {population!r}")

    if population == "unedited":
        cohort_paths = discover_cohort_baselines(bundle_root)
        cohort_label = "fine-tune baselines"
        if not cohort_paths:
            raise SystemExit(
                f"No baseline_structural_fleet_*.json under {bundle_root / FLEET_REL}; "
                "cannot render the unedited envelope figure."
            )
    else:
        cohort_paths = discover_cohort_rome(bundle_root)
        cohort_label = "ROME-edited fine-tunes"
        if not cohort_paths:
            raise SystemExit(
                f"No rome_structural_fleet_*.json under {bundle_root / FLEET_REL}; "
                "cannot render the edited envelope figure."
            )

    cohort_payloads: List[dict] = []
    for p in cohort_paths:
        try:
            cohort_payloads.append(load_json(p))
        except Exception as exc:
            print(f"[warn] could not load {p.name}: {exc}", file=sys.stderr)

    # Single Qwen3-8B reference per figure: baseline for "unedited", post-edit
    # for "edited". The other reference is intentionally omitted to keep each
    # panel a homogeneous population view.
    if population == "unedited":
        canon_path, _tag = discover_canonical(bundle_root, "baseline_structural")
        canon_pre = load_json(canon_path) if canon_path else None
        canon_post = None
        if canon_pre is None:
            print(f"[warn] no Qwen3-8B baseline_structural_*.json found; "
                  f"reference curve omitted.", file=sys.stderr)
        else:
            print(f"[info] Qwen3-8B baseline source: {canon_path.relative_to(bundle_root)}",
                  file=sys.stderr)
    else:
        canon_path, _tag = discover_canonical(bundle_root, "rome_structural")
        canon_pre = None
        canon_post = load_json(canon_path) if canon_path else None
        if canon_post is None:
            print(f"[warn] no Qwen3-8B rome_structural_*.json found; "
                  f"reference curve omitted.", file=sys.stderr)
        else:
            print(f"[info] Qwen3-8B + ROME source: {canon_path.relative_to(bundle_root)}",
                  file=sys.stderr)

    spec_source = (canon_post or canon_pre or
                   (cohort_payloads[0] if cohort_payloads else None))
    if spec_source is None:
        raise SystemExit("No payload available to derive panel specs.")
    # Pre-aggregated paper_graph_export JSONs do not carry per-test data,
    # so _payload_capabilities cannot detect spectral/attention support
    # from them. Force the full 13-panel spec set in that case.
    if is_paper_graph_export(spec_source):
        paper_specs = _stack_single_specs("full")
    else:
        paper_specs = _stack_single_specs(spec_source)

    n_panels = len(paper_specs)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols + 1   # +1 row for legend

    # Identical figsize and gridspec margins for both populations so the
    # un-edited and edited PDFs land at the same physical dimensions in
    # the appendix. Sizing is homogenized with figs 2/3
    # (artifacts_qwen3-8b_full / artifacts_gptj-6b_full): each interior
    # panel is roughly 4.1in wide x 2.55in tall, with the same fonts.
    panel_w = 4.1
    panel_h = 2.55
    legend_h = 0.65
    fig = plt.figure(figsize=(panel_w * ncols, panel_h * (nrows - 1) + legend_h + 0.6))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.55, wspace=0.30,
                          height_ratios=[1.0] * (nrows - 1) + [legend_h / panel_h],
                          left=0.055, right=0.99, top=0.955, bottom=0.045)

    cohort_n_used = 0
    for idx, spec in enumerate(paper_specs):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        n_used = render_panel(ax, spec, cohort_payloads, canon_pre, canon_post,
                              ell_star=ell_star, trim=trim)
        cohort_n_used = max(cohort_n_used, n_used)

    for k in range(n_panels, (nrows - 1) * ncols):
        r, c = divmod(k, ncols)
        ax_blank = fig.add_subplot(gs[r, c])
        ax_blank.axis("off")

    if population == "unedited":
        ref_handle = Line2D([0], [0], color=REFERENCE_COLOR, linewidth=PRE_LINEWIDTH,
                            linestyle=(0, (5, 2)), alpha=REFERENCE_ALPHA,
                            label="Qwen3-8B baseline (un-edited)")
    else:
        ref_handle = Line2D([0], [0], color=REFERENCE_COLOR, linewidth=POST_LINEWIDTH,
                            alpha=REFERENCE_ALPHA,
                            label=f"Qwen3-8B + ROME at $\\ell^*{{=}}{ell_star}$")

    legend_handles = [
        Patch(facecolor=FLEET_BAND_COLOR, edgecolor="none",
              label=f"{cohort_label.capitalize()} 5–95 pct (n up to {cohort_n_used} of {len(cohort_paths)})"),
        Line2D([0], [0], color=FLEET_MEDIAN_COLOR, linewidth=1.4,
               label=f"{cohort_label.capitalize()} median"),
        ref_handle,
        Patch(facecolor=TRIM_SHADE_COLOR, edgecolor="none", label="Trimmed boundary"),
        Line2D([0], [0], color=ELL_STAR_COLOR, linestyle=":", linewidth=1.0,
               label=f"$\\ell^*={ell_star}$"),
    ]
    ax_legend = fig.add_subplot(gs[-1, :])
    ax_legend.axis("off")
    ax_legend.legend(
        handles=legend_handles, loc="center", ncol=5, frameon=False, fontsize=10,
    )

    if population == "unedited":
        suptitle = ("Un-edited population: 99 fine-tunes vs. the "
                    "Qwen3-8B baseline")
    else:
        suptitle = ("ROME-edited population: 86 fine-tunes vs. the "
                    "Qwen3-8B + ROME reference at $\\ell^*=10$")
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.99)
    print(f"[{population}] envelopes use up to n={cohort_n_used} fine-tunes "
          f"(of {len(cohort_paths)} unique fine-tunes discovered)")
    return fig


# ----- output naming -----------------------------------------------------

def population_out_path(bundle_root: Path, population: str) -> Path:
    return bundle_root / "graphs" / "fleet_summary" / f"cohort_envelope_{population}.pdf"


def population_paper_img(paper_dir: Path, population: str) -> Path:
    return paper_dir / "img" / "detection" / f"cohort_envelope_{population}.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle-root", type=Path, default=Path("."))
    parser.add_argument("--population", choices=("unedited", "edited", "both"),
                        default="both",
                        help="which homogeneous figure to render. Default: both.")
    parser.add_argument("--out", type=Path, default=None,
                        help="explicit output path (only valid when --population != both)")
    parser.add_argument("--paper-dir", type=Path, default=None,
                        help="if given, also copy the PNG(s) into "
                             "<paper-dir>/img/detection/cohort_envelope_<population>.png")
    parser.add_argument("--ell-star", type=int, default=CANONICAL_TARGET_LAYER)
    parser.add_argument("--trim", type=int, default=DEFAULT_TRIM)
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve()
    populations = ("unedited", "edited") if args.population == "both" else (args.population,)
    if args.out is not None and args.population == "both":
        raise SystemExit("--out is only valid with a single --population")

    for pop in populations:
        out_path = (args.out or population_out_path(bundle_root, pop)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig = build_figure(bundle_root, ell_star=args.ell_star, trim=args.trim,
                           population=pop)
        # PDF is the canonical output (vector); a PNG raster is also
        # written next to it for quick browser preview.
        fig.savefig(out_path, bbox_inches="tight", metadata={"Creator": "render_cohort_envelope_figure"})
        fig.savefig(out_path.with_suffix(".png"), dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")
        if args.paper_dir is not None:
            paper_img = population_paper_img(args.paper_dir.resolve(), pop)
            paper_img.parent.mkdir(parents=True, exist_ok=True)
            paper_img.write_bytes(out_path.read_bytes())
            print(f"copied to {paper_img}")


if __name__ == "__main__":
    main()
