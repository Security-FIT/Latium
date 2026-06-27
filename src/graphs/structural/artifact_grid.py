"""
Legacy-compatible artifact-grid renderer over current structural artifacts.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
import numpy as np

from src.common.io import write_json
from src.graphs.context import RenderContext, RendererUnavailableError
from src.graphs.structural.materialize import matching_baseline, matrix_feature_table, target_layer_for
from src.graphs.structural.style import (
    BASELINE_COLOR,
    EDITED_BAND,
    EDITED_COLOR,
    ELL_STAR_COLOR,
    TRIM_SHADE,
    aggregate,
    apply_transform,
    model_label,
    setup_matplotlib_style,
    trim_for_model,
)
from src.results.naming import safe_slug


FEATURE_LABELS: dict[str, tuple[str, str]] = {
    "spectral_gap": ("SG", r"Spectral gap $\mathrm{SG}_\ell$"),
    "top1_energy": ("TE", r"Top-1 energy $\mathrm{TE}_\ell$"),
    "row_alignment": ("RA", r"Row alignment $\mathrm{RA}_\ell$"),
    "norm_cv": ("CV", r"Row-norm CV $\mathrm{CV}_\ell$"),
    "effective_rank": ("ER", r"Effective rank $\mathrm{ER}_\ell$"),
}

TRANSFORM_LABELS: dict[str, str] = {
    "raw": "Raw",
    "lz5": r"Local $z$-score (window 5)",
    "lz7": r"Local $z$-score (window 7)",
    "curv": r"Curvature $|x_{\ell-1}-2x_\ell+x_{\ell+1}|$",
}


def _sequence(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(item) for item in value)


def _shade_trim(ax: Any, layers: np.ndarray, trim: int) -> None:
    if trim <= 0 or layers.size == 0:
        return
    x_min, x_max = float(layers.min()), float(layers.max())
    ax.axvspan(x_min - 0.5, x_min + trim - 0.5, color=TRIM_SHADE, zorder=0)
    ax.axvspan(x_max - trim + 0.5, x_max + 0.5, color=TRIM_SHADE, zorder=0)


def _json_stats(layers: np.ndarray, stats: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, Any]:
    mean, std, count = stats
    return {
        "layers": [int(layer) for layer in layers],
        "mean": [None if not np.isfinite(value) else float(value) for value in mean],
        "std": [None if not np.isfinite(value) else float(value) for value in std],
        "count": [int(value) for value in count],
    }


def render_structural_artifact_grid(context: RenderContext) -> list[str]:
    output_dir = Path(context.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    options = dict(context.options or {})
    features = _sequence(
        options.get("features"),
        ("spectral_gap", "top1_energy", "row_alignment", "norm_cv", "effective_rank"),
    )
    transforms = _sequence(options.get("transforms"), ("raw", "lz5", "lz7", "curv"))
    formats = set(_sequence(options.get("formats"), ("png", "pdf", "json")))

    captures = list((context.captures or {}).get("matrix-features", ()))
    edited_payloads = [payload for payload in captures if payload.get("run", {}).get("edit_method")]
    if not edited_payloads:
        raise RendererUnavailableError("structural-artifact-grid requires edited matrix-features captures")
    executions = list(context.executions)

    setup_matplotlib_style()
    import matplotlib.pyplot as plt

    outputs: list[str] = []
    index: list[dict[str, Any]] = []
    for edited_payload in edited_payloads:
        edited = matrix_feature_table(edited_payload, features)
        baseline_payload = matching_baseline(edited_payload, captures)
        baseline = matrix_feature_table(baseline_payload, features) if baseline_payload else None
        target_layer = target_layer_for(
            executions,
            model=edited.model,
            plan_id=edited.plan_id,
            edit_method=edited.edit_method,
        )
        trim = trim_for_model(edited.model)
        n_rows = len(features)
        n_cols = len(transforms)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.1 * n_cols, 2.55 * n_rows), squeeze=False)
        data: dict[str, Any] = {
            "schema": "latium.structural_artifact_grid.v1",
            "model": edited.model,
            "plan_id": edited.plan_id,
            "edit_method": edited.edit_method,
            "target_layer": target_layer,
            "trim": trim,
            "features": list(features),
            "transforms": list(transforms),
            "series": {},
            "warnings": list(context.warnings),
        }

        for row, feature in enumerate(features):
            short, ylabel = FEATURE_LABELS.get(feature, (feature, feature))
            data["series"][feature] = {}
            for col, transform in enumerate(transforms):
                ax = axes[row, col]
                transformed = apply_transform(edited.values[feature], transform)
                edited_stats = aggregate(transformed)
                baseline_stats = None
                if baseline is not None and feature in baseline.values:
                    baseline_stats = aggregate(apply_transform(baseline.values[feature], transform))
                data["series"][feature][transform] = {
                    "edited": _json_stats(edited.layers, edited_stats),
                    "baseline": (
                        None
                        if baseline is None or baseline_stats is None
                        else _json_stats(baseline.layers, baseline_stats)
                    ),
                }

                _shade_trim(ax, edited.layers, trim)
                mean, std, _count = edited_stats
                ax.fill_between(edited.layers, mean - std, mean + std, color=EDITED_BAND, alpha=0.55, linewidth=0)
                line = ax.plot(edited.layers, mean, color=EDITED_COLOR, linewidth=1.4)[0]
                line.set_path_effects([pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()])
                if baseline is not None and baseline_stats is not None:
                    b_mean, _b_std, _b_count = baseline_stats
                    markevery = max(1, len(baseline.layers) // 12)
                    ax.plot(
                        baseline.layers,
                        b_mean,
                        color=BASELINE_COLOR,
                        linestyle="--",
                        linewidth=1.0,
                        marker="o",
                        markersize=2.2,
                        markerfacecolor="white",
                        markeredgecolor=BASELINE_COLOR,
                        markeredgewidth=0.7,
                        markevery=markevery,
                    )
                if target_layer is not None and edited.layers[0] <= int(target_layer) <= edited.layers[-1]:
                    ax.axvline(int(target_layer), color=ELL_STAR_COLOR, linestyle=":", linewidth=1.0, alpha=0.85)
                ax.set_title(
                    f"{short} -- {TRANSFORM_LABELS.get(transform, transform)}", fontsize=10, fontweight="bold", pad=4
                )
                ax.set_ylabel(ylabel if col == 0 else "", fontsize=9)
                if row == n_rows - 1:
                    ax.set_xlabel(r"Layer $\ell$", fontsize=7)
                ax.tick_params(axis="both", labelsize=8)
                ax.grid(True, linestyle=":", alpha=0.30)
                ax.margins(x=0.01)

        legend_handles = [
            Line2D([0], [0], color=EDITED_COLOR, linewidth=1.6, label="Edited mean"),
            Patch(facecolor=EDITED_BAND, edgecolor="none", alpha=0.55, label=r"Edited mean $\pm\,1\sigma$"),
            Line2D(
                [0],
                [0],
                color=BASELINE_COLOR,
                linestyle="--",
                linewidth=1.2,
                marker="o",
                markersize=3,
                markerfacecolor="white",
                markeredgecolor=BASELINE_COLOR,
                label="Unedited baseline mean",
            ),
            Line2D(
                [0],
                [0],
                color=ELL_STAR_COLOR,
                linestyle=":",
                linewidth=1.2,
                label=rf"Edited layer $\ell^*={target_layer}$",
            ),
            Patch(facecolor=TRIM_SHADE, edgecolor="#cbd5e1", label=f"Trimmed boundary ({trim} layers)"),
        ]
        fig.legend(
            handles=legend_handles, loc="lower center", ncol=5, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.005)
        )
        fig.suptitle(
            f"Per-layer artifact signals on {model_label(edited.model)} -- "
            f"{len(features)} artifacts $\\times$ {len(transforms)} depth transforms\n"
            f"means over $n={edited.case_count}$ ROME edits"
            + (f" and $n={baseline.case_count}$ unedited baseline cases" if baseline else "")
            + (f" at $\\ell^*={target_layer}$" if target_layer is not None else ""),
            fontsize=14,
            y=0.998,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 0.97))

        stem = (
            f"artifact_grid_{safe_slug(edited.model)}_{safe_slug(edited.plan_id)}_{safe_slug(str(edited.edit_method))}"
        )
        if "png" in formats:
            path = output_dir / f"{stem}.png"
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            outputs.append(str(path))
        if "pdf" in formats:
            path = output_dir / f"{stem}.pdf"
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            outputs.append(str(path))
        plt.close(fig)
        if "json" in formats:
            path = write_json(output_dir / f"{stem}.json", data)
            outputs.append(str(path))
        index.append(
            {"model": edited.model, "plan_id": edited.plan_id, "edit_method": edited.edit_method, "stem": stem}
        )

    outputs.append(str(write_json(output_dir / "artifact_grid_index.json", {"outputs": index})))
    return outputs


__all__ = ["render_structural_artifact_grid"]
