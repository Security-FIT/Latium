"""Six-panel per-layer matrix signals used by the CCS detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from src.common.arrays import local_zscore
from src.common.io import write_json
from src.graphs.context import RenderContext, RendererUnavailableError
from src.graphs.structural.materialize import matching_baseline, matrix_feature_table, target_layer_for
from src.graphs.structural.style import (
    BASELINE_COLOR,
    EDITED_BAND,
    EDITED_COLOR,
    ELL_STAR_COLOR,
    aggregate,
    model_label,
    setup_matplotlib_style,
)
from src.results.naming import safe_slug


FEATURES = ("spectral_gap", "top1_energy")
TRANSFORMS = ("raw", "lz5", "lz7")
FEATURE_LABELS = {
    "spectral_gap": r"Spectral gap $\mathrm{SG}_\ell$",
    "top1_energy": r"Top-1 energy $\mathrm{TE}_\ell$",
}
TRANSFORM_LABELS = {
    "raw": "Raw",
    "lz5": "Local z-score (window 5)",
    "lz7": "Local z-score (window 7)",
}
SUPPORTED_FORMATS = frozenset(("png", "pdf", "json"))


def _formats(options: dict[str, Any]) -> tuple[str, ...]:
    raw = options.get("formats", ("png", "pdf", "json"))
    if isinstance(raw, str):
        values = tuple(part.strip() for part in raw.split(",") if part.strip())
    else:
        values = tuple(str(part) for part in raw)
    if not values or set(values) - SUPPORTED_FORMATS:
        raise ValueError("structural-ccs-lines formats must be a nonempty list of png, pdf, json")
    return tuple(dict.fromkeys(values))


def _transform(values: np.ndarray, key: str) -> np.ndarray:
    if key == "raw":
        return values
    window = {"lz5": 5, "lz7": 7}[key]
    # Match CCS's center-excluded abs(local_zscore(...)) calculation per case.
    return local_zscore(values, window, axis=1, absolute=True)


def _stats(layers: np.ndarray, matrix: np.ndarray) -> dict[str, Any]:
    mean, std, count = aggregate(matrix)
    return {
        "layers": layers.astype(int).tolist(),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "count": count.astype(int).tolist(),
    }


def _require_complete_table(table: Any, *, label: str) -> None:
    if table.layers.size < 3:
        raise RendererUnavailableError(f"{label} needs at least three captured layers")
    for feature in FEATURES:
        if not np.all(np.isfinite(table.values[feature])):
            raise RendererUnavailableError(f"{label} has missing or nonfinite {feature} values")


def render_structural_ccs_lines(context: RenderContext) -> list[str]:
    options = dict(context.options or {})
    formats = _formats(options)
    captures = list((context.captures or {}).get("matrix-features", ()))
    edited_payloads = [payload for payload in captures if payload.get("run", {}).get("edit_method")]
    if not edited_payloads:
        raise RendererUnavailableError("structural-ccs-lines requires edited matrix-features captures")

    output_dir = Path(context.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib_style()
    import matplotlib.pyplot as plt

    outputs: list[str] = []
    index: list[dict[str, Any]] = []
    for edited_payload in edited_payloads:
        edited = matrix_feature_table(edited_payload, FEATURES)
        analysis_matches = [
            payload for payload in (context.analyses or {}).get("ccs-composite", ())
            if payload.get("run", {}).get("model") == edited.model
            and payload.get("run", {}).get("plan_id") == edited.plan_id
            and payload.get("run", {}).get("edit_method") == edited.edit_method
            and payload.get("status") == "complete"
        ]
        if not analysis_matches:
            raise RendererUnavailableError(
                f"no complete CCS analysis for {edited.model}/{edited.plan_id}/{edited.edit_method}"
            )
        baseline_payload = matching_baseline(edited_payload, captures)
        if baseline_payload is None:
            raise RendererUnavailableError(
                f"no matched unedited matrix-features baseline for {edited.model}/{edited.plan_id}/{edited.edit_method}"
            )
        baseline = matrix_feature_table(baseline_payload, FEATURES)
        _require_complete_table(edited, label="edited capture")
        _require_complete_table(baseline, label="unedited baseline")
        if not np.array_equal(edited.layers, baseline.layers):
            raise RendererUnavailableError("edited and baseline matrix-features layers do not match")
        target_layer = target_layer_for(
            list(context.executions),
            model=edited.model,
            plan_id=edited.plan_id,
            edit_method=edited.edit_method,
        )

        fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True, squeeze=False)
        data: dict[str, Any] = {
            "schema": "latium.structural_ccs_lines.v1",
            "model": edited.model,
            "plan_id": edited.plan_id,
            "edit_method": edited.edit_method,
            "target_layer": target_layer,
            "ccs_analysis_ids": [str(payload.get("artifact_id")) for payload in analysis_matches],
            "features": list(FEATURES),
            "transforms": list(TRANSFORMS),
            "edited_case_ids": list(edited.case_ids),
            "baseline_case_ids": list(baseline.case_ids),
            "series": {},
            "warnings": list(context.warnings),
        }

        for row, feature in enumerate(FEATURES):
            data["series"][feature] = {}
            for col, transform in enumerate(TRANSFORMS):
                ax = axes[row, col]
                edited_values = _transform(edited.values[feature], transform)
                baseline_values = _transform(baseline.values[feature], transform)
                edited_stats = _stats(edited.layers, edited_values)
                baseline_stats = _stats(baseline.layers, baseline_values)
                data["series"][feature][transform] = {
                    "edited_cases": edited_values.astype(float).tolist(),
                    "edited": edited_stats,
                    "baseline": baseline_stats,
                }

                for case_values in edited_values:
                    ax.plot(edited.layers, case_values, color=EDITED_COLOR, alpha=0.24, linewidth=0.8)
                edited_mean = np.asarray(edited_stats["mean"])
                edited_std = np.asarray(edited_stats["std"])
                ax.fill_between(
                    edited.layers,
                    edited_mean - edited_std,
                    edited_mean + edited_std,
                    color=EDITED_BAND,
                    alpha=0.18,
                    linewidth=0,
                )
                ax.plot(edited.layers, edited_mean, color=EDITED_COLOR, linewidth=2.0)
                ax.plot(
                    baseline.layers,
                    baseline_stats["mean"],
                    color=BASELINE_COLOR,
                    linestyle="--",
                    linewidth=1.6,
                )
                if target_layer is not None:
                    ax.axvline(target_layer, color=ELL_STAR_COLOR, linestyle=":", linewidth=1.2)
                ax.set_title(TRANSFORM_LABELS[transform], fontsize=10, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(FEATURE_LABELS[feature], fontsize=10)
                if row == len(FEATURES) - 1:
                    ax.set_xlabel(r"Layer $\ell$")
                ax.grid(True, linestyle=":", alpha=0.30)
                ax.margins(x=0.02)

        fig.legend(
            handles=[
                Line2D([0], [0], color=EDITED_COLOR, alpha=0.35, linewidth=1, label="Individual ROME edits"),
                Line2D([0], [0], color=EDITED_COLOR, linewidth=2, label="Edited mean"),
                Patch(facecolor=EDITED_BAND, alpha=0.25, label=r"Edited mean $\pm 1\sigma$"),
                Line2D([0], [0], color=BASELINE_COLOR, linestyle="--", label="Unedited mean"),
                Line2D([0], [0], color=ELL_STAR_COLOR, linestyle=":", label="Target layer"),
            ],
            loc="lower center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
        )
        fig.suptitle(
            f"CCS per-layer signals on {model_label(edited.model)} — "
            f"{edited.case_count} edits, {baseline.case_count} baseline cases",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))

        stem = f"ccs_lines_{safe_slug(edited.model)}_{safe_slug(edited.plan_id)}_{safe_slug(str(edited.edit_method))}"
        for image_format in ("png", "pdf"):
            if image_format in formats:
                path = output_dir / f"{stem}.{image_format}"
                fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
                outputs.append(str(path))
        plt.close(fig)
        if "json" in formats:
            outputs.append(str(write_json(output_dir / f"{stem}.json", data)))
        index.append({"model": edited.model, "plan_id": edited.plan_id, "edit_method": edited.edit_method, "stem": stem})

    outputs.append(str(write_json(output_dir / "ccs_lines_index.json", {"outputs": index})))
    return outputs


__all__ = ["render_structural_ccs_lines"]
