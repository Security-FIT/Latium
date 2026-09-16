"""Per-layer cohort and case plots for the relative ROME paper experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.common.io import write_json
from src.graphs.context import RenderContext, RendererUnavailableError
from src.graphs.structural.materialize import target_layer_for
from src.graphs.structural.relative_materialize import (
    CAPTURE_FOR, METHODS, RelativeCase, RelativeTable, matching_capture,
    matching_relative_baseline, materialize_relative, profile_stats, selected_fit,
)
from src.graphs.structural.style import (
    BASELINE_COLOR, EDITED_BAND, EDITED_COLOR, ELL_STAR_COLOR,
    model_label, setup_matplotlib_style,
)
from src.results.naming import safe_slug


_METHOD_NAMES = {
    "original-v3-relative-b0-v1": "Original V3",
    "v0-relative-b0-v1": "V0",
    "v0r-relative-b0-v1": "V0R",
    "v1-relative-b0-v1": "V1",
    "v2-relative-b0-v1": "V2",
    "lof-relative-b0-v1": "Cross-layer LOF",
    "decomposition-relative-b0-v1": "Cross-layer decomposition",
    "token-alignment-relative-b0-v1": "Token alignment",
}


def _sequence(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(item) for item in value)


def _clean(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _json_stats(values: np.ndarray, *, transform: str) -> dict[str, Any]:
    matrix = np.log1p(values) if transform == "log1p" else values
    return {key: _clean(value) if key != "count" else [int(v) for v in value]
            for key, value in profile_stats(matrix).items()}


def _fit_matrix(table: RelativeTable, kind: str) -> np.ndarray:
    values = np.full(table.values.shape, np.nan)
    position = {int(layer): col for col, layer in enumerate(table.layers)}
    for row, case in enumerate(table.cases):
        for layer, value in selected_fit(case, kind).items():
            if layer in position:
                values[row, position[layer]] = value
    return values


def _case_json(case: RelativeCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "status": case.status,
        "scores": {str(layer): value for layer, value in sorted(case.scores.items())},
        "eligible_layers": list(case.eligible_layers),
        "candidate_layer": case.candidate_layer,
        "original_localizer_layer": case.original_localizer_layer,
        "detected": case.detected,
        "gain": case.gain,
        "background_cost": case.background_cost,
        "anomaly_cost": case.anomaly_cost,
        "selected_background": case.background_fit,
        "selected_anomaly": case.anomaly_fit,
        "background_fit_log1p": {str(k): v for k, v in selected_fit(case, "background").items()},
        "anomaly_fit_log1p": {str(k): v for k, v in selected_fit(case, "anomaly").items()},
        "reason": case.reason,
    }


def _table_json(table: RelativeTable | None) -> dict[str, Any] | None:
    if table is None:
        return None
    return {
        "artifact_id": table.artifact_id,
        "config_hash": table.config_hash,
        "layers": [int(v) for v in table.layers],
        "case_count": len(table.cases),
        "raw": _json_stats(table.values, transform="raw"),
        "log1p": _json_stats(table.values, transform="log1p"),
        "background_fit_log1p": _json_stats(_fit_matrix(table, "background"), transform="raw"),
        "anomaly_fit_log1p": _json_stats(_fit_matrix(table, "anomaly"), transform="raw"),
        "cases": [_case_json(case) for case in table.cases],
    }


def _draw_series(
    ax: Any, table: RelativeTable, baseline: RelativeTable | None,
    *, transform: str, target: int | None, traces: bool, fits: bool,
) -> None:
    layers = table.layers
    values = np.log1p(table.values) if transform == "log1p" else table.values
    stats = profile_stats(values)
    mean, std = stats["mean"], stats["std"]
    if traces:
        for row in values:
            ax.plot(layers, row, color=EDITED_COLOR, alpha=0.10, linewidth=0.7, zorder=1)
    ax.fill_between(layers, stats["min"], stats["max"], color=EDITED_BAND, alpha=0.17,
                    label="Edited min–max")
    ax.fill_between(layers, mean - std, mean + std, color=EDITED_BAND, alpha=0.58,
                    label="Edited mean ± SD")
    ax.plot(layers, mean, color=EDITED_COLOR, linewidth=1.8, label="Edited mean")
    if baseline is not None and baseline.layers.size:
        baseline_values = np.log1p(baseline.values) if transform == "log1p" else baseline.values
        baseline_stats = profile_stats(baseline_values)
        ax.plot(baseline.layers, baseline_stats["mean"], color=BASELINE_COLOR,
                linestyle="--", linewidth=1.5, label="Unedited baseline")
        baseline_candidates = [case.candidate_layer for case in baseline.cases
                               if case.status == "complete" and case.candidate_layer is not None]
        if baseline_candidates:
            candidate = Counter(baseline_candidates).most_common(1)[0][0]
            ax.axvline(candidate, color=BASELINE_COLOR, linestyle=":", alpha=0.65,
                       label=f"Baseline candidate {candidate}")
    if fits and transform == "log1p":
        for kind, color, label in (
            ("background", "#2563eb", "Selected background fit"),
            ("anomaly", "#059669", "Selected anomaly fit"),
        ):
            fit = profile_stats(_fit_matrix(table, kind))
            if np.any(fit["count"]):
                ax.plot(layers, fit["mean"], color=color, alpha=0.85, linewidth=1.1,
                        label=label + " mean")
    if target is not None:
        ax.axvline(target, color=ELL_STAR_COLOR, linestyle=":", linewidth=1.3,
                   label=f"Edited layer {target}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("log1p(score)" if transform == "log1p" else "Score")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.margins(x=0.02)


def _cohort_figure(table: RelativeTable, baseline: RelativeTable | None,
                   target: int | None, traces: bool) -> Any:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.5, 7.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(4, 1), hspace=0.38)
    raw, decision = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    _draw_series(raw, table, baseline, transform="raw", target=target, traces=traces, fits=False)
    _draw_series(decision, table, baseline, transform="log1p", target=target, traces=traces, fits=True)
    raw.set_title("Raw score")
    decision.set_title("B0 decision space and selected fits")
    handles, labels = decision.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    candidates = Counter(case.candidate_layer for case in table.cases
                         if case.status == "complete" and case.candidate_layer is not None)
    count_axis = fig.add_subplot(grid[1, :])
    if table.layers.size:
        count_axis.bar(table.layers, [candidates.get(int(layer), 0) for layer in table.layers],
                       color=EDITED_COLOR, alpha=0.7, width=0.8)
        count_axis.set_xlim(table.layers[0] - 0.8, table.layers[-1] + 0.8)
        coverage_axis = count_axis.twinx()
        coverage_axis.plot(table.layers, profile_stats(table.values)["count"], color="#64748b",
                           marker=".", linewidth=1.0)
        coverage_axis.set_ylabel("Profiles at layer", color="#64748b")
        coverage_axis.set_ylim(bottom=0)
    if target is not None:
        count_axis.axvline(target, color=ELL_STAR_COLOR, linestyle=":")
    count_axis.set_ylabel("Candidates")
    count_axis.set_xlabel("Layer")
    count_axis.grid(True, axis="y", linestyle=":", alpha=0.35)
    available = sum(case.status == "complete" for case in table.cases)
    fig.suptitle(
        f"{_METHOD_NAMES[table.experiment_id]} • {model_label(table.model)} • {table.plan_id} • "
        f"{table.edit_method}   (profiles {len(table.cases)}, decisions {available}; "
        f"baseline {len(baseline.cases) if baseline else 0})", fontsize=13,
    )
    fig.subplots_adjust(top=0.90, bottom=0.16, left=0.07, right=0.98)
    return fig


def _case_is_error(case: RelativeCase, target: int | None) -> bool:
    return case.status != "complete" or case.detected is not True or (
        target is not None and case.candidate_layer != target
    )


def _case_pages(path: Path, table: RelativeTable, baseline: RelativeTable | None,
                target: int | None, mode: str) -> bool:
    if mode == "none":
        return False
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    selected = [case for case in table.cases if mode == "all" or _case_is_error(case, target)]
    # A false positive on the unedited model is also a diagnostic error.
    baseline_errors = [] if baseline is None else [
        case for case in baseline.cases if case.status == "complete" and case.detected is True
    ]
    if mode == "all" and baseline is not None:
        baseline_errors = list(baseline.cases)
    if not selected and not baseline_errors:
        return False
    with PdfPages(path) as pdf:
        for case, is_baseline in [(case, False) for case in selected] + [(case, True) for case in baseline_errors]:
            fig, ax = plt.subplots(figsize=(11.7, 6.5))
            layers = sorted(case.scores)
            values = [np.log1p(case.scores[layer]) for layer in layers]
            ax.plot(layers, values, color=BASELINE_COLOR if is_baseline else EDITED_COLOR,
                    marker="o", markersize=3, linewidth=1.5, label="Observed log1p(score)")
            for kind, color in (("background", "#2563eb"), ("anomaly", "#059669")):
                fit = selected_fit(case, kind)
                if fit:
                    ax.plot(sorted(fit), [fit[layer] for layer in sorted(fit)],
                            color=color, linestyle="--", linewidth=1.3, label=f"Selected {kind} fit")
            if case.candidate_layer is not None:
                ax.axvline(case.candidate_layer, color="#059669", linestyle=":",
                           label=f"Candidate {case.candidate_layer}")
            if target is not None and not is_baseline:
                ax.axvline(target, color=ELL_STAR_COLOR, linestyle=":", label=f"Target {target}")
            if case.original_localizer_layer is not None:
                ax.axvline(case.original_localizer_layer, color="#d97706", linestyle="-.",
                           alpha=0.65, label=f"Original localizer {case.original_localizer_layer}")
            ax.set(title=f"{_METHOD_NAMES[table.experiment_id]} • {'Unedited' if is_baseline else 'Edited'} "
                         f"case {case.case_id} • {case.status} • detected={case.detected}",
                   xlabel="Layer", ylabel="log1p(score)")
            ax.grid(True, linestyle=":", alpha=0.35)
            ax.legend(loc="best", fontsize=8)
            fig.text(0.09, 0.04,
                     f"gain={case.gain}   background cost={case.background_cost}   "
                     f"anomaly cost={case.anomaly_cost}   reason={case.reason or '—'}", fontsize=9)
            fig.tight_layout(rect=(0, 0.07, 1, 1))
            pdf.savefig(fig)
            plt.close(fig)
    return True


def _overview(path: Path, entries: list[tuple[RelativeTable, RelativeTable | None, int | None]],
              identifiers: tuple[str, ...]) -> bool:
    frequency = Counter(item[0].experiment_id for item in entries)
    # A method with multiple analysis configurations has no unique overview trace.
    subset = [item for item in entries if item[0].experiment_id in identifiers
              and frequency[item[0].experiment_id] == 1]
    if len(subset) < 2:
        return False
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), squeeze=False)
    for ax, identifier in zip(axes.flat, identifiers):
        item = next((item for item in subset if item[0].experiment_id == identifier), None)
        if item is None:
            ax.text(0.5, 0.5, "Unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        table, baseline, target = item
        _draw_series(ax, table, baseline, transform="log1p", target=target, traces=False, fits=False)
        ax.set_title(_METHOD_NAMES[identifier])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(f"Relative ROME methods • {model_label(subset[0][0].model)} • "
                 f"{subset[0][0].plan_id} • {subset[0][0].edit_method}")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return True


def render_rome_relative_profile_grid(context: RenderContext) -> list[str]:
    options = dict(context.options or {})
    formats = set(_sequence(options.get("formats"), ("png", "pdf", "json")))
    if not formats <= {"png", "pdf", "json"} or not formats:
        raise ValueError("formats must contain png, pdf, and/or json")
    case_pages = str(options.get("case_pages", "errors"))
    if case_pages not in {"none", "errors", "all"}:
        raise ValueError("case_pages must be none, errors, or all")
    trace_option = options.get("case_traces", "auto")
    if trace_option not in (True, False, "auto"):
        raise ValueError("case_traces must be true, false, or auto")
    output_dir = Path(context.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analyses = context.analyses or {}
    captures = context.captures or {}
    edited = [payload for producer in METHODS for payload in analyses.get(producer, ())
              if payload.get("run", {}).get("edit_method")]
    if not edited:
        raise RendererUnavailableError("no edited relative ROME analyses are available")

    setup_matplotlib_style()
    import matplotlib.pyplot as plt

    outputs: list[str] = []
    index: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[tuple[RelativeTable, RelativeTable | None, int | None]]] = defaultdict(list)
    for payload in edited:
        producer = str(payload.get("producer"))
        all_producer = list(analyses.get(producer, ()))
        baseline_payload = matching_relative_baseline(payload, all_producer)
        own_capture = matching_capture(payload, list(captures.get(CAPTURE_FOR[producer], ())))
        baseline_capture = (matching_capture(baseline_payload, list(captures.get(CAPTURE_FOR[producer], ())))
                            if baseline_payload else None)
        if baseline_payload and (own_capture is not None or baseline_capture is not None):
            if (own_capture is None or baseline_capture is None
                    or own_capture.get("config_hash") != baseline_capture.get("config_hash")):
                baseline_payload = None
                baseline_capture = None
        for experiment_id in METHODS[producer]:
            table = materialize_relative(payload, experiment_id, capture=own_capture)
            baseline = (materialize_relative(baseline_payload, experiment_id, capture=baseline_capture)
                        if baseline_payload else None)
            if baseline is not None and not np.isfinite(baseline.values).any():
                baseline = None
            stem = "relative_" + "_".join(safe_slug(str(value)) for value in (
                table.model, table.plan_id, table.edit_method, producer, experiment_id,
                table.config_hash or "nohash", table.artifact_id or "noartifact",
            ))
            entry: dict[str, Any] = {
                "model": table.model, "plan_id": table.plan_id, "edit_method": table.edit_method,
                "producer": producer, "experiment_id": experiment_id,
                "analysis_artifact_id": table.artifact_id,
                "baseline_artifact_id": baseline.artifact_id if baseline else None,
                "config_hash": table.config_hash, "stem": stem, "outputs": [],
            }
            if not table.layers.size or not np.isfinite(table.values).any():
                entry["status"] = "unavailable"
                entry["reason"] = "no usable per-layer scores"
                index.append(entry)
                continue
            target = target_layer_for(list(context.executions), model=table.model,
                                      plan_id=table.plan_id, edit_method=table.edit_method)
            data = {
                "schema": "latium.rome_relative_profiles.v1",
                "model": table.model, "plan_id": table.plan_id,
                "edit_method": table.edit_method, "producer": producer,
                "experiment_id": experiment_id, "target_layer": target,
                "edited": _table_json(table), "baseline": _table_json(baseline),
                "candidate_frequency": {str(layer): count for layer, count in sorted(Counter(
                    case.candidate_layer for case in table.cases
                    if case.status == "complete" and case.candidate_layer is not None
                ).items())},
                "warnings": list(context.warnings) + (["No matching unedited baseline analysis"] if baseline is None else []),
            }
            if "json" in formats:
                path = write_json(output_dir / f"{stem}.json", data)
                outputs.append(str(path))
                entry["outputs"].append(str(path.name))
            if formats & {"png", "pdf"}:
                traces = trace_option is True or (trace_option == "auto" and len(table.cases) <= 20)
                fig = _cohort_figure(table, baseline, target, traces)
                for fmt in ("png", "pdf"):
                    if fmt in formats:
                        path = output_dir / f"{stem}.{fmt}"
                        fig.savefig(path, dpi=180, facecolor="white")
                        outputs.append(str(path))
                        entry["outputs"].append(path.name)
                plt.close(fig)
            if case_pages != "none":
                path = output_dir / f"{stem}_cases.pdf"
                if _case_pages(path, table, baseline, target, case_pages):
                    outputs.append(str(path))
                    entry["outputs"].append(path.name)
            entry["status"] = "complete"
            index.append(entry)
            groups[(table.model, table.plan_id, str(table.edit_method))].append((table, baseline, target))
    for key, entries in groups.items():
        for name, ids in (
            ("directional", METHODS["rome-directional-experiments"]),
            ("global", ("original-v3-relative-b0-v1", *METHODS["rome-cross-layer-experiments"],
                        "token-alignment-relative-b0-v1")),
        ):
            path = output_dir / ("relative_overview_" + "_".join(safe_slug(value) for value in key) + f"_{name}.png")
            if "png" in formats and _overview(path, entries, ids):
                outputs.append(str(path))
    if not any(entry["status"] == "complete" for entry in index):
        raise RendererUnavailableError("relative ROME analyses contain no usable per-layer scores")
    index_path = write_json(output_dir / "relative_profile_index.json", {
        "schema": "latium.rome_relative_profile_index.v1", "methods": index,
        "outputs": [str(Path(path).name) for path in outputs],
    })
    outputs.append(str(index_path))
    return outputs


__all__ = ["render_rome_relative_profile_grid"]
