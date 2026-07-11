"""Explain the weighted-spectrum localizer and ROME-presence decisions."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.common.io import write_json
from src.graphs.context import RenderContext, RendererUnavailableError
from src.graphs.structural.materialize import run_key
from src.graphs.structural.style import ELL_STAR_COLOR, TRIM_SHADE, model_label, setup_matplotlib_style
from src.results.naming import safe_slug
from src.structural.detectors.weighted_spectrum import PROFILE_FIELDS, SCORE_FIELD


PRESENCE_PRODUCERS = (
    "rome-presence-blind-peak",
    "rome-presence-blind-footprint",
    "rome-presence-delta",
)

PROFILE_LABELS: dict[str, str] = {
    "relative_subspace_frobenius": "Whitened subspace Frobenius score",
    "rank2_energy": "Rank-2 residual energy",
    "bilateral_coherence": "Bilateral coherence",
    "bilateral_balance": "Bilateral balance",
}


def _sequence(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(item) for item in value)


def _case_map(payload: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if payload is None:
        return {}
    return {
        str(case.get("case_id")): case
        for case in payload.get("cases", ())
        if isinstance(case, dict) and case.get("case_id") is not None
    }


def _matching_analysis(
    source: dict[str, Any],
    candidates: tuple[dict[str, Any], ...],
) -> dict[str, Any] | None:
    matching = [payload for payload in candidates if run_key(payload) == run_key(source)]
    if not matching:
        return None
    same_hash = [payload for payload in matching if payload.get("config_hash") == source.get("config_hash")]
    if len(same_hash) == 1:
        return same_hash[0]
    if len(matching) == 1:
        return matching[0]
    source_config = source.get("config", {})
    same_trim = [
        payload
        for payload in matching
        if payload.get("config", {}).get("trim_first") == source_config.get("trim_first")
        and payload.get("config", {}).get("trim_last") == source_config.get("trim_last")
    ]
    if len(same_trim) == 1:
        return same_trim[0]
    raise RendererUnavailableError(
        f"cannot uniquely pair {source.get('artifact_id')} with {matching[0].get('producer')} analysis"
    )


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _layers(data: Mapping[str, Any]) -> list[int]:
    profiles = data.get("profiles", {})
    if not isinstance(profiles, Mapping):
        return []
    try:
        return sorted(int(layer) for layer in profiles)
    except (TypeError, ValueError):
        return []


def _series(mapping: Any, layers: list[int]) -> list[float]:
    values = mapping if isinstance(mapping, Mapping) else {}
    return [_float(values.get(str(layer), values.get(layer))) for layer in layers]


def _profile_series(data: Mapping[str, Any], field: str, layers: list[int]) -> list[float]:
    profiles = data.get("profiles", {})
    return [_float(profiles.get(str(layer), {}).get(field)) for layer in layers]


def _presence_cutoff(data: Mapping[str, Any]) -> float | None:
    evidence = data.get("evidence", {})
    if not isinstance(evidence, Mapping):
        return None
    median = _float(evidence.get("median"))
    scale = _float(evidence.get("effective_mad_scale", evidence.get("mad_scale")))
    threshold = _float(evidence.get("universal_threshold"))
    value = median + scale * threshold
    return float(value) if math.isfinite(value) else None


def _shade_excluded(ax: Any, layers: list[int], excluded: list[int]) -> None:
    available = set(layers)
    for layer in excluded:
        if layer in available:
            ax.axvspan(layer - 0.5, layer + 0.5, color=TRIM_SHADE, zorder=0)


def _markers(ax: Any, target: int | None, detected: int | None) -> None:
    if target is not None:
        ax.axvline(target, color=ELL_STAR_COLOR, linestyle="--", linewidth=1.25, label=f"ROME layer {target}")
    if detected is not None:
        ax.axvline(detected, color="#dc2626", linestyle=":", linewidth=1.5, label=f"detected {detected}")


def _plot_lines(
    ax: Any,
    layers: list[int],
    lines: list[tuple[str, list[float], str]],
    *,
    title: str,
    target: int | None,
    detected: int | None,
    excluded: list[int],
) -> None:
    _shade_excluded(ax, layers, excluded)
    for label, values, color in lines:
        ax.plot(layers, values, marker="o", markersize=2.8, linewidth=1.4, label=label, color=color)
    _markers(ax, target, detected)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(r"Layer $\ell$")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(fontsize=7, loc="best")


def _save_figure(fig: Any, stem: Path, formats: tuple[str, ...]) -> list[str]:
    outputs: list[str] = []
    for extension in ("png", "pdf"):
        if extension not in formats:
            continue
        path = stem.with_suffix(f".{extension}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        outputs.append(str(path))
    return outputs


def _plot_overview(
    record: dict[str, Any],
    stem: Path,
    formats: tuple[str, ...],
) -> list[str]:
    import matplotlib.pyplot as plt

    layers = record["layers"]
    localizer = record["localizer"]
    target = record["target_layer"]
    detected = record["detected_layer"]
    excluded = record["excluded_layers"]
    peak = record["blind_peak"]
    footprint = record["blind_footprint"]
    delta = record["delta"]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5), squeeze=False)
    _plot_lines(
        axes[0, 0],
        layers,
        [("final weighted-spectrum score", _series(localizer.get("layer_scores"), layers), "#7c3aed")],
        title="1. Localize the strongest affine-relative perturbation",
        target=target,
        detected=detected,
        excluded=excluded,
    )
    _plot_lines(
        axes[0, 1],
        layers,
        [
            ("rank-2 energy", _profile_series(localizer, "rank2_energy", layers), "#16a34a"),
            ("coherence", _profile_series(localizer, "bilateral_coherence", layers), "#be123c"),
            ("balance", _profile_series(localizer, "bilateral_balance", layers), "#4d7c0f"),
        ],
        title="2. Low-rank three-layer footprint",
        target=target,
        detected=detected,
        excluded=excluded,
    )

    evidence_ax = axes[1, 0]
    _shade_excluded(evidence_ax, layers, excluded)
    colors = {"blind peak": "#2563eb", "blind footprint": "#be123c"}
    for label, data in (("blind peak", peak), ("blind footprint", footprint)):
        values = _series(data.get("layer_evidence"), layers)
        evidence_ax.plot(layers, values, marker="o", markersize=2.8, linewidth=1.4, label=label, color=colors[label])
        cutoff = _presence_cutoff(data)
        if cutoff is not None:
            evidence_ax.axhline(cutoff, color=colors[label], linestyle="--", linewidth=1.0, alpha=0.75)
    _markers(evidence_ax, target, detected)
    evidence_ax.set_title("3. Training-free presence tests and universal cutoffs", fontsize=10, fontweight="bold")
    evidence_ax.set_xlabel(r"Layer $\ell$")
    evidence_ax.grid(True, linestyle=":", alpha=0.35)
    evidence_ax.legend(fontsize=7, loc="best")

    decision_ax = axes[1, 1]
    decision_ax.axis("off")
    decisions = [
        ("Localizer", f"layer {detected}", localizer.get("detection_score")),
        ("Blind peak", peak.get("verdict", "unavailable"), peak.get("detection_score")),
        ("Blind footprint", footprint.get("verdict", "unavailable"), footprint.get("detection_score")),
        ("Clean-delta", delta.get("verdict", "unavailable"), delta.get("detection_score")),
    ]
    y = 0.92
    decision_ax.text(0.02, y, "4. Final decision evidence", fontsize=11, fontweight="bold", va="top")
    y -= 0.13
    for name, verdict, score in decisions:
        positive = verdict == "rome_like" or name == "Localizer"
        color = "#15803d" if positive else "#b91c1c"
        numeric = _float(score)
        score_text = f"{numeric:.4g}" if math.isfinite(numeric) else "n/a"
        decision_ax.text(0.04, y, name, fontsize=10, fontweight="bold", va="top")
        decision_ax.text(0.35, y, str(verdict), fontsize=10, color=color, va="top")
        decision_ax.text(0.83, y, f"score {score_text}", fontsize=9, ha="right", va="top")
        y -= 0.12
    delta_evidence = delta.get("evidence", {})
    if isinstance(delta_evidence, Mapping):
        y -= 0.03
        decision_ax.text(
            0.04,
            y,
            "rank-1 residual / roundoff bound\n"
            f"{_float(delta_evidence.get('rank1_residual')):.4g} / "
            f"{_float(delta_evidence.get('roundoff_bound')):.4g}",
            fontsize=9,
            va="top",
        )
    decision_ax.text(
        0.04,
        0.07,
        "Shaded layers are excluded by boundary trimming.\n"
        "Dashed black = configured ROME layer; dotted red = localized layer.",
        fontsize=8,
        color="#475569",
        va="bottom",
    )

    fig.suptitle(
        f"ROME detector explanation — {model_label(record['model'])}, case {record['case_id']}\n"
        f"{record['edit_method']} at layer {target}; weighted-spectrum prediction {detected}",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    outputs = _save_figure(fig, stem, formats)
    plt.close(fig)
    return outputs


def _plot_all_profiles(
    record: dict[str, Any],
    stem: Path,
    formats: tuple[str, ...],
    profile_fields: tuple[str, ...],
) -> list[str]:
    import matplotlib.pyplot as plt

    layers = record["layers"]
    target = record["target_layer"]
    detected = record["detected_layer"]
    excluded = record["excluded_layers"]
    profiles = record["localizer"]
    columns = 3
    rows = math.ceil(len(profile_fields) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15.0, max(4.0, 2.75 * rows)), squeeze=False)
    colors = ("#7c3aed", "#2563eb", "#0891b2", "#ea580c", "#16a34a")
    for index, field in enumerate(profile_fields):
        ax = axes[index // columns, index % columns]
        _shade_excluded(ax, layers, excluded)
        ax.plot(
            layers,
            _profile_series(profiles, field, layers),
            marker="o",
            markersize=2.5,
            linewidth=1.25,
            color=colors[index % len(colors)],
        )
        _markers(ax, target, detected)
        ax.set_title(PROFILE_LABELS.get(field, field), fontsize=9, fontweight="bold")
        ax.set_xlabel(r"Layer $\ell$", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.32)
        ax.tick_params(axis="both", labelsize=7)
    for index in range(len(profile_fields), rows * columns):
        axes[index // columns, index % columns].axis("off")
    fig.suptitle(
        f"All weighted-spectrum layer statistics — {model_label(record['model'])}, case {record['case_id']}",
        fontsize=14,
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    outputs = _save_figure(fig, stem, formats)
    plt.close(fig)
    return outputs


def _layer_rows(record: dict[str, Any], profile_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    layers = record["layers"]
    localizer = record["localizer"]
    peak = record["blind_peak"]
    footprint = record["blind_footprint"]
    rows: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        profile = localizer.get("profiles", {}).get(str(layer), {})
        row: dict[str, Any] = {
            "model": record["model"],
            "plan_id": record["plan_id"],
            "edit_method": record["edit_method"],
            "case_id": record["case_id"],
            "config_hash": record["config_hash"],
            "layer": layer,
            "evaluated": layer in record["evaluated_layers"],
            "excluded": layer in record["excluded_layers"],
            "target": layer == record["target_layer"],
            "detected": layer == record["detected_layer"],
            "localizer_score": _series(localizer.get("layer_scores"), layers)[index],
            "blind_peak_evidence": _series(peak.get("layer_evidence"), layers)[index],
            "blind_footprint_evidence": _series(footprint.get("layer_evidence"), layers)[index],
        }
        row.update({field: _float(profile.get(field)) for field in profile_fields})
        rows.append(row)
    return rows


def _summary_row(record: dict[str, Any]) -> dict[str, Any]:
    localizer = record["localizer"]
    peak = record["blind_peak"]
    footprint = record["blind_footprint"]
    delta = record["delta"]
    target = record["target_layer"]
    detected = record["detected_layer"]
    return {
        "model": record["model"],
        "plan_id": record["plan_id"],
        "edit_method": record["edit_method"],
        "case_id": record["case_id"],
        "config_hash": record["config_hash"],
        "target_layer": target,
        "detected_layer": detected,
        "layer_distance": "" if target is None or detected is None else abs(target - detected),
        "localizer_score": localizer.get("detection_score"),
        "localizer_margin": localizer.get("margin"),
        "blind_peak_verdict": peak.get("verdict"),
        "blind_peak_evidence_ratio": peak.get("detection_score"),
        "blind_footprint_verdict": footprint.get("verdict"),
        "blind_footprint_evidence_ratio": footprint.get("detection_score"),
        "delta_verdict": delta.get("verdict"),
        "delta_evidence_ratio": delta.get("detection_score"),
        "delta_rank1_residual": delta.get("evidence", {}).get("rank1_residual"),
        "delta_roundoff_bound": delta.get("evidence", {}).get("roundoff_bound"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)
    return str(path)


def _plot_summary(rows: list[dict[str, Any]], stem: Path, formats: tuple[str, ...]) -> list[str]:
    if not rows:
        return []
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model"]), str(row["edit_method"])), []).append(row)
    labels: list[str] = []
    values = {"Exact layer": [], "Blind peak": [], "Blind footprint": [], "Clean delta": []}
    for (model, method), cases in grouped.items():
        labels.append(f"{model_label(model)}\n{method}")
        distances = [row["layer_distance"] for row in cases if row["layer_distance"] != ""]
        values["Exact layer"].append(100.0 * sum(distance == 0 for distance in distances) / len(distances) if distances else 0.0)
        values["Blind peak"].append(100.0 * sum(row["blind_peak_verdict"] == "rome_like" for row in cases) / len(cases))
        values["Blind footprint"].append(
            100.0 * sum(row["blind_footprint_verdict"] == "rome_like" for row in cases) / len(cases)
        )
        values["Clean delta"].append(100.0 * sum(row["delta_verdict"] == "rome_like" for row in cases) / len(cases))

    x = np.arange(len(labels), dtype=float)
    width = 0.19
    colors = ("#7c3aed", "#2563eb", "#be123c", "#15803d")
    fig, ax = plt.subplots(figsize=(max(9.0, len(labels) * 1.8), 5.3))
    for index, (name, percentages) in enumerate(values.items()):
        offset = (index - 1.5) * width
        ax.bar(x + offset, percentages, width=width, label=name, color=colors[index])
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0.0, 105.0)
    ax.set_ylabel("Cases (%)")
    ax.set_title("ROME localizer and presence-decision outcomes")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    outputs = _save_figure(fig, stem, formats)
    plt.close(fig)
    return outputs


def render_rome_detector_explainer(context: RenderContext) -> list[str]:
    """Render all per-layer detector statistics and all three presence decisions."""
    analyses = context.analyses or {}
    weighted_payloads = analyses.get("weighted-spectrum", ())
    if not weighted_payloads:
        raise RendererUnavailableError("rome-detector-explainer requires weighted-spectrum analyses")

    options = dict(context.options or {})
    formats = _sequence(options.get("formats"), ("png", "pdf", "json", "csv"))
    unsupported = sorted(set(formats) - {"png", "pdf", "json", "csv"})
    if unsupported:
        raise ValueError(f"unsupported rome-detector-explainer formats: {', '.join(unsupported)}")
    if not formats:
        raise ValueError("rome-detector-explainer requires at least one output format")
    requested_fields = _sequence(options.get("profile_fields"), PROFILE_FIELDS)
    unknown_fields = sorted(set(requested_fields) - set(PROFILE_FIELDS))
    if unknown_fields:
        raise ValueError(f"unknown weighted-spectrum profile fields: {', '.join(unknown_fields)}")
    if not requested_fields:
        raise ValueError("rome-detector-explainer requires at least one profile field")
    max_cases_raw = options.get("max_cases")
    max_cases = None if max_cases_raw in (None, "", 0, "0") else max(1, int(max_cases_raw))

    setup_matplotlib_style()
    output_dir = Path(context.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    index: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    rendered_cases = 0

    for weighted_payload in weighted_payloads:
        companions = {
            producer: _matching_analysis(weighted_payload, analyses.get(producer, ()))
            for producer in PRESENCE_PRODUCERS
        }
        missing = [producer for producer, payload in companions.items() if payload is None]
        if missing:
            raise RendererUnavailableError(
                f"weighted-spectrum analysis {weighted_payload.get('artifact_id')} lacks paired analyses: "
                + ", ".join(missing)
            )
        companion_cases = {producer: _case_map(payload) for producer, payload in companions.items()}
        run = weighted_payload.get("run", {})
        model = str(run.get("model", "model"))
        plan_id = str(run.get("plan_id", "plan"))
        edit_method = str(run.get("edit_method", "method"))
        for case_id, weighted_case in _case_map(weighted_payload).items():
            if max_cases is not None and rendered_cases >= max_cases:
                break
            if weighted_case.get("status") != "complete":
                continue
            paired = {producer: cases.get(case_id) for producer, cases in companion_cases.items()}
            if any(case is None or case.get("status") != "complete" for case in paired.values()):
                continue
            localizer = dict(weighted_case.get("data", {}))
            layers = _layers(localizer)
            if not layers:
                continue
            accuracy = weighted_case.get("accuracy", {})
            target_raw = accuracy.get("target_layer")
            detected_raw = accuracy.get("detected_layer", localizer.get("anomalous_layer"))
            record = {
                "schema": "latium.rome_detector_explainer.v1",
                "model": model,
                "plan_id": plan_id,
                "edit_method": edit_method,
                "case_id": case_id,
                "config_hash": str(weighted_payload.get("config_hash", "default")),
                "target_layer": None if target_raw in (None, "") else int(target_raw),
                "detected_layer": None if detected_raw in (None, "") else int(detected_raw),
                "layers": layers,
                "evaluated_layers": [int(layer) for layer in localizer.get("evaluated_layers", ())],
                "excluded_layers": [int(layer) for layer in localizer.get("excluded_layers", ())],
                "score_field": localizer.get("score_field", SCORE_FIELD),
                "localizer": localizer,
                "blind_peak": dict(paired["rome-presence-blind-peak"].get("data", {})),
                "blind_footprint": dict(paired["rome-presence-blind-footprint"].get("data", {})),
                "delta": dict(paired["rome-presence-delta"].get("data", {})),
                "warnings": list(context.warnings),
            }
            case_dir = (
                output_dir
                / safe_slug(model)
                / safe_slug(plan_id)
                / safe_slug(edit_method)
                / safe_slug(record["config_hash"][:12])
                / safe_slug(case_id)
            )
            overview_stem = case_dir / "detector-overview"
            profiles_stem = case_dir / "all-layer-statistics"
            case_outputs = [
                *_plot_overview(record, overview_stem, formats),
                *_plot_all_profiles(record, profiles_stem, formats, requested_fields),
            ]
            if "json" in formats:
                case_outputs.append(str(write_json(case_dir / "detector-data.json", record)))
            outputs.extend(case_outputs)
            summary = _summary_row(record)
            summary_rows.append(summary)
            layer_rows.extend(_layer_rows(record, requested_fields))
            index.append(
                {
                    **{
                        key: record[key]
                        for key in ("model", "plan_id", "edit_method", "case_id", "config_hash")
                    },
                    "target_layer": record["target_layer"],
                    "detected_layer": record["detected_layer"],
                    "outputs": [str(Path(path).relative_to(output_dir)) for path in case_outputs],
                    "verdicts": {
                        "blind_peak": record["blind_peak"].get("verdict"),
                        "blind_footprint": record["blind_footprint"].get("verdict"),
                        "delta": record["delta"].get("verdict"),
                    },
                }
            )
            rendered_cases += 1
        if max_cases is not None and rendered_cases >= max_cases:
            break

    if not index:
        raise RendererUnavailableError("rome-detector-explainer found no complete paired detector cases")

    summary_fields = list(summary_rows[0])
    layer_fields = list(layer_rows[0]) if layer_rows else []
    if "csv" in formats:
        outputs.append(_write_csv(output_dir / "detector-case-summary.csv", summary_rows, summary_fields))
        if layer_rows:
            outputs.append(_write_csv(output_dir / "detector-layer-statistics.csv", layer_rows, layer_fields))
    outputs.extend(_plot_summary(summary_rows, output_dir / "detector-outcomes", formats))
    outputs.append(
        str(
            write_json(
                output_dir / "index.json",
                {
                    "schema": "latium.rome_detector_explainer.index.v1",
                    "profile_fields": list(requested_fields),
                    "formats": list(formats),
                    "cases": index,
                    "summary": summary_rows,
                    "warnings": list(context.warnings),
                },
            )
        )
    )
    return outputs


__all__ = ["render_rome_detector_explainer"]
