"""Typed, lineage-aware views of saved relative ROME decisions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from src.graphs.context import RendererUnavailableError
from src.graphs.structural.materialize import run_key


METHODS: dict[str, tuple[str, ...]] = {
    "rome-profile-experiments": ("original-v3-relative-b0-v1",),
    "rome-directional-experiments": (
        "v0-relative-b0-v1", "v0r-relative-b0-v1", "v1-relative-b0-v1", "v2-relative-b0-v1",
    ),
    "rome-cross-layer-experiments": ("lof-relative-b0-v1", "decomposition-relative-b0-v1"),
    "rome-token-alignment-experiments": ("token-alignment-relative-b0-v1",),
}

CAPTURE_FOR: dict[str, str] = {
    "rome-profile-experiments": "gram-localization",
    "rome-directional-experiments": "gram-directional-error-v1",
    "rome-cross-layer-experiments": "gram-cross-layer-v1",
    "rome-token-alignment-experiments": "token-subspace-alignment-v1",
}

SCORE_FIELD: dict[str, str] = {
    "original-v3-relative-b0-v1": "diagonal_relative",
    "v0-relative-b0-v1": "original_score",
    "v0r-relative-b0-v1": "refined_ratio_score",
    "v1-relative-b0-v1": "centered_directional_score",
    "v2-relative-b0-v1": "standardized_directional_score",
    "lof-relative-b0-v1": "lof_score",
    "decomposition-relative-b0-v1": "decomposition_score",
    "token-alignment-relative-b0-v1": "alignment_score",
}


def _scores(raw: Any) -> dict[int, float]:
    if not isinstance(raw, Mapping):
        return {}
    values: dict[int, float] = {}
    for key, value in raw.items():
        try:
            layer, score = int(key), float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score) and score >= 0:
            values[layer] = score
    return values


def _integer(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class RelativeCase:
    case_id: str
    status: str
    scores: dict[int, float]
    eligible_layers: tuple[int, ...]
    candidate_layer: int | None
    original_localizer_layer: int | None
    detected: bool | None
    gain: float | None
    background_cost: float | None
    anomaly_cost: float | None
    background_fit: dict[str, Any] | None
    anomaly_fit: dict[str, Any] | None
    reason: str | None


@dataclass(frozen=True)
class RelativeTable:
    model: str
    plan_id: str
    edit_method: str | None
    producer: str
    experiment_id: str
    config_hash: str | None
    artifact_id: str | None
    cases: tuple[RelativeCase, ...]
    layers: np.ndarray
    values: np.ndarray


def matching_relative_baseline(edited: dict[str, Any], analyses: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Match the same producer/model/plan/config, never an adjacent run."""
    model, plan_id, _ = run_key(edited)
    producer, digest = edited.get("producer"), edited.get("config_hash")
    candidates = [
        payload for payload in analyses
        if run_key(payload) == (model, plan_id, None)
        and payload.get("producer") == producer
        and payload.get("config_hash") == digest
    ]
    if len(candidates) > 1:
        raise RendererUnavailableError(
            f"multiple unedited {producer} analyses match {model}/{plan_id}/{digest}"
        )
    return candidates[0] if candidates else None


def matching_capture(analysis: dict[str, Any], captures: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A capture fallback is permitted only when the analysis explicitly cites it."""
    input_ids = {
        str(ref.get("artifact_id")) for ref in analysis.get("inputs", ())
        if isinstance(ref, dict) and ref.get("artifact_id")
    }
    producer = CAPTURE_FOR.get(str(analysis.get("producer")))
    candidates = [
        capture for capture in captures
        if capture.get("artifact_id") in input_ids
        and capture.get("producer") == producer
        and run_key(capture) == run_key(analysis)
    ]
    if len(candidates) > 1:
        raise RendererUnavailableError("analysis references multiple matching profile captures")
    return candidates[0] if candidates else None


def _capture_case_scores(capture: dict[str, Any] | None, case_id: str, experiment_id: str) -> dict[int, float]:
    if capture is None:
        return {}
    field = SCORE_FIELD[experiment_id]
    for case in capture.get("cases", ()):
        if str(case.get("case_id")) != case_id:
            continue
        profiles = case.get("data", {}).get("profiles", {})
        if not isinstance(profiles, Mapping):
            return {}
        return _scores({layer: profile.get(field) for layer, profile in profiles.items() if isinstance(profile, Mapping)})
    return {}


def materialize_relative(
    analysis: dict[str, Any], experiment_id: str, *, capture: dict[str, Any] | None = None,
) -> RelativeTable:
    producer = str(analysis.get("producer"))
    if experiment_id not in METHODS.get(producer, ()):
        raise ValueError(f"{experiment_id} is not produced by {producer}")
    cases: list[RelativeCase] = []
    for case in analysis.get("cases", ()):
        case_id = str(case.get("case_id"))
        decision = case.get("data", {}).get("experiments", {}).get(experiment_id)
        decision = decision if isinstance(decision, dict) else {}
        scores = _scores(decision.get("layer_scores"))
        if not scores:
            scores = _capture_case_scores(capture, case_id, experiment_id)
        eligible = tuple(sorted({_integer(layer) for layer in decision.get("eligible_layers", ())} - {None}))
        if not eligible:
            eligible = tuple(sorted(scores))
        diagnostics = decision.get("diagnostics") or {}
        status = "complete" if decision.get("status") == "complete" else "unavailable"
        detected = decision.get("is_rome_like") if status == "complete" else None
        cases.append(RelativeCase(
            case_id=case_id, status=status, scores=scores, eligible_layers=eligible,
            candidate_layer=_integer(decision.get("candidate_layer")),
            original_localizer_layer=_integer(decision.get("original_localizer_layer")),
            detected=detected if isinstance(detected, bool) else None,
            gain=_number(decision.get("gain")),
            background_cost=_number(decision.get("background_cost")),
            anomaly_cost=_number(decision.get("anomaly_cost")),
            background_fit=diagnostics.get("selected_background"),
            anomaly_fit=diagnostics.get("selected_anomaly"),
            reason=diagnostics.get("reason") or case.get("error"),
        ))
    layers = np.asarray(sorted({layer for case in cases for layer in (*case.eligible_layers, *case.scores)}), dtype=int)
    values = np.full((len(cases), len(layers)), np.nan)
    lookup = {int(layer): col for col, layer in enumerate(layers)}
    for row, case in enumerate(cases):
        for layer, score in case.scores.items():
            values[row, lookup[layer]] = score
    model, plan_id, edit_method = run_key(analysis)
    return RelativeTable(
        model=str(model), plan_id=str(plan_id), edit_method=edit_method,
        producer=producer, experiment_id=experiment_id,
        config_hash=analysis.get("config_hash"), artifact_id=analysis.get("artifact_id"),
        cases=tuple(cases), layers=layers, values=values,
    )


def profile_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    """Per-layer statistics with explicit missing counts and no empty-slice warnings."""
    if values.ndim != 2:
        raise ValueError("profiles must be a two-dimensional matrix")
    count = np.isfinite(values).sum(axis=0)
    mean = np.full(values.shape[1], np.nan)
    std = np.full(values.shape[1], np.nan)
    low = np.full(values.shape[1], np.nan)
    high = np.full(values.shape[1], np.nan)
    for col in np.flatnonzero(count):
        observed = values[np.isfinite(values[:, col]), col]
        mean[col], std[col] = observed.mean(), observed.std()
        low[col], high[col] = observed.min(), observed.max()
    return {"mean": mean, "std": std, "min": low, "max": high, "count": count}


def selected_fit(case: RelativeCase, kind: str) -> dict[int, float]:
    """Rebuild the saved selected fit in B0's log1p decision space."""
    fit = case.background_fit if kind == "background" else case.anomaly_fit
    if not isinstance(fit, dict) or not case.eligible_layers:
        return {}
    layers = case.eligible_layers
    depth_range = layers[-1] - layers[0]
    if depth_range <= 0:
        return {}
    depth = np.asarray([(layer - layers[0]) / depth_range for layer in layers])
    family = fit.get("family")
    columns = [np.ones(len(layers)), depth]
    if family == "quadratic":
        columns.append(depth**2)
    elif family == "affine-plus-step":
        split = _integer(fit.get("split"))
        if split is None or not 0 <= split < len(layers):
            return {}
        columns.append((np.arange(len(layers)) >= split).astype(float))
    elif family != "affine":
        return {}
    if kind == "anomaly":
        candidate = _integer(fit.get("candidate_layer"))
        columns.append((np.asarray(layers) == candidate).astype(float))
    coefficients = fit.get("coefficients")
    if not isinstance(coefficients, (list, tuple)) or len(coefficients) != len(columns):
        return {}
    try:
        result = np.column_stack(columns) @ np.asarray(coefficients, dtype=float)
    except (TypeError, ValueError):
        return {}
    return {layer: float(value) for layer, value in zip(layers, result) if np.isfinite(value)}


__all__ = [
    "METHODS", "CAPTURE_FOR", "RelativeCase", "RelativeTable", "matching_relative_baseline",
    "matching_capture", "materialize_relative", "profile_stats", "selected_fit",
]
