"""
Typed views over structural artifacts for graph renderers.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.graphs.context import RendererUnavailableError


@dataclass(frozen=True)
class MatrixFeatureTable:
    model: str
    plan_id: str
    edit_method: str | None
    layers: np.ndarray
    values: dict[str, np.ndarray]
    case_count: int


def run_key(payload: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    run = payload.get("run", {})
    return run.get("model"), run.get("plan_id"), run.get("edit_method")


def target_layer_for(
    executions: list[dict[str, Any]],
    *,
    model: str,
    plan_id: str,
    edit_method: str | None,
) -> int | None:
    for payload in executions:
        if run_key(payload) != (model, plan_id, edit_method):
            continue
        value = payload.get("summary", {}).get("target_layer")
        return None if value in (None, "") else int(value)
    return None


def matrix_feature_table(
    payload: dict[str, Any],
    features: tuple[str, ...],
    *,
    family: str = "proj",
    require_success: bool = False,
) -> MatrixFeatureTable:
    run = payload.get("run", {})
    case_profiles: list[dict[int, dict[str, Any]]] = []
    for case in payload.get("cases", []):
        if not isinstance(case, dict) or case.get("status") != "complete":
            continue
        if require_success and not bool(case.get("edit", {}).get("success", True)):
            continue
        families = case.get("data", {}).get("families", {})
        profiles = families.get(family, {}) if isinstance(families, dict) else {}
        if not isinstance(profiles, dict):
            continue
        normalized: dict[int, dict[str, Any]] = {}
        for raw_layer, profile in profiles.items():
            if not isinstance(profile, dict):
                continue
            try:
                layer = int(raw_layer)
            except (TypeError, ValueError):
                continue
            normalized[layer] = profile
        if normalized:
            case_profiles.append(normalized)

    layers = sorted({layer for profiles in case_profiles for layer in profiles})
    if not layers:
        raise RendererUnavailableError(f"matrix-features has no usable {family} layer data")
    arrays = {feature: np.full((len(case_profiles), len(layers)), np.nan) for feature in features}
    for row, profiles in enumerate(case_profiles):
        for col, layer in enumerate(layers):
            profile = profiles.get(layer, {})
            for feature in features:
                try:
                    arrays[feature][row, col] = float(profile[feature])
                except (KeyError, TypeError, ValueError):
                    pass  # Missing values stay in their case's row as NaN.
    return MatrixFeatureTable(
        model=str(run.get("model", "model")),
        plan_id=str(run.get("plan_id", "plan")),
        edit_method=run.get("edit_method"),
        layers=np.asarray(layers, dtype=int),
        values=arrays,
        case_count=len(case_profiles),
    )


def matching_baseline(
    edited: dict[str, Any],
    captures: list[dict[str, Any]],
) -> dict[str, Any] | None:
    model, plan_id, _method = run_key(edited)
    candidates = [payload for payload in captures if run_key(payload) == (model, plan_id, None)]
    if not candidates:
        return None

    input_ids = _input_artifact_ids(edited)
    linked = [payload for payload in candidates if str(payload.get("artifact_id")) in input_ids]
    if len(linked) == 1:
        return linked[0]
    if len(linked) > 1:
        raise RendererUnavailableError("edited matrix-features capture references multiple matching baseline captures")

    edited_hash = edited.get("config_hash")
    if edited_hash:
        same_config = [payload for payload in candidates if payload.get("config_hash") == edited_hash]
        if len(same_config) == 1:
            return same_config[0]
        if len(same_config) > 1:
            raise RendererUnavailableError(
                "multiple baseline matrix-features captures share the edited config hash; "
                "cannot select a unique baseline"
            )

    if len(candidates) == 1 and not input_ids and not edited_hash:
        return candidates[0]

    raise RendererUnavailableError(
        "cannot match edited matrix-features capture to a unique baseline; "
        "expected an input artifact ref or a unique matching config_hash"
    )


def _input_artifact_ids(payload: dict[str, Any]) -> set[str]:
    refs = payload.get("inputs", ())
    if not isinstance(refs, list):
        return set()
    ids: set[str] = set()
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        artifact_id = ref.get("artifact_id")
        if artifact_id not in (None, ""):
            ids.add(str(artifact_id))
    return ids


__all__ = ["MatrixFeatureTable", "matching_baseline", "matrix_feature_table", "target_layer_for"]
