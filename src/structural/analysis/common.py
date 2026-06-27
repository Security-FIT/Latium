"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from src.structural.analysis.runtime import AnalysisContext, AnalysisUnavailableError


EPS = 1e-10


def map_cases(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(case.get("case_id")): case for case in cases if isinstance(case, dict) and case.get("case_id") is not None
    }


def execution_cases(context: AnalysisContext) -> dict[str, dict[str, Any]]:
    return map_cases(list(context.execution.get("cases", [])))


def capture_cases(
    context: AnalysisContext,
    capture_name: str,
) -> dict[str, dict[str, Any]]:
    return map_cases(context.captures.get(capture_name, []))


def eligible(
    execution_case: Optional[dict[str, Any]],
    capture_case: Optional[dict[str, Any]],
) -> bool:
    if not execution_case or not capture_case:
        return False
    if execution_case.get("status") != "complete":
        return False
    if capture_case.get("status", "complete") != "complete":
        return False
    edit = execution_case.get("edit", {})
    return bool(edit.get("success", True))


def required_capture_cases(
    context: AnalysisContext,
    execution_case: Optional[dict[str, Any]],
    case_id: str,
    capture_names: tuple[str, ...],
) -> tuple[Optional[dict[str, dict[str, Any]]], Optional[str]]:
    resolved: dict[str, dict[str, Any]] = {}
    for capture_name in capture_names:
        capture_case = capture_cases(context, capture_name).get(case_id)
        if not eligible(execution_case, capture_case):
            detail = capture_case.get("error") if capture_case else f"{capture_name} capture is missing"
            return None, str(detail or f"{capture_name} capture is unavailable")
        resolved[capture_name] = capture_case
    return resolved, None


def result_case(
    case_id: str,
    data: dict[str, Any],
    target_layer: Optional[int],
) -> dict[str, Any]:
    detected = data.get("anomalous_layer")
    return {
        "case_id": case_id,
        "status": "complete",
        "data": data,
        "error": None,
        "accuracy": {
            "target_layer": target_layer,
            "detected_layer": detected,
            "correct": (bool(detected == target_layer) if target_layer is not None and detected is not None else None),
        },
    }


def summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [case for case in cases if case.get("status") == "complete"]
    unavailable = [case for case in cases if case.get("status") == "unavailable"]
    errors = [case for case in cases if case.get("status") == "error"]
    evaluated = [case for case in complete if case.get("accuracy", {}).get("correct") is not None]
    correct = sum(bool(case["accuracy"]["correct"]) for case in evaluated)
    return {
        "cases_total": len(cases),
        "cases_complete": len(complete),
        "cases_unavailable": len(unavailable),
        "cases_error": len(errors),
        "cases_evaluated": len(evaluated),
        "correct": correct,
        "accuracy": correct / len(evaluated) if evaluated else 0.0,
    }


def run_case_analysis(
    context: AnalysisContext,
    capture_name: str,
    analyze: Callable[[dict[str, Any], str], dict[str, Any]],
) -> dict[str, Any]:
    execution = execution_cases(context)
    captures = capture_cases(context, capture_name)
    cases: list[dict[str, Any]] = []
    for case_id, execution_case in execution.items():
        capture_case = captures.get(case_id)
        if not eligible(execution_case, capture_case):
            cases.append(
                {
                    "case_id": case_id,
                    "status": "unavailable",
                    "data": {},
                    "error": "edit or capture unavailable",
                }
            )
            continue
        try:
            data = analyze(dict(capture_case.get("data", {})), case_id)
            cases.append(result_case(case_id, data, context.target_layer))
        except AnalysisUnavailableError as exc:
            cases.append(
                {
                    "case_id": case_id,
                    "status": "unavailable",
                    "data": {},
                    "error": str(exc),
                }
            )
        except Exception as exc:
            cases.append(
                {
                    "case_id": case_id,
                    "status": "error",
                    "data": {},
                    "error": str(exc),
                }
            )
    return {"cases": cases, "summary": summary(cases)}


def matrix_families(data: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    value = data.get("families")
    return value if isinstance(value, dict) else {}


def require_matrix_features(
    data: dict[str, Any],
    required_features: tuple[str, ...],
    *,
    family: str = "proj",
) -> dict[str, dict[str, float]]:
    families = matrix_families(data)
    profiles = families.get(family, {})
    if not profiles:
        raise AnalysisUnavailableError(f"matrix-features capture has no {family} profiles")
    available = set(data.get("features", ()))
    if not available:
        available = {name for profile in profiles.values() if isinstance(profile, dict) for name in profile}
    missing = [feature for feature in required_features if feature not in available]
    if missing:
        raise AnalysisUnavailableError(
            "matrix-features capture is missing required features "
            f"{', '.join(missing)}; recapture with a compatible feature_set is required"
        )
    incomplete_layers = [
        str(layer)
        for layer, profile in profiles.items()
        if not isinstance(profile, dict) or any(feature not in profile for feature in required_features)
    ]
    if incomplete_layers:
        preview = ", ".join(incomplete_layers[:8])
        suffix = "..." if len(incomplete_layers) > 8 else ""
        raise AnalysisUnavailableError(
            f"matrix-features capture is missing required features on layers {preview}{suffix}; recapture is required"
        )
    return profiles


def feature_z_scores(
    profiles: dict[str, dict[str, float]],
    names: tuple[str, ...],
) -> tuple[list[int], np.ndarray, dict[str, np.ndarray]]:
    layers = sorted((int(layer) for layer in profiles), key=int)
    arrays: dict[str, np.ndarray] = {}
    for name in names:
        values = np.array(
            [float(profiles[str(layer)].get(name, 0.0)) for layer in layers],
            dtype=np.float64,
        )
        arrays[name] = np.abs((values - values.mean()) / (values.std() + EPS))
    combined = np.mean(np.stack(list(arrays.values())), axis=0)
    return layers, combined, arrays
