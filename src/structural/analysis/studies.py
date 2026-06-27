"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.structural.analysis.common import (
    feature_z_scores,
    execution_cases,
    matrix_families,
    require_matrix_features,
    required_capture_cases,
    result_case,
    run_case_analysis,
    summary,
)
from src.structural.analysis.runtime import AnalysisContext, AnalysisUnavailableError
from src.structural.capture.matrix_features import PAPER_FEATURES


EPS = 1e-10


def analyze_ipr(context: AnalysisContext) -> dict[str, Any]:
    def analyze(data: dict[str, Any], _: str) -> dict[str, Any]:
        require_matrix_features(data, ("global_ipr", "row_ipr_mean", "row_ipr_std"))
        families = matrix_families(data)
        proj = families.get("proj", {})
        layers, scores, arrays = feature_z_scores(
            proj,
            ("global_ipr", "row_ipr_mean", "row_ipr_std"),
        )
        best = int(np.argmax(scores)) if scores.size else 0
        return {
            "anomalous_layer": int(layers[best]) if layers else None,
            "detection_score": float(scores[best]) if scores.size else 0.0,
            "families": families,
            "ipr_z_scores": {
                name: {str(layer): float(values[index]) for index, layer in enumerate(layers)}
                for name, values in arrays.items()
            },
        }

    return run_case_analysis(context, "matrix-features", analyze)


def analyze_symmetry(context: AnalysisContext) -> dict[str, Any]:
    def analyze(data: dict[str, Any], _: str) -> dict[str, Any]:
        profiles = require_matrix_features(data, ("top1_energy", "effective_rank", "stable_rank"))
        layers = sorted(int(layer) for layer in profiles)
        scores: dict[int, float] = {}
        for index, layer in enumerate(layers):
            mirror = layers[len(layers) - 1 - index]
            current = profiles[str(layer)]
            other = profiles[str(mirror)]
            scores[layer] = float(
                np.mean(
                    [
                        abs(float(current.get("top1_energy", 0.0)) - float(other.get("top1_energy", 0.0))),
                        abs(float(current.get("effective_rank", 0.0)) - float(other.get("effective_rank", 0.0))),
                        abs(float(current.get("stable_rank", 0.0)) - float(other.get("stable_rank", 0.0))),
                    ]
                )
            )
        best = max(layers, key=scores.get) if layers else None
        return {
            "anomalous_layer": best,
            "detection_score": scores.get(best, 0.0),
            "mirror_break_scores": {str(layer): value for layer, value in scores.items()},
        }

    return run_case_analysis(context, "matrix-features", analyze)


def analyze_interlayer(context: AnalysisContext) -> dict[str, Any]:
    def analyze(data: dict[str, Any], _: str) -> dict[str, Any]:
        profiles = require_matrix_features(data, ("top1_energy", "spectral_gap", "effective_rank", "norm_cv"))
        layers = sorted(int(layer) for layer in profiles)
        transitions: dict[str, float] = {}
        metrics = ("top1_energy", "spectral_gap", "effective_rank", "norm_cv")
        for previous, current in zip(layers, layers[1:]):
            left = profiles[str(previous)]
            right = profiles[str(current)]
            delta = np.asarray([float(right.get(metric, 0.0)) - float(left.get(metric, 0.0)) for metric in metrics])
            transitions[f"{previous}->{current}"] = float(np.linalg.norm(delta))
        best_edge = max(transitions, key=transitions.get) if transitions else None
        best_layer = int(best_edge.split("->")[1]) if best_edge else None
        return {
            "anomalous_layer": best_layer,
            "detection_score": transitions.get(best_edge, 0.0),
            "transitions": transitions,
        }

    return run_case_analysis(context, "matrix-features", analyze)


def analyze_attention(context: AnalysisContext) -> dict[str, Any]:
    execution = execution_cases(context)
    cases: list[dict[str, Any]] = []
    for case_id, execution_case in execution.items():
        required, reason = required_capture_cases(
            context,
            execution_case,
            case_id,
            ("matrix-features", "attention-features"),
        )
        if required is None:
            cases.append(
                {
                    "case_id": case_id,
                    "status": "unavailable",
                    "data": {},
                    "error": reason,
                }
            )
            continue
        matrix_case = required["matrix-features"]
        attention_case = required["attention-features"]
        try:
            proj = require_matrix_features(matrix_case["data"], PAPER_FEATURES)
        except AnalysisUnavailableError as exc:
            cases.append(
                {
                    "case_id": case_id,
                    "status": "unavailable",
                    "data": {},
                    "error": str(exc),
                }
            )
            continue
        attention = matrix_families(attention_case["data"])
        layers = sorted(int(layer) for layer in proj)
        scores: dict[int, float] = {}
        for layer in layers:
            proj_profile = proj[str(layer)]
            contrasts = []
            for family in attention.values():
                profile = family.get(str(layer), {})
                if profile:
                    contrasts.append(
                        abs(float(proj_profile.get("top1_energy", 0.0)) - float(profile.get("top1_energy", 0.0)))
                    )
            scores[layer] = float(np.mean(contrasts)) if contrasts else 0.0
        best = max(layers, key=scores.get) if layers else None
        cases.append(
            result_case(
                case_id,
                {
                    "anomalous_layer": best,
                    "detection_score": scores.get(best, 0.0),
                    "contrast_scores": {str(layer): value for layer, value in scores.items()},
                },
                context.target_layer,
            )
        )
    return {"cases": cases, "summary": summary(cases)}


def analyze_matrix_anomaly(context: AnalysisContext) -> dict[str, Any]:
    def analyze(data: dict[str, Any], _: str) -> dict[str, Any]:
        families = matrix_families(data)
        proj = families.get("proj", {})
        layers = sorted(int(layer) for layer in proj)
        metric_names = sorted({name for profile in proj.values() for name in profile})
        score_arrays = []
        for name in metric_names:
            values = np.asarray([float(proj[str(layer)].get(name, 0.0)) for layer in layers])
            score_arrays.append(np.abs((values - values.mean()) / (values.std() + EPS)))
        scores = np.mean(np.stack(score_arrays), axis=0) if score_arrays else np.zeros(len(layers))
        best = int(np.argmax(scores)) if scores.size else 0
        return {
            "anomalous_layer": int(layers[best]) if layers else None,
            "detection_score": float(scores[best]) if scores.size else 0.0,
            "families": families,
            "combined_scores": {str(layer): float(scores[index]) for index, layer in enumerate(layers)},
        }

    return run_case_analysis(context, "matrix-anomaly-features", analyze)
