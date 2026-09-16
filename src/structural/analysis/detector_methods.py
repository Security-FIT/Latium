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
    execution_cases,
    matrix_families,
    require_matrix_features,
    required_capture_cases,
    result_case,
    run_case_analysis,
    summary,
)
from src.structural.capture.matrix_features import (
    BLIND_FEATURES,
    EDIT_PRESENCE_FEATURES,
    PAPER_FEATURES,
    RANK1_FEATURES,
)
from src.structural.analysis.runtime import AnalysisContext, AnalysisUnavailableError
from src.structural.detectors.spectral import replay_spectral
from src.structural.detectors.bottom_rank_scoring import score_token_sweeps
from src.structural.detectors.local_scores import local_score_bank, rank01


def _required(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise KeyError(f"Missing analysis config key: {key}")
    return config[key]


def analyze_blind(context: AnalysisContext) -> dict[str, Any]:
    from src.structural.detectors.blind import detect_from_profiles

    def analyze(data: dict[str, Any], _: str) -> dict[str, Any]:
        profiles = require_matrix_features(data, BLIND_FEATURES)
        result = detect_from_profiles(profiles)
        result["detection_score"] = result["layer_anomaly_score"]
        return result

    return run_case_analysis(context, "matrix-features", analyze)


def analyze_ccs_composite(context: AnalysisContext) -> dict[str, Any]:
    from src.structural.detectors.composite import detect_layer

    execution = execution_cases(context)
    cases: list[dict[str, Any]] = []
    trim_first = int(_required(context.config, "trim_first"))
    trim_last = int(_required(context.config, "trim_last"))
    replay_config = {
        "top_k": int(_required(context.config, "top_k")),
        "trim_first": trim_first,
        "trim_last": trim_last,
        "neighbor_layers": int(_required(context.config, "neighbor_layers")),
        "rolling_window": int(_required(context.config, "rolling_window")),
        "local_windows": list(_required(context.config, "local_windows")),
        "boundary": int(_required(context.config, "boundary")),
    }
    for case_id, execution_case in execution.items():
        required, reason = required_capture_cases(
            context,
            execution_case,
            case_id,
            ("matrix-features", "spectral"),
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
        spectral_case = required["spectral"]
        try:
            profiles = require_matrix_features(matrix_case["data"], PAPER_FEATURES)
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
        try:
            spectral = replay_spectral(spectral_case["data"], replay_config)
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
        detected, method, info = detect_layer(
            {
                "blind_detection": {"layer_features": profiles},
                "spectral_detection": spectral,
            },
            trim_first=trim_first,
            trim_last=trim_last,
            small_window=int(_required(context.config, "small_window")),
            large_window=int(_required(context.config, "large_window")),
            te_window=int(_required(context.config, "te_window")),
            nc_window=int(_required(context.config, "nc_window")),
            feature_z_min=float(_required(context.config, "feature_z_min")),
            signal_a_confirm_z_min=float(_required(context.config, "signal_a_confirm_z_min")),
            signal_ab_boundary_width=int(_required(context.config, "signal_ab_boundary_width")),
            signal_ab_cluster_span=int(_required(context.config, "signal_ab_cluster_span")),
        )
        cases.append(
            result_case(
                case_id,
                {
                    "anomalous_layer": detected,
                    "method": method,
                    "signals": info,
                },
                context.target_layer,
            )
        )
    return {"cases": cases, "summary": summary(cases)}


def analyze_gpt_norm_cv(context: AnalysisContext) -> dict[str, Any]:
    from src.structural.detectors.gpt_norm_cv import detect

    trim_first = int(_required(context.config, "trim_first"))
    trim_last = int(_required(context.config, "trim_last"))

    def analyze(data: dict[str, Any], _: str) -> dict[str, Any]:
        profiles = require_matrix_features(data, ("norm_cv",))
        detected, method, info = detect(
            {"blind_detection": {"layer_features": profiles}},
            trim_first=trim_first,
            trim_last=trim_last,
        )
        return {
            "anomalous_layer": detected,
            "method": method,
            "signals": info,
        }

    return run_case_analysis(context, "matrix-features", analyze)


def _rank1_score(
    profiles: dict[str, dict[str, float]],
    trim_first: int,
    trim_last: int,
    local_windows: tuple[int, ...],
) -> dict[str, Any]:
    layers = sorted(int(layer) for layer in profiles)
    names = (
        "top1_energy",
        "top5_energy",
        "gap12",
        "effective_rank",
        "stable_rank",
        "rank1_residual",
    )
    arrays = [np.asarray([float(profiles[str(layer)].get(name, 0.0)) for layer in layers]) for name in names]
    arrays[3] *= -1
    arrays[4] *= -1
    arrays[5] *= -1
    raw = np.mean(np.stack([rank01(values) for values in arrays]), axis=0)
    local = local_score_bank(raw, windows=local_windows)["max_local_rank"]
    combined = 0.45 * raw + 0.55 * local
    candidates = np.arange(trim_first, len(layers) - trim_last)
    if candidates.size == 0:
        candidates = np.arange(len(layers))
    best = int(candidates[int(np.argmax(combined[candidates]))])
    return {
        "anomalous_layer": int(layers[best]),
        "detection_score": float(combined[best]),
        "raw_rank_score": {str(layer): float(raw[index]) for index, layer in enumerate(layers)},
        "combined_score": {str(layer): float(combined[index]) for index, layer in enumerate(layers)},
    }


def analyze_rank1(context: AnalysisContext) -> dict[str, Any]:
    trim_first = int(_required(context.config, "trim_first"))
    trim_last = int(_required(context.config, "trim_last"))
    local_windows = tuple(int(value) for value in _required(context.config, "local_windows"))
    return run_case_analysis(
        context,
        "matrix-features",
        lambda data, _: _rank1_score(
            require_matrix_features(data, RANK1_FEATURES),
            trim_first,
            trim_last,
            local_windows,
        ),
    )


def analyze_edit_presence(context: AnalysisContext) -> dict[str, Any]:
    from src.structural.detectors.edit_presence import detect_edit_presence_from_profiles

    local_windows = tuple(int(value) for value in _required(context.config, "local_windows"))
    detection_threshold = float(_required(context.config, "detection_threshold"))
    min_peak_robust_z = float(_required(context.config, "min_peak_robust_z"))
    min_margin = float(_required(context.config, "min_margin"))

    def analyze(data: dict[str, Any], _: str) -> dict[str, Any]:
        profiles = require_matrix_features(data, EDIT_PRESENCE_FEATURES)
        families = matrix_families(data)
        return detect_edit_presence_from_profiles(
            profiles,
            fc_metrics=families.get("fc"),
            detection_threshold=detection_threshold,
            min_peak_robust_z=min_peak_robust_z,
            min_margin=min_margin,
            local_windows=local_windows,
        )

    return run_case_analysis(context, "matrix-features", analyze)


def analyze_gram_localization(context: AnalysisContext) -> dict[str, Any]:
    from src.structural.detectors.rome_layer_localizer import detect_from_profiles

    def analyze(data: dict[str, Any], _: str) -> dict[str, Any]:
        result = detect_from_profiles(
            data.get("profiles", {}),
            layers=data.get("layers"),
            trim_fraction=float(data.get("trim_fraction", 0.10)),
        )
        localization = result["localization"]
        selected = localization["selected_layer"]
        result["anomalous_layer"] = selected
        result["detection_score"] = float(localization["layer_scores"][str(selected)])
        result["method"] = "diagonal-relative"
        return result

    return run_case_analysis(context, "gram-localization", analyze)


def _experiment_names(context: AnalysisContext) -> tuple[str, ...]:
    raw = _required(context.config, "experiments")
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise TypeError("experiments must be a list of identifiers")
    names = tuple(str(value) for value in raw)
    if not names:
        raise ValueError("experiments must not be empty")
    return names


def _add_experiment_summary(result: dict[str, Any], experiments: tuple[str, ...]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for identifier in experiments:
        complete = 0
        unavailable = 0
        positives = 0
        for case in result.get("cases", []):
            payload = case.get("data", {}).get("experiments", {}).get(identifier)
            if isinstance(payload, dict):
                complete += 1
                positives += int(payload.get("is_rome_like") is True)
            else:
                unavailable += 1
        summaries[identifier] = {
            "cases_complete": complete,
            "cases_unavailable": unavailable,
            "positive_decisions": positives,
        }
    result["summary"]["experiments"] = summaries
    return result


def analyze_rome_profile_experiments(context: AnalysisContext) -> dict[str, Any]:
    from src.structural.detectors.rome_layer_localizer import evaluate_profile_experiments

    experiments = _experiment_names(context)

    def analyze(data: dict[str, Any], _: str) -> dict[str, Any]:
        return {
            "method": "rome-profile-experiments",
            "experiments": evaluate_profile_experiments(
                data.get("profiles", {}),
                experiments=experiments,
                eligible_only=data.get("eligible_layers"),
            ),
        }

    return _add_experiment_summary(run_case_analysis(context, "gram-localization", analyze), experiments)


def analyze_rome_matrix_experiments(context: AnalysisContext) -> dict[str, Any]:
    from src.structural.detectors.rome_layer_localizer import evaluate_matrix_experiments

    experiments = _experiment_names(context)

    def analyze(data: dict[str, Any], _: str) -> dict[str, Any]:
        return {
            "method": "rome-matrix-experiments",
            "experiments": evaluate_matrix_experiments(data, experiments=experiments),
        }

    return _add_experiment_summary(run_case_analysis(context, "gram-experiments-v1", analyze), experiments)


def analyze_rome_directional_experiments(context: AnalysisContext) -> dict[str, Any]:
    from src.structural.detectors.rome_layer_localizer import evaluate_matrix_experiments

    experiments = _experiment_names(context)

    def analyze(data: dict[str, Any], _: str) -> dict[str, Any]:
        return {
            "method": "rome-directional-experiments",
            "experiments": evaluate_matrix_experiments(data, experiments=experiments),
        }

    result = run_case_analysis(context, "gram-directional-error-v1", analyze)
    return _add_experiment_summary(result, experiments)


def analyze_rome_control_experiment(context: AnalysisContext) -> dict[str, Any]:
    from src.structural.detectors.rome_layer_localizer import control_profile_mdl

    experiments = _experiment_names(context)
    unsupported = [name for name in experiments if name != "control-affine-mdl-v1"]
    if unsupported:
        raise ValueError(f"Unknown ROME control experiments: {', '.join(unsupported)}")
    execution = execution_cases(context)
    cases: list[dict[str, Any]] = []
    for case_id, execution_case in execution.items():
        required, reason = required_capture_cases(
            context,
            execution_case,
            case_id,
            ("gram-experiments-v1", "gram-control-v1"),
        )
        if required is None:
            cases.append({"case_id": case_id, "status": "unavailable", "data": {}, "error": reason})
            continue
        try:
            target_profiles = required["gram-experiments-v1"].get("data", {}).get("profiles", {})
            control_profiles = required["gram-control-v1"].get("data", {}).get("profiles", {})
            target = {
                str(layer): float(profile["original_score"])
                for layer, profile in target_profiles.items()
                if "original_score" in profile
            }
            control = {
                str(layer): float(profile["diagonal_relative"])
                for layer, profile in control_profiles.items()
                if "diagonal_relative" in profile
            }
            payload = {"control-affine-mdl-v1": control_profile_mdl(target, control)}
            cases.append(result_case(case_id, {"method": "rome-control-experiment", "experiments": payload}, None))
        except (ValueError, KeyError, TypeError) as exc:
            cases.append({"case_id": case_id, "status": "unavailable", "data": {}, "error": str(exc)})
    return _add_experiment_summary({"cases": cases, "summary": summary(cases)}, experiments)


def analyze_bottom_rank(context: AnalysisContext) -> dict[str, Any]:
    trim_first = int(_required(context.config, "trim_first"))
    trim_last = int(_required(context.config, "trim_last"))

    def analyze(data: dict[str, Any], _: str) -> dict[str, Any]:
        scored = score_token_sweeps(
            data.get("token_id_sweeps", {}),
            trim_first=trim_first,
            trim_last=trim_last,
        )
        return {
            "anomalous_layer": scored["anomalous_layer"],
            "detection_score": scored["detection_score"],
            "layer_scores": {str(layer): score for layer, score in scored["layer_scores"].items()},
            **data,
        }

    return run_case_analysis(context, "bottom-rank-tokens", analyze)
