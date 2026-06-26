"""
Shared spectral detector scoring and artifact replay.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.structural.detectors.local_scores import local_score_bank
from src.structural.detectors.spectral_primitives import (
    PCS_CROSS_NAMES,
    PCS_NAMES,
    hybrid_scores,
    pcs_cross_signals_from_rank_cumsums,
    pcs_signals_from_pairwise_cumsums,
    sv_ratio_energy,
    sv_z_energy,
)


def _required(config: Mapping[str, Any], key: str) -> Any:
    if key not in config:
        raise KeyError(f"Missing spectral analysis config key: {key}")
    return config[key]


def _analysis_unavailable(message: str) -> Exception:
    from src.structural.analysis.runtime import AnalysisUnavailableError

    return AnalysisUnavailableError(message)


def _score_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "top_k": int(_required(config, "top_k")),
        "rolling_window": max(1, int(_required(config, "rolling_window"))),
        "local_windows": tuple(int(value) for value in _required(config, "local_windows")),
        "boundary": max(0, int(_required(config, "boundary"))),
    }


def map_scores_to_layers(
    all_layers: Sequence[int],
    evaluated_layers: Sequence[int],
    values: np.ndarray,
    *,
    string_keys: bool = False,
) -> dict[Any, float]:
    result: dict[Any, float] = {str(layer) if string_keys else int(layer): 0.0 for layer in all_layers}
    for index, layer in enumerate(evaluated_layers):
        key = str(layer) if string_keys else int(layer)
        result[key] = float(values[index])
    return result


def select_best_layer(
    evaluated_layers: Sequence[int],
    scores: np.ndarray,
    boundary: int,
) -> tuple[int, int | None, float]:
    n_layers = len(evaluated_layers)
    if n_layers == 0 or scores.size == 0:
        return 0, None, 0.0
    width = max(0, int(boundary))
    candidates = np.arange(width, n_layers - width)
    if candidates.size == 0:
        candidates = np.arange(n_layers)
    best_index = int(candidates[int(np.argmax(scores[candidates]))])
    return best_index, int(evaluated_layers[best_index]), float(scores[best_index])


def empty_spectral_result(
    all_layers: Sequence[int],
    excluded_layers: Sequence[int],
    evaluated_layers: Sequence[int],
    config: Mapping[str, Any],
    *,
    string_keys: bool = False,
) -> dict[str, Any]:
    z = {str(layer) if string_keys else int(layer): 0.0 for layer in all_layers}
    return {
        "anomalous_layer": None,
        "detection_score": 0.0,
        "sv_z_scores": dict(z),
        "sv_ratio_scores": dict(z),
        "sv_z_rolling_z_scores": dict(z),
        "sv_ratio_rolling_z_scores": dict(z),
        "pcs_composite_rank_scores": dict(z),
        "sv_pcs_contradiction_scores": dict(z),
        "rome_hybrid_scores": dict(z),
        **{name: dict(z) for name in PCS_NAMES},
        **{name: dict(z) for name in PCS_CROSS_NAMES},
        "local_window_scores": {},
        "has_fc_weights": False,
        "config": dict(config),
        "excluded_layers": list(excluded_layers),
        "evaluated_layers": list(evaluated_layers),
    }


def _series(
    values: Mapping[str, np.ndarray],
    names: Sequence[str],
    n: int,
) -> dict[str, np.ndarray]:
    zeros = np.zeros(n, dtype=np.float64)
    return {name: np.asarray(values.get(name, zeros), dtype=np.float64) for name in names}


def score_spectral_inputs(
    *,
    all_layers: Sequence[int],
    evaluated_layers: Sequence[int],
    excluded_layers: Sequence[int],
    sv: np.ndarray,
    sv_fc: np.ndarray | None,
    pcs: Mapping[str, np.ndarray],
    pcs_cross: Mapping[str, np.ndarray],
    has_fc: bool,
    config: Mapping[str, Any],
    result_config: Mapping[str, Any] | None = None,
    string_keys: bool = False,
    pairwise_pcs: np.ndarray | None = None,
    include_pairwise_pcs: bool = False,
    emit_local_window_scores: bool = True,
) -> dict[str, Any]:
    scoring = _score_config(config)
    top_k = scoring["top_k"]
    sv = np.asarray(sv, dtype=np.float64)
    sv_z = sv_z_energy(sv, top_k)
    sv_fc_array = np.asarray(sv_fc if sv_fc is not None else np.empty((0, 0)), dtype=np.float64)
    sv_ratio = sv_ratio_energy(sv, sv_fc_array, top_k) if has_fc else np.zeros_like(sv_z)
    if sv_ratio.size != sv_z.size:
        sv_ratio = np.zeros_like(sv_z)

    pcs_values = _series(pcs, PCS_NAMES, len(sv_z))
    pcs_cross_values = _series(pcs_cross, PCS_CROSS_NAMES, len(sv_z))
    hybrid = hybrid_scores(
        sv_z,
        sv_ratio,
        pcs_values,
        pcs_cross_values,
        has_fc,
        scoring["rolling_window"],
    )
    scores = hybrid["rome_hybrid_scores"]
    _best_index, anomalous_layer, detection_score = select_best_layer(
        evaluated_layers,
        scores,
        scoring["boundary"],
    )

    def layer_map(values: np.ndarray) -> dict[Any, float]:
        return map_scores_to_layers(
            all_layers,
            evaluated_layers,
            values,
            string_keys=string_keys,
        )

    result: dict[str, Any] = {
        "anomalous_layer": anomalous_layer,
        "detection_score": detection_score,
        "sv_z_scores": layer_map(sv_z),
        "sv_ratio_scores": layer_map(sv_ratio),
        "has_fc_weights": bool(has_fc),
        "evaluated_layers": list(evaluated_layers),
        "excluded_layers": list(excluded_layers),
        "config": dict(result_config if result_config is not None else config),
    }
    if include_pairwise_pcs and pairwise_pcs is not None:
        result["pairwise_pcs"] = pairwise_pcs.tolist()

    for name, values in {**pcs_values, **pcs_cross_values, **hybrid}.items():
        result[name] = layer_map(values)

    if emit_local_window_scores:
        local_series = {
            "sv_z_scores": sv_z,
            "sv_ratio_scores": sv_ratio,
            "rome_hybrid_scores": scores,
            "pcs_next_jump_scores": pcs_values["pcs_next_jump_scores"],
            "pcs_neighbor_var_scores": pcs_values["pcs_neighbor_var_scores"],
            "pcs_next_curvature_scores": pcs_values["pcs_next_curvature_scores"],
        }
        result["local_window_scores"] = {
            name: {
                score_name: layer_map(score_values)
                for score_name, score_values in local_score_bank(
                    values,
                    windows=scoring["local_windows"],
                ).items()
            }
            for name, values in local_series.items()
        }
    else:
        result["local_window_scores"] = {}

    return result


def _layer_values(layer_map: Mapping[Any, Any], layers: Sequence[int]) -> list[Any]:
    return [layer_map[str(layer)] if str(layer) in layer_map else layer_map[int(layer)] for layer in layers]


def replay_spectral(
    data: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    layers = [int(layer) for layer in data.get("layers", [])]
    top_k = int(_required(config, "top_k"))
    stored_top_k = int(data.get("stored_top_k", 0))
    if top_k > stored_top_k:
        raise _analysis_unavailable(
            f"spectral top_k={top_k} exceeds captured top_k={stored_top_k}; recapture is required"
        )

    trim_first = max(0, int(_required(config, "trim_first")))
    trim_last = max(0, int(_required(config, "trim_last")))
    neighbor_layers = max(1, int(_required(config, "neighbor_layers")))

    sv_proj_map = data.get("sv_proj_topk", {})
    sv_proj = np.asarray(_layer_values(sv_proj_map, layers), dtype=np.float64)
    sv_fc_map = data.get("sv_fc_topk", {})
    has_fc = bool(sv_fc_map) and all(str(layer) in sv_fc_map or int(layer) in sv_fc_map for layer in layers)
    sv_fc = np.asarray(_layer_values(sv_fc_map, layers), dtype=np.float64) if has_fc else np.empty((0, 0))

    start = min(trim_first, len(layers))
    end = len(layers) - min(trim_last, len(layers) - start)
    evaluated_layers = layers[start:end]
    excluded_layers = layers[:start] + layers[end:]

    pcs, pairwise = pcs_signals_from_pairwise_cumsums(
        np.asarray(data.get("pcs_pairwise_dot_weight_cumsum", []), dtype=np.float64),
        np.asarray(data.get("pcs_flip_pairwise_weight_cumsum", []), dtype=np.float64),
        np.asarray(data.get("pcs_pairwise_weight_cumsum", []), dtype=np.float64),
        top_k=top_k,
        start=start,
        end=end,
        neighbor_layers=neighbor_layers,
    )
    pcs_cross = pcs_cross_signals_from_rank_cumsums(
        data.get("pcs_cross_dot_weight_cumsum", {}),
        data.get("pcs_cross_weight_cumsum", {}),
        layers,
        top_k=top_k,
        start=start,
        end=end,
    )

    return score_spectral_inputs(
        all_layers=layers,
        evaluated_layers=evaluated_layers,
        excluded_layers=excluded_layers,
        sv=sv_proj[start:end],
        sv_fc=sv_fc[start:end] if has_fc else sv_fc,
        pcs=pcs,
        pcs_cross=pcs_cross,
        has_fc=has_fc,
        config=config,
        string_keys=True,
        pairwise_pcs=pairwise,
        include_pairwise_pcs=True,
    )


def analyze_spectral(context: Any) -> dict[str, Any]:
    from src.structural.analysis.common import run_case_analysis

    return run_case_analysis(
        context,
        "spectral",
        lambda data, _: replay_spectral(data, context.config),
    )
