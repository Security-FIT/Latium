import math

import pytest

import torch

from src.structural.detectors.rome_layer_localizer import (
    capture_directional_error_weights,
    hidden_gram,
    neighbor_experiment_measurements,
    relative_profile_decision,
)


pytestmark = pytest.mark.unit


def _scores(values: list[float], layers: list[int] | None = None) -> dict[str, float]:
    coordinates = list(range(len(values))) if layers is None else layers
    return {str(layer): math.expm1(value) for layer, value in zip(coordinates, values)}


def test_b0_rejects_smooth_profiles_and_uses_actual_coordinates() -> None:
    layers = [1, 2, 4, 7, 11, 16, 22, 29]
    values = [0.2 + 0.7 * ((layer - 1) / 28.0) for layer in layers]

    result = relative_profile_decision(_scores(values, layers), eligible_layers=layers)

    assert result["status"] == "complete"
    assert result["rome_compatible_detected"] is False
    assert result["diagnostics"]["actual_layer_coordinates"] == layers
    assert result["diagnostics"]["scaled_layer_coordinates"][3] == pytest.approx(6.0 / 28.0)
    assert result["diagnostics"]["family_search_penalty"] == pytest.approx(2.0 * math.log(3.0))


def test_b0_detects_jointly_fitted_positive_excursion() -> None:
    values = [0.1 + 0.03 * layer for layer in range(12)]
    values[7] += 2.5

    result = relative_profile_decision(_scores(values))

    assert result["status"] == "complete"
    assert result["rome_compatible_detected"] is True
    assert result["candidate_layer"] == 7
    assert result["gain"] > 0.0
    assert result["diagnostics"]["selected_anomaly"]["amplitude"] > 0.0


def test_b0_background_can_explain_a_stage_transition() -> None:
    values = [0.1 + 0.02 * layer + (0.8 if layer >= 6 else 0.0) for layer in range(12)]

    result = relative_profile_decision(_scores(values))

    assert result["status"] == "complete"
    assert result["rome_compatible_detected"] is False
    assert result["diagnostics"]["selected_background"]["family"] == "affine-plus-step"
    assert result["diagnostics"]["step_split_count"] == 9


def test_b0_preserves_unavailable_diagnostics_for_incomplete_or_constant_profiles() -> None:
    incomplete = relative_profile_decision(
        _scores([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        eligible_layers=list(range(7)),
    )
    constant = relative_profile_decision(_scores([0.4] * 8))

    assert incomplete["status"] == "unavailable"
    assert incomplete["rome_compatible_detected"] is None
    assert "missing" in incomplete["diagnostics"]["reason"]
    assert constant["status"] == "unavailable"
    assert "constant" in constant["diagnostics"]["reason"]


def test_b0_verdict_is_independent_of_original_localizer_metadata() -> None:
    values = [0.1] * 10
    values[4] = 1.8
    first = relative_profile_decision(_scores(values), original_localizer_layer=2)
    second = relative_profile_decision(_scores(values), original_localizer_layer=8)

    assert first["rome_compatible_detected"] == second["rome_compatible_detected"]
    assert first["candidate_layer"] == second["candidate_layer"]
    assert first["background_cost"] == pytest.approx(second["background_cost"])
    assert first["anomaly_cost"] == pytest.approx(second["anomaly_cost"])


def test_directional_capture_excludes_candidate_neighbors_and_preserves_v0() -> None:
    weights = {layer: torch.eye(8) * (1.0 + 0.01 * layer) for layer in range(16)}
    weights[8] = weights[8].clone()
    weights[8][0, 0] += 1.0

    capture = capture_directional_error_weights(weights)
    profile = capture["profiles"]["8"]
    grams = {layer: hidden_gram(weight) for layer, weight in weights.items()}
    direct = neighbor_experiment_measurements(grams[7], grams[8], grams[9], layer=8)

    assert capture["capture_version"] == "gram-directional-error-v1"
    assert len(profile["reference_layers"]) == 6
    assert 8 not in profile["reference_layers"]
    assert 7 not in profile["reference_layers"]
    assert 9 not in profile["reference_layers"]
    assert profile["original_score"] == pytest.approx(direct["original_score"], rel=1e-5)
    assert len(profile["reference_projections"]) == 6


def test_unresolved_directional_standardization_does_not_discard_v0() -> None:
    weights = {layer: torch.eye(6) for layer in range(16)}
    weights[8] = torch.diag(torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

    profile = capture_directional_error_weights(weights)["profiles"]["8"]

    assert profile["v0_status"] == "ok"
    assert profile["original_score"] is not None
    assert profile["v2_status"] == "unavailable"
    assert profile["standardized_directional_score"] is None
