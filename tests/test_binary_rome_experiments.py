import math

import pytest

import torch

from src.structural.detectors.rome_layer_localizer import (
    capture_directional_error_weights,
    factor_cross_layer_kernel,
    hidden_gram,
    lof_scores_from_kernel,
    neighbor_experiment_measurements,
    relative_profile_decision,
    robust_decomposition_scores,
    token_subspace_alignment_profiles,
)
from src.structural.capture.artifacts import capture_case
from src.structural.capture.producers import CaptureContext, capture_gram_cross_layer


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


def test_cross_layer_capture_matches_explicit_gram_inner_products() -> None:
    weights = {
        layer: torch.arange(1, 13, dtype=torch.float64).reshape(3, 4) + layer
        for layer in range(7)
    }
    context = CaptureContext(
        proj_weights=weights,
        fc_weights=None,
        attention_weights={},
        probe_vector=None,
        token_predictor=None,
        changed_weights={},
        options={"gram_cross_layer_block_size": 2},
    )

    capture = capture_gram_cross_layer(context)
    grams = torch.stack([hidden_gram(weights[layer]) for layer in sorted(weights)])
    expected = torch.einsum("aij,bij->ab", grams, grams)

    assert capture["capture_version"] == "gram-cross-layer-v1"
    assert torch.tensor(capture["kernel"], dtype=torch.float64) == pytest.approx(expected)
    assert capture["runtime"]["block_size"] == 2
    assert capture["runtime"]["temporary_storage_bytes"] == grams.numel() * 8


def test_lof_handles_ties_identical_graphs_and_isolated_points() -> None:
    identical = torch.ones((7, 7), dtype=torch.float64)
    scores, diagnostics = lof_scores_from_kernel(identical)
    assert scores.tolist() == pytest.approx([1.0] * 7)
    assert diagnostics["identical_graph"] is True

    points = torch.tensor([[0.0], [0.0], [0.1], [0.2], [0.3], [0.4], [4.0]], dtype=torch.float64)
    kernel = points @ points.T
    scores, diagnostics = lof_scores_from_kernel(kernel)
    assert int(torch.argmax(scores).item()) == 6
    assert diagnostics["tie_expanded_neighbor_counts"][0] >= 5


def test_lof_rejects_unresolved_duplicate_subclusters() -> None:
    points = torch.tensor([[0.0]] * 6 + [[1.0], [2.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="duplicate subcluster"):
        lof_scores_from_kernel(points @ points.T)


def test_compressed_decomposition_matches_explicit_observations() -> None:
    base = torch.tensor([[1.0, 0.2, -0.1], [0.5, 0.1, -0.05]], dtype=torch.float64)
    coefficients = torch.linspace(0.5, 2.0, 10, dtype=torch.float64)[:, None]
    observations = coefficients @ base[:1]
    observations[6] += torch.tensor([4.0, -3.0, 2.0], dtype=torch.float64)

    explicit, explicit_diagnostics = robust_decomposition_scores(observations)
    factor, _ = factor_cross_layer_kernel(observations @ observations.T)
    compressed, compressed_diagnostics = robust_decomposition_scores(factor)

    assert explicit == pytest.approx(compressed, rel=1e-5, abs=1e-6)
    assert explicit_diagnostics["converged"] is True
    assert compressed_diagnostics["converged"] is True
    assert int(torch.argmax(compressed).item()) == 6


def test_decomposition_reports_convergence_and_trivial_failures() -> None:
    data = torch.arange(35, dtype=torch.float64).reshape(7, 5)
    with pytest.raises(ValueError, match="did not converge"):
        robust_decomposition_scores(data, max_iterations=1, tolerance=1e-15)
    with pytest.raises(ValueError, match="trivial"):
        robust_decomposition_scores(torch.zeros((7, 3), dtype=torch.float64))


def _alignment_fixture() -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    base = torch.randn((4, 6), generator=generator, dtype=torch.float64)
    weights = {layer: base + 0.01 * layer for layer in range(6)}
    head = torch.randn((11, 4), generator=generator, dtype=torch.float64)
    return weights, head


def test_token_alignment_bounds_orientation_and_batch_agreement() -> None:
    weights, head = _alignment_fixture()
    linear = token_subspace_alignment_profiles(
        weights,
        head,
        projection_layout="linear-output-input",
        output_head_layout="linear-output-input",
        vocabulary_batch_size=3,
    )
    conv = token_subspace_alignment_profiles(
        {layer: weight.T for layer, weight in weights.items()},
        head.T,
        projection_layout="conv1d-input-output",
        output_head_layout="conv1d-input-output",
        vocabulary_batch_size=100,
    )

    for layer in weights:
        left = linear["profiles"][str(layer)]
        right = conv["profiles"][str(layer)]
        assert 0.0 <= left["alignment_score"] <= 1.0
        assert left["alignment_score"] == pytest.approx(right["alignment_score"])
        assert left["maximizing_token_id"] == right["maximizing_token_id"]


def test_token_alignment_is_invariant_to_input_rotation() -> None:
    weights, head = _alignment_fixture()
    rotation, _ = torch.linalg.qr(torch.randn((6, 6), generator=torch.Generator().manual_seed(9), dtype=torch.float64))
    rotated = {layer: weight @ rotation for layer, weight in weights.items()}
    first = token_subspace_alignment_profiles(
        weights,
        head,
        projection_layout="linear-output-input",
        output_head_layout="linear-output-input",
    )
    second = token_subspace_alignment_profiles(
        rotated,
        head,
        projection_layout="linear-output-input",
        output_head_layout="linear-output-input",
    )

    assert [first["profiles"][str(layer)]["alignment_score"] for layer in weights] == pytest.approx(
        [second["profiles"][str(layer)]["alignment_score"] for layer in weights]
    )


def test_token_alignment_rejects_incompatible_heads_and_capture_marks_missing_access_unavailable() -> None:
    weights, _head = _alignment_fixture()
    with pytest.raises(ValueError, match="does not match head"):
        token_subspace_alignment_profiles(
            weights,
            torch.randn((10, 5), dtype=torch.float64),
            projection_layout="linear-output-input",
            output_head_layout="linear-output-input",
        )
    context = CaptureContext(
        proj_weights=weights,
        fc_weights=None,
        attention_weights={},
        probe_vector=None,
        token_predictor=None,
        changed_weights={},
        options={},
    )
    result = capture_case("token-subspace-alignment-v1", context, case_id="clean")
    assert result["status"] == "unavailable"
    assert "output-head" in result["error"]
