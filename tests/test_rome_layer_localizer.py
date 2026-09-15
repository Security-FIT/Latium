import pytest
import torch

from src.structural.detectors.rome_layer_localizer import (
    RomeLayerLocalizer,
    capture_experiment_weights,
    control_profile_mdl,
    detect_from_profiles,
    eligible_layers,
    evaluate_matrix_experiments,
    evaluate_profile_experiments,
    hidden_gram,
    neighbor_experiment_measurements,
    quadratic_experiment_measurements,
    score_layer,
)


pytestmark = pytest.mark.unit


def test_eligible_layers_uses_fractional_trim_with_neighbor_guard() -> None:
    assert eligible_layers(list(range(10))) == list(range(1, 9))
    assert eligible_layers(list(range(20))) == list(range(2, 18))


def test_profile_localization_is_deterministic_and_prefers_lower_tie() -> None:
    profiles = {
        str(layer): {"diagonal_relative": score}
        for layer, score in enumerate([0.0, 1.0, 7.0, 7.0, 2.0, 0.0])
    }
    result = detect_from_profiles(profiles, layers=list(range(6)))
    assert result["localization"]["selected_layer"] == 2
    assert result["localization"]["margin"] == 0.0


def test_profile_localization_rejects_incomplete_scores() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        detect_from_profiles({"1": {"diagonal_relative": 1.0}}, layers=list(range(5)))


def test_hidden_gram_is_orientation_invariant_and_normalized() -> None:
    weight = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    direct = hidden_gram(weight)
    transposed = hidden_gram(weight.T)
    assert torch.allclose(direct, transposed)
    assert torch.isclose(torch.trace(direct), torch.tensor(1.0))


def test_direct_localizer_finds_single_layer_perturbation() -> None:
    weights = {layer: torch.eye(6) for layer in range(7)}
    weights[3] = torch.diag(torch.tensor([4.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

    result = RomeLayerLocalizer().localize(weights)

    assert result["localization"]["selected_layer"] == 3
    assert result["localization"]["margin"] > 0


def test_neighbor_agreement_distinguishes_isolated_change_from_step() -> None:
    background = torch.eye(4) / 4.0
    change = torch.diag(torch.tensor([0.03, -0.01, -0.01, -0.01]))

    isolated = neighbor_experiment_measurements(
        background, background + change, background, layer=3
    )
    step = neighbor_experiment_measurements(
        background, background, background + change, layer=3
    )

    assert isolated["neighbor_agreement"] == pytest.approx(1.0, abs=1e-5)
    assert step["neighbor_agreement"] == pytest.approx(0.0, abs=1e-6)
    assert 0.0 <= isolated["bounded_contrast_score"] <= 2.0 * (2.0 ** 0.5)


def test_quadratic_predictor_removes_exact_quadratic_background() -> None:
    center = torch.eye(4) / 4.0
    curvature = torch.diag(torch.tensor([0.003, -0.001, -0.001, -0.001]))
    result = quadratic_experiment_measurements(
        center + 4 * curvature,
        center + curvature,
        center,
        center + curvature,
        center + 4 * curvature,
        layer=4,
    )
    assert result["quadratic_residual_frobenius"] == pytest.approx(0.0, abs=1e-7)
    assert result["quadratic_score"] == pytest.approx(0.0, abs=1e-6)


def test_experiment_capture_preserves_baseline_and_independent_variants() -> None:
    weights = {layer: torch.eye(6) for layer in range(9)}
    weights[4] = torch.diag(torch.tensor([4.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    capture = capture_experiment_weights(
        weights,
        groups=("neighbors", "quadratic", "footprint"),
        trim_fraction=0.0,
    )
    grams = {layer: hidden_gram(weight) for layer, weight in weights.items()}
    reference = (grams[3] + grams[5]) * 0.5

    assert capture["profiles"]["4"]["original_score"] == pytest.approx(
        score_layer(grams[4], reference, layer=4), rel=1e-6
    )
    results = evaluate_matrix_experiments(
        capture,
        experiments=(
            "baseline-affine-mdl-v1",
            "agreement-affine-mdl-v1",
            "quadratic-neighbor-affine-mdl-v1",
            "signed-footprint-mdl-v1",
        ),
    )
    assert all(result["candidate_layer"] == 4 for result in results.values())
    assert all(result["is_rome_like"] is True for result in results.values())


def test_profile_and_control_models_find_positive_excursion() -> None:
    profiles = {
        str(layer): {"diagonal_relative": 5.0 if layer == 4 else 0.1}
        for layer in range(1, 8)
    }
    results = evaluate_profile_experiments(
        profiles,
        experiments=("affine-mdl-v1", "quadratic-mdl-v1", "affine-step-mdl-v1"),
    )
    assert all(result["candidate_layer"] == 4 for result in results.values())
    control = control_profile_mdl(
        {layer: profile["diagonal_relative"] for layer, profile in profiles.items()},
        {str(layer): 0.1 for layer in range(1, 8)},
    )
    assert control["candidate_layer"] == 4
    assert control["is_rome_like"] is True
