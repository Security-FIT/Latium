import pytest
import torch

from src.structural.detectors.rome_layer_localizer import (
    RomeLayerLocalizer,
    detect_from_profiles,
    eligible_layers,
    hidden_gram,
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
