from __future__ import annotations

import torch

from src.structural.detectors.rome_presence import (
    ATTRIBUTION_SCOPE,
    detect_rome_compatible_edit,
    gram_delta_evidence,
)
from src.structural.detectors.rome_presence_resident import RomeDetector


def _weights() -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    return {
        layer: torch.randn(8, 12, generator=generator)
        for layer in range(12)
    }


def test_b0_accepts_a_synthetic_rank_one_rome_style_update() -> None:
    clean = _weights()
    suspect = dict(clean)
    generator = torch.Generator().manual_seed(23)
    suspect[5] = clean[5] + (
        torch.randn(8, 1, generator=generator)
        @ torch.randn(1, 12, generator=generator)
    )

    result = detect_rome_compatible_edit(suspect, clean)

    assert result["is_rome_compatible"] is True
    assert result["verdict"] == "rome_compatible_low_rank_edit"
    assert result["selected_layer"] == 5
    assert result["attribution_scope"] == ATTRIBUTION_SCOPE
    assert "rome_like" not in result


def test_b0_rejects_no_change() -> None:
    clean = _weights()

    result = detect_rome_compatible_edit(dict(clean), clean)

    assert result == {
        "available": True,
        "is_rome_compatible": False,
        "verdict": "no_detectable_change",
        "selected_layer": None,
        "change_magnitude": 0.0,
        "magnitude_bound": 0.0,
        "rank2_tail_ratio": 0.0,
        "tail_ratio_bound": 0.0,
        "attribution_scope": ATTRIBUTION_SCOPE,
    }


def test_b0_is_storage_transpose_invariant() -> None:
    clean = _weights()
    suspect = dict(clean)
    generator = torch.Generator().manual_seed(29)
    suspect[5] = clean[5] + (
        torch.randn(8, 1, generator=generator)
        @ torch.randn(1, 12, generator=generator)
    )

    direct = detect_rome_compatible_edit(suspect, clean)
    transposed = detect_rome_compatible_edit(
        {layer: weight.T for layer, weight in suspect.items()},
        {layer: weight.T for layer, weight in clean.items()},
    )

    assert direct["is_rome_compatible"] is True
    assert transposed["is_rome_compatible"] is True
    assert transposed["selected_layer"] == direct["selected_layer"]


def test_b0_numerical_bounds_depend_on_dtype_and_dimension_not_model_name() -> None:
    clean = _weights()
    suspect = dict(clean)
    suspect[5] = clean[5] + torch.ones(8, 1) @ torch.ones(1, 12)

    evidence = gram_delta_evidence(suspect[5], clean[5], layer=5)

    assert evidence["magnitude_bound"] > 0.0
    assert evidence["tail_ratio_bound"] > 0.0
    assert "model" not in evidence
    assert "family" not in evidence


def test_minimal_resident_api_returns_localization_and_b0() -> None:
    generator = torch.Generator().manual_seed(31)
    base = torch.randn(8, 12, generator=generator)
    clean = {
        layer: base + 0.002 * layer * torch.randn(8, 12, generator=generator)
        for layer in range(12)
    }
    suspect = {layer: weight.clone() for layer, weight in clean.items()}
    suspect[5] += 0.35 * (
        torch.randn(8, 1, generator=generator)
        @ torch.randn(1, 12, generator=generator)
    )

    result = RomeDetector().detect(suspect, clean)

    assert result["localization"]["selected_layer"] == 5
    assert result["clean_reference_presence"]["is_rome_compatible"] is True
