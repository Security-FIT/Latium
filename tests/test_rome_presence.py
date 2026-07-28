from __future__ import annotations

import pytest
import torch

from src.structural.detectors.rome_presence_resident import RomeDetector
from src.structural.detectors.weighted_spectrum import SCHEMA_VERSION


def _suspect_weights() -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(31)
    base = torch.randn(8, 12, generator=generator)
    weights = {
        layer: base + 0.002 * layer * torch.randn(8, 12, generator=generator)
        for layer in range(12)
    }
    weights[5] += 0.35 * (
        torch.randn(8, 1, generator=generator)
        @ torch.randn(1, 12, generator=generator)
    )
    return weights


def test_resident_api_accepts_exactly_one_checkpoint() -> None:
    detector = RomeDetector()
    suspect = _suspect_weights()

    result = detector.detect_one_checkpoint(suspect)

    assert result["schema_version"] == "rome-detector-minimal-v3" == SCHEMA_VERSION
    assert result["localization"]["selected_layer"] == 5
    assert "clean_reference_presence" not in result
    assert "is_rome_compatible" not in result


def test_detect_alias_is_also_single_checkpoint_only() -> None:
    detector = RomeDetector()
    suspect = _suspect_weights()

    assert detector.detect(suspect) == detector.detect_one_checkpoint(suspect)
    with pytest.raises(TypeError):
        detector.detect(suspect, suspect)  # type: ignore[call-arg]


def test_one_checkpoint_result_does_not_expose_edit_request_metadata() -> None:
    result = RomeDetector().detect_one_checkpoint(_suspect_weights())
    serialized_keys = repr(result)

    assert "case_id" not in serialized_keys
    assert "target_layer" not in serialized_keys
    assert "configured_layer" not in serialized_keys
    assert "causal" not in serialized_keys
    assert "covariance" not in serialized_keys
