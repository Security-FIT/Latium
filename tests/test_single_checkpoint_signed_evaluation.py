from __future__ import annotations

from scripts.evaluate_single_checkpoint_signed import (
    calibrate_equal_family_cutoff,
    threshold_metrics,
)


def test_signed_calibration_selects_one_global_direction_and_cutoff() -> None:
    records = [
        {"family": "a", "label": "rome", "signed_residual_consistency": 0.1},
        {"family": "a", "label": "clean", "signed_residual_consistency": 0.9},
        {"family": "b", "label": "rome", "signed_residual_consistency": 0.2},
        {"family": "b", "label": "hard_negative", "signed_residual_consistency": 1.0},
    ]

    result = calibrate_equal_family_cutoff(records)

    assert result["direction"] == "below"
    assert 0.2 < result["cutoff"] < 0.9
    assert result["pooled"]["sensitivity"] == 1.0
    assert result["pooled"]["specificity"] == 1.0


def test_signed_threshold_has_strict_deterministic_tie_behavior() -> None:
    records = [
        {"family": "a", "label": "rome", "signed_residual_consistency": 0.5},
        {"family": "a", "label": "clean", "signed_residual_consistency": 0.5},
    ]

    below = threshold_metrics(records, cutoff=0.5, direction="below")
    above = threshold_metrics(records, cutoff=0.5, direction="above")

    assert below["tp"] == above["tp"] == 0
    assert below["tn"] == above["tn"] == 1
