from __future__ import annotations

import math

import pytest

from src.structural.experiments.single_checkpoint_rome import (
    calibrate_equal_family_cutoff,
    dtype_scale_bound,
    local_prominence_statistic,
    peak_prominence_statistic,
    robust_peak_statistic,
)


def test_robust_peak_is_positive_affine_invariant() -> None:
    scores = {str(layer): value for layer, value in enumerate((1.0, 2.0, 3.0, 20.0, 4.0))}
    transformed = {layer: 17.0 + 9.0 * value for layer, value in scores.items()}

    direct = robust_peak_statistic(scores, eligible_layers=range(5))
    affine = robust_peak_statistic(transformed, eligible_layers=range(5))

    assert direct["selected_layer"] == affine["selected_layer"] == 3
    assert direct["z_peak"] == pytest.approx(affine["z_peak"])


def test_zero_mad_uses_finite_dtype_bound() -> None:
    result = robust_peak_statistic(
        {str(layer): 4.0 for layer in range(8)},
        eligible_layers=range(8),
    )

    assert result["selected_layer"] == 0
    assert result["mad"] == 0.0
    assert result["scale_bound"] == dtype_scale_bound([4.0] * 8)
    assert result["z_peak"] == 0.0
    assert math.isfinite(result["z_peak"])


def test_peak_prominence_uses_second_highest_and_deterministic_ties() -> None:
    result = peak_prominence_statistic(
        {"0": 1.0, "1": 7.0, "2": 3.0, "3": 7.0},
        eligible_layers=range(4),
    )

    assert result["selected_layer"] == 1
    assert result["second_layer"] == 3
    assert result["peak_prominence"] == 0.0


def test_local_prominence_uses_adjacent_depth_not_global_runner_up() -> None:
    result = local_prominence_statistic(
        {"0": 9.0, "1": 2.0, "2": 10.0, "3": 4.0, "4": 8.0},
        eligible_layers=range(5),
    )

    assert result["selected_layer"] == 2
    assert result["neighbor_layer"] == 3
    assert result["neighbor_peak"] == 4.0
    assert result["local_prominence"] == pytest.approx(1.5)


def test_equal_family_calibration_uses_one_global_cutoff() -> None:
    records = [
        {"family": "a", "label": "clean", "z_peak": 1.0},
        {"family": "a", "label": "rome", "z_peak": 8.0},
        {"family": "b", "label": "clean", "z_peak": 2.0},
        {"family": "b", "label": "rome", "z_peak": 9.0},
    ]

    result = calibrate_equal_family_cutoff(records)

    assert 2.0 < result["cutoff"] < 8.0
    assert result["pooled"]["sensitivity"] == 1.0
    assert result["pooled"]["specificity"] == 1.0
    assert {item["cutoff"] for item in result["per_family"]} == {result["cutoff"]}


def test_robust_peak_rejects_missing_or_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="Missing"):
        robust_peak_statistic({"0": 1.0}, eligible_layers=(0, 1))
    with pytest.raises(ValueError, match="finite"):
        robust_peak_statistic({"0": float("nan")}, eligible_layers=(0,))
