from __future__ import annotations

import pytest

from src.structural.experiments.two_stat_rome import (
    calibrate_two_stat_rule,
    checkpoint_statistics,
    predict_two_stat,
    two_stat_margin,
)


def _capture(values: list[tuple[float, float]]) -> dict:
    return {
        "eligible_layers": list(range(len(values))),
        "profiles": {
            str(layer): {
                "relative_subspace_frobenius": score,
                "signed_residual_consistency": signed,
            }
            for layer, (score, signed) in enumerate(values)
        },
    }


def test_checkpoint_statistics_preserve_deterministic_m3_peak() -> None:
    result = checkpoint_statistics(_capture([(1.0, 0.9), (4.0, 0.2), (4.0, 0.7), (2.0, 0.8)]))

    assert result["selected_layer"] == 1
    assert result["signed_residual_consistency"] == 0.2
    assert result["peak"] == 4.0
    assert result["global_prominence"] == 0.0
    assert result["local_prominence"] == pytest.approx(0.0)


def test_two_stat_calibration_uses_one_rule_for_all_families() -> None:
    records = [
        {
            "family": family,
            "label": label,
            "signed_residual_consistency": signed,
            "robust_z": robust_z,
        }
        for family, label, signed, robust_z in (
            ("a", "rome", 0.2, 5.0),
            ("a", "clean", 0.9, 1.0),
            ("a", "hard_negative", 0.1, 9.0),
            ("b", "rome", 0.3, 4.0),
            ("b", "clean", 1.0, 1.5),
            ("b", "hard_negative", 0.15, 8.0),
        )
    ]

    rule = calibrate_two_stat_rule(records, secondary="robust_z")

    assert all(predict_two_stat(record, rule) == (record["label"] == "rome") for record in records)
    assert all((two_stat_margin(record, rule) > 0.0) == (record["label"] == "rome") for record in records)
