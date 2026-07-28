from __future__ import annotations

import numpy as np

from scripts.evaluate_single_checkpoint_linear import _fit_model, _predict


def test_two_feature_linear_model_is_deterministic_and_uses_no_identity() -> None:
    records = [
        {
            "family": family,
            "label": label,
            "signed_residual_consistency": signed,
            "robust_z": robust_z,
        }
        for family, label, signed, robust_z in (
            ("a", "rome", 0.2, 4.0),
            ("a", "rome", 0.3, 5.0),
            ("a", "clean", 0.9, 1.0),
            ("a", "hard_negative", 0.1, 9.0),
            ("b", "rome", 0.25, 4.5),
            ("b", "rome", 0.35, 5.5),
            ("b", "clean", 1.0, 1.5),
            ("b", "hard_negative", 0.15, 8.0),
        )
    ]

    first = _fit_model(records, secondary="robust_z")
    second = _fit_model(records, secondary="robust_z")
    first_prediction, first_probability = _predict(records, first)
    second_prediction, second_probability = _predict(records, second)

    assert np.array_equal(first_prediction, second_prediction)
    assert np.array_equal(first_probability, second_probability)
    assert first["standardized_coefficients"].shape == (2,)
