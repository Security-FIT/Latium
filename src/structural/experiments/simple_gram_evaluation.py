"""Evaluation helpers for the opt-in simple-Gram experiment."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.structural.experiments.simple_gram import PROFILE_FIELDS


SPIKE_FIELDS = (
    "robust_peak",
    "global_prominence",
    "local_prominence",
)


def localization_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate exact and within-one localization over successful edits."""
    successful = [record for record in records if bool(record["edit_success"])]
    output: dict[str, Any] = {
        "successful_edits": len(successful),
        "candidates": {},
    }
    for field in PROFILE_FIELDS:
        by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in successful:
            by_model[str(record["model"])].append(record)
        per_model: dict[str, Any] = {}
        exact_total = 0
        within_total = 0
        macro_exact: list[float] = []
        for model, model_records in sorted(by_model.items()):
            exact = sum(
                int(record["selected_layers"][field]) == int(record["target_layer"])
                for record in model_records
            )
            within = sum(
                abs(
                    int(record["selected_layers"][field])
                    - int(record["target_layer"])
                )
                <= 1
                for record in model_records
            )
            count = len(model_records)
            exact_total += exact
            within_total += within
            macro_exact.append(exact / count)
            per_model[model] = {
                "successful": count,
                "exact": exact,
                "within_one": within,
            }
        output["candidates"][field] = {
            "exact": exact_total,
            "exact_rate": exact_total / len(successful) if successful else 0.0,
            "within_one": within_total,
            "within_one_rate": (
                within_total / len(successful) if successful else 0.0
            ),
            "equal_model_macro_exact": (
                float(np.mean(macro_exact)) if macro_exact else 0.0
            ),
            "per_model": per_model,
        }
    return output


def _balanced_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> tuple[float, float, float]:
    positive = labels
    negative = ~labels
    sensitivity = (
        float(predictions[positive].mean()) if bool(positive.any()) else 0.0
    )
    specificity = (
        float((~predictions[negative]).mean()) if bool(negative.any()) else 0.0
    )
    return sensitivity, specificity, (sensitivity + specificity) / 2.0


def calibrate_global_cutoff(
    records: Sequence[Mapping[str, Any]],
    *,
    statistic: str,
) -> float:
    """Fit one high-is-positive cutoff with equal-family weighting."""
    if statistic not in SPIKE_FIELDS:
        raise ValueError(f"Unknown spike statistic: {statistic}")
    if not records:
        raise ValueError("Cutoff calibration requires records")
    values = np.asarray(
        [float(record["statistics"][statistic]) for record in records],
        dtype=np.float64,
    )
    labels = np.asarray(
        [bool(record["is_positive"]) for record in records],
        dtype=bool,
    )
    families = np.asarray(
        [str(record["family"]) for record in records],
        dtype=object,
    )
    unique = np.unique(values)
    middle = (unique[:-1] + unique[1:]) / 2.0
    cutoffs = np.concatenate(
        (
            [np.nextafter(unique[0], -np.inf)],
            middle,
            [np.nextafter(unique[-1], np.inf)],
        )
    )
    best: tuple[tuple[float, float, float, float], float] | None = None
    for cutoff in cutoffs:
        predictions = values > cutoff
        family_metrics = [
            _balanced_accuracy(
                labels[families == family],
                predictions[families == family],
            )
            for family in sorted(set(families))
        ]
        if any(
            not bool(labels[families == family].any())
            or not bool((~labels[families == family]).any())
            for family in sorted(set(families))
        ):
            raise ValueError(
                "Every calibration family needs positive and negative records"
            )
        macro_balanced = float(
            np.mean([metrics[2] for metrics in family_metrics])
        )
        macro_specificity = float(
            np.mean([metrics[1] for metrics in family_metrics])
        )
        macro_sensitivity = float(
            np.mean([metrics[0] for metrics in family_metrics])
        )
        objective = (
            macro_balanced,
            macro_specificity,
            macro_sensitivity,
            float(cutoff),
        )
        if best is None or objective > best[0]:
            best = (objective, float(cutoff))
    if best is None:
        raise RuntimeError("No calibration cutoff was evaluated")
    return best[1]


def leave_one_family_out_presence(
    records: Sequence[Mapping[str, Any]],
    *,
    statistic: str,
) -> dict[str, Any]:
    """Evaluate fold-specific global cutoffs without family-specific routing."""
    families = sorted({str(record["family"]) for record in records})
    predictions: list[dict[str, Any]] = []
    for held_out in families:
        training = [
            record for record in records if str(record["family"]) != held_out
        ]
        testing = [
            record for record in records if str(record["family"]) == held_out
        ]
        cutoff = calibrate_global_cutoff(training, statistic=statistic)
        predictions.extend(
            {
                "family": held_out,
                "is_positive": bool(record["is_positive"]),
                "predicted": float(record["statistics"][statistic]) > cutoff,
                "cutoff": cutoff,
            }
            for record in testing
        )

    labels = np.asarray(
        [prediction["is_positive"] for prediction in predictions],
        dtype=bool,
    )
    predicted = np.asarray(
        [prediction["predicted"] for prediction in predictions],
        dtype=bool,
    )
    sensitivity, specificity, balanced = _balanced_accuracy(labels, predicted)
    per_family: dict[str, Any] = {}
    family_balanced: list[float] = []
    for family in families:
        mask = np.asarray(
            [prediction["family"] == family for prediction in predictions],
            dtype=bool,
        )
        family_metrics = _balanced_accuracy(labels[mask], predicted[mask])
        family_balanced.append(family_metrics[2])
        per_family[family] = {
            "sensitivity": family_metrics[0],
            "specificity": family_metrics[1],
            "balanced_accuracy": family_metrics[2],
        }
    return {
        "statistic": statistic,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "equal_family_macro_balanced_accuracy": float(
            np.mean(family_balanced)
        ),
        "worst_family_balanced_accuracy": min(family_balanced, default=0.0),
        "per_family": per_family,
        "predictions": predictions,
    }


__all__ = [
    "SPIKE_FIELDS",
    "calibrate_global_cutoff",
    "leave_one_family_out_presence",
    "localization_summary",
]
