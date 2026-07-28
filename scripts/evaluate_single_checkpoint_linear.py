#!/usr/bin/env python3
"""Evaluate a tiny two-feature linear baseline after transparent rules fail."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from scripts.evaluate_single_checkpoint_signed import (
    _blocked_interval,
    _load_hard_negatives,
    _load_structural_records,
    _metrics_from_predictions,
)
from src.structural.experiments.two_stat_rome import SECONDARY_STATISTICS


SCHEMA_VERSION = "rome-single-checkpoint-linear-evaluation-v1"
LINEAR_VERSION = "rome-single-checkpoint-two-feature-logistic-v1"


def _sample_weights(records: list[dict[str, Any]]) -> np.ndarray:
    """Give every family/class cell equal total development weight."""
    cells: dict[tuple[str, bool], int] = {}
    for record in records:
        key = (str(record["family"]), record["label"] == "rome")
        cells[key] = cells.get(key, 0) + 1
    return np.asarray(
        [1.0 / cells[(str(record["family"]), record["label"] == "rome")] for record in records],
        dtype=np.float64,
    )


def _fit_model(
    records: list[dict[str, Any]],
    *,
    secondary: str,
) -> dict[str, Any]:
    features = np.asarray(
        [[record["signed_residual_consistency"], record[secondary]] for record in records],
        dtype=np.float64,
    )
    labels = np.asarray(
        [record["label"] == "rome" for record in records],
        dtype=np.int64,
    )
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.maximum(scale, np.finfo(np.float64).eps)
    standardized = (features - center) / scale
    estimator = LogisticRegression(
        C=1.0,
        max_iter=1_000,
        random_state=20260728,
        solver="lbfgs",
    )
    estimator.fit(
        standardized,
        labels,
        sample_weight=_sample_weights(records),
    )
    probabilities = estimator.predict_proba(standardized)[:, 1]
    cutoff = _calibrate_probability_cutoff(records, probabilities)
    return {
        "secondary": secondary,
        "center": center,
        "scale": scale,
        "estimator": estimator,
        "cutoff": cutoff,
        "standardized_coefficients": estimator.coef_[0],
        "intercept": float(estimator.intercept_[0]),
    }


def _calibrate_probability_cutoff(
    records: list[dict[str, Any]],
    probabilities: np.ndarray,
) -> float:
    unique = np.unique(probabilities)
    cutoffs = np.concatenate(
        (
            [np.nextafter(unique[0], -math.inf)],
            (unique[:-1] + unique[1:]) / 2.0,
            [np.nextafter(unique[-1], math.inf)],
        )
    )
    families = sorted({str(record["family"]) for record in records})
    best: tuple[tuple[float, float, float], float] | None = None
    labels = np.asarray(
        [record["label"] == "rome" for record in records],
        dtype=bool,
    )
    for cutoff in cutoffs:
        predicted = probabilities > cutoff
        balanced = []
        for family in families:
            family_mask = np.asarray(
                [str(record["family"]) == family for record in records],
                dtype=bool,
            )
            positive = family_mask & labels
            negative = family_mask & ~labels
            sensitivity = float(predicted[positive].mean())
            specificity = float((~predicted[negative]).mean())
            balanced.append((sensitivity + specificity) / 2.0)
        pooled_specificity = float((~predicted[~labels]).mean())
        objective = (
            float(np.mean(balanced)),
            float(np.min(balanced)),
            pooled_specificity,
        )
        if best is None or objective > best[0]:
            best = (objective, float(cutoff))
    if best is None:
        raise ValueError("Probability calibration records are required")
    return best[1]


def _predict(
    records: list[dict[str, Any]],
    fitted: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(
        [
            [
                record["signed_residual_consistency"],
                record[str(fitted["secondary"])],
            ]
            for record in records
        ],
        dtype=np.float64,
    )
    standardized = (features - fitted["center"]) / fitted["scale"]
    probabilities = fitted["estimator"].predict_proba(standardized)[:, 1]
    return probabilities > float(fitted["cutoff"]), probabilities


def _auroc(records: list[dict[str, Any]]) -> float:
    positive = [record["probability"] for record in records if record["label"] == "rome"]
    negative = [record["probability"] for record in records if record["label"] != "rome"]
    wins = sum(1.0 if left > right else 0.5 if left == right else 0.0 for left in positive for right in negative)
    return wins / (len(positive) * len(negative))


def _evaluate_secondary(
    records: list[dict[str, Any]],
    *,
    secondary: str,
) -> dict[str, Any]:
    families = sorted({str(record["family"]) for record in records})
    predictions: list[dict[str, Any]] = []
    folds = []
    for held_family in families:
        training = [record for record in records if record["family"] != held_family]
        testing = [record for record in records if record["family"] == held_family]
        fitted = _fit_model(training, secondary=secondary)
        predicted, probabilities = _predict(testing, fitted)
        folds.append(
            {
                "held_family": held_family,
                "cutoff": fitted["cutoff"],
                "standardized_coefficients": fitted["standardized_coefficients"].tolist(),
                "intercept": fitted["intercept"],
            }
        )
        predictions.extend(
            {
                **record,
                "predicted": bool(prediction),
                "probability": float(probability),
            }
            for record, prediction, probability in zip(
                testing,
                predicted,
                probabilities,
                strict=True,
            )
        )
    per_family = [
        {
            "family": family,
            **_metrics_from_predictions([record for record in predictions if record["family"] == family]),
        }
        for family in families
    ]
    pooled = _metrics_from_predictions(predictions)
    macro = float(np.mean([row["balanced_accuracy"] for row in per_family]))
    categories = sorted(
        {record["negative_category"] for record in predictions if record["negative_category"] is not None}
    )
    per_negative_category = []
    for category in categories:
        selected = [record for record in predictions if record["negative_category"] == category]
        per_negative_category.append(
            {
                "category": category,
                "count": len(selected),
                "specificity": sum(not record["predicted"] for record in selected) / len(selected),
            }
        )
    return {
        "candidate_version": LINEAR_VERSION,
        "secondary": secondary,
        "pooled": pooled,
        "auroc": _auroc(predictions),
        "equal_family_macro_balanced_accuracy": macro,
        "worst_family_balanced_accuracy": min(row["balanced_accuracy"] for row in per_family),
        "family_blocked_95ci": {
            metric: _blocked_interval(per_family, metric)
            for metric in ("balanced_accuracy", "sensitivity", "specificity")
        },
        "per_family": per_family,
        "per_negative_category": per_negative_category,
        "folds": folds,
        "completion_gate_passed": bool(
            pooled["sensitivity"] >= 0.95
            and pooled["specificity"] >= 0.95
            and macro >= 0.95
            and min(row["balanced_accuracy"] for row in per_family) >= 0.80
            and all(row["specificity"] >= 0.90 for row in per_negative_category)
        ),
        "records": predictions,
    }


def evaluate(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    records: list[dict[str, Any]] = []
    for model in manifest["models"]:
        structural, _failures = _load_structural_records(model)
        records.extend(structural)
        records.extend(_load_hard_negatives(model))
    candidates = [_evaluate_secondary(records, secondary=secondary) for secondary in SECONDARY_STATISTICS]
    winner = max(
        candidates,
        key=lambda result: (
            result["equal_family_macro_balanced_accuracy"],
            result["worst_family_balanced_accuracy"],
            result["pooled"]["specificity"],
            -SECONDARY_STATISTICS.index(result["secondary"]),
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_baseline": False,
        "split": "leave-one-exposed-family-out-development",
        "features": "signed_residual_consistency plus one M3 scalar; no model identity",
        "candidates": [
            {key: value for key, value in candidate.items() if key != "records"} for candidate in candidates
        ],
        "selected_secondary": winner["secondary"],
        "completion_gate_passed": bool(winner["completion_gate_passed"]),
        "selected_records": winner["records"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.manifest)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "selected_records"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
