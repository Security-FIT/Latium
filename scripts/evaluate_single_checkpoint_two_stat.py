#!/usr/bin/env python3
"""Evaluate the final transparent two-statistic single-checkpoint rules."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluate_single_checkpoint_signed import (
    _blocked_interval,
    _load_hard_negatives,
    _load_structural_records,
    _metrics_from_predictions,
)
from src.structural.experiments.two_stat_rome import (
    SECONDARY_STATISTICS,
    TWO_STAT_VERSION,
    calibrate_two_stat_rule,
    predict_two_stat,
    two_stat_margin,
)


SCHEMA_VERSION = "rome-single-checkpoint-two-stat-evaluation-v1"


def _margin_auroc(records: list[dict[str, Any]]) -> float:
    positive = [float(record["decision_margin"]) for record in records if record["label"] == "rome"]
    negative = [float(record["decision_margin"]) for record in records if record["label"] != "rome"]
    wins = sum(1.0 if left > right else 0.5 if left == right else 0.0 for left in positive for right in negative)
    return wins / (len(positive) * len(negative))


def _evaluate_secondary(
    records: list[dict[str, Any]],
    *,
    secondary: str,
) -> dict[str, Any]:
    families = sorted({str(record["family"]) for record in records})
    predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for held_family in families:
        training = [record for record in records if record["family"] != held_family]
        testing = [record for record in records if record["family"] == held_family]
        rule = calibrate_two_stat_rule(training, secondary=secondary)
        folds.append({"held_family": held_family, **rule})
        predictions.extend(
            {
                **record,
                "predicted": predict_two_stat(record, rule),
                "decision_margin": two_stat_margin(record, rule),
            }
            for record in testing
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
        "candidate_version": TWO_STAT_VERSION,
        "secondary": secondary,
        "pooled": pooled,
        "auroc": _margin_auroc(predictions),
        "equal_family_macro_balanced_accuracy": macro,
        "worst_family_balanced_accuracy": min(row["balanced_accuracy"] for row in per_family),
        "family_blocked_95ci": {
            metric: _blocked_interval(per_family, metric)
            for metric in ("balanced_accuracy", "sensitivity", "specificity")
        },
        "per_family": per_family,
        "per_negative_category": per_negative_category,
        "folds": folds,
        "all_development_calibration": calibrate_two_stat_rule(
            records,
            secondary=secondary,
        ),
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
    failed_rome = 0
    for model in manifest["models"]:
        structural, failures = _load_structural_records(model)
        records.extend(structural)
        records.extend(_load_hard_negatives(model))
        failed_rome += failures
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
    positive = [record for record in records if record["label"] == "rome"]
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_baseline": False,
        "split": "leave-one-exposed-family-out-development",
        "counts": {
            "records": len(records),
            "successful_rome": len(positive),
            "failed_rome_excluded": failed_rome,
            "standalone_clean": sum(record["label"] == "clean" for record in records),
            "hard_negative": sum(record["label"] == "hard_negative" for record in records),
        },
        "localization": {
            "exact": sum(record["localization_correct"] for record in positive) / len(positive),
            "within_one": sum(record["localization_within_one"] for record in positive) / len(positive),
        },
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
