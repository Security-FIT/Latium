#!/usr/bin/env python3
"""Evaluate the opt-in signed M3 statistic with family-blocked calibration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.structural.experiments.single_checkpoint_rome import (
    SIGNED_CONSISTENCY_VERSION,
    selected_signed_consistency,
)


SCHEMA_VERSION = "rome-single-checkpoint-signed-evaluation-v1"
STATISTIC = "signed_residual_consistency"


def _predict(value: float, cutoff: float, direction: str) -> bool:
    if direction == "below":
        return float(value) < float(cutoff)
    if direction == "above":
        return float(value) > float(cutoff)
    raise ValueError(f"Unsupported direction: {direction}")


def threshold_metrics(
    records: list[dict[str, Any]],
    *,
    cutoff: float,
    direction: str,
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for record in records:
        truth = record["label"] == "rome"
        predicted = _predict(record[STATISTIC], cutoff, direction)
        if truth and predicted:
            tp += 1
        elif truth:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    balanced = (
        (sensitivity + specificity) / 2.0 if math.isfinite(sensitivity) and math.isfinite(specificity) else math.nan
    )
    return {
        "cutoff": float(cutoff),
        "direction": direction,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced,
    }


def calibrate_equal_family_cutoff(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Choose one direction and cutoff by equal-family balanced accuracy."""
    if not records:
        raise ValueError("Calibration records are required")
    values = sorted({float(record[STATISTIC]) for record in records})
    candidates = [float(np.nextafter(values[0], -math.inf))]
    candidates.extend((left + right) / 2.0 for left, right in zip(values, values[1:], strict=False))
    candidates.append(float(np.nextafter(values[-1], math.inf)))
    families = sorted({str(record["family"]) for record in records})
    evaluated: list[tuple[tuple[float, float, float], str, float]] = []
    for direction in ("below", "above"):
        for cutoff in candidates:
            per_family = [
                threshold_metrics(
                    [record for record in records if str(record["family"]) == family],
                    cutoff=cutoff,
                    direction=direction,
                )
                for family in families
            ]
            valid = [
                float(item["balanced_accuracy"])
                for item in per_family
                if math.isfinite(float(item["balanced_accuracy"]))
            ]
            macro = float(np.mean(valid)) if valid else -math.inf
            worst = float(np.min(valid)) if valid else -math.inf
            pooled = threshold_metrics(
                records,
                cutoff=cutoff,
                direction=direction,
            )
            evaluated.append(
                (
                    (macro, worst, float(pooled["specificity"])),
                    direction,
                    cutoff,
                )
            )
    objective, direction, cutoff = max(
        evaluated,
        key=lambda item: (item[0], item[1] == "below", -abs(item[2])),
    )
    return {
        "cutoff": cutoff,
        "direction": direction,
        "equal_family_balanced_accuracy": objective[0],
        "worst_family_balanced_accuracy": objective[1],
        "pooled": threshold_metrics(
            records,
            cutoff=cutoff,
            direction=direction,
        ),
    }


def _case(payload: dict[str, Any], case_id: str) -> dict[str, Any]:
    return next(case for case in payload["cases"] if str(case["case_id"]) == str(case_id))


def _load_structural_records(model: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    root = Path(model["run_root"])
    plan = root / "plans" / model["model_key"] / model["plan_id"]
    baseline_payload = json.loads((plan / "baseline/captures/single-checkpoint-signed.json").read_text())
    edited_payload = json.loads((plan / "methods/rome/captures/single-checkpoint-signed.json").read_text())
    execution = json.loads((plan / "methods/rome/execution.json").read_text())
    baseline_case = _case(baseline_payload, "baseline")
    baseline_statistic = selected_signed_consistency(baseline_case["data"])
    records = [
        {
            "specimen_id": f"{model['model_key']}:clean",
            "model": model["model_key"],
            "family": model["family"],
            "label": "clean",
            "negative_category": "standalone_clean",
            **baseline_statistic,
        }
    ]
    failures = 0
    for execution_case in execution["cases"]:
        edit = execution_case.get("edit") or {}
        if execution_case.get("status") != "complete" or not bool(edit.get("success")):
            failures += 1
            continue
        case_id = str(execution_case["case_id"])
        capture_case = _case(edited_payload, case_id)
        statistic = selected_signed_consistency(capture_case["data"])
        target_layers = (edit.get("modified_weights") or {}).get("proj") or []
        target_layer = int(target_layers[0]) if target_layers else None
        records.append(
            {
                "specimen_id": f"{model['model_key']}:rome:{case_id}",
                "model": model["model_key"],
                "family": model["family"],
                "label": "rome",
                "negative_category": None,
                "case_id": case_id,
                "target_layer": target_layer,
                "localization_correct": statistic["selected_layer"] == target_layer,
                "localization_within_one": target_layer is not None
                and abs(statistic["selected_layer"] - target_layer) <= 1,
                **statistic,
            }
        )
    return records, failures


def _load_hard_negatives(model: dict[str, Any]) -> list[dict[str, Any]]:
    raw_path = model.get("hard_negative_bundle")
    if not raw_path:
        return []
    payload = json.loads(Path(raw_path).read_text())
    if payload["model_key"] != model["model_key"]:
        raise ValueError("Hard-negative bundle model does not match manifest")
    return [
        {
            "specimen_id": record["specimen_id"],
            "model": model["model_key"],
            "family": model["family"],
            "label": "hard_negative",
            "negative_category": record["negative_category"],
            **selected_signed_consistency(record["capture"]),
        }
        for record in payload["records"]
        if record["label"] == "hard_negative"
    ]


def _metrics_from_predictions(records: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(record["label"] == "rome" and record["predicted"] for record in records)
    fn = sum(record["label"] == "rome" and not record["predicted"] for record in records)
    fp = sum(record["label"] != "rome" and record["predicted"] for record in records)
    tn = sum(record["label"] != "rome" and not record["predicted"] for record in records)
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    return {
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
    }


def _auroc(records: list[dict[str, Any]], *, direction: str) -> float:
    sign = -1.0 if direction == "below" else 1.0
    positive = [sign * float(record[STATISTIC]) for record in records if record["label"] == "rome"]
    negative = [sign * float(record[STATISTIC]) for record in records if record["label"] != "rome"]
    if not positive or not negative:
        return math.nan
    wins = sum(1.0 if left > right else 0.5 if left == right else 0.0 for left in positive for right in negative)
    return wins / (len(positive) * len(negative))


def _blocked_interval(
    per_family: list[dict[str, Any]],
    metric: str,
    *,
    seed: int = 20260728,
    iterations: int = 10_000,
) -> list[float]:
    values = np.asarray([record[metric] for record in per_family], dtype=np.float64)
    generator = np.random.default_rng(seed)
    distribution = generator.choice(
        values,
        size=(iterations, len(values)),
        replace=True,
    ).mean(axis=1)
    return [float(value) for value in np.quantile(distribution, (0.025, 0.975))]


def evaluate(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    records: list[dict[str, Any]] = []
    failed_rome = 0
    for model in manifest["models"]:
        structural, failures = _load_structural_records(model)
        records.extend(structural)
        records.extend(_load_hard_negatives(model))
        failed_rome += failures
    families = sorted({record["family"] for record in records})
    predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for held_family in families:
        training = [record for record in records if record["family"] != held_family]
        testing = [record for record in records if record["family"] == held_family]
        calibration = calibrate_equal_family_cutoff(training)
        folds.append(
            {
                "held_family": held_family,
                "cutoff": calibration["cutoff"],
                "direction": calibration["direction"],
            }
        )
        predictions.extend(
            {
                **record,
                "predicted": _predict(
                    record[STATISTIC],
                    calibration["cutoff"],
                    calibration["direction"],
                ),
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
    macro = float(np.mean([record["balanced_accuracy"] for record in per_family]))
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
    positive = [record for record in predictions if record["label"] == "rome"]
    all_development = calibrate_equal_family_cutoff(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_version": SIGNED_CONSISTENCY_VERSION,
        "scientific_baseline": False,
        "split": "leave-one-exposed-family-out-development",
        "counts": {
            "records": len(records),
            "successful_rome": len(positive),
            "failed_rome_excluded": failed_rome,
            "standalone_clean": sum(record["label"] == "clean" for record in records),
            "hard_negative": sum(record["label"] == "hard_negative" for record in records),
        },
        "pooled": pooled,
        "auroc": _auroc(
            records,
            direction=all_development["direction"],
        ),
        "equal_family_macro_balanced_accuracy": macro,
        "worst_family_balanced_accuracy": min(record["balanced_accuracy"] for record in per_family),
        "family_blocked_95ci": {
            metric: _blocked_interval(per_family, metric)
            for metric in ("balanced_accuracy", "sensitivity", "specificity")
        },
        "localization": {
            "exact": sum(record["localization_correct"] for record in positive) / len(positive),
            "within_one": sum(record["localization_within_one"] for record in positive) / len(positive),
        },
        "per_family": per_family,
        "per_negative_category": per_negative_category,
        "folds": folds,
        "all_development_calibration": all_development,
        "completion_gate_passed": bool(
            pooled["sensitivity"] >= 0.95
            and pooled["specificity"] >= 0.95
            and macro >= 0.95
            and min(record["balanced_accuracy"] for record in per_family) >= 0.80
            and all(record["specificity"] >= 0.90 for record in per_negative_category)
        ),
        "records": predictions,
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
            {key: value for key, value in result.items() if key != "records"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
