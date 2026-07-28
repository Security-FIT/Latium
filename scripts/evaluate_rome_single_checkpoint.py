#!/usr/bin/env python3
"""Evaluate transparent suspect-only ROME statistics on a checkpoint manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.structural.experiments.single_checkpoint_rome import (
    LOCAL_PROMINENCE_VERSION,
    PEAK_PROMINENCE_VERSION,
    ROBUST_PEAK_VERSION,
    calibrate_equal_family_cutoff,
    local_prominence_statistic,
    peak_prominence_statistic,
    robust_peak_statistic,
    threshold_metrics,
)


def _case(payload: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in payload.get("cases", []):
        if str(case.get("case_id")) == str(case_id):
            return case
    raise ValueError(f"Case {case_id!r} is absent from {payload.get('artifact_id')}")


def _profile(record: dict[str, Any]) -> tuple[dict[str, float], list[int]]:
    payload = json.loads(Path(record["source_artifact"]).read_text())
    case = _case(payload, str(record["case_id"]))
    if case.get("status") != "complete":
        raise ValueError(f"Specimen {record['specimen_id']} is not complete")
    data = case.get("data") or {}
    if record["label"] == "clean":
        profiles = data.get("profiles") or {}
        scores = {str(layer): float(profile["relative_subspace_frobenius"]) for layer, profile in profiles.items()}
        eligible = [int(layer) for layer in data.get("eligible_layers", ())]
    else:
        localization = data.get("localization") or {}
        scores = {str(layer): float(value) for layer, value in (localization.get("layer_scores") or {}).items()}
        eligible = [int(layer) for layer in localization.get("eligible_layers", ())]
    if not scores or not eligible:
        raise ValueError(f"Specimen {record['specimen_id']} has no complete M3 profile")
    return scores, eligible


def _auc(records: list[dict[str, Any]], statistic: str) -> float:
    positive = [float(record[statistic]) for record in records if record["label"] == "rome"]
    negative = [float(record[statistic]) for record in records if record["label"] != "rome"]
    if not positive or not negative:
        return math.nan
    wins = sum(1.0 if pos > neg else 0.5 if pos == neg else 0.0 for pos in positive for neg in negative)
    return wins / (len(positive) * len(negative))


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


def _family_blocked_interval(
    per_family: list[dict[str, Any]],
    metric: str,
    *,
    iterations: int = 10_000,
    seed: int = 20260728,
) -> list[float]:
    values = np.asarray([float(row[metric]) for row in per_family], dtype=np.float64)
    generator = np.random.default_rng(seed)
    sampled = generator.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(sampled, (0.025, 0.975))]


def evaluate(manifest_path: Path, *, candidate: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    records: list[dict[str, Any]] = []
    for specimen in manifest["specimens"]:
        scores, eligible = _profile(specimen)
        if candidate == "robust-peak":
            statistics = robust_peak_statistic(scores, eligible_layers=eligible)
            statistic = "z_peak"
            candidate_version = ROBUST_PEAK_VERSION
        elif candidate == "peak-prominence":
            statistics = peak_prominence_statistic(scores, eligible_layers=eligible)
            statistic = "peak_prominence"
            candidate_version = PEAK_PROMINENCE_VERSION
        elif candidate == "local-prominence":
            statistics = local_prominence_statistic(scores, eligible_layers=eligible)
            statistic = "local_prominence"
            candidate_version = LOCAL_PROMINENCE_VERSION
        else:
            raise ValueError(f"Unknown candidate: {candidate}")
        records.append({**specimen, **statistics})

    families = sorted({str(record["family"]) for record in records})
    predictions: list[dict[str, Any]] = []
    fold_calibrations: list[dict[str, Any]] = []
    for held_family in families:
        training = [record for record in records if record["family"] != held_family]
        testing = [record for record in records if record["family"] == held_family]
        calibration = calibrate_equal_family_cutoff(training, statistic=statistic)
        cutoff = float(calibration["cutoff"])
        fold_calibrations.append(
            {
                "held_family": held_family,
                "cutoff": cutoff,
                "training_equal_family_balanced_accuracy": calibration["equal_family_balanced_accuracy"],
            }
        )
        predictions.extend(
            {
                **record,
                "held_family_cutoff": cutoff,
                "predicted": float(record[statistic]) > cutoff,
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
    macro_balanced = float(np.mean([row["balanced_accuracy"] for row in per_family]))
    all_development_calibration = calibrate_equal_family_cutoff(
        records,
        statistic=statistic,
    )
    return {
        "schema_version": "rome-single-checkpoint-evaluation-v1",
        "candidate_version": candidate_version,
        "statistic": statistic,
        "scientific_baseline": False,
        "split": "leave-one-exposed-family-out-development",
        "manifest": str(manifest_path),
        "counts": {
            "specimens": len(records),
            "rome_positive": sum(record["label"] == "rome" for record in records),
            "standalone_clean": sum(record["label"] == "clean" for record in records),
            "hard_negative": sum(record["label"] not in {"rome", "clean"} for record in records),
        },
        "pooled": pooled,
        "auroc": _auc(records, statistic),
        "equal_family_macro_balanced_accuracy": macro_balanced,
        "worst_family_balanced_accuracy": min(row["balanced_accuracy"] for row in per_family),
        "family_blocked_95ci": {
            "balanced_accuracy": _family_blocked_interval(per_family, "balanced_accuracy"),
            "sensitivity": _family_blocked_interval(per_family, "sensitivity"),
            "specificity": _family_blocked_interval(per_family, "specificity"),
        },
        "per_family": per_family,
        "fold_calibrations": fold_calibrations,
        "all_development_calibration": all_development_calibration,
        "completion_gate_passed": bool(
            pooled["sensitivity"] >= 0.95
            and pooled["specificity"] >= 0.95
            and macro_balanced >= 0.95
            and min(row["balanced_accuracy"] for row in per_family) >= 0.80
        ),
        "records": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=("robust-peak", "peak-prominence", "local-prominence"),
        required=True,
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    result = evaluate(args.manifest, candidate=args.candidate)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.ledger is not None:
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger_row = {
            key: result[key]
            for key in (
                "candidate_version",
                "split",
                "counts",
                "pooled",
                "auroc",
                "equal_family_macro_balanced_accuracy",
                "worst_family_balanced_accuracy",
                "family_blocked_95ci",
                "completion_gate_passed",
            )
        }
        with args.ledger.open("a") as handle:
            handle.write(json.dumps(ledger_row, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
