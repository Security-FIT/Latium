#!/usr/bin/env python3
"""Evaluate binary ROME-presence rules from saved Latium artifacts.

This script never loads a model.  It treats completed ROME cases as positives
and the one saved clean baseline profile per model as a small negative sanity
check.  The latter is not a substitute for a held-out negative cohort.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.structural.detectors.spectral import replay_spectral
from src.structural.detectors.rome_layer_localizer import PROFILE_EXPERIMENTS, evaluate_profile_experiments

try:
    from src.structural.detectors.rome_presence import detect_rome_presence_blind
except ImportError:  # Legacy detector was removed; manifest-backed experiments remain available.
    detect_rome_presence_blind = None


MAD_NORMAL_SCALE = 1.482602218505602
EPS = np.finfo(np.float64).eps
EXPERIMENT_PRODUCERS = {
    "rome-profile-experiments",
    "rome-matrix-experiments",
    "rome-control-experiment",
    "rome-directional-experiments",
    "rome-cross-layer-experiments",
    "rome-token-alignment-experiments",
}
EXPERIMENT_CAPTURE = {
    "rome-profile-experiments": "gram-localization",
    "rome-matrix-experiments": "gram-experiments-v1",
    "rome-control-experiment": "gram-experiments-v1",
    "rome-directional-experiments": "gram-directional-error-v1",
    "rome-cross-layer-experiments": "gram-cross-layer-v1",
    "rome-token-alignment-experiments": "token-subspace-alignment-v1",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = manifest.get("artifacts", {})
    return [value for value in artifacts.values() if isinstance(value, dict)]


def find_record(
    records: Iterable[dict[str, Any]],
    *,
    kind: str,
    producer: str,
    edit_method: str | None,
) -> dict[str, Any] | None:
    matches = [
        record
        for record in records
        if record.get("kind") == kind
        and record.get("producer") == producer
        and record.get("edit_method") == edit_method
    ]
    if not matches:
        return None
    if len(matches) > 1:
        matches.sort(key=lambda record: str(record.get("updated_at", "")))
    return matches[-1]


def load_record(run_root: Path, record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not record:
        return None
    path = run_root / str(record["path"])
    return load_json(path) if path.is_file() else None


def complete_cases(artifact: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not artifact:
        return []
    return [case for case in artifact.get("cases", []) if case.get("status") == "complete"]


def baseline_profiles(artifact: dict[str, Any] | None) -> dict[str, dict[str, float]] | None:
    cases = complete_cases(artifact)
    if not cases:
        return None
    profiles = cases[0].get("data", {}).get("profiles")
    return profiles if isinstance(profiles, dict) and profiles else None


def universal_outlier(values: np.ndarray) -> dict[str, float | bool]:
    """Match the training-free robust extreme test used by ROME presence."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Expected a finite, non-empty one-dimensional score array")
    center = float(np.median(values))
    scale = float(MAD_NORMAL_SCALE * np.median(np.abs(values - center)))
    peak = float(np.max(values))
    effective_scale = max(scale, EPS * max(1.0, abs(center), abs(peak)))
    robust_z = max(0.0, (peak - center) / effective_scale)
    threshold = math.sqrt(2.0 * math.log(max(2, int(values.size))))
    return {
        "is_rome_like": bool(robust_z > threshold),
        "robust_z": robust_z,
        "threshold": threshold,
        "evidence_ratio": robust_z / threshold,
    }


def spectral_presence(data: dict[str, Any], *, log_transform: bool) -> dict[str, float | bool]:
    scores = data.get("rome_hybrid_scores", {})
    evaluated = [int(layer) for layer in data.get("evaluated_layers", [])]
    values = np.asarray(
        [float(scores.get(str(layer), scores.get(layer, 0.0))) for layer in evaluated],
        dtype=np.float64,
    )
    if log_transform:
        values = np.log1p(np.maximum(values, 0.0))
    return universal_outlier(values)


def summarize_positive_cases(artifact: dict[str, Any] | None) -> dict[str, Any]:
    cases = complete_cases(artifact)
    positives = [case for case in cases if bool(case.get("data", {}).get("is_rome_like"))]
    return {
        "positive_cases": len(cases),
        "true_positives": len(positives),
        "verdicts": dict(Counter(str(case.get("data", {}).get("verdict", "unknown")) for case in cases)),
    }


def analysis_row(
    *,
    model: str,
    detector: str,
    positive_cases: int,
    true_positives: int,
    clean_result: dict[str, Any] | None,
    verdicts: dict[str, int],
) -> dict[str, Any]:
    clean_positive = None if clean_result is None else bool(clean_result.get("is_rome_like"))
    return {
        "model": model,
        "detector": detector,
        "positive_cases": positive_cases,
        "true_positives": true_positives,
        "sensitivity": true_positives / positive_cases if positive_cases else None,
        "clean_cases": 0 if clean_result is None else 1,
        "clean_false_positives": int(clean_positive) if clean_positive is not None else 0,
        "clean_is_rome_like": clean_positive,
        "clean_evidence_ratio": None if clean_result is None else clean_result.get(
            "gain", clean_result.get("detection_score", clean_result.get("evidence_ratio"))
        ),
        "verdicts": verdicts,
    }


def evaluate_model(model_dir: Path) -> list[dict[str, Any]]:
    run_root = model_dir / "detection"
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = load_json(manifest_path)
    records = artifact_records(manifest)
    model = model_dir.name

    weighted_baseline = load_record(
        run_root,
        find_record(records, kind="capture", producer="weighted-spectrum", edit_method=None),
    )
    profiles = baseline_profiles(weighted_baseline)
    rows: list[dict[str, Any]] = []

    for producer, strategy in (() if detect_rome_presence_blind is None else (
        ("rome-presence-blind-peak", "peak"),
        ("rome-presence-blind-footprint", "footprint"),
    )):
        artifact = load_record(
            run_root,
            find_record(records, kind="analysis", producer=producer, edit_method="rome"),
        )
        positive = summarize_positive_cases(artifact)
        config = artifact.get("config", {}) if artifact else {}
        clean = None
        if profiles:
            clean = detect_rome_presence_blind(
                profiles,
                trim_first=int(config.get("trim_first", 5)),
                trim_last=int(config.get("trim_last", 5)),
                strategy=strategy,
            )
        rows.append(
            analysis_row(
                model=model,
                detector=producer,
                clean_result=clean,
                **positive,
            )
        )

    delta_artifact = load_record(
        run_root,
        find_record(records, kind="analysis", producer="rome-presence-delta", edit_method="rome"),
    )
    delta = summarize_positive_cases(delta_artifact)
    rows.append(
        analysis_row(
            model=model,
            detector="rome-presence-delta",
            clean_result=None,
            **delta,
        )
    )

    spectral_artifact = load_record(
        run_root,
        find_record(records, kind="analysis", producer="spectral", edit_method="rome"),
    )
    spectral_baseline = load_record(
        run_root,
        find_record(records, kind="capture", producer="spectral", edit_method=None),
    )
    spectral_clean_data = None
    if spectral_artifact and spectral_baseline:
        baseline_cases = complete_cases(spectral_baseline)
        if baseline_cases:
            spectral_clean_data = replay_spectral(
                dict(baseline_cases[0].get("data", {})),
                dict(spectral_artifact.get("config", {})),
            )

    spectral_cases = complete_cases(spectral_artifact)
    for log_transform in (False, True):
        detector = "spectral-hybrid-universal-log" if log_transform else "spectral-hybrid-universal-raw"
        decisions = [
            spectral_presence(case.get("data", {}), log_transform=log_transform)
            for case in spectral_cases
        ]
        clean = (
            spectral_presence(spectral_clean_data, log_transform=log_transform)
            if spectral_clean_data is not None
            else None
        )
        rows.append(
            analysis_row(
                model=model,
                detector=detector,
                positive_cases=len(decisions),
                true_positives=sum(bool(decision["is_rome_like"]) for decision in decisions),
                clean_result=clean,
                verdicts={
                    "rome_like": sum(bool(decision["is_rome_like"]) for decision in decisions),
                    "no_universal_outlier": sum(not bool(decision["is_rome_like"]) for decision in decisions),
                },
            )
        )
    return rows


def _experiment_decisions(artifact: dict[str, Any] | None, identifier: str) -> tuple[list[dict[str, Any]], int]:
    decisions: list[dict[str, Any]] = []
    unavailable = 0
    for case in (artifact or {}).get("cases", []):
        if case.get("status") != "complete":
            unavailable += 1
            continue
        value = case.get("data", {}).get("experiments", {}).get(identifier)
        if isinstance(value, dict) and isinstance(value.get("is_rome_like"), bool):
            decisions.append(value)
        else:
            unavailable += 1
    return decisions, unavailable


def evaluate_structural_run(run_root: Path) -> list[dict[str, Any]]:
    """Evaluate manifest-backed ROME experiments without loading a model."""
    manifest = load_json(run_root / "manifest.json")
    records = artifact_records(manifest)
    rows: list[dict[str, Any]] = []
    method_records = [
        record for record in records
        if record.get("kind") == "analysis"
        and record.get("producer") in EXPERIMENT_PRODUCERS
        and record.get("edit_method") == "rome"
    ]
    for record in method_records:
        artifact = load_record(run_root, record)
        if artifact is None:
            continue
        matching_baselines = [
            candidate for candidate in records
            if candidate.get("kind") == "analysis"
            and candidate.get("producer") == record.get("producer")
            and candidate.get("edit_method") is None
            and candidate.get("model") == record.get("model")
            and candidate.get("plan_id") == record.get("plan_id")
            and candidate.get("config_hash") == record.get("config_hash")
        ]
        baseline = load_record(run_root, matching_baselines[-1]) if matching_baselines else None
        identifiers = sorted({
            str(identifier)
            for case in artifact.get("cases", [])
            for identifier in case.get("data", {}).get("experiments", {})
        })
        for identifier in identifiers:
            decisions, unavailable = _experiment_decisions(artifact, identifier)
            clean_decisions, clean_unavailable = _experiment_decisions(baseline, identifier)
            if not decisions and not clean_decisions:
                continue
            clean = clean_decisions[0] if clean_decisions else None
            rows.append(analysis_row(
                model=str(record.get("model")),
                detector=identifier,
                positive_cases=len(decisions),
                true_positives=sum(bool(decision["is_rome_like"]) for decision in decisions),
                clean_result=clean,
                verdicts={
                    "rome_like": sum(bool(decision["is_rome_like"]) for decision in decisions),
                    "not_rome_like": sum(not bool(decision["is_rome_like"]) for decision in decisions),
                    "unavailable": unavailable,
                    "clean_unavailable": clean_unavailable,
                },
            ))
    return rows


def collect_structural_specimens(run_root: Path) -> list[dict[str, Any]]:
    """Normalize old and new manifest analyses into one specimen contract."""
    manifest = load_json(run_root / "manifest.json")
    records = artifact_records(manifest)
    execution_by_key: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for record in records:
        if record.get("kind") != "execution":
            continue
        artifact = load_record(run_root, record)
        if artifact is not None:
            execution_by_key[(
                str(record.get("model")),
                str(record.get("plan_id")),
                record.get("edit_method"),
            )] = artifact

    specimens: list[dict[str, Any]] = []
    seen_clean: set[tuple[str, str]] = set()
    capture_cache: dict[tuple[str, str, str | None, str], dict[str, Any]] = {}
    for capture_record in records:
        if capture_record.get("kind") != "capture":
            continue
        producer = str(capture_record.get("producer"))
        if producer not in set(EXPERIMENT_CAPTURE.values()):
            continue
        loaded = load_record(run_root, capture_record)
        if loaded is not None:
            capture_cache[(
                str(capture_record.get("model")),
                str(capture_record.get("plan_id")),
                capture_record.get("edit_method"),
                producer,
            )] = loaded
    analysis_records = [
        record for record in records
        if record.get("kind") == "analysis" and record.get("producer") in EXPERIMENT_PRODUCERS
    ]
    for record in analysis_records:
        artifact = load_record(run_root, record)
        if artifact is None:
            continue
        model = str(record.get("model"))
        plan_id = str(record.get("plan_id"))
        edit_method = record.get("edit_method")
        capture_producer = EXPERIMENT_CAPTURE[str(record.get("producer"))]
        capture_artifact = capture_cache.get((model, plan_id, edit_method, capture_producer), {})
        capture_cases = {str(case.get("case_id")): case for case in capture_artifact.get("cases", [])}
        execution = execution_by_key.get((model, plan_id, edit_method), {})
        execution_cases = {str(case.get("case_id")): case for case in execution.get("cases", [])}
        ground_truth_layer = execution.get("summary", {}).get("target_layer")
        for case in artifact.get("cases", []):
            case_id = str(case.get("case_id"))
            experiments = case.get("data", {}).get("experiments", {}) if case.get("status") == "complete" else {}
            for identifier, decision in experiments.items():
                identifier = str(identifier)
                if edit_method is None:
                    deduplication_key = (model, identifier)
                    if deduplication_key in seen_clean:
                        continue
                    seen_clean.add(deduplication_key)
                execution_case = execution_cases.get(case_id, {})
                capture_data = capture_cases.get(case_id, {}).get("data", {})
                edit = execution_case.get("edit", {}) if isinstance(execution_case.get("edit"), dict) else {}
                complete = isinstance(decision, dict) and isinstance(decision.get("is_rome_like"), bool)
                candidate = decision.get("candidate_layer") if isinstance(decision, dict) else None
                detected = decision.get("is_rome_like") if complete else None
                identity = f"{model}:{plan_id}:{case_id}:{'rome' if edit_method else 'clean'}"
                specimens.append({
                    "specimen_id": identity,
                    "experiment_id": identifier,
                    "decision_version": decision.get("decision_version") if isinstance(decision, dict) else None,
                    "status": "complete" if complete else "unavailable",
                    "rome_compatible_detected": detected,
                    "candidate_layer": candidate,
                    "original_localizer_layer": (
                        decision.get("original_localizer_layer") if isinstance(decision, dict) else None
                    ),
                    "eligible_layers": decision.get("eligible_layers", []) if isinstance(decision, dict) else [],
                    "layer_scores": decision.get("layer_scores", {}) if isinstance(decision, dict) else {},
                    "background_cost": decision.get("background_cost") if isinstance(decision, dict) else None,
                    "anomaly_cost": decision.get("anomaly_cost") if isinstance(decision, dict) else None,
                    "gain": decision.get("gain") if isinstance(decision, dict) else None,
                    "diagnostics": decision.get("diagnostics", {}) if isinstance(decision, dict) else {},
                    "runtime": capture_data.get("runtime", {}),
                    "provenance": capture_data.get("provenance", {}),
                    "model": model,
                    "base_lineage": model,
                    "ground_truth_positive": edit_method is not None,
                    "ground_truth_layer": ground_truth_layer,
                    "negative_category": None if edit_method is not None else "clean",
                    "edit_applied": edit_method is not None,
                    "behavioral_success": edit.get("success"),
                    "localization_exact": (
                        bool(candidate == ground_truth_layer)
                        if candidate is not None and ground_truth_layer is not None else None
                    ),
                    "localization_within_one": (
                        bool(abs(int(candidate) - int(ground_truth_layer)) <= 1)
                        if candidate is not None and ground_truth_layer is not None else None
                    ),
                })
    return specimens


def summarize_specimens(specimens: list[dict[str, Any]]) -> dict[str, Any]:
    """Build confusion, availability, localization, and overlapping error sets."""
    per_experiment: dict[str, Any] = {}
    errors: dict[str, set[str]] = {}
    outcomes: dict[str, dict[str, bool]] = {}
    for experiment in sorted({str(item["experiment_id"]) for item in specimens}):
        selected = [item for item in specimens if item["experiment_id"] == experiment]
        complete = [item for item in selected if item["status"] == "complete"]
        tp = sum(item["ground_truth_positive"] and item["rome_compatible_detected"] for item in complete)
        fn = sum(item["ground_truth_positive"] and not item["rome_compatible_detected"] for item in complete)
        fp = sum(not item["ground_truth_positive"] and item["rome_compatible_detected"] for item in complete)
        tn = sum(not item["ground_truth_positive"] and not item["rome_compatible_detected"] for item in complete)
        localization = [item for item in complete if item["ground_truth_positive"] and item["candidate_layer"] is not None]
        error_ids = {
            item["specimen_id"]
            for item in complete
            if bool(item["rome_compatible_detected"]) != bool(item["ground_truth_positive"])
        }
        errors[experiment] = error_ids
        outcomes[experiment] = {
            item["specimen_id"]: bool(item["rome_compatible_detected"]) == bool(item["ground_truth_positive"])
            for item in complete
        }
        category_counts: Counter[str] = Counter()
        category_false: Counter[str] = Counter()
        for item in complete:
            if item["ground_truth_positive"]:
                continue
            category = str(item.get("negative_category") or "unknown")
            category_counts[category] += 1
            category_false[category] += int(bool(item["rome_compatible_detected"]))
        per_experiment[experiment] = {
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "availability": len(complete) / len(selected) if selected else None,
            "available": len(complete),
            "unavailable": len(selected) - len(complete),
            "applied_edits": sum(bool(item["edit_applied"]) for item in selected),
            "behavioral_successes": sum(item.get("behavioral_success") is True for item in selected),
            "localization_exact": sum(item["localization_exact"] is True for item in localization),
            "localization_within_one": sum(item["localization_within_one"] is True for item in localization),
            "localization_cases": len(localization),
            "runtime_seconds": {
                "total": sum(float(item.get("runtime", {}).get("seconds", 0.0)) for item in complete),
                "measured_cases": sum("seconds" in item.get("runtime", {}) for item in complete),
            },
            "maximum_reported_storage_bytes": max(
                (
                    int(item.get("runtime", {}).get("temporary_storage_bytes", 0))
                    or int(item.get("runtime", {}).get("source_tensor_bytes", 0))
                    for item in complete
                ),
                default=0,
            ),
            "negative_categories": {
                category: {
                    "count": count,
                    "false_positives": category_false[category],
                }
                for category, count in sorted(category_counts.items())
            },
            "error_specimens": sorted(error_ids),
        }
    overlap = {
        first: {
            second: len(errors[first] & errors[second])
            for second in sorted(errors)
        }
        for first in sorted(errors)
    }
    reference = outcomes.get("original-v3-relative-b0-v1", {})
    changes: dict[str, Any] = {}
    if reference:
        for experiment, experiment_outcomes in outcomes.items():
            if experiment == "original-v3-relative-b0-v1":
                continue
            shared = set(reference) & set(experiment_outcomes)
            changes[experiment] = {
                "fixes": sorted(identity for identity in shared if not reference[identity] and experiment_outcomes[identity]),
                "regressions": sorted(identity for identity in shared if reference[identity] and not experiment_outcomes[identity]),
            }
    return {
        "per_experiment": per_experiment,
        "overlapping_error_counts": overlap,
        "changes_from_original_relative": changes,
    }


def historical_local_prominence(input_root: Path) -> dict[str, Any]:
    """Keep the published hard-negative baseline in a separate named cohort."""
    expected = {"tp": 43, "fn": 51, "fp": 50, "tn": 155, "positive_cases": 94, "negative_cases": 205}
    candidates = list(input_root.rglob("evaluation-with-hard-negatives.json")) if input_root.is_dir() else []
    return {
        "cohort": "historical-v3-hard-negatives",
        "detector": "local-prominence",
        "counts": expected,
        "source_artifact_found": bool(candidates),
        "source_artifact": str(candidates[0]) if candidates else None,
        "pooled_with_relative_experiments": False,
    }


def _profile_decisions(capture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return evaluate_profile_experiments(
        capture.get("profiles", {}),
        experiments=tuple(PROFILE_EXPERIMENTS),
        eligible_only=capture.get("eligible_layers"),
    )


def evaluate_archived_profiles(input_root: Path) -> list[dict[str, Any]]:
    """Replay scalar experiments on the exposed simple-Gram development artifacts."""
    hard_dir = (
        input_root
        if input_root.name == "rome-simple-gram-hard-negatives-v1"
        else input_root / "rome-simple-gram-hard-negatives-v1"
    )
    search_root = input_root.parent if input_root.name == "rome-simple-gram-hard-negatives-v1" else input_root
    if not hard_dir.is_dir():
        return []

    positives: dict[str, list[tuple[dict[str, Any], bool | None]]] = {}
    negatives: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for path in search_root.glob("rome-simple-gram-n20-*-v1/plans/*/*/methods/rome/captures/simple-gram-experiment.json"):
        artifact = load_json(path)
        family = str(artifact.get("run", {}).get("model", "unknown"))
        execution_path = path.parent.parent / "execution.json"
        execution = load_json(execution_path) if execution_path.is_file() else {}
        success_by_case = {
            str(case.get("case_id")): (
                bool(case.get("edit", {}).get("success"))
                if isinstance(case.get("edit"), dict) and "success" in case.get("edit", {})
                else None
            )
            for case in execution.get("cases", [])
        }
        for case in complete_cases(artifact):
            capture = case.get("data", {})
            if isinstance(capture.get("profiles"), dict):
                positives.setdefault(family, []).append(
                    (capture, success_by_case.get(str(case.get("case_id"))))
                )

    hard_families: set[str] = set()
    for path in sorted(hard_dir.glob("*.json")):
        if path.name.startswith("evaluation-"):
            continue
        payload = load_json(path)
        family = str(payload.get("model_key", path.stem))
        hard_families.add(family)
        for record in payload.get("records", []):
            capture = record.get("capture", {})
            category = str(record.get("negative_category", "unknown"))
            if isinstance(capture.get("profiles"), dict):
                negatives.setdefault(family, []).append((category, capture))

    for path in search_root.glob("rome-simple-gram-n20-*-v1/plans/*/*/baseline/captures/simple-gram-experiment.json"):
        artifact = load_json(path)
        family = str(artifact.get("run", {}).get("model", "unknown"))
        if family in hard_families:
            continue
        cases = complete_cases(artifact)
        if cases and isinstance(cases[0].get("data", {}).get("profiles"), dict):
            negatives.setdefault(family, []).append(("standalone_clean", cases[0]["data"]))

    rows: list[dict[str, Any]] = []
    for family in sorted(set(positives) | set(negatives)):
        positive_results: dict[str, list[dict[str, Any]]] = {name: [] for name in PROFILE_EXPERIMENTS}
        negative_results: dict[str, list[tuple[str, dict[str, Any]]]] = {name: [] for name in PROFILE_EXPERIMENTS}
        unavailable: Counter[str] = Counter()
        applied_count = len(positives.get(family, []))
        efficacy_failed = sum(success is False for _capture, success in positives.get(family, []))
        for capture, success in positives.get(family, []):
            if success is False:
                continue
            try:
                decisions = _profile_decisions(capture)
            except (ValueError, KeyError, TypeError):
                unavailable["positive"] += 1
                continue
            for name, decision in decisions.items():
                if isinstance(decision.get("is_rome_like"), bool):
                    positive_results[name].append(decision)
                else:
                    unavailable[f"positive:{name}"] += 1
        for category, capture in negatives.get(family, []):
            try:
                decisions = _profile_decisions(capture)
            except (ValueError, KeyError, TypeError):
                unavailable[category] += 1
                continue
            for name, decision in decisions.items():
                if isinstance(decision.get("is_rome_like"), bool):
                    negative_results[name].append((category, decision))
                else:
                    unavailable[f"{category}:{name}"] += 1

        for identifier in PROFILE_EXPERIMENTS:
            positive = positive_results[identifier]
            negative = negative_results[identifier]
            false_by_category = Counter(
                category for category, decision in negative if bool(decision["is_rome_like"])
            )
            counts_by_category = Counter(category for category, _decision in negative)
            row = analysis_row(
                model=family,
                detector=identifier,
                positive_cases=len(positive),
                true_positives=sum(bool(decision["is_rome_like"]) for decision in positive),
                clean_result=None,
                verdicts={
                    "false_positives_by_category": dict(false_by_category),
                    "negative_counts_by_category": dict(counts_by_category),
                    "unavailable": dict(unavailable),
                },
            )
            row["clean_cases"] = len(negative)
            row["clean_false_positives"] = sum(false_by_category.values())
            row["applied_cases"] = applied_count
            row["efficacy_failed_cases"] = efficacy_failed
            rows.append(row)
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for detector in sorted({str(row["detector"]) for row in rows}):
        selected = [row for row in rows if row["detector"] == detector]
        positives = sum(int(row["positive_cases"]) for row in selected)
        true_positives = sum(int(row["true_positives"]) for row in selected)
        clean_cases = sum(int(row["clean_cases"]) for row in selected)
        false_positives = sum(int(row["clean_false_positives"]) for row in selected)
        sensitivities = [float(row["sensitivity"]) for row in selected if row["sensitivity"] is not None]
        clean_specificities = [
            1.0 - int(row["clean_false_positives"]) / int(row["clean_cases"])
            for row in selected if int(row["clean_cases"])
        ]
        category_counts: Counter[str] = Counter()
        category_false_positives: Counter[str] = Counter()
        for row in selected:
            verdicts = row.get("verdicts", {})
            category_counts.update(verdicts.get("negative_counts_by_category", {}))
            category_false_positives.update(verdicts.get("false_positives_by_category", {}))
        family_totals: dict[str, dict[str, int]] = {}
        for row in selected:
            total = family_totals.setdefault(
                str(row["model"]), {"positive": 0, "tp": 0, "negative": 0, "fp": 0}
            )
            total["positive"] += int(row["positive_cases"])
            total["tp"] += int(row["true_positives"])
            total["negative"] += int(row["clean_cases"])
            total["fp"] += int(row["clean_false_positives"])
        interval_sensitivity = None
        interval_specificity = None
        if family_totals:
            families = sorted(family_totals)
            seed = int.from_bytes(hashlib.sha256(detector.encode()).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            boot_sensitivity: list[float] = []
            boot_specificity: list[float] = []
            for _ in range(1000):
                sampled = rng.choice(families, size=len(families), replace=True)
                positive_n = sum(family_totals[str(family)]["positive"] for family in sampled)
                positive_tp = sum(family_totals[str(family)]["tp"] for family in sampled)
                negative_n = sum(family_totals[str(family)]["negative"] for family in sampled)
                negative_fp = sum(family_totals[str(family)]["fp"] for family in sampled)
                if positive_n:
                    boot_sensitivity.append(positive_tp / positive_n)
                if negative_n:
                    boot_specificity.append(1.0 - negative_fp / negative_n)
            if boot_sensitivity:
                interval_sensitivity = [float(value) for value in np.quantile(boot_sensitivity, [0.025, 0.975])]
            if boot_specificity:
                interval_specificity = [float(value) for value in np.quantile(boot_specificity, [0.025, 0.975])]
        aggregates.append(
            {
                "detector": detector,
                "models": len({str(row["model"]) for row in selected}),
                "positive_cases": positives,
                "true_positives": true_positives,
                "sensitivity": true_positives / positives if positives else None,
                "clean_cases": clean_cases,
                "clean_false_positives": false_positives,
                "clean_false_positive_rate": false_positives / clean_cases if clean_cases else None,
                "equal_family_macro_sensitivity": sum(sensitivities) / len(sensitivities) if sensitivities else None,
                "equal_family_macro_specificity": (
                    sum(clean_specificities) / len(clean_specificities) if clean_specificities else None
                ),
                "worst_family_sensitivity": min(sensitivities) if sensitivities else None,
                "worst_family_specificity": min(clean_specificities) if clean_specificities else None,
                "lineage_bootstrap_95_sensitivity": interval_sensitivity,
                "lineage_bootstrap_95_specificity": interval_specificity,
                "per_negative_category": {
                    category: {
                        "count": count,
                        "false_positives": category_false_positives[category],
                        "false_positive_rate": category_false_positives[category] / count,
                    }
                    for category, count in sorted(category_counts.items())
                },
            }
        )
    return aggregates


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def markdown(
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    input_root: Path,
    *,
    specimen_summary: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# Offline binary ROME-presence evaluation",
        "",
        f"Source: `{input_root.as_posix()}`",
        "",
        "Completed efficacy-successful ROME cases are treated as positives when efficacy metadata is available.",
        "Negative categories and unavailable cases are retained from the input artifacts.",
        "",
        "## Aggregate",
        "",
        "| Detector | Models | ROME TP / cases | Sensitivity | Clean FP / baselines | Clean FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['detector']} | {row['models']} | {row['true_positives']}/{row['positive_cases']} | "
            f"{pct(row['sensitivity'])} | {row['clean_false_positives']}/{row['clean_cases']} | "
            f"{pct(row['clean_false_positive_rate'])} |"
        )
    category_rows = [
        (row["detector"], category, values)
        for row in aggregates
        for category, values in row.get("per_negative_category", {}).items()
    ]
    if category_rows:
        lines.extend([
            "",
            "## Negative categories",
            "",
            "| Detector | Category | FP / cases | FPR |",
            "|---|---|---:|---:|",
        ])
        for detector, category, values in category_rows:
            lines.append(
                f"| {detector} | {category} | {values['false_positives']}/{values['count']} | "
                f"{pct(values['false_positive_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## Per model",
            "",
            "| Model | Detector | ROME TP / cases | Sensitivity | Clean verdict | Clean evidence ratio |",
            "|---|---|---:|---:|---|---:|",
        ]
    )
    if specimen_summary and specimen_summary.get("per_experiment"):
        lines.extend([
            "",
            "## Specimen-level relative decisions",
            "",
            "| Experiment | TP | FN | FP | TN | Available | Exact / localized | Within one / localized |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for identifier, values in sorted(specimen_summary["per_experiment"].items()):
            lines.append(
                f"| {identifier} | {values['tp']} | {values['fn']} | {values['fp']} | {values['tn']} | "
                f"{values['available']}/{values['available'] + values['unavailable']} | "
                f"{values['localization_exact']}/{values['localization_cases']} | "
                f"{values['localization_within_one']}/{values['localization_cases']} |"
            )
    for row in sorted(rows, key=lambda item: (str(item["model"]), str(item["detector"]))):
        clean = row["clean_is_rome_like"]
        clean_label = "—" if clean is None else ("ROME-like" if clean else "clean")
        ratio = row["clean_evidence_ratio"]
        ratio_label = "—" if ratio is None else f"{float(ratio):.3f}"
        lines.append(
            f"| {row['model']} | {row['detector']} | {row['true_positives']}/{row['positive_cases']} | "
            f"{pct(row['sensitivity'])} | {clean_label} | {ratio_label} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `*-mdl-v1` rows are the named score-profile experiments implemented by the ROME research localizer.",
            "- `rome-presence-blind-peak` and `blind-footprint` are existing suspect-only rules over saved weighted-spectrum profiles.",
            "- `spectral-hybrid-universal-*` applies the same median/MAD universal extreme test to saved `rome_hybrid_scores`; it is an offline candidate, not a production detector.",
            "- `rome-presence-delta` needs a matching clean checkpoint. No independent clean-delta negative artifact exists; baseline-vs-itself would be negative by construction, so no clean FPR is reported.",
            "- Results are exposed development evidence. Categories absent from the input artifacts are not inferred or counted as clean.",
            "",
        ]
    )
    return "\n".join(lines)


def csv_value(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    specimens: list[dict[str, Any]] = []
    if (args.input_root / "manifest.json").is_file():
        rows.extend(evaluate_structural_run(args.input_root))
        specimens.extend(collect_structural_specimens(args.input_root))
    else:
        rows.extend(evaluate_archived_profiles(args.input_root))
        if not rows:
            for model_dir in sorted(path for path in args.input_root.iterdir() if path.is_dir()):
                rows.extend(evaluate_model(model_dir))
    if not rows:
        raise SystemExit(f"No usable manifests below {args.input_root}")

    aggregates = aggregate_rows(rows)
    specimen_summary = summarize_specimens(specimens)
    historical = historical_local_prominence(args.input_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps({
            "models": sorted({row['model'] for row in rows}),
            "aggregate": aggregates,
            "rows": rows,
            "historical_cohorts": [historical],
            "specimen_summary": specimen_summary,
            "specimens": specimens,
        }, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        markdown(rows, aggregates, args.input_root, specimen_summary=specimen_summary),
        encoding="utf-8",
    )
    (args.output_dir / "specimens.json").write_text(json.dumps(specimens, indent=2), encoding="utf-8")
    with (args.output_dir / "per-model.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: csv_value(value) for key, value in row.items()} for row in rows)
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
