#!/usr/bin/env python3
"""Evaluate versioned ROME math recaptures without touching source artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from omegaconf import OmegaConf

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from src.structural.experiments.rome_math_ablation import (
    CANDIDATE_FIELDS,
    EVALUATION_SCHEMA_VERSION,
    evaluate_capture_data,
)


ALLOWED_SPLITS = {"development", "held_out_family", "final_frozen"}
SELECTION_CANDIDATES = ("M0", "M1", "M2", "M3")
NONINFERIORITY_MARGIN = 0.025
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20_260_728


def enumerate_artifacts(root: Path) -> list[Path]:
    """Use ripgrep's ignored-file-aware inventory, as required by the protocol."""
    result = subprocess.run(
        ["rg", "--no-ignore", "--files", str(root)],
        check=True,
        text=True,
        capture_output=True,
    )
    return sorted(
        Path(line) for line in result.stdout.splitlines() if line.endswith("/captures/rome-math-ablation.json")
    )


def enumerate_run_roots(roots: Iterable[Path]) -> list[Path]:
    """Enumerate only explicitly supplied run roots without cohort mixing."""
    return sorted(
        {
            path
            for root in roots
            for path in enumerate_artifacts(root)
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(
    path: Path,
) -> tuple[dict[object, dict[str, str]], dict[str, Any]]:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Recapture manifest must be a mapping")
    if raw.get("schema_version") != "rome-math-ablation-recapture-v1":
        raise ValueError("Unsupported recapture manifest schema")
    models: dict[object, dict[str, str]] = {}
    for record in raw.get("models", []):
        if not isinstance(record, dict):
            raise ValueError("Every recapture model entry must be a mapping")
        model = str(record["model"])
        run_id = record.get("run_id")
        split = str(record["split"])
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Unsupported split {split!r} for {model}")
        key: object = (str(run_id), model) if run_id is not None else model
        models[key] = {
            "family": str(record["family"]),
            "split": split,
        }
    return models, raw


def _capture_key(document: Mapping[str, Any]) -> tuple[str, str, str, str | None]:
    run = document["run"]
    return (
        str(run["run_id"]),
        str(run["model"]),
        str(run["plan_id"]),
        None if run.get("edit_method") is None else str(run["edit_method"]),
    )


def _case_map(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): dict(case) for case in document.get("cases", [])}


def _execution_document(capture_path: Path) -> dict[str, Any]:
    execution_path = capture_path.parent.parent / "execution.json"
    if not execution_path.exists():
        return {}
    return _load_json(execution_path)


def collect_cases(
    artifact_paths: Iterable[Path],
    *,
    model_metadata: Mapping[object, Mapping[str, str]],
    blind_candidate: str,
    blind_cutoff: float | None,
) -> list[dict[str, Any]]:
    captures: dict[tuple[str, str, str, str | None], tuple[Path, dict[str, Any]]] = {}
    for path in artifact_paths:
        document = _load_json(path)
        if document.get("kind") != "capture" or document.get("producer") != "rome-math-ablation":
            continue
        captures[_capture_key(document)] = (path, document)

    cases: list[dict[str, Any]] = []
    ordered_captures = sorted(
        captures.items(),
        key=lambda item: tuple("" if value is None else value for value in item[0]),
    )
    for key, (path, suspect_document) in ordered_captures:
        run_id, model, plan_id, edit_method = key
        if edit_method is None:
            continue
        metadata = model_metadata.get((run_id, model)) or model_metadata.get(model)
        if metadata is None:
            raise ValueError(f"Recapture manifest does not declare model {model!r}")
        baseline_pair = captures.get((run_id, model, plan_id, None))
        if baseline_pair is None:
            raise ValueError(f"Missing baseline capture for {run_id}/{model}/{plan_id}")
        baseline_path, baseline_document = baseline_pair
        baseline_cases = _case_map(baseline_document)
        baseline_case = baseline_cases.get("baseline")
        if baseline_case is None or baseline_case.get("status") != "complete":
            raise ValueError(f"Baseline capture is not complete: {baseline_path}")
        execution = _execution_document(path)
        execution_summary = dict(execution.get("summary", {}))
        execution_cases = _case_map(execution)
        target_layer = execution_summary.get("target_layer")
        num_layers = execution_summary.get("num_layers")

        for case in suspect_document.get("cases", []):
            case_id = str(case["case_id"])
            edit_success = execution_cases.get(case_id, {}).get("edit", {}).get("success")
            common = {
                "source_run": run_id,
                "source_artifact": str(path),
                "source_baseline_artifact": str(baseline_path),
                "model_identifier": model,
                "family": metadata["family"],
                "split": metadata["split"],
                "plan_id": plan_id,
                "edit_method": edit_method,
                "case_id": case_id,
                "target_layer": (None if target_layer is None else int(target_layer)),
                "num_layers": None if num_layers is None else int(num_layers),
                "edit_success": (bool(edit_success) if isinstance(edit_success, bool) else None),
            }
            if case.get("status") != "complete":
                cases.append(
                    {
                        **common,
                        "status": str(case.get("status", "unavailable")),
                        "error": case.get("error"),
                        "candidates": {},
                        "binary": {},
                        "localized_layer": None,
                        "presence_peak_layer": None,
                    }
                )
                continue
            try:
                evaluated = evaluate_capture_data(
                    baseline_case["data"],
                    case["data"],
                    blind_candidate=blind_candidate,
                    blind_cutoff=blind_cutoff,
                )
            except Exception as exc:
                cases.append(
                    {
                        **common,
                        "status": "error",
                        "error": str(exc),
                        "candidates": {},
                        "binary": {},
                        "localized_layer": None,
                        "presence_peak_layer": None,
                    }
                )
                continue
            cases.append(
                {
                    **common,
                    "status": "complete",
                    "error": None,
                    "eligible_layers": evaluated["eligible_layers"],
                    "candidates": evaluated["candidates"],
                    "binary": evaluated["binary"],
                    "localized_layer": evaluated["localized_layer"],
                    "presence_peak_layer": evaluated["presence_peak_layer"],
                    "runtime_seconds": float(case["data"].get("runtime_seconds", 0.0)),
                    "estimated_peak_bytes": int(case["data"].get("estimated_peak_bytes", 0)),
                }
            )
    return cases


def _candidate_metrics(
    cases: list[dict[str, Any]],
    candidate: str,
) -> dict[str, Any]:
    inventory = [case for case in cases if case.get("target_layer") is not None]
    available = [case for case in inventory if case["status"] == "complete" and candidate in case["candidates"]]
    exact = sum(case["candidates"][candidate]["selected_layer"] == case["target_layer"] for case in available)
    successful = [case for case in available if case.get("edit_success") is True]
    successful_exact = sum(
        case["candidates"][candidate]["selected_layer"] == case["target_layer"] for case in successful
    )
    within_one = sum(
        abs(case["candidates"][candidate]["selected_layer"] - case["target_layer"]) <= 1 for case in available
    )

    def grouped_metrics(field: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for value in sorted({str(case[field]) for case in inventory}):
            group_inventory = [case for case in inventory if str(case[field]) == value]
            group_cases = [case for case in available if str(case[field]) == value]
            group_successful = [case for case in group_cases if case.get("edit_success") is True]
            group_exact = sum(
                case["candidates"][candidate]["selected_layer"] == case["target_layer"] for case in group_cases
            )
            group_successful_exact = sum(
                case["candidates"][candidate]["selected_layer"] == case["target_layer"] for case in group_successful
            )
            groups[value] = {
                "correct": group_exact,
                "correct_successful_edits": group_successful_exact,
                "available": len(group_cases),
                "successful_edits": len(group_successful),
                "total": len(group_inventory),
                "requested_accuracy": (group_exact / len(group_inventory) if group_inventory else None),
                "successful_edit_accuracy": (
                    group_successful_exact / len(group_successful) if group_successful else None
                ),
            }
        return groups

    per_family = grouped_metrics("family")
    per_model = grouped_metrics("model_identifier")
    family_accuracies = [
        float(record["requested_accuracy"])
        for record in per_family.values()
        if record["requested_accuracy"] is not None
    ]
    model_accuracies = [
        float(record["requested_accuracy"]) for record in per_model.values() if record["requested_accuracy"] is not None
    ]
    margins = [float(case["candidates"][candidate]["margin"]) for case in available]
    return {
        "correct": exact,
        "correct_successful_edits": successful_exact,
        "available": len(available),
        "successful_edits": len(successful),
        "total": len(inventory),
        "exact_accuracy": exact / len(inventory) if inventory else None,
        "requested_accuracy": exact / len(inventory) if inventory else None,
        "successful_edit_accuracy": (successful_exact / len(successful) if successful else None),
        "within_one_accuracy": within_one / len(inventory) if inventory else None,
        "macro_family_accuracy": (sum(family_accuracies) / len(family_accuracies) if family_accuracies else None),
        "macro_model_accuracy": sum(model_accuracies) / len(model_accuracies) if model_accuracies else None,
        "per_family": per_family,
        "per_model": per_model,
        "mean_margin": sum(margins) / len(margins) if margins else None,
        "failures": [
            {
                "source_run": case["source_run"],
                "model_identifier": case["model_identifier"],
                "family": case["family"],
                "case_id": case["case_id"],
                "status": case["status"],
                "target_layer": case["target_layer"],
                "selected_layer": case["candidates"].get(candidate, {}).get("selected_layer"),
                "target_depth": _target_depth(case),
            }
            for case in inventory
            if case["status"] != "complete"
            or candidate not in case["candidates"]
            or case["candidates"][candidate]["selected_layer"] != case["target_layer"]
        ],
    }


def _b0_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        case
        for case in cases
        if case["status"] == "complete"
        and case.get("edit_success") is True
        and isinstance(case.get("binary", {}).get("B0", {}).get("is_rome_like"), bool)
    ]
    detected = sum(bool(case["binary"]["B0"]["is_rome_like"]) for case in successful)
    failures = [
        {
            "source_run": case["source_run"],
            "model_identifier": case["model_identifier"],
            "family": case["family"],
            "case_id": case["case_id"],
            "target_layer": case["target_layer"],
            "selected_layer": case["binary"]["B0"].get("selected_layer"),
            "verdict": case["binary"]["B0"].get("verdict"),
        }
        for case in successful
        if not case["binary"]["B0"]["is_rome_like"]
    ]
    return {
        "scope": "successful ROME edits only",
        "claim": "ROME-compatible low-rank edit sensitivity; no specificity or ROME attribution",
        "true_positive": detected,
        "successful_edits_evaluated": len(successful),
        "sensitivity": detected / len(successful) if successful else None,
        "failures": failures,
    }


def _target_depth(case: Mapping[str, Any]) -> str:
    target = case.get("target_layer")
    count = case.get("num_layers")
    if target is None or count in (None, 0):
        return "unknown"
    relative = float(target) / max(1.0, float(count) - 1.0)
    if relative < 1.0 / 3.0:
        return "early"
    if relative < 2.0 / 3.0:
        return "middle"
    return "late"


def _disagreements(cases: list[dict[str, Any]]) -> dict[str, int]:
    candidates = list(CANDIDATE_FIELDS)
    output: dict[str, int] = {}
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            output[f"{left}__{right}"] = sum(
                case["candidates"][left]["selected_layer"] != case["candidates"][right]["selected_layer"]
                for case in cases
                if case["status"] == "complete" and left in case["candidates"] and right in case["candidates"]
            )
    return output


def _selection_analysis(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the predeclared equal-family paired non-inferiority rule."""
    inventory = [case for case in cases if case.get("target_layer") is not None]
    families = sorted({str(case["family"]) for case in inventory})
    if not inventory or not families:
        return {
            "status": "not_evaluated_no_cases",
            "selected_candidate": None,
        }

    correctness: dict[str, dict[str, np.ndarray]] = {}
    for family in families:
        family_cases = [case for case in inventory if str(case["family"]) == family]
        correctness[family] = {}
        for candidate in SELECTION_CANDIDATES:
            correctness[family][candidate] = np.asarray(
                [
                    float(
                        case.get("status") == "complete"
                        and case.get("candidates", {}).get(candidate, {}).get("selected_layer")
                        == case["target_layer"]
                    )
                    for case in family_cases
                ],
                dtype=np.float64,
            )

    observed = {
        candidate: float(
            np.mean(
                [
                    correctness[family][candidate].mean()
                    for family in families
                ]
            )
        )
        for candidate in SELECTION_CANDIDATES
    }
    best = max(
        SELECTION_CANDIDATES,
        key=lambda candidate: (
            observed[candidate],
            -SELECTION_CANDIDATES.index(candidate),
        ),
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = np.empty(
        (BOOTSTRAP_ITERATIONS, len(SELECTION_CANDIDATES)),
        dtype=np.float64,
    )
    family_count = len(families)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        family_draw = rng.integers(0, family_count, size=family_count)
        cluster_differences = np.empty(
            (family_count, len(SELECTION_CANDIDATES)),
            dtype=np.float64,
        )
        for draw_index, family_index in enumerate(family_draw):
            family = families[int(family_index)]
            sample_count = int(correctness[family][best].size)
            case_draw = rng.integers(0, sample_count, size=sample_count)
            best_values = correctness[family][best][case_draw]
            for candidate_index, candidate in enumerate(SELECTION_CANDIDATES):
                candidate_values = correctness[family][candidate][case_draw]
                cluster_differences[draw_index, candidate_index] = float(
                    np.mean(candidate_values - best_values)
                )
        differences[iteration] = cluster_differences.mean(axis=0)

    comparisons: dict[str, dict[str, Any]] = {}
    for candidate_index, candidate in enumerate(SELECTION_CANDIDATES):
        lower, upper = np.quantile(
            differences[:, candidate_index],
            (0.025, 0.975),
        )
        comparisons[candidate] = {
            "observed_macro_accuracy": observed[candidate],
            "observed_difference_from_best": observed[candidate] - observed[best],
            "paired_hierarchical_bootstrap_95_ci": [
                float(lower),
                float(upper),
            ],
            "noninferior": bool(float(lower) > -NONINFERIORITY_MARGIN),
        }

    selected = next(
        (
            candidate
            for candidate in SELECTION_CANDIDATES
            if comparisons[candidate]["noninferior"]
        ),
        None,
    )
    return {
        "status": "provisional_development_selection",
        "selection_order": list(SELECTION_CANDIDATES),
        "equal_family_weighting": True,
        "unavailable_cases_count_as_incorrect_for_all_candidates": True,
        "best_observed_candidate": best,
        "selected_candidate": selected,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "bootstrap": {
            "type": "paired_hierarchical_family_then_case",
            "confidence": 0.95,
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "families": families,
        },
        "comparisons": comparisons,
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, Any] = {}
    for split in sorted(ALLOWED_SPLITS):
        split_cases = [case for case in cases if case["split"] == split]
        by_split[split] = {
            "cases_total": len(split_cases),
            "cases_complete": sum(case["status"] == "complete" for case in split_cases),
            "edit_success_count": sum(case.get("edit_success") is True for case in split_cases),
            "cases_unavailable_or_error": sum(case["status"] != "complete" for case in split_cases),
            "localization": {candidate: _candidate_metrics(split_cases, candidate) for candidate in CANDIDATE_FIELDS},
            "B0": _b0_metrics(split_cases),
            "candidate_disagreements": _disagreements(split_cases),
            "candidate_selection": _selection_analysis(split_cases),
            "runtime_seconds": sum(float(case.get("runtime_seconds", 0.0)) for case in split_cases),
            "peak_estimated_bytes": max(
                (int(case.get("estimated_peak_bytes", 0)) for case in split_cases),
                default=0,
            ),
        }
    return by_split


def _csv_rows(cases: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for case in cases:
        if case["status"] != "complete":
            yield {
                **{
                    key: case.get(key)
                    for key in (
                        "source_run",
                        "source_artifact",
                        "model_identifier",
                        "family",
                        "split",
                        "plan_id",
                        "edit_method",
                        "case_id",
                        "status",
                        "target_layer",
                        "edit_success",
                    )
                },
                "candidate_version": "",
                "selected_layer": "",
                "score": "",
                "margin": "",
                "localized_layer": "",
                "presence_peak_layer": "",
                "B0": "",
                "B1": "",
                "B2": "",
            }
            continue
        for candidate, result in case["candidates"].items():
            yield {
                **{
                    key: case.get(key)
                    for key in (
                        "source_run",
                        "source_artifact",
                        "model_identifier",
                        "family",
                        "split",
                        "plan_id",
                        "edit_method",
                        "case_id",
                        "status",
                        "target_layer",
                        "edit_success",
                    )
                },
                "candidate_version": candidate,
                "selected_layer": result["selected_layer"],
                "score": result["score"],
                "margin": result["margin"],
                "localized_layer": case["localized_layer"],
                "presence_peak_layer": case["presence_peak_layer"],
                "B0": bool(case["binary"]["B0"]["is_rome_like"]),
                "B1": (
                    case["binary"]["B1"]["is_rome_like"]
                    if isinstance(case["binary"]["B1"].get("is_rome_like"), bool)
                    else case["binary"]["B1"].get("status", "not_evaluated_uncalibrated")
                ),
                "B2": bool(case["binary"]["B2"]["is_rome_like"]),
            }


def _write_outputs(
    payload: Mapping[str, Any],
    *,
    json_out: Path,
    csv_out: Path,
    force: bool,
) -> None:
    for path in (json_out, csv_out):
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = list(_csv_rows(list(payload["cases"])))
    fieldnames = [
        "source_run",
        "source_artifact",
        "model_identifier",
        "family",
        "split",
        "plan_id",
        "edit_method",
        "case_id",
        "status",
        "target_layer",
        "edit_success",
        "candidate_version",
        "selected_layer",
        "score",
        "margin",
        "localized_layer",
        "presence_peak_layer",
        "B0",
        "B1",
        "B2",
    ]
    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=Path("analysis_out"))
    parser.add_argument(
        "--run-root",
        action="append",
        type=Path,
        help=(
            "Exact run root to evaluate; repeat for multiple per-model roots. "
            "When supplied, --analysis-root is not searched."
        ),
    )
    parser.add_argument(
        "--recapture-manifest",
        type=Path,
        default=Path("manifests/rome_math_ablation_recapture.yaml"),
    )
    parser.add_argument("--blind-candidate", choices=("M0", "M1", "M2"), default="M0")
    parser.add_argument(
        "--blind-cutoff",
        type=float,
        help="One globally calibrated development-family cutoff; required unless --localization-only.",
    )
    parser.add_argument(
        "--localization-only",
        action="store_true",
        help="Evaluate M0--M3 and B0 without evaluating uncalibrated B1.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("analysis_out/rome-math-ablation-evaluation/results.json"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("analysis_out/rome-math-ablation-evaluation/results.csv"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_evaluation_mode(*, localization_only: bool, blind_cutoff: float | None) -> float | None:
    if localization_only:
        if blind_cutoff is not None:
            raise ValueError("--blind-cutoff must be omitted with --localization-only")
        return None
    if blind_cutoff is None:
        raise ValueError("--blind-cutoff is required unless --localization-only is set")
    if not math.isfinite(blind_cutoff) or blind_cutoff < 0.0:
        raise ValueError("--blind-cutoff must be finite and non-negative")
    return float(blind_cutoff)


def main() -> int:
    args = parse_args()
    blind_cutoff = validate_evaluation_mode(
        localization_only=bool(args.localization_only),
        blind_cutoff=args.blind_cutoff,
    )
    metadata, manifest = _load_manifest(args.recapture_manifest)
    source_roots = list(args.run_root or [args.analysis_root])
    paths = enumerate_run_roots(source_roots)
    cases = collect_cases(
        paths,
        model_metadata=metadata,
        blind_candidate=args.blind_candidate,
        blind_cutoff=blind_cutoff,
    )
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "scientific_baseline": False,
        "source_roots": [str(path) for path in source_roots],
        "recapture_manifest": str(args.recapture_manifest),
        "split_policy": manifest.get("split_policy"),
        "calibration": {
            "blind_candidate": args.blind_candidate,
            "blind_cutoff": blind_cutoff,
            "B1_status": (
                "not_evaluated_uncalibrated" if args.localization_only else "globally_calibrated_development_cutoff"
            ),
            "rule": (
                "localization-only; B1 not evaluated"
                if args.localization_only
                else "calibrate on development families, then freeze"
            ),
        },
        "threat_models": {
            "B0": "clean-reference ROME-compatible low-rank Gram edit",
            "B1": "blind ROME suspicion; requires held-out hard-negative validation",
            "B2": "current blind footprint control; uncalibrated universal bound",
        },
        "cases": cases,
        "summary": summarize(cases),
    }
    _write_outputs(
        payload,
        json_out=args.json_out,
        csv_out=args.csv_out,
        force=args.force,
    )
    print(f"wrote {args.json_out}")
    print(f"wrote {args.csv_out}")
    print(f"cases: {len(cases)}; capture artifacts: {len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
