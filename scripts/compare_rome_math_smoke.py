#!/usr/bin/env python3
"""Compare repeated ROME math smoke captures within dtype/dimension bounds."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.evaluate_rome_math_ablation import enumerate_artifacts
from src.structural.experiments.rome_math_ablation import (
    CANDIDATE_FIELDS,
    evaluate_capture_data,
    numerical_tolerance,
)


SMOKE_CANDIDATES = ("M0", "M1", "M2", "M3")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_map(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): dict(case) for case in document.get("cases", [])}


def evaluate_smoke_run(root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    captures: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for path in enumerate_artifacts(root):
        document = _load_json(path)
        run = document["run"]
        key = (
            str(run["model"]),
            str(run["plan_id"]),
            None if run.get("edit_method") is None else str(run["edit_method"]),
        )
        captures[key] = document

    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (model, plan_id, edit_method), suspect in sorted(
        captures.items(),
        key=lambda item: tuple("" if value is None else value for value in item[0]),
    ):
        if edit_method is None:
            continue
        baseline = captures.get((model, plan_id, None))
        if baseline is None:
            raise ValueError(f"Missing baseline capture for {model}/{plan_id}")
        baseline_case = _case_map(baseline).get("baseline")
        if baseline_case is None or baseline_case.get("status") != "complete":
            raise ValueError(f"Incomplete baseline capture for {model}/{plan_id}")
        for case in suspect.get("cases", []):
            case_id = str(case["case_id"])
            if case.get("status") != "complete":
                raise ValueError(
                    f"Incomplete smoke capture for {model}/{plan_id}/{case_id}: "
                    f"{case.get('status')}"
                )
            records[(model, plan_id, case_id)] = evaluate_capture_data(
                baseline_case["data"],
                case["data"],
                blind_candidate="M0",
                blind_cutoff=None,
            )
    return records


def evaluate_smoke_roots(
    roots: Iterable[Path],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for root in roots:
        current = evaluate_smoke_run(root)
        overlap = set(records).intersection(current)
        if overlap:
            raise ValueError(
                f"Duplicate smoke cases across run roots: {sorted(overlap)}"
            )
        records.update(current)
    return records


def _score_tolerance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    candidate: str,
    layer: int,
) -> float:
    field = CANDIDATE_FIELDS[candidate]
    left_profile = left["profiles"][str(layer)]
    right_profile = right["profiles"][str(layer)]
    dtype_name = str(left_profile["compute_dtype"])
    if dtype_name != str(right_profile["compute_dtype"]):
        raise ValueError(f"Compute dtype changed for {candidate} at layer {layer}")
    dtype = getattr(torch, dtype_name)
    dimension = max(
        *(int(value) for value in left_profile["weight_shape"]),
        *(int(value) for value in right_profile["weight_shape"]),
    )
    scale = max(
        abs(float(left_profile[field])),
        abs(float(right_profile[field])),
    )
    # Each score is a separate floating-point computation, so compare against
    # the sum of their recorded dtype/dimension-derived roundoff bounds.
    return 2.0 * numerical_tolerance(dtype, dimension, scale)


def compare_smoke_records(
    left: Mapping[tuple[str, str, str], Mapping[str, Any]],
    right: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        return {
            "status": "failed",
            "reason": "case_inventory_mismatch",
            "missing_from_a": [list(key) for key in missing_left],
            "missing_from_b": [list(key) for key in missing_right],
            "comparisons": [],
        }

    comparisons: list[dict[str, Any]] = []
    passed = True
    for key in sorted(left):
        left_case = left[key]
        right_case = right[key]
        for candidate in SMOKE_CANDIDATES:
            left_result = left_case["candidates"][candidate]
            right_result = right_case["candidates"][candidate]
            same_layer = left_result["selected_layer"] == right_result["selected_layer"]
            layer = int(left_result["selected_layer"])
            tolerance = (
                _score_tolerance(
                    left_case,
                    right_case,
                    candidate=candidate,
                    layer=layer,
                )
                if same_layer
                else 0.0
            )
            score_difference = abs(
                float(left_result["score"]) - float(right_result["score"])
            )
            finite = all(
                math.isfinite(float(result["score"]))
                for result in (left_result, right_result)
            )
            comparison_passed = bool(
                same_layer and finite and score_difference <= tolerance
            )
            passed = passed and comparison_passed
            comparisons.append(
                {
                    "model": key[0],
                    "plan_id": key[1],
                    "case_id": key[2],
                    "candidate": candidate,
                    "selected_layer_a": left_result["selected_layer"],
                    "selected_layer_b": right_result["selected_layer"],
                    "score_a": float(left_result["score"]),
                    "score_b": float(right_result["score"]),
                    "absolute_difference": score_difference,
                    "numerical_tolerance": tolerance,
                    "passed": comparison_passed,
                }
            )
    return {
        "status": "passed" if passed else "failed",
        "tolerance_policy": (
            "sum of dtype/dimension-derived numerical_tolerance bounds "
            "for the two repeated computations"
        ),
        "cases": len(left),
        "comparisons": comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", action="append", type=Path, required=True)
    parser.add_argument("--run-b", action="append", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare_smoke_records(
        evaluate_smoke_roots(args.run_a),
        evaluate_smoke_roots(args.run_b),
    )
    document = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out is None:
        print(document, end="")
    else:
        if args.json_out.exists() and not args.force:
            raise FileExistsError(
                f"Refusing to overwrite existing output: {args.json_out}"
            )
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(document, encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
