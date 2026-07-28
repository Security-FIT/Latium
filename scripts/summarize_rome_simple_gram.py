#!/usr/bin/env python3
"""Summarize simple-Gram captures without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.structural.experiments.simple_gram import PROFILE_FIELDS
from src.structural.experiments.simple_gram_evaluation import (
    SPIKE_FIELDS,
    leave_one_family_out_presence,
    localization_summary,
)


CAPTURE_NAME = "simple-gram-experiment.json"
HARD_NEGATIVE_SCHEMA = "rome-simple-gram-hard-negatives-v1"


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _case_map(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): case for case in document.get("cases", ())}


def collect_run(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect localization and presence rows from one structural run root."""
    localization: list[dict[str, Any]] = []
    presence: list[dict[str, Any]] = []
    for capture_path in sorted(
        run_root.glob(
            f"plans/*/*/methods/rome/captures/{CAPTURE_NAME}"
        )
    ):
        plan_root = capture_path.parents[3]
        model = capture_path.parts[capture_path.parts.index("plans") + 1]
        execution = _case_map(
            _read(plan_root / "methods" / "rome" / "execution.json")
        )
        captures = _case_map(_read(capture_path))
        baseline_path = (
            plan_root
            / "baseline"
            / "captures"
            / CAPTURE_NAME
        )
        baseline_cases = _case_map(_read(baseline_path))
        baseline = baseline_cases.get("baseline")
        if baseline and baseline.get("status") == "complete":
            data = baseline["data"]
            for field in PROFILE_FIELDS:
                presence.append(
                    {
                        "model": model,
                        "family": model,
                        "specimen_id": f"{model}:clean",
                        "is_positive": False,
                        "negative_category": "standalone_clean",
                        "field": field,
                        "statistics": data["spike_statistics"][field],
                    }
                )

        for case_id, captured in captures.items():
            executed = execution.get(case_id, {})
            edit = executed.get("edit") or {}
            if (
                captured.get("status") != "complete"
                or executed.get("status") != "complete"
                or not edit
            ):
                continue
            modified = edit.get("modified_weights", {}).get("proj", ())
            if len(modified) != 1:
                continue
            data = captured["data"]
            edit_success = bool(edit.get("success", False))
            localization.append(
                {
                    "model": model,
                    "family": model,
                    "case_id": case_id,
                    "edit_success": edit_success,
                    "target_layer": int(modified[0]),
                    "selected_layers": {
                        field: int(
                            data["localization"][field]["selected_layer"]
                        )
                        for field in PROFILE_FIELDS
                    },
                }
            )
            if edit_success:
                for field in PROFILE_FIELDS:
                    presence.append(
                        {
                            "model": model,
                            "family": model,
                            "specimen_id": f"{model}:rome:{case_id}",
                            "case_id": case_id,
                            "is_positive": True,
                            "negative_category": None,
                            "field": field,
                            "statistics": data["spike_statistics"][field],
                        }
                    )
    return localization, presence


def collect_hard_negative_bundle(path: Path) -> list[dict[str, Any]]:
    """Load simple-Gram hard negatives without treating them as references."""
    payload = _read(path)
    if payload.get("schema_version") != HARD_NEGATIVE_SCHEMA:
        raise ValueError(f"Unexpected hard-negative schema in {path}")
    model = str(payload["model_key"])
    records: list[dict[str, Any]] = []
    for record in payload.get("records", ()):
        if record.get("label") != "hard_negative":
            continue
        capture = record.get("capture") or {}
        statistics = capture.get("spike_statistics") or {}
        if any(field not in statistics for field in PROFILE_FIELDS):
            raise ValueError(
                f"Incomplete simple-Gram hard-negative capture in {path}"
            )
        for field in PROFILE_FIELDS:
            records.append(
                {
                    "model": model,
                    "family": model,
                    "specimen_id": str(record["specimen_id"]),
                    "is_positive": False,
                    "negative_category": str(record["negative_category"]),
                    "field": field,
                    "statistics": statistics[field],
                }
            )
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(
    run_roots: list[Path],
    *,
    hard_negative_bundles: list[Path] | None = None,
) -> dict[str, Any]:
    localization: list[dict[str, Any]] = []
    presence: list[dict[str, Any]] = []
    for run_root in run_roots:
        run_localization, run_presence = collect_run(run_root)
        localization.extend(run_localization)
        presence.extend(run_presence)
    bundles = hard_negative_bundles or []
    for path in bundles:
        presence.extend(collect_hard_negative_bundle(path))

    binary: dict[str, Any] = {}
    for field in PROFILE_FIELDS:
        field_records = [
            record for record in presence if record["field"] == field
        ]
        binary[field] = {}
        if len({record["family"] for record in field_records}) < 2:
            binary[field]["status"] = "not_evaluated_insufficient_families"
            continue
        for statistic in SPIKE_FIELDS:
            binary[field][statistic] = leave_one_family_out_presence(
                field_records,
                statistic=statistic,
            )
    return {
        "schema_version": "rome-simple-gram-evaluation-v1",
        "scientific_baseline": False,
        "run_roots": [str(path) for path in run_roots],
        "hard_negative_bundles": [
            {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for path in bundles
        ],
        "localization": localization_summary(localization),
        "binary": binary,
        "claim_boundary": (
            "One-checkpoint development screening. A ROME provenance claim "
            "requires high specificity on magnitude-matched rank-one edits; "
            "the detector receives no reference checkpoint."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_roots",
        nargs="+",
        type=Path,
        help="Structural run roots containing simple-gram-experiment captures.",
    )
    parser.add_argument(
        "--hard-negative-bundle",
        action="append",
        default=[],
        type=Path,
        help="Repeat for each simple-Gram hard-negative bundle.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = summarize(
        args.run_roots,
        hard_negative_bundles=args.hard_negative_bundle,
    )
    encoded = (
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    result["content_sha256_before_hash_field"] = hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()
    encoded = (
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
