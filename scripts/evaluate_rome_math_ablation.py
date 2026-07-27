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


def enumerate_artifacts(root: Path) -> list[Path]:
    """Use ripgrep's ignored-file-aware inventory, as required by the protocol."""
    result = subprocess.run(
        ["rg", "--no-ignore", "--files", str(root)],
        check=True,
        text=True,
        capture_output=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line.endswith("/captures/rome-math-ablation.json")]


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


def _execution_summary(capture_path: Path) -> dict[str, Any]:
    execution_path = capture_path.parent.parent / "execution.json"
    if not execution_path.exists():
        return {}
    return dict(_load_json(execution_path).get("summary", {}))


def collect_cases(
    artifact_paths: Iterable[Path],
    *,
    model_metadata: Mapping[object, Mapping[str, str]],
    blind_candidate: str,
    blind_cutoff: float,
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
        execution_summary = _execution_summary(path)
        target_layer = execution_summary.get("target_layer")
        num_layers = execution_summary.get("num_layers")

        for case in suspect_document.get("cases", []):
            case_id = str(case["case_id"])
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
    within_one = sum(
        abs(case["candidates"][candidate]["selected_layer"] - case["target_layer"]) <= 1 for case in available
    )
    per_family: dict[str, dict[str, Any]] = {}
    for family in sorted({case["family"] for case in inventory}):
        family_inventory = [case for case in inventory if case["family"] == family]
        family_cases = [case for case in available if case["family"] == family]
        family_exact = sum(
            case["candidates"][candidate]["selected_layer"] == case["target_layer"] for case in family_cases
        )
        per_family[family] = {
            "correct": family_exact,
            "available": len(family_cases),
            "total": len(family_inventory),
            "accuracy": family_exact / len(family_inventory) if family_inventory else None,
        }
    family_accuracies = [float(record["accuracy"]) for record in per_family.values() if record["accuracy"] is not None]
    margins = [float(case["candidates"][candidate]["margin"]) for case in available]
    return {
        "correct": exact,
        "available": len(available),
        "total": len(inventory),
        "exact_accuracy": exact / len(inventory) if inventory else None,
        "within_one_accuracy": within_one / len(inventory) if inventory else None,
        "macro_family_accuracy": (sum(family_accuracies) / len(family_accuracies) if family_accuracies else None),
        "per_family": per_family,
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


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, Any] = {}
    for split in sorted(ALLOWED_SPLITS):
        split_cases = [case for case in cases if case["split"] == split]
        by_split[split] = {
            "cases_total": len(split_cases),
            "cases_complete": sum(case["status"] == "complete" for case in split_cases),
            "cases_unavailable_or_error": sum(case["status"] != "complete" for case in split_cases),
            "localization": {candidate: _candidate_metrics(split_cases, candidate) for candidate in CANDIDATE_FIELDS},
            "candidate_disagreements": _disagreements(split_cases),
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
                    )
                },
                "candidate_version": candidate,
                "selected_layer": result["selected_layer"],
                "score": result["score"],
                "margin": result["margin"],
                "localized_layer": case["localized_layer"],
                "presence_peak_layer": case["presence_peak_layer"],
                "B0": bool(case["binary"]["B0"]["is_rome_like"]),
                "B1": bool(case["binary"]["B1"]["is_rome_like"]),
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
        "--recapture-manifest",
        type=Path,
        default=Path("manifests/rome_math_ablation_recapture.yaml"),
    )
    parser.add_argument("--blind-candidate", choices=("M0", "M1", "M2"), default="M0")
    parser.add_argument(
        "--blind-cutoff",
        type=float,
        required=True,
        help="One globally calibrated development-family cutoff; never per model/family.",
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


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.blind_cutoff) or args.blind_cutoff < 0.0:
        raise ValueError("--blind-cutoff must be finite and non-negative")
    metadata, manifest = _load_manifest(args.recapture_manifest)
    paths = enumerate_artifacts(args.analysis_root)
    cases = collect_cases(
        paths,
        model_metadata=metadata,
        blind_candidate=args.blind_candidate,
        blind_cutoff=args.blind_cutoff,
    )
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "scientific_baseline": False,
        "source_root": str(args.analysis_root),
        "recapture_manifest": str(args.recapture_manifest),
        "split_policy": manifest.get("split_policy"),
        "calibration": {
            "blind_candidate": args.blind_candidate,
            "blind_cutoff": float(args.blind_cutoff),
            "rule": "calibrate on development families, then freeze",
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
