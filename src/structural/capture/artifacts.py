"""
Execution and capture artifact helpers for structural runs.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Optional

from src.common.io import to_serializable
from src.results import ArtifactWriter, RunLayout, build_artifact, config_hash
from src.results.ids import capture_id, execution_id
from src.structural.capture.matrix_features import resolve_matrix_features
from src.structural.capture.registry import CAPTURES
from src.structural.capture.producers import CaptureContext
from src.structural.config import ModelRunPlan, StructuralBenchmarkConfig


def capture_options(
    config: StructuralBenchmarkConfig,
) -> dict[str, Any]:
    variants = config.effective_analysis_variants
    return {
        "spectral_top_k": max(int(variant.spectral_top_k) for variant in variants),
        "matrix_feature_set": str(config.matrix_feature_set),
        "matrix_features": tuple(config.matrix_features),
        "matrix_svd_top_k": int(config.matrix_svd_top_k),
        "bottom_rank_sweep_ranks": tuple(config.bottom_rank_sweep_ranks),
        "bottom_rank_top_svd_rank": int(config.bottom_rank_top_svd_rank),
        "bottom_rank_boundary": int(config.bottom_rank_boundary),
    }


def analysis_variant_metadata(config: StructuralBenchmarkConfig) -> list[dict[str, Any]]:
    return [variant.to_dict() for variant in config.effective_analysis_variants]


def _fallback_case_selection(
    config: StructuralBenchmarkConfig,
    plan: ModelRunPlan,
) -> dict[str, Any]:
    return {
        "mode": "manifest" if config.case_index_file else "contiguous_slice",
        "start_idx": int(plan.start_idx),
        "end_idx": int(plan.end_idx),
        "case_index_file": config.case_index_file,
    }


def execution_config(
    config: StructuralBenchmarkConfig,
    plan: ModelRunPlan,
    edit_method: Optional[str],
    *,
    model_context: Optional[Mapping[str, Any]] = None,
    case_selection: Optional[Mapping[str, Any]] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "edit_method": edit_method or "baseline",
        "seed": int(config.seed),
        "model": to_serializable(model_context or {"model_key": plan.model_key}),
        "case_selection": to_serializable(
            dict(case_selection) if case_selection is not None else _fallback_case_selection(config, plan)
        ),
        "capture": {
            "profile": config.capture_profile,
            "enabled": list(config.enable_captures),
            "disabled": list(config.disable_captures),
            "options": to_serializable(dict(options or capture_options(config))),
        },
    }


def capture_config(
    capture_name: str,
    options: dict[str, Any],
    *,
    profile: str,
) -> dict[str, Any]:
    if capture_name == "spectral":
        relevant_options = {
            "spectral_top_k": int(options["spectral_top_k"]),
        }
    elif capture_name == "matrix-features":
        relevant_options = {
            "feature_set": str(options.get("matrix_feature_set", "paper")),
            "features": list(
                resolve_matrix_features(
                    str(options.get("matrix_feature_set", "paper")),
                    options.get("matrix_features", ()),
                )
            ),
            "svd_top_k": int(options.get("matrix_svd_top_k", 50)),
        }
    elif capture_name == "bottom-rank-tokens":
        relevant_options = {
            "bottom_rank_sweep_ranks": list(options["bottom_rank_sweep_ranks"]),
            "bottom_rank_top_svd_rank": int(options["bottom_rank_top_svd_rank"]),
            "bottom_rank_boundary": int(options["bottom_rank_boundary"]),
        }
    else:
        relevant_options = {}
    return {
        "capture": capture_name,
        "profile": profile,
        "options": relevant_options,
    }


def write_execution(
    writer: ArtifactWriter,
    layout: RunLayout,
    *,
    run_id: str,
    model: str,
    plan: ModelRunPlan,
    edit_method: Optional[str],
    config: dict[str, Any],
    cases: list[dict[str, Any]],
    target_layer: int,
    num_layers: int,
    force: bool,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    artifact_id = execution_id(model, plan.plan_id, edit_method)
    complete = sum(case.get("status") == "complete" for case in cases)
    successful_edits = sum(
        bool(case.get("edit", {}).get("success")) for case in cases if case.get("status") == "complete"
    )
    payload = build_artifact(
        artifact_id=artifact_id,
        kind="execution",
        producer=edit_method or "baseline",
        run_id=run_id,
        model=model,
        plan_id=plan.plan_id,
        edit_method=edit_method,
        status="complete" if complete == len(cases) else "error",
        config=config,
        config_hash=config_hash(config),
        inputs=[],
        created_at=datetime.now().isoformat(),
        cases=cases,
        summary={
            "target_layer": int(target_layer),
            "num_layers": int(num_layers),
            "cases_total": len(cases),
            "cases_complete": complete,
            "edit_success_count": successful_edits,
            "edit_success_rate": successful_edits / complete if complete else 0.0,
        },
        error=None if complete == len(cases) else "one or more cases failed",
    )
    if metadata:
        payload["record_metadata"] = to_serializable(metadata)
    path = layout.execution_path(model, plan.plan_id, edit_method=edit_method)
    return writer.write(path, payload, force=force)


def capture_inputs(
    capture_name: str,
    execution_record: dict[str, Any],
    baseline_record: Optional[dict[str, Any]],
) -> list[dict[str, str]]:
    inputs = [
        {
            "artifact_id": execution_record["artifact_id"],
            "content_hash": execution_record["content_hash"],
        }
    ]
    if CAPTURES.get(capture_name).requires_baseline:
        if baseline_record is None:
            raise ValueError(f"baseline record is required for capture {capture_name}")
        inputs.append(
            {
                "artifact_id": baseline_record["artifact_id"],
                "content_hash": baseline_record["content_hash"],
            }
        )
    return inputs


def write_capture(
    writer: ArtifactWriter,
    layout: RunLayout,
    *,
    run_id: str,
    model: str,
    plan: ModelRunPlan,
    edit_method: Optional[str],
    capture_name: str,
    capture_config: dict[str, Any],
    cases: list[dict[str, Any]],
    inputs: list[dict[str, str]],
    force: bool,
) -> dict[str, Any]:
    artifact_id = capture_id(model, plan.plan_id, capture_name, edit_method)
    complete = sum(case.get("status") == "complete" for case in cases)
    unavailable = sum(case.get("status") == "unavailable" for case in cases)
    errors = sum(case.get("status") == "error" for case in cases)
    if errors:
        status = "error"
    elif complete:
        status = "complete"
    else:
        status = "unavailable"
    payload = build_artifact(
        artifact_id=artifact_id,
        kind="capture",
        producer=capture_name,
        run_id=run_id,
        model=model,
        plan_id=plan.plan_id,
        edit_method=edit_method,
        status=status,
        config=capture_config,
        config_hash=config_hash(capture_config),
        inputs=inputs,
        created_at=datetime.now().isoformat(),
        cases=cases,
        summary={
            "cases_total": len(cases),
            "cases_complete": complete,
            "cases_unavailable": unavailable,
            "cases_error": errors,
        },
        error=(
            None
            if status == "complete"
            else "capture unavailable"
            if status == "unavailable"
            else "one or more capture cases failed"
        ),
    )
    path = layout.capture_path(
        model,
        plan.plan_id,
        capture_name,
        edit_method=edit_method,
    )
    return writer.write(path, payload, force=force)


def capture_case(
    capture_name: str,
    context: CaptureContext,
    *,
    case_id: str,
) -> dict[str, Any]:
    spec = CAPTURES.get(capture_name)
    if spec.requires_probe and context.probe_vector is None:
        return {
            "case_id": case_id,
            "status": "unavailable",
            "data": {},
            "error": "capture requires an edit probe vector",
        }
    try:
        data = spec.load()(context)
        return {
            "case_id": case_id,
            "status": "complete",
            "data": to_serializable(data),
            "error": None,
        }
    except Exception as exc:
        return {
            "case_id": case_id,
            "status": "error",
            "data": {},
            "error": str(exc),
        }


__all__ = [
    "analysis_variant_metadata",
    "capture_case",
    "capture_config",
    "capture_inputs",
    "capture_options",
    "execution_config",
    "write_capture",
    "write_execution",
]
