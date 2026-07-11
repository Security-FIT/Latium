"""
Baseline execution and capture artifact creation.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from src.results import ArtifactWriter, RunLayout, config_hash
from src.results.ids import capture_id
from src.structural.capture.artifacts import (
    analysis_variant_metadata,
    capture_case,
    capture_config,
    execution_config,
    write_capture,
    write_execution,
)
from src.structural.capture.registry import CAPTURES
from src.structural.capture.producers import CaptureContext
from src.structural.config import ModelRunPlan, StructuralBenchmarkConfig


def baseline_artifacts(
    *,
    writer: ArtifactWriter,
    layout: RunLayout,
    config: StructuralBenchmarkConfig,
    plan: ModelRunPlan,
    model: str,
    handler: Any,
    capture_names: tuple[str, ...],
    options: dict[str, Any],
    case_selection: Mapping[str, Any],
    model_context: Mapping[str, Any],
    baseline_proj: dict[int, torch.Tensor],
    baseline_fc: Optional[dict[int, torch.Tensor]],
    baseline_attention: dict[str, dict[int, torch.Tensor]],
) -> dict[str, dict[str, Any]]:
    resolved_execution_config = execution_config(
        config,
        plan,
        None,
        model_context=model_context,
        case_selection=case_selection,
        options=options,
    )
    execution_record = write_execution(
        writer,
        layout,
        run_id=plan.run_id,
        model=model,
        plan=plan,
        edit_method=None,
        config=resolved_execution_config,
        cases=[
            {
                "case_id": "baseline",
                "status": "complete",
                "edit": {"method": "baseline", "success": True},
                "error": None,
            }
        ],
        target_layer=int(handler._layer),
        num_layers=int(handler.num_of_layers),
        force=config.force,
        metadata={"analysis_variants": analysis_variant_metadata(config)},
    )
    records: dict[str, dict[str, Any]] = {"execution": execution_record}
    for capture_name in capture_names:
        resolved_capture_config = capture_config(
            capture_name,
            options,
            profile=config.capture_profile,
        )
        artifact_id = capture_id(model, plan.plan_id, capture_name, None)
        inputs = [
            {
                "artifact_id": execution_record["artifact_id"],
                "content_hash": execution_record["content_hash"],
            }
        ]
        current = writer.current(
            artifact_id,
            expected_config_hash=config_hash(resolved_capture_config),
            inputs=inputs,
        )
        if not config.force and current is not None:
            records[capture_name] = current
            continue
        spec = CAPTURES.get(capture_name)
        if spec.requires_probe:
            cases = [
                {
                    "case_id": "baseline",
                    "status": "unavailable",
                    "data": {},
                    "error": "baseline has no edit probe vector",
                }
            ]
        else:
            context = CaptureContext(
                proj_weights=baseline_proj,
                fc_weights=baseline_fc,
                attention_weights=baseline_attention,
                probe_vector=None,
                token_predictor=None,
                changed_weights={},
                options=options,
                baseline_proj_weights=baseline_proj,
                baseline_fc_weights=baseline_fc,
            )
            cases = [capture_case(capture_name, context, case_id="baseline")]
        records[capture_name] = write_capture(
            writer,
            layout,
            run_id=plan.run_id,
            model=model,
            plan=plan,
            edit_method=None,
            capture_name=capture_name,
            capture_config=resolved_capture_config,
            cases=cases,
            inputs=inputs,
            force=config.force,
        )
    return records


__all__ = ["baseline_artifacts"]
