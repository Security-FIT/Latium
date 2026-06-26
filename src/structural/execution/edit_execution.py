"""
Per-case edit execution, restoration, and edited-state capture.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import logging
import traceback
from collections import defaultdict
from typing import Any, Callable, Mapping, Optional

import torch

from src.common.io import to_serializable
from src.editing.base import EditOutcome
from src.results import ArtifactWriter, RunLayout, config_hash
from src.results.ids import capture_id, execution_id
from src.structural.capture.artifacts import (
    analysis_variant_metadata,
    capture_case,
    capture_config,
    capture_inputs,
    execution_config,
    write_capture,
    write_execution,
)
from src.structural.capture.producers import CaptureContext, token_predictor_from_handler
from src.structural.config import ModelRunPlan, StructuralBenchmarkConfig
from src.worker_progress import effective_progress_interval


LOGGER = logging.getLogger(__name__)


def modified_weights(
    handler: Any,
    baseline_proj: dict[int, torch.Tensor],
    baseline_fc: Optional[dict[int, torch.Tensor]],
    baseline_attention: dict[str, dict[int, torch.Tensor]],
    proj_template: str,
    fc_template: Optional[str],
    outcome: EditOutcome,
) -> tuple[
    dict[int, torch.Tensor],
    Optional[dict[int, torch.Tensor]],
    dict[str, dict[int, torch.Tensor]],
]:
    from src.structural.execution.weight_extraction import extract_attention_weights

    modified_proj = dict(baseline_proj)
    modified_fc = dict(baseline_fc) if baseline_fc is not None else None
    modified_attention = {family: dict(weights) for family, weights in baseline_attention.items()}

    if "proj" in outcome.modified_weights:
        changed = outcome.modified_weights["proj"]
        layers = range(handler.num_of_layers) if changed is None else (int(layer) for layer in changed)
        for layer in layers:
            modified_proj[int(layer)] = (
                handler._get_module(proj_template.format(int(layer))).weight.detach().clone().cpu()
            )

    if "fc" in outcome.modified_weights and fc_template:
        changed = outcome.modified_weights["fc"]
        layers = range(handler.num_of_layers) if changed is None else (int(layer) for layer in changed)
        modified_fc = modified_fc or {}
        for layer in layers:
            modified_fc[int(layer)] = handler._get_module(fc_template.format(int(layer))).weight.detach().clone().cpu()

    if "attention" in outcome.modified_weights:
        modified_attention = extract_attention_weights(handler, proj_template)

    return modified_proj, modified_fc, modified_attention


def restore(handler: Any, outcome: Optional[EditOutcome]) -> None:
    handler.remove_hooks()
    if outcome is None:
        return
    for module_name, old_weight in outcome.restorations.items():
        handler._get_module(module_name).weight = torch.nn.Parameter(old_weight)


def run_edit_method(
    *,
    writer: ArtifactWriter,
    layout: RunLayout,
    config: StructuralBenchmarkConfig,
    plan: ModelRunPlan,
    model: str,
    handler: Any,
    test_cases: list[dict[str, Any]],
    edit_method_name: str,
    capture_names: tuple[str, ...],
    options: dict[str, Any],
    baseline_records: dict[str, dict[str, Any]],
    baseline_proj: dict[int, torch.Tensor],
    baseline_fc: Optional[dict[int, torch.Tensor]],
    baseline_attention: dict[str, dict[int, torch.Tensor]],
    proj_template: str,
    fc_template: Optional[str],
    case_selection: Optional[Mapping[str, Any]] = None,
    model_context: Optional[Mapping[str, Any]] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    method_loader: Optional[Callable[[str], Any]] = None,
    capture_case_fn: Optional[Callable[..., dict[str, Any]]] = None,
    traceback_formatter: Optional[Callable[[], str]] = None,
) -> dict[str, Any]:
    if method_loader is None:
        from src.editing.registry import get_edit_method as method_loader

    capture_one = capture_case if capture_case_fn is None else capture_case_fn
    format_traceback = traceback.format_exc if traceback_formatter is None else traceback_formatter

    method = method_loader(edit_method_name)
    resolved_execution_config = execution_config(
        config,
        plan,
        edit_method_name,
        model_context=model_context,
        case_selection=case_selection,
        options=options,
    )
    execution_artifact_id = execution_id(model, plan.plan_id, edit_method_name)
    expected_execution_hash = config_hash(resolved_execution_config)
    existing_execution = writer.current(
        execution_artifact_id,
        expected_config_hash=expected_execution_hash,
        inputs=[],
    )
    execution_current = existing_execution is not None

    capture_configs = {
        capture_name: capture_config(
            capture_name,
            options,
            profile=config.capture_profile,
        )
        for capture_name in capture_names
    }
    captures_current = execution_current
    if execution_current:
        assert existing_execution is not None
        execution_ref = {
            "artifact_id": execution_artifact_id,
            "content_hash": existing_execution["content_hash"],
        }
        for capture_name in capture_names:
            baseline_record = baseline_records.get(capture_name)
            inputs = capture_inputs(capture_name, execution_ref, baseline_record)
            captures_current = (
                captures_current
                and writer.current(
                    capture_id(model, plan.plan_id, capture_name, edit_method_name),
                    expected_config_hash=config_hash(capture_configs[capture_name]),
                    inputs=inputs,
                )
                is not None
            )
    if captures_current and not config.force:
        LOGGER.info(
            "Skipping current capture: model=%s plan=%s method=%s",
            model,
            plan.plan_id,
            edit_method_name,
        )
        return {"skipped": True, "method": edit_method_name}

    execution_cases: list[dict[str, Any]] = []
    captured_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    interval = effective_progress_interval(len(test_cases), config.progress_interval)

    for index, case in enumerate(test_cases, start=1):
        case_id = str(case["case_id"])
        outcome: Optional[EditOutcome] = None
        try:
            outcome = method.apply(handler, case)
            metrics = method.evaluate(handler, case, outcome)
            outcome.metrics.update(metrics)
            if "efficacy_score" in metrics:
                outcome.success = bool(float(metrics["efficacy_score"]) >= 1.0)
            execution_cases.append(
                {
                    "case_id": case_id,
                    "subject": case.get("subject"),
                    "status": "complete",
                    "edit": {
                        "method": edit_method_name,
                        "success": bool(outcome.success),
                        "metrics": to_serializable(outcome.metrics),
                        "metadata": to_serializable(outcome.metadata),
                        "modified_weights": to_serializable(outcome.modified_weights),
                    },
                    "error": None,
                }
            )

            modified_proj, modified_fc, modified_attention = modified_weights(
                handler,
                baseline_proj,
                baseline_fc,
                baseline_attention,
                proj_template,
                fc_template,
                outcome,
            )
            capture_context = CaptureContext(
                proj_weights=modified_proj,
                fc_weights=modified_fc,
                attention_weights=modified_attention,
                probe_vector=outcome.probe_vector,
                token_predictor=token_predictor_from_handler(handler),
                changed_weights=dict(outcome.modified_weights),
                options=options,
            )
            for capture_name in capture_names:
                captured_cases[capture_name].append(capture_one(capture_name, capture_context, case_id=case_id))
        except Exception as exc:
            LOGGER.warning(
                "Case failed: model=%s method=%s case=%s error=%s",
                model,
                edit_method_name,
                case_id,
                exc,
            )
            if not config.log_skip_traceback:
                LOGGER.warning("%s", format_traceback())
            execution_cases.append(
                {
                    "case_id": case_id,
                    "subject": case.get("subject"),
                    "status": "error",
                    "edit": {"method": edit_method_name, "success": False},
                    "error": str(exc),
                }
            )
            for capture_name in capture_names:
                captured_cases[capture_name].append(
                    {
                        "case_id": case_id,
                        "status": "unavailable",
                        "data": {},
                        "error": "edit execution failed",
                    }
                )
        finally:
            restore(handler, outcome)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if index % interval == 0 or index == len(test_cases):
            if progress_callback is not None:
                progress_callback(model, index, len(test_cases))

    execution_record = write_execution(
        writer,
        layout,
        run_id=plan.run_id,
        model=model,
        plan=plan,
        edit_method=edit_method_name,
        config=resolved_execution_config,
        cases=execution_cases,
        target_layer=int(handler._layer),
        num_layers=int(handler.num_of_layers),
        force=config.force,
        metadata={"analysis_variants": analysis_variant_metadata(config)},
    )
    for capture_name in capture_names:
        baseline_record = baseline_records.get(capture_name)
        inputs = capture_inputs(capture_name, execution_record, baseline_record)
        write_capture(
            writer,
            layout,
            run_id=plan.run_id,
            model=model,
            plan=plan,
            edit_method=edit_method_name,
            capture_name=capture_name,
            capture_config=capture_configs[capture_name],
            cases=captured_cases[capture_name],
            inputs=inputs,
            force=config.force,
        )
    return {
        "skipped": False,
        "method": edit_method_name,
        "cases": len(execution_cases),
    }


__all__ = ["modified_weights", "restore", "run_edit_method"]
