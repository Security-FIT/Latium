#!/usr/bin/env python3
"""
Model-resident edit execution and primitive capture orchestration.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

import torch

from src.common.config import get_config_value as _get, plain
from src.common.linalg import clear_linalg_caches
from src.handlers.rome import ModelHandler
from src.results import ArtifactWriter, RunLayout
from src.structural.capture.baseline import baseline_artifacts
from src.structural.capture.registry import required_weight_families, resolve_captures
from src.structural.execution.case_selection import load_test_cases
from src.structural.config import ModelRunPlan, StructuralBenchmarkConfig
from src.structural.execution.covariance import find_second_moment_files
from src.structural.capture.artifacts import capture_options
from src.structural.execution.edit_execution import run_edit_method
from src.structural.planning import build_model_run_plans, normalize_models_arg
from src.structural.execution.weight_extraction import extract_attention_weights, extract_weights
from src.structural.execution.weights import build_cfg, get_fc_template, load_model_config
from src.runtime import set_global_seed
from src.worker_progress import effective_progress_interval, write_worker_progress


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

MODEL_EXECUTION_FIELDS = (
    "name",
    "layer",
    "layer_name_template",
    "fc_layer_name_template",
    "rewrite_module_tmp",
    "layer_module_tmp",
    "mlp_module_tmp",
    "attn_module_tmp",
    "ln_f_module",
    "lm_head_module",
    "fact_token",
    "lr",
    "kl_factor",
    "weight_decay",
    "epochs",
    "k_N",
    "v_N",
    "prefix_range",
    "optimize_pcs",
    "prefix_mode",
    "prefix_source",
    "prefix_cache_path",
    "prefix_cache_size",
    "prefix_enforce_latin",
    "prefix_min_words",
)


def _model_context(
    cfg: Any,
    *,
    model_key: str,
    proj_template: str,
    fc_template: Optional[str],
    num_layers: int,
    second_moment_allow_autocompute: bool,
) -> dict[str, Any]:
    model_cfg = cfg.model
    settings = {
        field: plain(value)
        for field in MODEL_EXECUTION_FIELDS
        if (value := _get(model_cfg, field, None)) not in (None, "")
    }
    return {
        "model_key": model_key,
        "model_name": str(_get(model_cfg, "name", model_key)),
        "target_layer": int(_get(model_cfg, "layer", 0) or 0),
        "num_layers": int(num_layers),
        "layer_name_template": proj_template,
        "fc_layer_name_template": fc_template,
        "settings": settings,
        "runtime": {
            "second_moment_allow_autocompute": bool(second_moment_allow_autocompute),
        },
    }


def _update_progress(
    config: StructuralBenchmarkConfig,
    *,
    model: str,
    completed: int,
    total: int,
    status: str = "running",
) -> None:
    if not config.progress_file:
        return
    write_worker_progress(
        config.progress_file,
        {
            "worker_id": config.worker_id or "",
            "status": status,
            "current_model": model,
            "current_model_progress": f"{completed}/{total}",
            "progress_interval": effective_progress_interval(
                total,
                config.progress_interval,
            ),
        },
        preserve_existing=True,
    )


def _extract_capture_weights(
    handler: ModelHandler,
    *,
    model_key: str,
    proj_template: str,
    fc_template: Optional[str],
    capture_names: tuple[str, ...],
) -> tuple[
    dict[int, torch.Tensor],
    Optional[dict[int, torch.Tensor]],
    dict[str, dict[int, torch.Tensor]],
]:
    """Copy only matrix families consumed by the selected capture producers."""
    families = required_weight_families(capture_names)
    projection = extract_weights(handler, proj_template) if "proj" in families else {}
    fc: Optional[dict[int, torch.Tensor]] = None
    if "fc" in families and fc_template:
        try:
            fc = extract_weights(handler, fc_template)
        except (KeyError, ValueError):
            LOGGER.warning("FC weights unavailable for %s", model_key)
    attention = (
        extract_attention_weights(handler, proj_template) if "attention" in families else {}
    )
    return projection, fc, attention


def _run_methods_for_plan(
    *,
    writer: ArtifactWriter,
    layout: RunLayout,
    config: StructuralBenchmarkConfig,
    plan: ModelRunPlan,
    model_key: str,
    handler: ModelHandler,
    test_cases: list[dict[str, Any]],
    case_selection: dict[str, Any],
    capture_names: tuple[str, ...],
    options: dict[str, Any],
    model_context: dict[str, Any],
    baseline_records: dict[str, dict[str, Any]],
    baseline_proj: dict[int, torch.Tensor],
    baseline_fc: Optional[dict[int, torch.Tensor]],
    baseline_attention: dict[str, dict[int, torch.Tensor]],
    proj_template: str,
    fc_template: Optional[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for edit_method in config.edit_methods:
        results.append(
            run_edit_method(
                writer=writer,
                layout=layout,
                config=config,
                plan=plan,
                model=model_key,
                handler=handler,
                test_cases=test_cases,
                edit_method_name=edit_method,
                capture_names=capture_names,
                options=options,
                baseline_records=baseline_records,
                baseline_proj=baseline_proj,
                baseline_fc=baseline_fc,
                baseline_attention=baseline_attention,
                proj_template=proj_template,
                fc_template=fc_template,
                case_selection=case_selection,
                model_context=model_context,
                progress_callback=lambda model, completed, total: _update_progress(
                    config,
                    model=model,
                    completed=completed,
                    total=total,
                ),
            )
        )
    return results


def run_capture(config: StructuralBenchmarkConfig) -> dict[str, Any]:
    set_global_seed(config.seed)
    models = tuple(normalize_models_arg(config.models))
    capture_names = resolve_captures(
        config.capture_profile,
        enabled=config.enable_captures,
        disabled=config.disable_captures,
    )
    run_id = config.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    layout = RunLayout.from_output(config.output_dir, run_id).ensure()
    writer = ArtifactWriter(
        layout.root,
        run_id=run_id,
        metadata={
            "models": list(models),
            "edit_methods": list(config.edit_methods),
            "capture_profile": config.capture_profile,
            "capture_producers": list(capture_names),
        },
    )
    plans = build_model_run_plans(config, run_id=run_id)
    plans_by_model: dict[str, list[ModelRunPlan]] = defaultdict(list)
    for plan in plans:
        plans_by_model[plan.model_key].append(plan)

    results: dict[str, Any] = {
        "run_id": run_id,
        "run_root": str(layout.root),
        "models": {},
    }
    test_case_cache: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}

    for model_key, model_plans in plans_by_model.items():
        model_cfg = load_model_config(model_key)
        second_moments, second_moment_dir = find_second_moment_files(model_cfg)
        if "rome" in config.edit_methods and not second_moments:
            message = (
                f"Missing second moment stats for model={model_cfg.name} layer={model_cfg.layer} in {second_moment_dir}"
            )
            if config.fail_on_missing_second_moment:
                raise FileNotFoundError(message)
            LOGGER.warning("%s", message)
            results["models"][model_key] = {"status": "skipped", "error": message}
            continue

        cfg = build_cfg(
            model_key,
            runtime={
                "hf_token": config.hf_token,
                "prefix_log_all": config.prefix_log_all,
                "second_moment_allow_autocompute": config.second_moment_allow_autocompute,
                "log_skip_traceback": config.log_skip_traceback,
            },
            seed=config.seed,
        )
        LOGGER.info("Loading %s", cfg.model.name)
        handler = ModelHandler(cfg)
        proj_template = handler._layer_name_template
        configured_fc = str(getattr(cfg.model, "fc_layer_name_template", "") or "").strip()
        fc_template = configured_fc or get_fc_template(proj_template)
        baseline_proj, baseline_fc, baseline_attention = _extract_capture_weights(
            handler,
            model_key=model_key,
            proj_template=proj_template,
            fc_template=fc_template,
            capture_names=capture_names,
        )
        model_context = _model_context(
            cfg,
            model_key=model_key,
            proj_template=proj_template,
            fc_template=fc_template,
            num_layers=int(handler.num_of_layers),
            second_moment_allow_autocompute=config.second_moment_allow_autocompute,
        )

        model_results: list[dict[str, Any]] = []
        try:
            for plan in model_plans:
                cache_key = config.case_index_file or f"start:{plan.start_idx}:count:{config.n_tests}"
                if cache_key not in test_case_cache:
                    test_cases, case_selection = load_test_cases(
                        config.n_tests,
                        plan.start_idx,
                        dataset_name=config.case_dataset_name,
                        split=config.case_dataset_split,
                        case_index_file=config.case_index_file,
                    )
                    test_case_cache[cache_key] = (test_cases, case_selection)
                test_cases, case_selection = test_case_cache[cache_key]
                options = capture_options(config)
                baseline_records = baseline_artifacts(
                    writer=writer,
                    layout=layout,
                    config=config,
                    plan=plan,
                    model=model_key,
                    handler=handler,
                    capture_names=capture_names,
                    options=options,
                    case_selection=case_selection,
                    model_context=model_context,
                    baseline_proj=baseline_proj,
                    baseline_fc=baseline_fc,
                    baseline_attention=baseline_attention,
                )
                methods = _run_methods_for_plan(
                    writer=writer,
                    layout=layout,
                    config=config,
                    plan=plan,
                    model_key=model_key,
                    handler=handler,
                    test_cases=test_cases,
                    case_selection=case_selection,
                    capture_names=capture_names,
                    options=options,
                    model_context=model_context,
                    baseline_records=baseline_records,
                    baseline_proj=baseline_proj,
                    baseline_fc=baseline_fc,
                    baseline_attention=baseline_attention,
                    proj_template=proj_template,
                    fc_template=fc_template,
                )
                model_results.append(
                    {
                        "plan_id": plan.plan_id,
                        "methods": methods,
                    }
                )
        finally:
            clear_linalg_caches()
            del handler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results["models"][model_key] = {
            "status": "complete",
            "plans": model_results,
        }

    return results


__all__ = ["run_capture"]
