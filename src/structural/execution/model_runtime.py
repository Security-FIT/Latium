#!/usr/bin/env python3
"""
Model-resident edit execution and primitive capture orchestration.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

from src.common.config import get_config_value as _get, plain
from src.common.linalg import clear_linalg_caches
from src.graphs.registry import resolve_renderers
from src.handlers.rome import ModelHandler
from src.results import ArtifactWriter, RunLayout
from src.structural.analysis.registry import resolve_analyses
from src.structural.capture.baseline import baseline_artifacts
from src.structural.capture.registry import required_weight_families, resolve_capture_plan
from src.structural.execution.case_selection import load_test_cases
from src.structural.config import ModelRunPlan, StructuralBenchmarkConfig
from src.structural.execution.covariance import find_second_moment_files
from src.structural.capture.artifacts import capture_options
from src.structural.execution.edit_execution import run_edit_method
from src.structural.planning import build_model_run_plans, normalize_models_arg
from src.structural.execution.weight_extraction import (
    extract_attention_weights,
    extract_token_alignment_access,
    extract_weights,
)
from src.structural.execution.weights import build_cfg, get_fc_template, load_model_config
from src.runtime import set_global_seed
from src.worker_progress import effective_progress_interval, write_worker_progress


def _source_revision() -> dict[str, Any]:
    """Record enough source identity to distinguish experimental runs."""
    root = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--", "src", "scripts", "tests"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        tracked_files = (
            root / "src/structural/detectors/rome_layer_localizer.py",
            root / "src/structural/capture/producers.py",
            root / "src/structural/analysis/detector_methods.py",
            root / "scripts/evaluate_binary_rome_presence.py",
        )
        source_digest = hashlib.sha256()
        for path in tracked_files:
            source_digest.update(str(path.relative_to(root)).encode())
            source_digest.update(path.read_bytes())
        return {
            "git_revision": revision,
            "tracked_source_dirty": bool(diff),
            "tracked_source_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "rome_experiment_source_sha256": source_digest.hexdigest(),
        }
    except (OSError, subprocess.SubprocessError):
        return {
            "git_revision": None,
            "tracked_source_dirty": None,
            "tracked_source_diff_sha256": None,
            "rome_experiment_source_sha256": None,
        }


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
    output_head_weight: Optional[torch.Tensor] = None,
    projection_layout: Optional[str] = None,
    output_head_layout: Optional[str] = None,
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
                output_head_weight=output_head_weight,
                projection_layout=projection_layout,
                output_head_layout=output_head_layout,
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
    analysis_names = (
        resolve_analyses(
            config.analysis_preset,
            enabled=config.enable_analyses,
            disabled=config.disable_analyses,
        )
        if config.run_analysis
        else ()
    )
    renderer_names = (
        resolve_renderers(
            config.renderer_preset,
            enabled=config.enable_renderers,
            disabled=config.disable_renderers,
        )
        if config.render_graphs
        else ()
    )
    capture_plan = resolve_capture_plan(
        config.capture_profile,
        enabled=config.enable_captures,
        disabled=config.disable_captures,
        analyses=analysis_names,
        renderers=renderer_names,
        matrix_feature_set=config.matrix_feature_set,
        matrix_features=config.matrix_features,
    )
    capture_names = capture_plan.names
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
            "source": _source_revision(),
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
        weight_families = required_weight_families(capture_names)
        baseline_proj = extract_weights(handler, proj_template) if "proj" in weight_families else {}
        baseline_fc: Optional[dict[int, torch.Tensor]] = None
        if "fc" in weight_families and fc_template:
            try:
                baseline_fc = extract_weights(handler, fc_template)
            except (KeyError, ValueError):
                LOGGER.warning("FC weights unavailable for %s", model_key)
        baseline_attention = (
            extract_attention_weights(handler, proj_template) if "attention" in weight_families else {}
        )
        output_head_weight: Optional[torch.Tensor] = None
        projection_layout: Optional[str] = None
        output_head_layout: Optional[str] = None
        if "token-subspace-alignment-v1" in capture_names:
            try:
                output_head_weight, projection_layout, output_head_layout = extract_token_alignment_access(
                    handler, proj_template
                )
            except ValueError as exc:
                LOGGER.warning("Token alignment access unavailable for %s: %s", model_key, exc)
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
                options = capture_options(config, matrix_features=capture_plan.matrix_features)
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
                    output_head_weight=output_head_weight,
                    projection_layout=projection_layout,
                    output_head_layout=output_head_layout,
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
                    output_head_weight=output_head_weight,
                    projection_layout=projection_layout,
                    output_head_layout=output_head_layout,
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
