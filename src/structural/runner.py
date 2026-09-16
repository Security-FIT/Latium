"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from src.editing.registry import EDIT_METHODS
from src.graphs.registry import resolve_renderers
from src.structural.analysis.registry import resolve_analyses
from src.structural.analysis.runtime import run_analyses
from src.structural.capture.registry import resolve_capture_plan
from src.structural.config import StructuralBenchmarkConfig
from src.tracking import tracking_session


def _coerce_structural_config(
    config: StructuralBenchmarkConfig | Mapping[str, Any],
) -> StructuralBenchmarkConfig:
    if isinstance(config, StructuralBenchmarkConfig):
        return config
    if isinstance(config, Mapping):
        return StructuralBenchmarkConfig(**dict(config))
    raise TypeError(f"Unsupported structural benchmark config: {type(config)!r}")


def validate_structural_config(config: StructuralBenchmarkConfig) -> None:
    if not config.edit_methods:
        raise ValueError("At least one editing method is required")
    for identifier in config.edit_methods:
        EDIT_METHODS.get(identifier)
    analysis_names = resolve_analyses(
        config.analysis_preset,
        enabled=config.enable_analyses,
        disabled=config.disable_analyses,
    )
    renderer_names = resolve_renderers(
        config.renderer_preset,
        enabled=config.enable_renderers,
        disabled=config.disable_renderers,
    )
    capture_plan = resolve_capture_plan(
        config.capture_profile,
        enabled=config.enable_captures,
        disabled=config.disable_captures,
        analyses=analysis_names if config.run_analysis else (),
        renderers=renderer_names if config.render_graphs else (),
        matrix_feature_set=config.matrix_feature_set,
        matrix_features=config.matrix_features,
    )
    if not config.run_analysis and not capture_plan.names:
        raise ValueError(
            "Capture-only runs require a capture profile or an explicitly enabled capture"
        )


def run_structural_capture(
    config: StructuralBenchmarkConfig | Mapping[str, Any],
) -> dict[str, Any]:
    resolved = _coerce_structural_config(config)
    validate_structural_config(resolved)
    if not resolved.run_id:
        resolved = resolved.with_run_id(datetime.now().strftime("%Y%m%d_%H%M%S"))
    from src.structural.execution.model_runtime import run_capture

    with tracking_session(resolved, job_type="structural-capture") as tracker:
        tracker.set_state(**{"monitor/stage": "capture", "run_id": resolved.run_id})
        return run_capture(resolved)


def run_structural_analysis(
    run_root: str,
    *,
    preset: str = "paper",
    enabled: tuple[str, ...] = (),
    disabled: tuple[str, ...] = (),
    method_configs: Mapping[str, Mapping[str, Any]] | None = None,
    force: bool = False,
    continue_on_error: bool = False,
    tracking_config: StructuralBenchmarkConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    with tracking_session(tracking_config or {}, job_type="structural-analysis") as tracker:
        tracker.set_state(**{"monitor/stage": "analysis", "run_root": run_root})
        return run_analyses(
            run_root,
            preset=preset,
            selected=enabled,
            disabled=disabled,
            method_configs=method_configs,
            force=force,
            continue_on_error=continue_on_error,
        )


def run_structural_benchmark(
    config: StructuralBenchmarkConfig | Mapping[str, Any],
) -> dict[str, Any]:
    resolved = _coerce_structural_config(config)
    if not resolved.run_id:
        resolved = resolved.with_run_id(datetime.now().strftime("%Y%m%d_%H%M%S"))
    with tracking_session(resolved, job_type="structural-benchmark") as tracker:
        capture_result = run_structural_capture(resolved)
        if not resolved.run_analysis:
            return {"capture": capture_result, "analysis": None}
        analysis_result = run_structural_analysis(
            str(capture_result["run_root"]),
            preset=resolved.analysis_preset,
            enabled=resolved.enable_analyses,
            disabled=resolved.disable_analyses,
            method_configs=resolved.analysis_method_configs,
            force=resolved.force,
            continue_on_error=resolved.analysis_continue_on_error,
        )
        render_result = None
        if resolved.render_graphs:
            from src.graphs.runtime import render_run

            tracker.set_state(**{"monitor/stage": "render", "monitor/substage": "rendering"})
            render_result = render_run(
                str(capture_result["run_root"]),
                preset=resolved.renderer_preset,
                enabled=resolved.enable_renderers,
                disabled=resolved.disable_renderers,
                force=resolved.force,
                continue_on_error=resolved.render_continue_on_error,
            )
        return {
            "capture": capture_result,
            "analysis": analysis_result,
            "render": render_result,
        }


__all__ = [
    "run_structural_analysis",
    "run_structural_benchmark",
    "run_structural_capture",
    "validate_structural_config",
]
