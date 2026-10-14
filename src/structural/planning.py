"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional, Sequence
from src.structural.config import ModelRunPlan, StructuralBenchmarkConfig


def normalize_models_arg(models: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for entry in models:
        name = str(entry).strip()
        if "," in name or ";" in name:
            raise ValueError("models must be a YAML list; comma/semicolon strings are not supported")
        if name and name not in seen:
            seen.add(name)
            normalized.append(name)
    return normalized


def build_model_run_plans(
    config: StructuralBenchmarkConfig,
    *,
    run_id: Optional[str] = None,
) -> list[ModelRunPlan]:
    resolved_run_id = run_id or config.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    plans: list[ModelRunPlan] = []
    cohort = None
    if config.case_index_file:
        from src.counterfact_selection import load_case_manifest
        cohort = load_case_manifest(config.case_index_file)

    for model_key in config.models:
        for run_idx in range(1, int(config.runs_per_model) + 1):
            start_idx = int(config.start_idx) + int(config.run_start_idx_step) * (run_idx - 1)
            end_idx = start_idx + max(0, int(config.n_tests) - 1)
            plan_id = f'cases{start_idx}-{end_idx}_r{run_idx:02d}'
            if cohort is not None:
                if start_idx < 0 or start_idx + config.n_tests > cohort['count']:
                    raise ValueError('Planned range exceeds case manifest')
                plan_id = f"cf_{cohort['manifest_hash'][:12]}_m{start_idx:04d}-{start_idx + config.n_tests:04d}_r{run_idx:02d}"
            plans.append(
                ModelRunPlan(
                    model_key=model_key,
                    run_id=resolved_run_id,
                    plan_id=plan_id,
                    run_index=run_idx,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
            )
    return plans

def build_plan_summary(config, *, run_id=None):
    from src.structural.analysis.registry import resolve_analyses
    from src.structural.capture.registry import resolve_capture_plan
    from src.graphs.registry import resolve_renderers
    resolved = run_id or config.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    analyses = resolve_analyses(config.analysis_preset, enabled=config.enable_analyses, disabled=config.disable_analyses)
    renderers = resolve_renderers(config.renderer_preset, enabled=config.enable_renderers, disabled=config.disable_renderers)
    captures = resolve_capture_plan(config.capture_profile, enabled=config.enable_captures, disabled=config.disable_captures,
                                   analyses=analyses if config.run_analysis else (),
                                   renderers=renderers if config.render_graphs else ())
    return {**{k: list(v) if isinstance(v, tuple) else v for k, v in config.to_dict().items()}, 'output_dir': str(config.output_dir), 'run_id': resolved,
            'resolved_captures': list(captures.names), 'resolved_analyses': list(analyses),
            'resolved_renderers': list(renderers),
            'planned_runs': [plan.to_record() for plan in build_model_run_plans(config, run_id=resolved)]}
