"""
Hydra-to-structural config adapter.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig

from src.common.config import (
    dict_section as _dict_section,
    mapping_section as _section,
    optional_str as _optional_str,
    plain as _plain,
    strict_bool,
    string_list as _string_list,
)
from src.structural.config import StructuralBenchmarkConfig
from src.structural.planning import (
    analysis_variant_settings,
    parse_int_values,
)

SPECS: Mapping[str, tuple[str, str]] = {
    "models": ("run", "models"),
    "edit_methods": ("run", "edit_methods"),
    "n_tests": ("run", "n_tests"),
    "start_idx": ("run", "start_idx"),
    "case_index_file": ("run", "case_index_file"),
    "run_start_idx_step": ("run", "run_start_idx_step"),
    "runs_per_model": ("run", "runs_per_model"),
    "output_dir": ("run", "output_dir"),
    "run_id": ("run", "run_id"),
    "progress_file": ("run", "progress_file"),
    "progress_interval": ("run", "progress_interval"),
    "worker_id": ("run", "worker_id"),
    "fail_on_missing_second_moment": ("run", "fail_on_missing_second_moment"),
    "force": ("run", "force"),
    "capture_profile": ("capture", "profile"),
    "enable_captures": ("capture", "enable"),
    "disable_captures": ("capture", "disable"),
    "analysis_preset": ("analysis", "preset"),
    "enable_analyses": ("analysis", "enable"),
    "disable_analyses": ("analysis", "disable"),
    "analysis_continue_on_error": ("analysis", "continue_on_error"),
    "render_graphs": ("render", "enabled"),
    "render_continue_on_error": ("render", "continue_on_error"),
    "renderer_preset": ("render", "renderer_preset"),
    "enable_renderers": ("render", "enable"),
    "disable_renderers": ("render", "disable"),
}


def _collect(structural: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field, (section, key) in SPECS.items():
        source = _section(structural, section)
        if key in source:
            out[field] = source[key]
    return out


def _bottom_rank_settings(structural: Mapping[str, Any]) -> dict[str, Any]:
    defaults = StructuralBenchmarkConfig()
    bottom_rank = _section(_section(structural, "analysis"), "bottom_rank")
    return {
        "bottom_rank_sweep_ranks": tuple(
            parse_int_values(
                bottom_rank.get("sweep_ranks"),
                default=defaults.bottom_rank_sweep_ranks,
                min_value=1,
            )
        ),
        "bottom_rank_top_svd_rank": int(bottom_rank.get("top_svd_rank", defaults.bottom_rank_top_svd_rank)),
        "bottom_rank_boundary": int(bottom_rank.get("boundary", defaults.bottom_rank_boundary)),
    }


def _matrix_feature_settings(structural: Mapping[str, Any]) -> dict[str, Any]:
    defaults = StructuralBenchmarkConfig()
    matrix_features = _section(_section(structural, "capture"), "matrix_features")
    return {
        "matrix_feature_set": str(matrix_features.get("feature_set", defaults.matrix_feature_set)),
        "matrix_features": tuple(_string_list(matrix_features.get("features"), defaults.matrix_features)),
        "matrix_svd_top_k": int(matrix_features.get("svd_top_k", defaults.matrix_svd_top_k)),
    }


def _runtime_settings(cfg: DictConfig) -> dict[str, Any]:
    runtime = cfg.runtime
    return {
        "seed": int(cfg.seed),
        "hf_token": _optional_str(runtime.hf_token),
        "prefix_log_all": strict_bool(runtime.prefix_log_all, name="runtime.prefix_log_all"),
        "second_moment_allow_autocompute": strict_bool(
            runtime.second_moment_allow_autocompute,
            name="runtime.second_moment_allow_autocompute",
        ),
        "log_skip_traceback": strict_bool(runtime.log_skip_traceback, name="runtime.log_skip_traceback"),
    }


def structural_config_from_hydra(
    cfg: DictConfig,
    *,
    run_analysis: bool,
) -> StructuralBenchmarkConfig:
    structural = _plain(cfg.structural) or {}
    dataset_facts = _plain(cfg.dataset_facts)
    if not isinstance(dataset_facts, dict):
        raise ValueError("dataset_facts config is required for structural commands")

    analysis = _section(structural, "analysis")
    return StructuralBenchmarkConfig(
        **_collect(structural),
        case_dataset_name=str(dataset_facts["name"]),
        case_dataset_split=str(dataset_facts["split"]),
        **analysis_variant_settings(structural),
        **_matrix_feature_settings(structural),
        **_bottom_rank_settings(structural),
        analysis_method_configs=_dict_section(analysis, "methods"),
        run_analysis=run_analysis,
        **_runtime_settings(cfg),
    )


__all__ = ["structural_config_from_hydra"]
