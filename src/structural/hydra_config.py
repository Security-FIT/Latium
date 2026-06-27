"""
Hydra-to-structural config adapter.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.common.config import (
    dict_section as _dict_section,
    mapping_section as _section,
    optional_str as _optional_str,
    plain as _plain,
    string_list as _string_list,
)
from src.structural.config import StructuralBenchmarkConfig
from src.structural.planning import (
    analysis_variant_settings,
    normalize_models_arg,
    parse_int_values,
)

Converter = Callable[[Any], Any]
Spec = tuple[str, str, Any, Converter]


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _to_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def _to_str_tuple(value: Any) -> tuple[str, ...]:
    return tuple(_string_list(value))


def _to_models(value: Any) -> tuple[str, ...]:
    return tuple(normalize_models_arg(_string_list(value)))


def _int_at_least(minimum: int) -> Converter:
    return lambda value: max(minimum, int(value))


SPECS: Mapping[str, Spec] = {
    "models": ("run", "models", (), _to_models),
    "edit_methods": ("run", "edit_methods", ("rome",), _to_str_tuple),
    "n_tests": ("run", "n_tests", 30, _int_at_least(0)),
    "start_idx": ("run", "start_idx", 0, _int_at_least(0)),
    "case_index_file": ("run", "case_index_file", None, _optional_str),
    "run_start_idx_step": ("run", "run_start_idx_step", 0, _int_at_least(0)),
    "runs_per_model": ("run", "runs_per_model", 1, _int_at_least(1)),
    "output_dir": ("run", "output_dir", "./analysis_out", _to_path),
    "run_id": ("run", "run_id", None, _optional_str),
    "progress_file": ("run", "progress_file", None, _optional_str),
    "progress_interval": ("run", "progress_interval", 10, _int_at_least(1)),
    "worker_id": ("run", "worker_id", None, _optional_str),
    "fail_on_missing_second_moment": ("run", "fail_on_missing_second_moment", False, _to_bool),
    "force": ("run", "force", False, _to_bool),
    "capture_profile": ("capture", "profile", "spectral", str),
    "enable_captures": ("capture", "enable", (), _to_str_tuple),
    "disable_captures": ("capture", "disable", (), _to_str_tuple),
    "analysis_preset": ("analysis", "preset", "paper", str),
    "enable_analyses": ("analysis", "enable", (), _to_str_tuple),
    "disable_analyses": ("analysis", "disable", (), _to_str_tuple),
    "render_graphs": ("render", "enabled", False, _to_bool),
    "renderer_preset": ("render", "renderer_preset", "none", str),
    "enable_renderers": ("render", "enable", (), _to_str_tuple),
    "disable_renderers": ("render", "disable", (), _to_str_tuple),
}


def _collect(structural: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field, (section, key, default, convert) in SPECS.items():
        out[field] = convert(_section(structural, section).get(key, default))
    return out


def _bottom_rank_settings(structural: Mapping[str, Any]) -> dict[str, Any]:
    bottom_rank = _section(_section(structural, "analysis"), "bottom_rank")
    return {
        "bottom_rank_sweep_ranks": tuple(
            parse_int_values(
                bottom_rank.get("sweep_ranks"),
                default=(4, 8, 16, 32),
                min_value=1,
            )
        ),
        "bottom_rank_top_svd_rank": max(
            1,
            int(bottom_rank.get("top_svd_rank", 64)),
        ),
        "bottom_rank_boundary": max(0, int(bottom_rank.get("boundary", 2))),
    }


def _matrix_feature_settings(structural: Mapping[str, Any]) -> dict[str, Any]:
    matrix_features = _section(_section(structural, "capture"), "matrix_features")
    return {
        "matrix_feature_set": str(matrix_features.get("feature_set", "paper")),
        "matrix_features": tuple(_string_list(matrix_features.get("features"))),
        "matrix_svd_top_k": max(1, int(matrix_features.get("svd_top_k", 50))),
    }


def _runtime_settings(cfg: DictConfig) -> dict[str, Any]:
    runtime = cfg.runtime
    return {
        "seed": int(cfg.seed),
        "hf_token": _optional_str(runtime.hf_token),
        "prefix_log_all": _to_bool(runtime.prefix_log_all),
        "second_moment_allow_autocompute": _to_bool(runtime.second_moment_allow_autocompute),
        "log_skip_traceback": _to_bool(runtime.log_skip_traceback),
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
