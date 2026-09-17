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
    is_sequence as _is_sequence,
    mapping_section as _section,
    optional_str as _optional_str,
    plain as _plain,
    strict_bool,
    string_list as _string_list,
)
from src.structural.config import StructuralBenchmarkConfig, strict_int
from src.structural.planning import (
    analysis_variant_settings,
    normalize_models_arg,
    parse_int_values,
)

Converter = Callable[[Any], Any]
Spec = tuple[str, str, Any, Converter]


def _to_bool(value: Any) -> bool:
    return strict_bool(value, name="structural boolean")


def _to_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def _to_str_tuple(value: Any) -> tuple[str, ...]:
    raw = _plain(value)
    if raw is None:
        return ()
    if not _is_sequence(raw):
        raise TypeError("structural list values must use native YAML lists")
    output: list[str] = []
    for item in raw:
        normalized = str(item).strip() if item is not None else ""
        if "," in normalized or ";" in normalized:
            raise ValueError("structural list values must not contain comma/semicolon pseudo-lists")
        if normalized and normalized not in output:
            output.append(normalized)
    return tuple(output)


def _to_models(value: Any) -> tuple[str, ...]:
    return tuple(normalize_models_arg(_to_str_tuple(value)))


def _int_at_least(minimum: int) -> Converter:
    def convert(value: Any) -> int:
        return strict_int(_plain(value), name="structural integer value", minimum=minimum)

    return convert


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
    "tracking_provider": ("tracking", "provider", "none", str),
    "tracking_project": ("tracking", "project", "latium", str),
    "tracking_entity": ("tracking", "entity", None, _optional_str),
    "tracking_mode": ("tracking", "mode", "online", str),
    "tracking_run_name": ("tracking", "run_name", None, _optional_str),
    "tracking_group": ("tracking", "group", None, _optional_str),
    "tracking_tags": ("tracking", "tags", (), _to_str_tuple),
    "tracking_heartbeat_seconds": ("tracking", "heartbeat_seconds", 60, _int_at_least(1)),
    "fail_on_missing_second_moment": ("run", "fail_on_missing_second_moment", False, _to_bool),
    "force": ("run", "force", False, _to_bool),
    "capture_profile": ("capture", "profile", "none", str),
    "enable_captures": ("capture", "enable", (), _to_str_tuple),
    "disable_captures": ("capture", "disable", (), _to_str_tuple),
    "analysis_preset": ("analysis", "preset", "paper", str),
    "enable_analyses": ("analysis", "enable", (), _to_str_tuple),
    "disable_analyses": ("analysis", "disable", (), _to_str_tuple),
    "analysis_continue_on_error": ("analysis", "continue_on_error", False, _to_bool),
    "render_graphs": ("render", "enabled", False, _to_bool),
    "renderer_preset": ("render", "renderer_preset", "ccs-report", str),
    "renderer_style_preset": ("render", "style_preset", "default", str),
    "enable_renderers": ("render", "enable", (), _to_str_tuple),
    "disable_renderers": ("render", "disable", (), _to_str_tuple),
    "render_continue_on_error": ("render", "continue_on_error", False, _to_bool),
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
        "bottom_rank_top_svd_rank": _int_at_least(1)(bottom_rank.get("top_svd_rank", 64)),
        "bottom_rank_boundary": _int_at_least(0)(bottom_rank.get("boundary", 2)),
    }


def _matrix_feature_settings(structural: Mapping[str, Any]) -> dict[str, Any]:
    matrix_features = _section(_section(structural, "capture"), "matrix_features")
    return {
        "matrix_feature_set": str(matrix_features.get("feature_set", "paper")),
        "matrix_features": tuple(_string_list(matrix_features.get("features"))),
        "matrix_svd_top_k": _int_at_least(1)(matrix_features.get("svd_top_k", 50)),
    }


def _rome_experiment_settings(structural: Mapping[str, Any]) -> dict[str, Any]:
    experiments = _section(_section(structural, "capture"), "rome_experiments")
    return {
        "rome_experiment_groups": tuple(_string_list(experiments.get("groups", ["neighbors"]))),
    }


def _renderer_settings(structural: Mapping[str, Any]) -> dict[str, Any]:
    return {"renderer_options": _dict_section(_section(structural, "render"), "renderers")}


def _runtime_settings(cfg: DictConfig) -> dict[str, Any]:
    runtime = cfg.runtime
    return {
        "seed": strict_int(cfg.seed, name="seed"),
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
        **_rome_experiment_settings(structural),
        **_renderer_settings(structural),
        **_bottom_rank_settings(structural),
        analysis_method_configs=_dict_section(analysis, "methods"),
        run_analysis=run_analysis,
        **_runtime_settings(cfg),
    )


__all__ = ["structural_config_from_hydra"]
