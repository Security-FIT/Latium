"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Optional, Sequence

from src.common.config import (
    is_sequence as _is_sequence,
    mapping_section as _section,
)
from src.structural.config import AnalysisVariantConfig, ModelRunPlan, StructuralBenchmarkConfig


def _native_values(raw: Any, default: Sequence[Any], *, name: str) -> list[Any]:
    source = default if raw is None else raw
    if not _is_sequence(source):
        raise TypeError(f"{name} must be a YAML list")
    return list(source)


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


def _strict_int(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} values must be integers")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{name} values must be at least {minimum}")
    return resolved


def parse_local_windows(raw: Any, default: Sequence[int] = (3, 5, 7)) -> list[int]:
    values = _native_values(raw, default, name="local_windows")
    output: list[int] = []
    for item in values:
        window = _strict_int(item, name="local_windows", minimum=1)
        if window % 2 == 0:
            raise ValueError("local_windows values must be odd")
        if window not in output:
            output.append(window)
    if not output:
        raise ValueError("local_windows must not be empty")
    return output


def parse_int_values(
    raw: Any,
    default: Sequence[int],
    *,
    min_value: int = 0,
    force_odd: bool = False,
) -> list[int]:
    values = _native_values(raw, default, name="integer sweep")
    output: list[int] = []
    for item in values:
        value = _strict_int(item, name="integer sweep", minimum=min_value)
        if force_odd and value % 2 == 0:
            raise ValueError("integer sweep values must be odd")
        if value not in output:
            output.append(value)
    if not output:
        raise ValueError("integer sweep must not be empty")
    return output


def parse_trim_values(raw: Any, default: Sequence[Optional[int]]) -> list[Optional[int]]:
    values = _native_values(raw, default, name="trim sweep")
    output: list[Optional[int]] = []
    for item in values:
        value = None if item is None else _strict_int(item, name="trim sweep", minimum=0)
        if value not in output:
            output.append(value)
    if not output:
        raise ValueError("trim sweep must not be empty")
    return output


def parse_local_window_sets(
    raw: Any,
    default: Sequence[Sequence[int]] = ((3, 5, 7),),
) -> list[tuple[int, ...]]:
    values = _native_values(raw, default, name="local_window_sets")
    output: list[tuple[int, ...]] = []
    for item in values:
        if not _is_sequence(item):
            raise TypeError("local_window_sets must be a YAML list of lists")
        windows = tuple(parse_local_windows(item))
        if windows not in output:
            output.append(windows)
    if not output:
        raise ValueError("local_window_sets must not be empty")
    return output


def _expand_for_zip(values: Sequence[object], target_len: int, arg_name: str) -> list[object]:
    if len(values) == target_len:
        return list(values)
    if len(values) == 1:
        return list(values) * target_len
    raise ValueError(f'In sweep zip mode, {arg_name} must have length 1 or {target_len} (got {len(values)}).')


def build_analysis_variants(
    *,
    spectral_top_k_values: Sequence[int],
    trim_first_values: Sequence[Optional[int]],
    trim_last_values: Sequence[Optional[int]],
    spectral_neighbor_layers_values: Sequence[int],
    spectral_rolling_window_values: Sequence[int],
    local_window_sets: Sequence[Sequence[int]],
    mode: str = 'zip',
    max_configs: Optional[int] = None,
) -> list[AnalysisVariantConfig]:
    topks = parse_int_values(spectral_top_k_values, default=(50,), min_value=1)
    trim_firsts = parse_trim_values(trim_first_values, default=(None,))
    trim_lasts = parse_trim_values(trim_last_values, default=(None,))
    neighbors = parse_int_values(spectral_neighbor_layers_values, default=(1,), min_value=1)
    rollings = parse_int_values(spectral_rolling_window_values, default=(5,), min_value=1, force_odd=True)

    window_sets: list[tuple[int, ...]] = []
    seen_windows: set[tuple[int, ...]] = set()
    for seq in local_window_sets:
        normalized = tuple(parse_local_windows(seq, default=(3, 5, 7)))
        if normalized and normalized not in seen_windows:
            seen_windows.add(normalized)
            window_sets.append(normalized)
    if not window_sets:
        window_sets = [tuple(parse_local_windows(None, default=(3, 5, 7)))]

    if mode not in {'zip', 'product'}:
        raise ValueError(f'Unsupported sweep mode: {mode}')

    if mode == 'product':
        iterable = itertools.product(topks, trim_firsts, trim_lasts, neighbors, rollings, window_sets)
    else:
        target = max(len(topks), len(trim_firsts), len(trim_lasts), len(neighbors), len(rollings), len(window_sets))
        iterable = zip(
            _expand_for_zip(topks, target, '--sweep-spectral-top-k'),
            _expand_for_zip(trim_firsts, target, '--sweep-trim-first'),
            _expand_for_zip(trim_lasts, target, '--sweep-trim-last'),
            _expand_for_zip(neighbors, target, '--sweep-spectral-neighbor-layers'),
            _expand_for_zip(rollings, target, '--sweep-spectral-rolling-window'),
            _expand_for_zip(window_sets, target, '--sweep-local-window-sets'),
        )

    configs: list[AnalysisVariantConfig] = []
    seen: set[tuple[object, ...]] = set()
    if max_configs is not None and int(max_configs) < 1:
        raise ValueError("max_configs must be at least 1 or null")
    limit = None if max_configs is None else int(max_configs)
    for top_k, trim_first, trim_last, neighbor_layers, rolling_window, window_set in iterable:
        config = AnalysisVariantConfig(
            spectral_top_k=int(top_k),
            trim_first=None if trim_first is None else int(trim_first),
            trim_last=None if trim_last is None else int(trim_last),
            spectral_neighbor_layers=int(neighbor_layers),
            spectral_rolling_window=int(rolling_window),
            local_windows=tuple(int(w) for w in window_set),
        )
        key = (
            config.spectral_top_k,
            config.trim_first,
            config.trim_last,
            config.spectral_neighbor_layers,
            config.spectral_rolling_window,
            config.local_windows,
        )
        if key in seen:
            continue
        seen.add(key)
        configs.append(config)
        if limit is not None and len(configs) >= limit:
            break

    return configs or [AnalysisVariantConfig()]


def analysis_variant_settings(structural: Mapping[str, Any]) -> dict[str, Any]:
    defaults = AnalysisVariantConfig()
    analysis = _section(structural, "analysis")
    variants = _section(analysis, "variants")
    sweep = _section(variants, "sweep")

    local_windows = tuple(
        parse_local_windows(
            variants.get("local_windows", defaults.local_windows),
            default=defaults.local_windows,
        )
    )
    spectral_top_k = variants.get("spectral_top_k", defaults.spectral_top_k)
    trim_first = variants.get("trim_first", defaults.trim_first)
    trim_last = variants.get("trim_last", defaults.trim_last)
    neighbor_layers = variants.get("spectral_neighbor_layers", defaults.spectral_neighbor_layers)
    rolling_window = variants.get("spectral_rolling_window", defaults.spectral_rolling_window)

    return {
        "spectral_top_k": int(spectral_top_k),
        "trim_first": None if trim_first is None else int(trim_first),
        "trim_last": None if trim_last is None else int(trim_last),
        "spectral_neighbor_layers": int(neighbor_layers),
        "spectral_rolling_window": int(rolling_window),
        "local_windows": local_windows,
        "analysis_variants": tuple(
            build_analysis_variants(
                spectral_top_k_values=parse_int_values(
                    sweep.get("spectral_top_k"),
                    default=[int(spectral_top_k)],
                    min_value=1,
                ),
                trim_first_values=parse_trim_values(
                    sweep.get("trim_first"),
                    default=[None if trim_first is None else int(trim_first)],
                ),
                trim_last_values=parse_trim_values(
                    sweep.get("trim_last"),
                    default=[None if trim_last is None else int(trim_last)],
                ),
                spectral_neighbor_layers_values=parse_int_values(
                    sweep.get("spectral_neighbor_layers"),
                    default=[int(neighbor_layers)],
                    min_value=1,
                ),
                spectral_rolling_window_values=parse_int_values(
                    sweep.get("spectral_rolling_window"),
                    default=[int(rolling_window)],
                    min_value=1,
                    force_odd=True,
                ),
                local_window_sets=parse_local_window_sets(
                    sweep.get("local_window_sets"),
                    default=[local_windows],
                ),
                mode=str(sweep.get("mode", "zip")),
                max_configs=(None if sweep.get("max_configs") is None else int(sweep["max_configs"])),
            )
        ),
    }


def format_optional_int(value: object) -> str:
    if value is None:
        return 'auto'
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return 'auto'


def analysis_variant_slug(cfg: AnalysisVariantConfig | dict[str, object]) -> str:
    if isinstance(cfg, AnalysisVariantConfig):
        local_windows = [int(w) for w in cfg.local_windows]
        spectral_top_k = cfg.spectral_top_k
        trim_first = cfg.trim_first
        trim_last = cfg.trim_last
        neighbor_layers = cfg.spectral_neighbor_layers
        rolling_window = cfg.spectral_rolling_window
    else:
        local_windows = parse_local_windows(cfg.get('local_windows', (3, 5, 7)), default=(3, 5, 7))
        spectral_top_k = int(cfg.get('spectral_top_k', 50))
        trim_first = cfg.get('trim_first')
        trim_last = cfg.get('trim_last')
        neighbor_layers = int(cfg.get('spectral_neighbor_layers', 1))
        rolling_window = int(cfg.get('spectral_rolling_window', 5))

    local_window_slug = '-'.join(str(w) for w in local_windows)
    return (
        f'tk{int(spectral_top_k)}'
        f'_tf{format_optional_int(trim_first)}'
        f'_tl{format_optional_int(trim_last)}'
        f'_nl{int(neighbor_layers)}'
        f'_rw{int(rolling_window)}'
        f'_lw{local_window_slug}'
    )


def build_model_run_plans(
    config: StructuralBenchmarkConfig,
    *,
    run_id: Optional[str] = None,
) -> list[ModelRunPlan]:
    resolved_run_id = run_id or config.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    plans: list[ModelRunPlan] = []

    for model_key in config.models:
        for run_idx in range(1, int(config.runs_per_model) + 1):
            start_idx = int(config.start_idx) + int(config.run_start_idx_step) * (run_idx - 1)
            end_idx = start_idx + max(0, int(config.n_tests) - 1)
            plan_id = f'cases{start_idx}-{end_idx}_r{run_idx:02d}'
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


def build_plan_summary(
    config: StructuralBenchmarkConfig,
    *,
    run_id: Optional[str] = None,
) -> dict[str, object]:
    resolved_run_id = run_id or config.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    plans = build_model_run_plans(config, run_id=resolved_run_id)
    return {
        'run_id': resolved_run_id,
        'models': list(config.models),
        'n_tests': int(config.n_tests),
        'start_idx': int(config.start_idx),
        'edit_methods': list(config.edit_methods),
        'capture_profile': config.capture_profile,
        'enable_captures': list(config.enable_captures),
        'disable_captures': list(config.disable_captures),
        'matrix_feature_set': config.matrix_feature_set,
        'matrix_features': list(config.matrix_features),
        'matrix_svd_top_k': int(config.matrix_svd_top_k),
        'analysis_preset': config.analysis_preset,
        'enable_analyses': list(config.enable_analyses),
        'disable_analyses': list(config.disable_analyses),
        'run_analysis': config.run_analysis,
        'renderer_preset': config.renderer_preset,
        'enable_renderers': list(config.enable_renderers),
        'disable_renderers': list(config.disable_renderers),
        'render_graphs': config.render_graphs,
        'output_dir': str(config.output_dir),
        'runs_per_model': int(config.runs_per_model),
        'run_start_idx_step': int(config.run_start_idx_step),
        'case_index_file': config.case_index_file,
        'analysis_variants': [
            {
                "identifier": analysis_variant_slug(variant),
                "config": variant.to_dict(),
            }
            for variant in config.effective_analysis_variants
        ],
        'planned_runs': [plan.to_record() for plan in plans],
    }
