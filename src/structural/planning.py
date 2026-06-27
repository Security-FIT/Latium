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
    optional_int as _optional_int,
)
from src.structural.config import AnalysisVariantConfig, ModelRunPlan, StructuralBenchmarkConfig


def _comma_tokens(raw: object) -> list[object]:
    if raw is None:
        return [None]
    if _is_sequence(raw):
        tokens: list[object] = []
        for item in raw:
            tokens.extend(_comma_tokens(item))
        return tokens
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(',') if part.strip()]
    return [raw]


def _looks_like_window_scalar(value: object) -> bool:
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        text = value.strip()
        return bool(text) and ',' not in text and ';' not in text
    return False


def normalize_models_arg(models: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for entry in models:
        for part in str(entry).split(','):
            name = part.strip()
            if name and name not in seen:
                seen.add(name)
                normalized.append(name)
    return normalized


def parse_local_windows(raw: Any, default: Sequence[int] = (3, 5, 7)) -> list[int]:
    if raw is None:
        return [int(w) for w in default]
    parts = _comma_tokens(raw)
    if not parts:
        return [int(w) for w in default]

    out: list[int] = []
    seen: set[int] = set()
    for part in parts:
        try:
            value = max(1, int(part))
        except (TypeError, ValueError):
            continue
        if value % 2 == 0:
            value += 1
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out or [int(w) for w in default]


def parse_int_values(
    raw: Any,
    default: Sequence[int],
    *,
    min_value: int = 0,
    force_odd: bool = False,
) -> list[int]:
    parts = [str(v) for v in default] if raw is None else _comma_tokens(raw)

    out: list[int] = []
    seen: set[int] = set()
    for part in parts:
        try:
            value = int(part)
        except (TypeError, ValueError):
            continue
        if value < min_value:
            value = min_value
        if force_odd and value % 2 == 0:
            value += 1
        if value not in seen:
            seen.add(value)
            out.append(value)

    if out:
        return out

    fallback: list[int] = []
    for value in default:
        normalized = max(min_value, int(value))
        if force_odd and normalized % 2 == 0:
            normalized += 1
        fallback.append(normalized)
    return fallback or ([1] if min_value <= 1 else [min_value])


def parse_trim_values(raw: Any, default: Sequence[Optional[int]]) -> list[Optional[int]]:
    source = (
        [None if value is None else str(value) for value in default]
        if raw is None
        else [item for item in _comma_tokens(raw)]
    )

    out: list[Optional[int]] = []
    seen: set[str] = set()
    for item in source:
        if item is None:
            value: Optional[int] = None
        else:
            token = str(item).strip().lower()
            if token in {'auto', 'none', 'default'}:
                value = None
            else:
                try:
                    value = max(0, int(token))
                except ValueError:
                    continue

        key = 'auto' if value is None else str(value)
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out or [None]


def parse_local_window_sets(
    raw: Any,
    default: Sequence[Sequence[int]] = ((3, 5, 7),),
) -> list[tuple[int, ...]]:
    default_base = tuple(default[0]) if default else (3, 5, 7)
    if raw is None:
        chunks: list[object] = [tuple(seq) for seq in default]
    elif _is_sequence(raw):
        raw_items = list(raw)
        if raw_items and all(_looks_like_window_scalar(item) for item in raw_items):
            chunks = [raw_items]
        else:
            chunks = []
            for item in raw_items:
                if isinstance(item, str):
                    chunks.extend(chunk.strip() for chunk in item.split(';') if chunk.strip())
                else:
                    chunks.append(item)
    else:
        chunks = [chunk.strip() for chunk in str(raw).split(';') if chunk.strip()]

    out: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for chunk in chunks:
        values = tuple(parse_local_windows(chunk, default=default_base))
        if values and values not in seen:
            seen.add(values)
            out.append(values)
    return out or [tuple(parse_local_windows(None, default=default_base))]


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
    topks = [max(1, int(v)) for v in spectral_top_k_values] or [50]
    trim_firsts = [None if v is None else max(0, int(v)) for v in trim_first_values] or [None]
    trim_lasts = [None if v is None else max(0, int(v)) for v in trim_last_values] or [None]
    neighbors = [max(1, int(v)) for v in spectral_neighbor_layers_values] or [1]
    rollings = [max(1, int(v)) for v in spectral_rolling_window_values] or [5]

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
    limit = None if max_configs is None else max(1, int(max_configs))
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
    analysis = _section(structural, "analysis")
    variants = _section(analysis, "variants")
    sweep = _section(variants, "sweep")

    local_windows = tuple(
        parse_local_windows(
            variants.get("local_windows", (3, 5, 7)),
            default=(3, 5, 7),
        )
    )
    spectral_top_k = variants.get("spectral_top_k", 50)
    trim_first = variants.get("trim_first")
    trim_last = variants.get("trim_last")
    neighbor_layers = variants.get("spectral_neighbor_layers", 1)
    rolling_window = variants.get("spectral_rolling_window", 5)

    return {
        "spectral_top_k": max(1, int(spectral_top_k)),
        "trim_first": _optional_int(trim_first),
        "trim_last": _optional_int(trim_last),
        "spectral_neighbor_layers": max(1, int(neighbor_layers)),
        "spectral_rolling_window": max(1, int(rolling_window)),
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
                    default=[_optional_int(trim_first)],
                ),
                trim_last_values=parse_trim_values(
                    sweep.get("trim_last"),
                    default=[_optional_int(trim_last)],
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
                max_configs=_optional_int(sweep.get("max_configs")),
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
        local_windows = parse_local_windows(cfg.get('local_windows', '3,5,7'), default=(3, 5, 7))
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
        for run_idx in range(1, max(1, int(config.runs_per_model)) + 1):
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
