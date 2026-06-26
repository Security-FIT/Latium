"""
Structural runtime configuration models.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional


@dataclass(frozen=True)
class AnalysisVariantConfig:
    spectral_top_k: int = 50
    trim_first: Optional[int] = None
    trim_last: Optional[int] = None
    spectral_neighbor_layers: int = 1
    spectral_rolling_window: int = 5
    local_windows: tuple[int, ...] = (3, 5, 7)

    def to_dict(self) -> dict[str, object]:
        return {
            "spectral_top_k": int(self.spectral_top_k),
            "trim_first": None if self.trim_first is None else int(self.trim_first),
            "trim_last": None if self.trim_last is None else int(self.trim_last),
            "spectral_neighbor_layers": int(self.spectral_neighbor_layers),
            "spectral_rolling_window": int(self.spectral_rolling_window),
            "local_windows": [int(window) for window in self.local_windows],
        }


@dataclass(frozen=True)
class ModelRunPlan:
    model_key: str
    run_id: str
    plan_id: str
    run_index: int
    start_idx: int
    end_idx: int

    def to_record(self) -> dict[str, object]:
        return {
            "model_key": self.model_key,
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "run_index": self.run_index,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
        }


_DEFAULTS: dict[str, Any] = {
    "models": (),
    "edit_methods": ("rome",),
    "n_tests": 30,
    "start_idx": 0,
    "case_index_file": None,
    "run_start_idx_step": 0,
    "runs_per_model": 1,
    "output_dir": Path("./analysis_out"),
    "run_id": None,
    "progress_file": None,
    "progress_interval": 10,
    "worker_id": None,
    "fail_on_missing_second_moment": False,
    "force": False,
    "case_dataset_name": "",
    "case_dataset_split": "",
    "capture_profile": "spectral",
    "enable_captures": (),
    "disable_captures": (),
    "spectral_top_k": 50,
    "trim_first": None,
    "trim_last": None,
    "spectral_neighbor_layers": 1,
    "spectral_rolling_window": 5,
    "local_windows": (3, 5, 7),
    "bottom_rank_sweep_ranks": (4, 8, 16, 32),
    "bottom_rank_top_svd_rank": 64,
    "bottom_rank_boundary": 2,
    "analysis_variants": (),
    "analysis_method_configs": {},
    "analysis_preset": "paper",
    "enable_analyses": (),
    "disable_analyses": (),
    "run_analysis": True,
    "render_graphs": False,
    "renderer_preset": "none",
    "enable_renderers": (),
    "disable_renderers": (),
    "seed": 0,
    "hf_token": None,
    "prefix_log_all": False,
    "second_moment_allow_autocompute": False,
    "log_skip_traceback": False,
}

_STR_TUPLE_FIELDS = {
    "models",
    "edit_methods",
    "enable_captures",
    "disable_captures",
    "enable_analyses",
    "disable_analyses",
    "enable_renderers",
    "disable_renderers",
}
_INT_TUPLE_FIELDS = {"local_windows", "bottom_rank_sweep_ranks"}
_INT_FIELDS = {
    "n_tests",
    "start_idx",
    "run_start_idx_step",
    "runs_per_model",
    "progress_interval",
    "spectral_top_k",
    "spectral_neighbor_layers",
    "spectral_rolling_window",
    "bottom_rank_top_svd_rank",
    "bottom_rank_boundary",
    "seed",
}
_OPTIONAL_INT_FIELDS = {"trim_first", "trim_last"}
_BOOL_FIELDS = {
    "fail_on_missing_second_moment",
    "force",
    "run_analysis",
    "render_graphs",
    "prefix_log_all",
    "second_moment_allow_autocompute",
    "log_skip_traceback",
}


class StructuralBenchmarkConfig:
    """Normalized structural runtime config with flat compatibility fields."""

    __slots__ = ("_values",)

    def __init__(self, **values: Any) -> None:
        unknown = sorted(set(values) - set(_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown structural config fields: {', '.join(unknown)}")
        normalized = {
            field: _normalize_field(field, values[field] if field in values else _default_value(field))
            for field in _DEFAULTS
        }
        object.__setattr__(self, "_values", MappingProxyType(normalized))

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __repr__(self) -> str:
        args = ", ".join(f"{key}={value!r}" for key, value in self._values.items())
        return f"{self.__class__.__name__}({args})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StructuralBenchmarkConfig) and self.to_dict() == other.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def with_run_id(self, run_id: str) -> "StructuralBenchmarkConfig":
        return StructuralBenchmarkConfig(**{**self._values, "run_id": run_id})

    @property
    def effective_analysis_variants(self) -> tuple[AnalysisVariantConfig, ...]:
        if self.analysis_variants:
            return tuple(self.analysis_variants)
        return (
            AnalysisVariantConfig(
                spectral_top_k=int(self.spectral_top_k),
                trim_first=self.trim_first,
                trim_last=self.trim_last,
                spectral_neighbor_layers=int(self.spectral_neighbor_layers),
                spectral_rolling_window=int(self.spectral_rolling_window),
                local_windows=tuple(int(window) for window in self.local_windows),
            ),
        )


def _default_value(field: str) -> Any:
    value = _DEFAULTS[field]
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, Path):
        return Path(value)
    return value


def _normalize_field(field: str, value: Any) -> Any:
    if field in _STR_TUPLE_FIELDS:
        return tuple(str(item) for item in _as_sequence(value) if item not in (None, ""))
    if field in _INT_TUPLE_FIELDS:
        return tuple(int(item) for item in _as_sequence(value))
    if field in _INT_FIELDS:
        return int(value)
    if field in _OPTIONAL_INT_FIELDS:
        return None if value is None else int(value)
    if field in _BOOL_FIELDS:
        return _bool(value)
    if field == "output_dir":
        return value if isinstance(value, Path) else Path(str(value))
    if field == "analysis_variants":
        return tuple(_analysis_variant(item) for item in _as_sequence(value))
    if field == "analysis_method_configs":
        return {str(key): dict(item) for key, item in dict(value or {}).items() if isinstance(item, Mapping)}
    return value


def _analysis_variant(value: Any) -> AnalysisVariantConfig:
    if isinstance(value, AnalysisVariantConfig):
        return value
    if isinstance(value, Mapping):
        return AnalysisVariantConfig(**dict(value))
    raise TypeError(f"Unsupported analysis variant config: {type(value)!r}")


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(value)
    return (value,)


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


__all__ = [
    "AnalysisVariantConfig",
    "ModelRunPlan",
    "StructuralBenchmarkConfig",
]
