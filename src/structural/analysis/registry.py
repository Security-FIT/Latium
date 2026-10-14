"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Sequence
from src.registry import NamedRegistry, RegistryEntry, load_object, resolve_preset_selection
from src.registry import supports_model as _supports_model


@dataclass(frozen=True)
class AnalysisSpec(RegistryEntry):
    category: str = ""
    runner: str = ""
    required_captures: tuple[str, ...] = ()
    config_fields: tuple[str, ...] = ()
    variant_fields: tuple[tuple[str, str], ...] = ()
    model_families: tuple[str, ...] = ("all",)

    def load(self) -> Callable[..., dict[str, Any]]:
        return load_object(self.runner)

ANALYSES = NamedRegistry([AnalysisSpec(
    "gram-localization", "Localize a ROME-style edit from a single-checkpoint profile.",
    "detection", "src.structural.analysis.detector_methods:analyze_gram_localization",
    ("gram-localization",),
)])
ANALYSIS_PRESETS = {"none": (), "gram-localization": ("gram-localization",), "full": ("gram-localization",)}

def supports_model(spec: AnalysisSpec, model: str) -> bool:
    return _supports_model(spec, model)


def resolve_analyses(
    preset: str,
    *,
    enabled: Sequence[str] = (),
    disabled: Sequence[str] = (),
) -> tuple[str, ...]:
    return resolve_preset_selection(
        ANALYSIS_PRESETS,
        ANALYSES,
        preset,
        enabled=enabled,
        disabled=disabled,
        preset_label="analysis preset",
    )
