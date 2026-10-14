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


@dataclass(frozen=True)
class RendererSpec(RegistryEntry):
    runner: str = ""
    model_families: tuple[str, ...] = ("all",)
    requires_execution: bool = False
    required_captures: tuple[str, ...] = ()
    optional_captures: tuple[str, ...] = ()
    required_analyses: tuple[str, ...] = ()
    optional_analyses: tuple[str, ...] = ()
    option_keys: tuple[str, ...] = ()
    requires_analyses: bool = False

    def load(self) -> Callable[[Any], list[str]]:
        return load_object(self.runner)


RENDERERS = NamedRegistry([RendererSpec(
    "gram-report", "GRAM cases, checkpoint baselines and ROME efficacy metrics.",
    "src.graphs.gram:render_gram_report", requires_execution=True,
    required_captures=("gram-localization",), required_analyses=("gram-localization",),
    option_keys=("formats",),
)])
RENDERER_PRESETS = {"none": (), "gram-report": ("gram-report",), "full": ("gram-report",)}


def resolve_renderers(
    preset: str,
    *,
    enabled: Sequence[str] = (),
    disabled: Sequence[str] = (),
) -> tuple[str, ...]:
    return resolve_preset_selection(
        RENDERER_PRESETS,
        RENDERERS,
        preset,
        enabled=enabled,
        disabled=disabled,
        preset_label="renderer preset",
    )
