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

    def load(self) -> Callable[[dict[str, Any]], list[str]]:
        return load_object(self.runner)


RENDERERS = NamedRegistry(
    [
        RendererSpec(
            "paper",
            "Machine-readable paper analysis summary.",
            "src.graphs.renderers:render_paper",
        ),
        RendererSpec(
            "detector",
            "Artifact-only detector summary and accuracy graph.",
            "src.graphs.renderers:render_detector",
        ),
        RendererSpec(
            "run-summary",
            "Run-level aggregate summaries.",
            "src.graphs.renderers:render_run_summary",
        ),
    ]
)

RENDERER_PRESETS: dict[str, tuple[str, ...]] = {
    "none": (),
    "paper": ("paper", "detector"),
    "full": RENDERERS.identifiers(),
}


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
