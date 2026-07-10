"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

from src.registry import NamedRegistry, RegistryEntry, load_object, resolve_preset_selection

if TYPE_CHECKING:
    from src.graphs.context import RenderContext


@dataclass(frozen=True)
class RendererSpec(RegistryEntry):
    runner: str = ""
    requires_execution: bool = False
    requires_analyses: bool = False
    required_captures: tuple[str, ...] = ()
    optional_captures: tuple[str, ...] = ()
    required_analyses: tuple[str, ...] = ()
    optional_analyses: tuple[str, ...] = ()
    option_keys: tuple[str, ...] = ()
    schema_version: str = "1"

    def __post_init__(self) -> None:
        declared = (
            self.requires_execution
            or self.requires_analyses
            or self.required_captures
            or self.optional_captures
            or self.required_analyses
            or self.optional_analyses
        )
        if not declared:
            raise ValueError(f"Renderer {self.identifier!r} must declare its artifact inputs")

    def load(self) -> Callable[["RenderContext"], list[str]]:
        return load_object(self.runner)


RENDERERS = NamedRegistry(
    [
        RendererSpec(
            "paper",
            "Machine-readable paper analysis summary.",
            "src.graphs.renderers:render_paper",
            requires_analyses=True,
        ),
        RendererSpec(
            "detector",
            "Artifact-only detector summary and accuracy graph.",
            "src.graphs.renderers:render_detector",
            requires_analyses=True,
        ),
        RendererSpec(
            "run-summary",
            "Run-level aggregate summaries.",
            "src.graphs.renderers:render_run_summary",
            requires_analyses=True,
        ),
        RendererSpec(
            "rome-success",
            "ROME execution success rates and score summaries.",
            "src.graphs.renderers:render_rome_success",
            requires_execution=True,
        ),
        RendererSpec(
            "detector-window",
            "Detector layer-window accuracy and distance summaries.",
            "src.graphs.renderers:render_detector_window",
            requires_analyses=True,
        ),
        RendererSpec(
            "detector-signals",
            "Per-analysis detector signal profile plots.",
            "src.graphs.renderers:render_detector_signals",
            requires_analyses=True,
        ),
        RendererSpec(
            "structural-artifact-grid",
            "Legacy-compatible 5x4 per-layer artifact grid from matrix features.",
            "src.graphs.structural.artifact_grid:render_structural_artifact_grid",
            requires_execution=True,
            required_captures=("matrix-features",),
            option_keys=("features", "transforms", "formats"),
            schema_version="1",
        ),
    ]
)

RENDERER_PRESETS: dict[str, tuple[str, ...]] = {
    "none": (),
    "paper": ("paper", "detector", "rome-success", "detector-window"),
    "structural-paper": ("structural-artifact-grid",),
    "structural-full": ("structural-artifact-grid",),
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
