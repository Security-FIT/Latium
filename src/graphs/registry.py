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
    schema_version: str = "1"

    def load(self) -> Callable[[Any], list[str]]:
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
        RendererSpec(
            "rome-success",
            "ROME execution success rates and score summaries.",
            "src.graphs.renderers:render_rome_success",
        ),
        RendererSpec(
            "detector-window",
            "Detector layer-window accuracy and distance summaries.",
            "src.graphs.renderers:render_detector_window",
        ),
        RendererSpec(
            "detector-signals",
            "Per-analysis detector signal profile plots.",
            "src.graphs.renderers:render_detector_signals",
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
        RendererSpec(
            "rome-detector-explainer",
            "All per-layer weighted-spectrum statistics and ROME-presence decisions.",
            "src.graphs.structural.rome_detector:render_rome_detector_explainer",
            requires_execution=True,
            required_analyses=(
                "weighted-spectrum",
                "rome-presence-blind-peak",
                "rome-presence-blind-footprint",
                "rome-presence-delta",
            ),
            option_keys=("formats", "max_cases", "profile_fields"),
            schema_version="1",
        ),
    ]
)

RENDERER_PRESETS: dict[str, tuple[str, ...]] = {
    "none": (),
    "paper": ("paper", "detector", "rome-success", "detector-window"),
    "structural-paper": ("structural-artifact-grid",),
    "structural-full": ("structural-artifact-grid",),
    "rome-presence": ("rome-detector-explainer", "rome-success", "detector-window"),
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
