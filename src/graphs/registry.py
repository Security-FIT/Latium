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
from src.structural.capture.matrix_features import PAPER_FEATURES


@dataclass(frozen=True)
class RendererSpec(RegistryEntry):
    runner: str = ""
    model_families: tuple[str, ...] = ("all",)
    requires_execution: bool = False
    required_captures: tuple[str, ...] = ()
    optional_captures: tuple[str, ...] = ()
    required_analyses: tuple[str, ...] = ()
    optional_analyses: tuple[str, ...] = ()
    required_matrix_features: tuple[str, ...] = ()
    option_keys: tuple[str, ...] = ()
    requires_analyses: bool = False

    def load(self) -> Callable[[Any], list[str]]:
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
            "Fixed 5x4 per-layer artifact grid from matrix features.",
            "src.graphs.structural.artifact_grid:render_structural_artifact_grid",
            requires_execution=True,
            required_captures=("matrix-features",),
            required_matrix_features=PAPER_FEATURES,
            option_keys=("features", "transforms", "formats"),
        ),
        RendererSpec(
            "rome-relative-profile-grid",
            "Per-layer relative ROME profiles, B0 fits, baseline, and case diagnostics.",
            "src.graphs.structural.relative_profiles:render_rome_relative_profile_grid",
            requires_execution=True,
            optional_captures=(
                "gram-localization", "gram-directional-error-v1",
                "gram-cross-layer-v1", "token-subspace-alignment-v1",
            ),
            optional_analyses=(
                "rome-profile-experiments", "rome-directional-experiments",
                "rome-cross-layer-experiments", "rome-token-alignment-experiments",
            ),
            option_keys=("formats", "case_pages", "case_traces"),
        ),
    ]
)

RENDERER_PRESETS: dict[str, tuple[str, ...]] = {
    "none": (),
    "paper": ("paper", "detector", "rome-success", "detector-window"),
    "structural-paper": ("structural-artifact-grid",),
    "structural-full": ("structural-artifact-grid",),
    "rome-relative-paper": ("rome-relative-profile-grid",),
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
