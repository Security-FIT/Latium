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
from src.structural.capture.matrix_features import resolve_matrix_features


@dataclass(frozen=True)
class CaptureSpec(RegistryEntry):
    """Describe capture inputs and whether baseline data is produced or overlaid."""

    producer: str = ""
    requires_probe: bool = False
    captures_baseline: bool = True
    requires_baseline: bool = True
    model_families: tuple[str, ...] = ("all",)
    weight_families: tuple[str, ...] = ("proj", "fc")

    def load(self) -> Callable[..., dict[str, Any]]:
        return load_object(self.producer)


@dataclass(frozen=True)
class CapturePlan:
    """Resolved captures and scalar features needed by one structural run."""

    names: tuple[str, ...]
    matrix_features: tuple[str, ...]


CAPTURES = NamedRegistry(
    [
        CaptureSpec(
            "spectral",
            "Reusable singular-value and principal-component primitives.",
            "src.structural.capture.producers:capture_spectral",
        ),
        CaptureSpec(
            "matrix-features",
            "Reusable per-layer matrix, rank, norm, and IPR profiles.",
            "src.structural.capture.producers:capture_matrix_features",
        ),
        CaptureSpec(
            "attention-features",
            "Reusable attention-family matrix profiles.",
            "src.structural.capture.producers:capture_attention_features",
            weight_families=("attention",),
        ),
        CaptureSpec(
            "matrix-anomaly-features",
            "Reusable experimental matrix anomaly profiles.",
            "src.structural.capture.producers:capture_matrix_anomaly_features",
        ),
        CaptureSpec(
            "gram-localization",
            "Single-checkpoint profile for localizing a ROME-style edit.",
            "src.structural.capture.producers:capture_gram_localization",
            requires_baseline=False,
            weight_families=("proj",),
        ),
        CaptureSpec(
            "bottom-rank-tokens",
            "Per-layer tail-response token sweeps for bottom-rank analysis.",
            "src.structural.capture.producers:capture_bottom_rank_tokens",
            requires_probe=True,
            captures_baseline=False,
            requires_baseline=False,
            weight_families=("proj",),
        ),
    ]
)

CAPTURE_PROFILES: dict[str, tuple[str, ...]] = {
    "none": (),
    "spectral": ("spectral",),
    "matrix": ("matrix-features",),
    "paper": ("spectral", "matrix-features"),
    "gram-localization": ("gram-localization",),
    "full": CAPTURES.identifiers(),
}


def required_weight_families(capture_names: Sequence[str]) -> frozenset[str]:
    return frozenset(
        family
        for capture_name in capture_names
        for family in CAPTURES.get(capture_name).weight_families
    )


def captures_require_probe(capture_names: Sequence[str]) -> bool:
    return any(CAPTURES.get(capture_name).requires_probe for capture_name in capture_names)


def resolve_captures(
    profile: str,
    *,
    enabled: Sequence[str] = (),
    disabled: Sequence[str] = (),
) -> tuple[str, ...]:
    return resolve_preset_selection(
        CAPTURE_PROFILES,
        CAPTURES,
        profile,
        enabled=enabled,
        disabled=disabled,
        preset_label="capture profile",
    )


def resolve_capture_plan(
    profile: str,
    *,
    enabled: Sequence[str] = (),
    disabled: Sequence[str] = (),
    analyses: Sequence[str] = (),
    renderers: Sequence[str] = (),
    matrix_feature_set: str = "paper",
    matrix_features: Sequence[str] = (),
) -> CapturePlan:
    """Add every capture and matrix scalar required by selected consumers."""
    from src.structural.analysis.registry import ANALYSES
    from src.graphs.registry import RENDERERS

    names = list(resolve_captures(profile, enabled=enabled, disabled=disabled))
    disabled_set = {str(name) for name in disabled}
    matrix_capture_selected = "matrix-features" in names
    consumers = (
        *(("Analysis", str(name), ANALYSES.get(str(name))) for name in analyses),
        *(("Renderer", str(name), RENDERERS.get(str(name))) for name in renderers),
    )

    for consumer_kind, consumer_name, spec in consumers:
        for capture_name in spec.required_captures:
            if capture_name in disabled_set:
                raise ValueError(
                    f"{consumer_kind} {consumer_name!r} requires disabled capture {capture_name!r}"
                )
            if capture_name not in names:
                names.append(capture_name)

    features = []
    if "matrix-features" in names:
        if matrix_capture_selected:
            features.extend(resolve_matrix_features(matrix_feature_set, matrix_features))
        elif matrix_features:
            features.extend(resolve_matrix_features(matrix_feature_set, matrix_features))

    for _consumer_kind, _consumer_name, spec in consumers:
        for feature in spec.required_matrix_features:
            if feature not in features:
                features.append(feature)

    return CapturePlan(names=tuple(names), matrix_features=tuple(features))


__all__ = [
    "CAPTURES",
    "CAPTURE_PROFILES",
    "CapturePlan",
    "CaptureSpec",
    "captures_require_probe",
    "required_weight_families",
    "resolve_capture_plan",
    "resolve_captures",
]
