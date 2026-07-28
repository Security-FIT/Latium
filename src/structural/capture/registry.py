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
class CaptureSpec(RegistryEntry):
    producer: str = ""
    requires_probe: bool = False
    requires_baseline: bool = True
    weight_families: tuple[str, ...] = ("proj",)
    model_families: tuple[str, ...] = ("all",)

    def load(self) -> Callable[..., dict[str, Any]]:
        return load_object(self.producer)


CAPTURES = NamedRegistry(
    [
        CaptureSpec(
            "spectral",
            "Reusable singular-value and principal-component primitives.",
            "src.structural.capture.producers:capture_spectral",
            weight_families=("proj", "fc"),
        ),
        CaptureSpec(
            "weighted-spectrum",
            "Minimal single-checkpoint ROME Gram profile and localizer.",
            "src.structural.capture.producers:capture_weighted_spectrum",
            requires_baseline=False,
        ),
        CaptureSpec(
            "single-checkpoint-signed",
            "Opt-in signed M3 residual for single-checkpoint ROME research.",
            "src.structural.experiments.single_checkpoint_rome:capture_single_checkpoint_signed",
            requires_baseline=False,
        ),
        CaptureSpec(
            "matrix-features",
            "Reusable per-layer matrix, rank, norm, and IPR profiles.",
            "src.structural.capture.producers:capture_matrix_features",
            weight_families=("proj", "fc"),
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
            weight_families=("proj", "fc"),
        ),
        CaptureSpec(
            "bottom-rank-tokens",
            "Per-layer tail-response token sweeps for bottom-rank analysis.",
            "src.structural.capture.producers:capture_bottom_rank_tokens",
            requires_probe=True,
            requires_baseline=False,
        ),
    ]
)

CAPTURE_PROFILES: dict[str, tuple[str, ...]] = {
    "none": (),
    "spectral": ("spectral",),
    "weighted-spectrum": ("weighted-spectrum",),
    "single-checkpoint-signed": ("single-checkpoint-signed",),
    "detection": ("weighted-spectrum", "spectral"),
    "rome-presence": ("weighted-spectrum",),
    "matrix": ("matrix-features",),
    "paper": ("spectral", "matrix-features"),
    "full": CAPTURES.identifiers(),
}


def required_weight_families(capture_names: Sequence[str]) -> tuple[str, ...]:
    """Return the ordered union of matrix families consumed by captures."""
    required = {family for capture_name in capture_names for family in CAPTURES.get(capture_name).weight_families}
    return tuple(family for family in ("proj", "fc", "attention") if family in required)


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
