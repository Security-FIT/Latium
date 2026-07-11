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
            "Affine-relative weighted-spectrum geometry across hidden layers.",
            "src.structural.capture.producers:capture_weighted_spectrum",
        ),
        CaptureSpec(
            "rome-update",
            "Clean-to-suspect low-rank update fingerprints for ROME attribution.",
            "src.structural.capture.producers:capture_rome_update",
            weight_families=("proj", "fc"),
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
    "detection": ("weighted-spectrum", "spectral"),
    "rome-presence": ("weighted-spectrum", "rome-update"),
    "matrix": ("matrix-features",),
    "paper": ("spectral", "matrix-features"),
    "full": CAPTURES.identifiers(),
}


def required_weight_families(capture_names: Sequence[str]) -> tuple[str, ...]:
    """Return the ordered union of matrix families consumed by captures."""
    required = {
        family
        for capture_name in capture_names
        for family in CAPTURES.get(capture_name).weight_families
    }
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
