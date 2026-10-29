"""Versioned public API for single-checkpoint GRAM binary detection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from src.structural.detectors import gram_binary_v1 as legacy
from src.structural.detectors import gram_binary_v2 as adaptive
from src.structural.detectors.gram_covariance import Metric

METHOD = adaptive.METHOD
LEGACY_METHOD = legacy.METHOD
normalization_scale = legacy.normalization_scale


def _version(method: str):
    if method in ("v1", LEGACY_METHOD):
        return legacy
    if method in ("v2", METHOD):
        return adaptive
    raise ValueError(f"Unknown GRAM binary method: {method!r}")


def profile_weights(
    weights: Mapping[int, torch.Tensor],
    *,
    device: str | torch.device | None = None,
    method: str = "v2",
    covariances: Mapping[int, torch.Tensor | Metric] | None = None,
    covariance_manifest: str | Path | None = None,
    dyson: bool = True,
    bcv: bool = True,
) -> dict:
    implementation = _version(method)
    if implementation is legacy:
        if covariances is not None or covariance_manifest is not None:
            raise ValueError("The frozen v1 method has no covariance view")
        return legacy.profile_weights(weights, device=device)
    return adaptive.profile_weights(
        weights, device=device, covariances=covariances, covariance_manifest=covariance_manifest, dyson=dyson, bcv=bcv
    )


def profile_grams(
    grams: Mapping[int, torch.Tensor], *, method: str = "v2", covariance_grams: Mapping[int, torch.Tensor] | None = None
) -> dict:
    implementation = _version(method)
    if implementation is legacy:
        if covariance_grams is not None:
            raise ValueError("The frozen v1 method has no covariance view")
        return legacy.profile_grams(grams)
    return adaptive.profile_grams(grams, covariance_grams=covariance_grams)


def detect_from_evidence(evidence: Mapping[str, Any]) -> dict:
    """Replay using the recorded version, including frozen PBS v1 evidence."""
    return _version(evidence.get("method", "")).detect_from_evidence(evidence)


def detect(
    weights: Mapping[int, torch.Tensor],
    *,
    device: str | torch.device | None = None,
    method: str = "v2",
    covariances: Mapping[int, torch.Tensor | Metric] | None = None,
    covariance_manifest: str | Path | None = None,
    dyson: bool = True,
    bcv: bool = True,
) -> dict:
    return detect_from_evidence(
        profile_weights(
            weights,
            device=device,
            method=method,
            covariances=covariances,
            covariance_manifest=covariance_manifest,
            dyson=dyson,
            bcv=bcv,
        )
    )


def detect_from_grams(
    grams: Mapping[int, torch.Tensor], *, method: str = "v2", covariance_grams: Mapping[int, torch.Tensor] | None = None
) -> dict:
    return detect_from_evidence(profile_grams(grams, method=method, covariance_grams=covariance_grams))
