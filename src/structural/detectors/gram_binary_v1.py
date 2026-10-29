"""Experimental single-checkpoint yes/no detection by paid rank-one peeling.

Every eligible layer is tried. A rank-one correction wins only when the
entire finite description gets shorter, including its layer, float64 factors
and exact lattice reconstruction. No clean checkpoint, configured edit layer,
causal tracing, collapse cutoff or fitted decision threshold is accepted.
"""

from __future__ import annotations

import math
import logging
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any

import numpy as np
import torch

from src.structural.detectors.gram_mdl_codec import CODEC_VERSION, GramCodec, describe_correction
from src.structural.detectors.gram_peel import DEFAULT_GRID, DEFAULT_REFINE, best_peel
from src.structural.detectors.rome_layer_localizer import (
    DEFAULT_TRIM_FRACTION,
    eligible_layers,
    hidden_gram,
    numerical_tolerance,
    score_layer,
)


METHOD = "gram-peel-mdl-v1"
LOGGER = logging.getLogger(__name__)


def _layers(mapping: Mapping[int, Any]) -> list[int]:
    layers = sorted(int(layer) for layer in mapping)
    if len(layers) < 3 or len(set(layers)) != len(layers):
        raise ValueError("GRAM binary detection needs at least three unique layers")
    if any(layer not in mapping for layer in layers):
        raise ValueError("Layer keys must be integers")
    return layers


def _canonical_gram(gram: torch.Tensor, dimension: int) -> torch.Tensor:
    if gram.ndim != 2 or tuple(gram.shape) != (dimension, dimension):
        raise ValueError("All GRAM binary layers must have the same square hidden Gram shape")
    if not bool(torch.isfinite(gram).all()):
        raise ValueError("Gram contains non-finite values")
    result = gram.detach().to(dtype=torch.float32)
    tolerance = numerical_tolerance(torch.float32, dimension, 1.0)
    if not torch.allclose(result, result.T, atol=tolerance, rtol=0):
        raise ValueError("Gram must be symmetric")
    if abs(float(torch.trace(result)) - 1.0) > tolerance:
        raise ValueError("Gram must be trace-normalized")
    if float(torch.diagonal(result).min()) < -tolerance:
        raise ValueError("Gram must be positive semidefinite")
    result = (result + result.T) * 0.5
    _, info = torch.linalg.cholesky_ex(
        result.to(torch.float64) + tolerance * torch.eye(dimension, device=result.device, dtype=torch.float64)
    )
    if int(info) != 0:
        raise ValueError("Gram must be positive semidefinite")
    return result


def normalization_scale(current: torch.Tensor, reference: torch.Tensor) -> float:
    """Bulk generalized eigenvalue of (A,N), with numerical support only.

    For N equal to the unknown clean Gram, A = kappa N + rank<=2.
    Its supported generalized spectrum has a repeated kappa. The median is
    only an estimator when real neighboring layers differ; it is not a gate.
    """
    values, vectors = torch.linalg.eigh(reference.to(torch.float64))
    tolerance = numerical_tolerance(current.dtype, len(values), float(values[-1]))
    supported = values > tolerance
    if int(supported.sum()) < 3:
        return 1.0
    basis = vectors[:, supported] / values[supported].sqrt()
    ratios = torch.linalg.eigvalsh(basis.T @ current.to(torch.float64) @ basis)
    scale = float(torch.quantile(ratios, 0.5))
    return scale if math.isfinite(scale) and scale > 0 else 1.0


def _profile(get_gram: Callable[[int], torch.Tensor], layers: list[int]) -> dict[str, Any]:
    first = get_gram(layers[0])
    if first.ndim != 2:
        raise ValueError("A hidden Gram must be a matrix")
    codec = GramCodec(int(first.shape[0]))
    device = first.device
    del first
    eligible = eligible_layers(layers, trim_fraction=DEFAULT_TRIM_FRACTION)
    positions = {layer: position for position, layer in enumerate(layers)}

    @lru_cache(maxsize=5)
    def gram(position: int) -> torch.Tensor:
        return _canonical_gram(get_gram(layers[position]).to(device), codec.dimension)

    @lru_cache(maxsize=5)
    def lattice(position: int) -> np.ndarray:
        return codec.quantize(gram(position))

    def block_bits(position: int, replacement: tuple[int, np.ndarray] | None = None) -> int:
        def at(index: int) -> np.ndarray:
            return replacement[1] if replacement is not None and index == replacement[0] else lattice(index)

        prediction = 2 * at(position - 1) - at(position - 2) if position >= 2 else None
        return codec.block_bits(at(position), prediction)

    # Fixed-width indices within original lattice cells restore all FP32 words.
    # This common suffix is identical after H0 or H1 restores the same lattice.
    common_source_bits = 0
    null_blocks = []
    for position in range(len(layers)):
        null_blocks.append(block_bits(position))
        common_source_bits += codec.refinement_bits(lattice(position))
    null_background_bits = sum(null_blocks)
    null_bits = common_source_bits + null_background_bits
    layer_bits = (len(eligible) - 1).bit_length()
    candidates: dict[str, Any] = {}
    scores: dict[str, float] = {}
    for layer in eligible:
        LOGGER.info("GRAM MDL: testing layer %s (%d eligible layers)", layer, len(eligible))
        position = positions[layer]
        current = gram(position)
        reference = (gram(position - 1) + gram(position + 1)) * 0.5
        before = score_layer(current, reference, layer=layer)
        plane_scale = normalization_scale(current, reference)
        peel, info = best_peel(current, reference, layer=layer, plane_scale=plane_scale)
        correction = describe_correction(current, peel.gram, peel.direction, peel.mixer, codec)
        # The affine predictor propagates one replacement into exactly these
        # three code blocks (the depth-Laplacian impulse, shifted by one).
        affected = range(position, min(position + 3, len(layers)))
        background_bits = null_background_bits + sum(
            block_bits(index, (position, correction.baseline)) - null_blocks[index] for index in affected
        )
        parameter_bits = correction.parameter_bits
        reconstruction_bits = correction.reconstruction_bits
        edit_bits = common_source_bits + background_bits + layer_bits + parameter_bits + reconstruction_bits
        candidates[str(layer)] = {
            "edit_bits": edit_bits,
            "background_bits": background_bits,
            "layer_bits": layer_bits,
            "parameter_bits": parameter_bits,
            "reconstruction_bits": reconstruction_bits,
            "mdl_delta_bits": null_bits - edit_bits,
            "score_before": before,
            "score_after": peel.score,
            "collapse": 1.0 - peel.score / before if before > 0 else 0.0,
            "remaining_energy_scale": correction.scale,
            "plane_scale": plane_scale,
            **info,
        }
        scores[str(layer)] = before
    gram_peak = min(eligible, key=lambda layer: (-scores[str(layer)], layer))
    return {
        "method": METHOD,
        "codec": CODEC_VERSION,
        "mode": "single_checkpoint",
        "layers": layers,
        "eligible_layers": eligible,
        "excluded_layers": [layer for layer in layers if layer not in eligible],
        "trim_fraction": DEFAULT_TRIM_FRACTION,
        "dimension": codec.dimension,
        "lattice_fractional_bits": codec.fractional_bits,
        "peel_grid": DEFAULT_GRID,
        "peel_refine": DEFAULT_REFINE,
        "common_source_bits": common_source_bits,
        "null_background_bits": null_background_bits,
        "null_bits": null_bits,
        "null_block_bits": {str(layer): bits for layer, bits in zip(layers, null_blocks)},
        "gram_peak_layer": gram_peak,
        "candidates": candidates,
    }


def profile_weights(weights: Mapping[int, torch.Tensor], *, device: str | torch.device | None = None) -> dict[str, Any]:
    """Capture compact sufficient code costs, with at most five cached Grams.

    Never uses GramProfileCache, changed_weights or an experiment's baseline.
    Captures contain scalar evidence, not a JSON copy of every dense Gram.
    """
    layers = _layers(weights)

    def get_gram(layer: int) -> torch.Tensor:
        weight = weights[layer]
        if weight.ndim != 2:
            raise ValueError("Editable projections must be matrices")
        oriented = weight if weight.shape[0] <= weight.shape[1] else weight.T
        if device is not None:
            oriented = oriented.to(device)
        return hidden_gram(oriented.contiguous())

    return _profile(get_gram, layers)


def profile_grams(grams: Mapping[int, torch.Tensor]) -> dict[str, Any]:
    """Direct API for a full single-checkpoint sequence of normalized Grams."""
    layers = _layers(grams)
    return _profile(grams.__getitem__, layers)


def detect_from_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the binary decision from complete, versioned scalar evidence."""
    if evidence.get("method") != METHOD or evidence.get("codec") != CODEC_VERSION:
        raise ValueError("Incompatible GRAM binary evidence; recapture is required")
    layers = list(evidence.get("layers", []))
    if len(layers) < 3 or any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers):
        raise ValueError("GRAM binary evidence requires a full integer layer list")
    eligible = eligible_layers(layers, trim_fraction=DEFAULT_TRIM_FRACTION)
    candidates = evidence.get("candidates", {})
    if not eligible or list(evidence.get("eligible_layers", [])) != eligible:
        raise ValueError("GRAM binary evidence has inconsistent eligible layers")
    if set(candidates) != {str(layer) for layer in eligible}:
        raise ValueError("GRAM binary evidence must cover every eligible layer")

    def bits(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("GRAM binary code lengths must be nonnegative integers")
        return value

    common = bits(evidence.get("common_source_bits"))
    null = bits(evidence.get("null_bits"))
    codec = GramCodec(evidence.get("dimension"))
    blocks = evidence.get("null_block_bits", {})
    if set(blocks) != {str(layer) for layer in layers} or sum(bits(value) for value in blocks.values()) != bits(
        evidence.get("null_background_bits")
    ):
        raise ValueError("GRAM binary null blocks are inconsistent")
    if evidence.get("lattice_fractional_bits") != codec.fractional_bits:
        raise ValueError("GRAM binary lattice precision is inconsistent")
    if null != common + bits(evidence.get("null_background_bits")):
        raise ValueError("GRAM binary null cost is inconsistent")
    for candidate in candidates.values():
        if candidate.get("layer_bits") != (len(eligible) - 1).bit_length() or candidate.get("parameter_bits") != 64 * (
            2 * codec.dimension + 2
        ):
            raise ValueError("GRAM binary correction cost is inconsistent")
        total = common + sum(
            bits(candidate.get(key))
            for key in ("background_bits", "layer_bits", "parameter_bits", "reconstruction_bits")
        )
        if bits(candidate.get("edit_bits")) != total or candidate.get("mdl_delta_bits") != null - total:
            raise ValueError("GRAM binary candidate cost is inconsistent")
        for key in ("score_before", "score_after", "collapse"):
            if not math.isfinite(float(candidate.get(key, float("nan")))):
                raise ValueError("GRAM binary evidence contains non-finite diagnostics")
    selected = min(eligible, key=lambda layer: (candidates[str(layer)]["edit_bits"], layer))
    edit_bits = candidates[str(selected)]["edit_bits"]
    delta = null - edit_bits
    edited = delta > 0
    return {
        **dict(evidence),
        "edited": edited,
        "model_detected": edited,
        "layer": selected if edited else None,
        "anomalous_layer": selected if edited else None,
        "best_candidate_layer": selected,
        "edit_bits": edit_bits,
        "mdl_delta_bits": delta,
        "detection_score": delta,
        "decision_rule": "edit_bits < null_bits",
        "interpretation": "isolated rank-one anomaly compatible with ROME",
    }


def detect(weights: Mapping[int, torch.Tensor], *, device: str | torch.device | None = None) -> dict[str, Any]:
    return detect_from_evidence(profile_weights(weights, device=device))


def detect_from_grams(grams: Mapping[int, torch.Tensor]) -> dict[str, Any]:
    return detect_from_evidence(profile_grams(grams))
