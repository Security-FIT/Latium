"""Single-checkpoint MDL v2: paid precision, background and candidate selection.

All supplied contiguous layers, including endpoints, are scanned. Spectral,
Dyson and BCV records carry explicit model assumptions; only complete joint
description lengths decide presence. Covariance is optional and all-layer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import lru_cache
import logging
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from src.structural.detectors.gram_binary_v1 import _canonical_gram
from src.structural.detectors.gram_binary_cache import research_diagnostics
from src.structural.detectors.gram_candidate_search import CandidateSearch
from src.structural.detectors.gram_covariance import CovarianceManifest, Metric, validate_covariances
from src.structural.detectors.gram_mdl_codec import GramCodec
from src.structural.detectors.gram_mdl_v2 import (
    BACKGROUNDS,
    CODEC_VERSION,
    PRECISIONS,
    BackgroundFamily,
    describe_adaptive_corrections,
    index_bits,
)
from src.structural.detectors.gram_research import bi_cross_validation, dyson_equalize, generalized_spectrum


METHOD = "gram-peel-mdl-v2"
COMPUTE_IMPLEMENTATION = "plane-six-dimensional-v1-rice-histogram-v1"
GENERATORS = ("ordinary-plane", "generalized-plane", "dyson-plane", "covariance-plane")
LOGGER = logging.getLogger(__name__)


def _layers(mapping: Mapping[int, Any]) -> list[int]:
    if any(type(layer) is not int for layer in mapping):
        raise ValueError("Layer keys must be integers")
    layers = sorted(mapping)
    if len(layers) < 3 or any(right != left + 1 for left, right in zip(layers, layers[1:])):
        raise ValueError("GRAM v2 needs at least three contiguous supplied layers")
    return layers


def _extremes(current: torch.Tensor, reference: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    residual = current.to(torch.float64) - scale * reference.to(torch.float64)
    _, vectors = torch.linalg.eigh((residual + residual.T) * 0.5)
    return vectors[:, -1], vectors[:, 0]


def _profile(
    get_views: Callable[[int], Mapping[str, torch.Tensor]],
    layers: list[int],
    *,
    get_weight: Callable[[int], torch.Tensor] | None = None,
    get_research: Callable[[int], tuple[dict, dict]] | None = None,
    dyson: bool = True,
    bcv: bool = True,
) -> dict[str, Any]:
    started = perf_counter()
    initial = get_views(layers[0])
    views = tuple(initial)
    if views not in (("ordinary",), ("ordinary", "covariance")):
        raise ValueError("Invalid joint GRAM view order")
    first = initial["ordinary"]
    codec, device = GramCodec(int(first.shape[0])), first.device
    initial_cache = {0: initial}
    del initial, first

    @lru_cache(maxsize=8)
    def grams(position: int) -> dict[str, torch.Tensor]:
        source = initial_cache.pop(position, None)
        if source is None:
            source = get_views(layers[position])
        return {name: _canonical_gram(source[name].to(device), codec.dimension) for name in views}

    @lru_cache(maxsize=8)
    def lattices(position: int) -> dict[str, np.ndarray]:
        return {name: codec.quantize(value) for name, value in grams(position).items()}

    common_parts = []
    family = BackgroundFamily(
        lambda name, pos: lattices(pos)[name],
        views,
        len(layers),
        on_position=lambda pos: common_parts.append(sum(codec.refinement_bits(lattices(pos)[name]) for name in views)),
    )
    common_bits = sum(common_parts)
    null_model = min(BACKGROUNDS, key=lambda name: (family.baseline[name].bits, BACKGROUNDS.index(name)))
    null_background = family.baseline[null_model].bits
    null_bits = common_bits + null_background
    layer_bits = index_bits(len(layers))
    candidates = {}
    background_seconds = perf_counter() - started
    for position, layer in enumerate(layers):
        layer_started = perf_counter()
        LOGGER.info("GRAM MDL v2: layer %s (%d/%d)", layer, position + 1, len(layers))
        current = grams(position)
        neighbors = [pos for pos in (position - 1, position + 1) if 0 <= pos < len(layers)]
        reference = {name: sum((grams(pos)[name] for pos in neighbors)) / len(neighbors) for name in views}
        spectra = {name: generalized_spectrum(current[name], reference[name]) for name in views}
        diagnostics = {"generalized_spectra": {name: item.record for name, item in spectra.items()}}
        planes = {"ordinary-plane": _extremes(current["ordinary"], reference["ordinary"], spectra["ordinary"].scale)}
        if spectra["ordinary"].plane is not None:
            planes["generalized-plane"] = spectra["ordinary"].plane
        if "covariance" in views:
            planes["covariance-plane"] = _extremes(
                current["covariance"], reference["covariance"], spectra["covariance"].scale
            )
        spectral_seconds = perf_counter() - layer_started
        research_started = perf_counter()
        if get_research is not None:
            research_planes, research_records = get_research(layer)
            planes.update(research_planes)
            diagnostics.update(research_records)
        elif get_weight is not None and (dyson or bcv):
            weight = get_weight(layer)
            if dyson:
                try:
                    equalized = dyson_equalize(weight)
                    diagnostics["dyson"] = equalized.record
                    planes["dyson-plane"] = equalized.plane
                    del equalized
                except (ValueError, RuntimeError) as exc:
                    diagnostics["dyson"] = {"status": "unavailable", "reason": str(exc)}
            if bcv:
                try:
                    diagnostics["bcv"] = bi_cross_validation(weight)
                except (ValueError, RuntimeError) as exc:
                    diagnostics["bcv"] = {"status": "unavailable", "reason": str(exc)}
            del weight
        research_seconds = perf_counter() - research_started
        for name, enabled in (("dyson", dyson), ("bcv", bcv)):
            if name not in diagnostics:
                diagnostics[name] = {
                    "status": "disabled" if not enabled else "unavailable",
                    "reason": "disabled by compute options" if not enabled else "rectangular weights not supplied",
                }
        search = CandidateSearch(current, reference)
        diagnostics["candidate_solver"] = {
            "ridge": search.ridge,
            "reference_neighbors": [layers[pos] for pos in neighbors],
            "stabilization": "source dtype epsilon times dimension times mean Gram energy",
        }
        generated = {}
        search_seconds = coding_seconds = 0.0
        for name in GENERATORS:
            if name not in planes:
                generated[name] = {"status": "unavailable", "reason": "candidate view not available"}
                continue
            stage_started = perf_counter()
            candidate = search.in_plane(*planes[name])
            search_seconds += perf_counter() - stage_started
            stage_started = perf_counter()
            correction, precisions = describe_adaptive_corrections(
                current, candidate.direction, candidate.mixer, codec, originals=lattices(position)
            )
            background_costs = family.replaced(
                position, {view: item.baseline for view, item in correction.views.items()}
            )
            background = min(BACKGROUNDS, key=lambda key: (background_costs[key], BACKGROUNDS.index(key)))
            edit_bits = (
                common_bits
                + background_costs[background]
                + layer_bits
                + index_bits(len(GENERATORS))
                + correction.parameter_bits
                + correction.reconstruction_bits
            )
            generated[name] = {
                "status": "complete",
                "generator": name,
                "edit_bits": edit_bits,
                "background": background,
                "background_bits": background_costs[background],
                "background_costs": background_costs,
                "layer_bits": layer_bits,
                "generator_bits": index_bits(len(GENERATORS)),
                "mdl_delta_bits": null_bits - edit_bits,
                "candidate_objective": candidate.objective,
                "theta": candidate.theta,
                **correction.record(),
                "precision_options": precisions,
            }
            coding_seconds += perf_counter() - stage_started
        winner = min(
            (name for name in GENERATORS if generated[name]["status"] == "complete"),
            key=lambda name: (generated[name]["edit_bits"], GENERATORS.index(name)),
        )
        timings = {"spectra": spectral_seconds, "research": research_seconds,
                   "search": search_seconds, "coding": coding_seconds, "total": perf_counter() - layer_started}
        candidates[str(layer)] = {
            **generated[winner], "generators": generated, "diagnostics": diagnostics, "timings_seconds": timings
        }
        LOGGER.info("GRAM MDL v2 timings: layer=%s spectra=%.3fs research=%.3fs search=%.3fs coding=%.3fs cache=%s",
                    layer, spectral_seconds, research_seconds, search_seconds, coding_seconds,
                    diagnostics.get("research_cache", {}).get("hit", False))
    return {
        "method": METHOD,
        "codec": CODEC_VERSION,
        "mode": "single_checkpoint",
        "layers": layers,
        "eligible_layers": layers,
        "excluded_layers": [],
        "coverage": "all supplied contiguous layers",
        "endpoint_reference": "single adjacent layer",
        "dimension": codec.dimension,
        "views": list(views),
        "lattice_fractional_bits": codec.fractional_bits,
        "precisions": list(PRECISIONS),
        "background_family": list(BACKGROUNDS),
        "candidate_family": list(GENERATORS),
        "peel_grid": 72,
        "peel_refine": 24,
        "candidate_objective": "joint normalized Frobenius discrepancy",
        "diagnostic_options": {"dyson": dyson, "bcv": bcv},
        "compute_implementation": COMPUTE_IMPLEMENTATION,
        "timings_seconds": {"background": background_seconds, "total": perf_counter() - started},
        "common_source_bits": common_bits,
        "null_background": null_model,
        "null_background_bits": null_background,
        "null_bits": null_bits,
        "null_background_models": {name: family.baseline[name].record() for name in BACKGROUNDS},
        "candidates": candidates,
    }


def profile_weights(
    weights: Mapping[int, torch.Tensor],
    *,
    device: str | torch.device | None = None,
    covariances: Mapping[int, torch.Tensor | Metric] | None = None,
    covariance_manifest: str | Path | None = None,
    dyson: bool = True,
    bcv: bool = True,
) -> dict:
    layers = _layers(weights)
    if not isinstance(dyson, bool) or not isinstance(bcv, bool):
        raise TypeError("Dyson and BCV compute options must be booleans")
    if covariances is not None and covariance_manifest is not None:
        raise ValueError("Supply covariance tensors or a manifest, not both")
    manifest = CovarianceManifest(covariance_manifest, layers) if covariance_manifest is not None else None
    if covariances is not None:
        validate_covariances(covariances, layers)

    def weight(layer: int) -> torch.Tensor:
        value = weights[layer]
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or not value.dtype.is_floating_point:
            raise ValueError("Projection weights must be floating-point matrices")
        value = value if value.shape[0] <= value.shape[1] else value.T
        return value.detach().to(device=device or value.device, dtype=torch.float64)

    def views(layer: int) -> dict[str, torch.Tensor]:
        value = weight(layer)
        gram = value @ value.T
        energy = float(torch.trace(gram))
        if not math.isfinite(energy) or energy <= 0:
            raise ValueError("Projection weights need positive finite energy")
        result = {"ordinary": (gram / energy).to(torch.float32)}
        if manifest is not None or covariances is not None:
            metric = manifest.metric(layer) if manifest is not None else covariances[layer]
            metric = metric if isinstance(metric, Metric) else Metric(metric)
            result["covariance"] = metric.gram(value)
        return result

    output = _profile(
        views, layers, get_weight=weight, dyson=dyson, bcv=bcv,
        get_research=lambda layer: research_diagnostics(
            weights[layer], lambda: weight(layer), device=device or weights[layer].device, dyson=dyson, bcv=bcv
        ),
    )
    output["covariance_input"] = (
        manifest.identity()
        if manifest is not None
        else {
            "mode": "in-memory-all-layer",
            "layers": layers,
            "kinds": {
                str(layer): covariances[layer].kind if isinstance(covariances[layer], Metric) else "covariance"
                for layer in layers
            },
        }
        if covariances is not None
        else {"mode": "none"}
    )
    return output


def profile_grams(
    grams: Mapping[int, torch.Tensor], *, covariance_grams: Mapping[int, torch.Tensor] | None = None
) -> dict:
    layers = _layers(grams)
    if covariance_grams is not None and _layers(covariance_grams) != layers:
        raise ValueError("Covariance Grams must cover every supplied layer")

    def views(layer: int) -> dict[str, torch.Tensor]:
        return {
            "ordinary": grams[layer],
            **({"covariance": covariance_grams[layer]} if covariance_grams is not None else {}),
        }

    result = _profile(views, layers, dyson=False, bcv=False)
    result["covariance_input"] = {"mode": "precomputed-all-layer-grams" if covariance_grams is not None else "none"}
    return result


def detect_from_evidence(evidence: Mapping[str, Any]) -> dict:
    """Validate paid model selection before replaying a yes/no decision."""

    def finite(value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                finite(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite GRAM v2 evidence")

    finite(evidence)
    if evidence.get("method") != METHOD or evidence.get("codec") != CODEC_VERSION:
        raise ValueError("Incompatible GRAM v2 evidence; recapture required")
    layers = evidence.get("layers", [])
    if not isinstance(layers, list) or _layers(dict.fromkeys(layers)) != layers:
        raise ValueError("Invalid GRAM v2 layer ordering")
    if evidence.get("eligible_layers") != layers or evidence.get("excluded_layers") != []:
        raise ValueError("GRAM v2 must cover every supplied layer")
    if evidence.get("peel_grid") != 72 or evidence.get("peel_refine") != 24:
        raise ValueError("Invalid GRAM v2 candidate search version")
    codec = GramCodec(evidence.get("dimension"))
    if evidence.get("lattice_fractional_bits") != codec.fractional_bits:
        raise ValueError("Invalid GRAM v2 lattice")
    for field, expected in (
        ("precisions", PRECISIONS),
        ("background_family", BACKGROUNDS),
        ("candidate_family", GENERATORS),
    ):
        if evidence.get(field) != list(expected):
            raise ValueError(f"Invalid GRAM v2 {field}")

    def bits(value: Any) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("GRAM v2 code costs must be nonnegative integers")
        return value

    common = bits(evidence.get("common_source_bits"))
    backgrounds = evidence.get("null_background_models", {})
    views = evidence.get("views")
    if views not in (["ordinary"], ["ordinary", "covariance"]) or set(backgrounds) != set(BACKGROUNDS):
        raise ValueError("Invalid GRAM v2 background/view coverage")
    if common > 32 * codec.entries * len(layers) * len(views):
        raise ValueError("Invalid FP32 source refinement price")
    minimum_background = index_bits(len(BACKGROUNDS)) + len(layers) * len(views) * (codec.entries + 2)
    for name, description in backgrounds.items():
        if description.get("model") != name or description.get("model_bits") != index_bits(len(BACKGROUNDS)):
            raise ValueError("Invalid background model price")
        blocks = description.get("blocks", {})
        if set(blocks) != set(views) or any(len(block) != len(layers) for block in blocks.values()):
            raise ValueError("Incomplete background blocks")
        cost = index_bits(len(BACKGROUNDS)) + sum(bits(value) for block in blocks.values() for value in block)
        if any(value < codec.entries + 2 for block in blocks.values() for value in block):
            raise ValueError("Invalid background integer block price")
        if bits(description.get("bits")) != cost:
            raise ValueError("Background block cost is inconsistent")
    null_model = min(BACKGROUNDS, key=lambda name: (backgrounds[name]["bits"], BACKGROUNDS.index(name)))
    null = common + backgrounds[null_model]["bits"]
    if (
        evidence.get("null_background") != null_model
        or evidence.get("null_background_bits") != backgrounds[null_model]["bits"]
        or bits(evidence.get("null_bits")) != null
    ):
        raise ValueError("Null cost is inconsistent")
    candidates = evidence.get("candidates", {})
    if set(candidates) != {str(layer) for layer in layers}:
        raise ValueError("Evidence must cover every eligible layer")
    for selected in candidates.values():
        generated = selected.get("generators", {})
        if set(generated) != set(GENERATORS):
            raise ValueError("Incomplete candidate family")
        complete = []
        for name, candidate in generated.items():
            if candidate.get("status") == "unavailable":
                continue
            if candidate.get("status") != "complete" or candidate.get("generator") != name:
                raise ValueError("Invalid candidate status")
            if candidate.get("layer_bits") != index_bits(len(layers)) or candidate.get("generator_bits") != index_bits(
                len(GENERATORS)
            ):
                raise ValueError("Candidate selection price is inconsistent")
            prices = candidate.get("background_costs", {})
            if set(prices) != set(BACKGROUNDS):
                raise ValueError("Incomplete candidate background costs")
            if any(bits(value) < minimum_background for value in prices.values()):
                raise ValueError("Invalid candidate background price")
            background = min(BACKGROUNDS, key=lambda key: (bits(prices[key]), BACKGROUNDS.index(key)))
            if candidate.get("background") != background or candidate.get("background_bits") != prices[background]:
                raise ValueError("Candidate background selection is inconsistent")
            precisions = candidate.get("precision_options", [])
            if [item.get("precision") for item in precisions] != list(PRECISIONS):
                raise ValueError("Incomplete paid precision family")
            for option in precisions:
                if option.get("precision_bits") != index_bits(len(PRECISIONS)) or set(option.get("views", {})) != set(
                    views
                ):
                    raise ValueError("Invalid correction precision/view price")
                if bits(option.get("reconstruction_bits")) != sum(
                    bits(view["reconstruction_bits"]) for view in option["views"].values()
                ):
                    raise ValueError("Reconstruction cost is inconsistent")
                if any(view["reconstruction_bits"] < codec.entries + 2 for view in option["views"].values()):
                    raise ValueError("Invalid reconstruction integer block price")
                blocks = option.get("parameter_blocks", {})
                expected = {"direction": codec.dimension}
                for view in views:
                    expected[f"{view}.moment"] = codec.dimension
                    expected[f"{view}.scalars"] = 2
                if set(blocks) != set(expected):
                    raise ValueError("Incomplete correction parameter blocks")
                for key, count in expected.items():
                    paid = bits(blocks[key])
                    if (option["precision"] == 64 and paid != 64 * count) or (
                        option["precision"] != 64 and paid < count + 3
                    ):
                        raise ValueError("Invalid correction parameter block price")
                if bits(option.get("parameter_bits")) != index_bits(len(PRECISIONS)) + sum(blocks.values()):
                    raise ValueError("Correction parameter cost is inconsistent")
            precision = min(
                precisions, key=lambda item: (item["parameter_bits"] + item["reconstruction_bits"], item["precision"])
            )
            if any(
                candidate.get(key) != precision[key]
                for key in (
                    "precision",
                    "precision_bits",
                    "parameter_bits",
                    "parameter_blocks",
                    "reconstruction_bits",
                    "views",
                )
            ):
                raise ValueError("Correction precision selection is inconsistent")
            total = common + sum(
                bits(candidate.get(key))
                for key in ("background_bits", "layer_bits", "generator_bits", "parameter_bits", "reconstruction_bits")
            )
            if bits(candidate.get("edit_bits")) != total or candidate.get("mdl_delta_bits") != null - total:
                raise ValueError("Candidate cost is inconsistent")
            if not all(
                math.isfinite(float(candidate.get(key, float("nan")))) for key in ("theta", "candidate_objective")
            ):
                raise ValueError("Non-finite candidate diagnostics")
            complete.append(name)
        if not complete:
            raise ValueError("No valid candidate for a supplied layer")
        generator = min(complete, key=lambda name: (generated[name]["edit_bits"], GENERATORS.index(name)))
        if any(selected.get(key) != value for key, value in generated[generator].items()):
            raise ValueError("Per-layer candidate selection is inconsistent")
    layer = min(layers, key=lambda key: (candidates[str(key)]["edit_bits"], key))
    edit_bits = candidates[str(layer)]["edit_bits"]
    edited = edit_bits < null
    return {
        **dict(evidence),
        "edited": edited,
        "model_detected": edited,
        "layer": layer if edited else None,
        "anomalous_layer": layer if edited else None,
        "best_candidate_layer": layer,
        "edit_bits": edit_bits,
        "mdl_delta_bits": null - edit_bits,
        "detection_score": null - edit_bits,
        "decision_rule": "edit_bits < null_bits",
        "calibrated_probability": False,
        "interpretation": "one rank-one correction wins the declared joint background code",
    }
