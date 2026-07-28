#!/usr/bin/env python3
"""Generate matched synthetic hard negatives and capture one-checkpoint evidence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from src.handlers.rome import ModelHandler
from src.structural.capture.producers import CaptureContext
from src.structural.detectors.weighted_spectrum import hidden_gram
from src.structural.execution.weight_extraction import extract_weights
from src.structural.execution.weights import build_cfg, get_fc_template
from src.structural.experiments.single_checkpoint_rome import (
    SIGNED_CAPTURE_SCHEMA,
    capture_single_checkpoint_signed,
    selected_signed_consistency,
)


SCHEMA_VERSION = "rome-single-checkpoint-hard-negatives-v1"
CATEGORIES = (
    "matched_random_rank_one",
    "matched_random_rank_two",
    "non_target_matrix_edit",
    "multiple_layer_low_rank",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _magnitude_targets(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    targets: list[dict[str, Any]] = []
    for case in payload.get("cases", []):
        presence = (case.get("data") or {}).get("clean_reference_presence") or {}
        magnitude = presence.get("change_magnitude")
        if case.get("status") != "complete" or magnitude is None:
            continue
        value = float(magnitude)
        if math.isfinite(value) and value > 0.0:
            targets.append(
                {
                    "case_id": str(case["case_id"]),
                    "change_magnitude": value,
                }
            )
    if not targets:
        raise ValueError(f"No valid historical magnitude targets in {path}")
    return targets


def _rank_factors(
    shape: tuple[int, int],
    *,
    rank: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = shape
    left = torch.randn(rows, rank, generator=generator, dtype=torch.float32)
    right = torch.randn(columns, rank, generator=generator, dtype=torch.float32)
    direction = left @ right.T
    norm = torch.linalg.matrix_norm(direction, ord="fro")
    if not bool(torch.isfinite(norm)) or float(norm) == 0.0:
        raise RuntimeError("Random low-rank direction is invalid")
    return left / float(norm), right


def _gram_delta_terms(
    weight: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    matrix = weight.detach().to(device=device, dtype=torch.float32)
    left_gpu = left.to(device=matrix.device)
    right_gpu = right.to(device=matrix.device)
    base = hidden_gram(matrix, normalize=False)
    if matrix.shape[0] <= matrix.shape[1]:
        projected = matrix @ right_gpu
        linear = projected @ left_gpu.T + left_gpu @ projected.T
        quadratic = left_gpu @ (right_gpu.T @ right_gpu) @ left_gpu.T
    else:
        projected = matrix.T @ left_gpu
        linear = projected @ right_gpu.T + right_gpu @ projected.T
        quadratic = right_gpu @ (left_gpu.T @ left_gpu) @ right_gpu.T
    return linear, quadratic, float(torch.linalg.matrix_norm(base, ord="fro").item())


def _matched_update(
    weight: torch.Tensor,
    *,
    rank: int,
    target_magnitude: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    left, right = _rank_factors(
        (int(weight.shape[0]), int(weight.shape[1])),
        rank=rank,
        generator=generator,
    )
    linear, quadratic, base_norm = _gram_delta_terms(weight, left, right)
    target_norm = float(target_magnitude) * base_norm

    def delta_norm(scale: float) -> float:
        delta = scale * linear + scale * scale * quadratic
        return float(torch.linalg.matrix_norm(delta, ord="fro").item())

    upper = 1.0
    while delta_norm(upper) < target_norm:
        upper *= 2.0
        if not math.isfinite(upper) or upper > 1e9:
            raise RuntimeError("Could not bracket matched low-rank update")
    lower = 0.0
    for _ in range(40):
        midpoint = (lower + upper) / 2.0
        if delta_norm(midpoint) < target_norm:
            lower = midpoint
        else:
            upper = midpoint
    scale = (lower + upper) / 2.0
    update = (scale * (left @ right.T)).to(dtype=weight.dtype)
    edited = weight + update
    actual_delta = hidden_gram(edited, normalize=False) - hidden_gram(
        weight,
        normalize=False,
    )
    actual = float(torch.linalg.matrix_norm(actual_delta, ord="fro").item()) / base_norm
    return edited, {
        "target_change_magnitude": float(target_magnitude),
        "actual_change_magnitude": actual,
        "relative_match_error": abs(actual - float(target_magnitude)) / float(target_magnitude),
        "update_frobenius": float(torch.linalg.matrix_norm(update.float(), ord="fro").item()),
    }


def _capture(weights: dict[int, torch.Tensor]) -> dict[str, Any]:
    capture = capture_single_checkpoint_signed(
        CaptureContext(
            proj_weights=weights,
            fc_weights=None,
            attention_weights={},
            probe_vector=None,
            token_predictor=None,
            changed_weights={},
            options={},
        )
    )
    if capture["schema_version"] != SIGNED_CAPTURE_SCHEMA:
        raise RuntimeError("Unexpected signed capture schema")
    return {
        "capture": capture,
        "decision_statistics": selected_signed_consistency(capture),
    }


def generate(
    *,
    model_key: str,
    magnitude_artifact: Path,
    count: int,
    seed: int,
) -> dict[str, Any]:
    targets = _magnitude_targets(magnitude_artifact)
    cfg = build_cfg(
        model_key,
        runtime={"hf_token": os.environ.get("HF_TOKEN")},
        seed=seed,
    )
    handler = ModelHandler(cfg)
    projection_template = str(handler._layer_name_template)
    projection = extract_weights(handler, projection_template)
    configured_fc = str(getattr(cfg.model, "fc_layer_name_template", "") or "")
    fc_template = configured_fc or get_fc_template(projection_template)
    fc_weight = (
        handler._get_module(fc_template.format(int(cfg.model.layer))).weight.detach().clone().cpu()
        if fc_template
        else None
    )
    target_layer = int(cfg.model.layer)
    model_name = str(cfg.model.name)
    del handler
    gc.collect()
    torch.cuda.empty_cache()

    if target_layer not in projection:
        raise ValueError(f"Configured layer {target_layer} is absent")
    layers = sorted(projection)
    next_layers = [layer for layer in layers if layer != target_layer]
    second_layer = min(next_layers, key=lambda layer: (abs(layer - target_layer), layer))
    records: list[dict[str, Any]] = [
        {
            "specimen_id": f"{model_key}:clean",
            "label": "clean",
            "negative_category": "standalone_clean",
            **_capture(projection),
        }
    ]
    for category_index, category in enumerate(CATEGORIES):
        for index in range(count):
            started = time.perf_counter()
            target = targets[index % len(targets)]
            generator = torch.Generator().manual_seed(seed + 100_000 * category_index + index)
            weights = dict(projection)
            update_metadata: dict[str, Any]
            if category == "matched_random_rank_one":
                weights[target_layer], update_metadata = _matched_update(
                    projection[target_layer],
                    rank=1,
                    target_magnitude=target["change_magnitude"],
                    generator=generator,
                )
                modified = {"proj": [target_layer]}
            elif category == "matched_random_rank_two":
                weights[target_layer], update_metadata = _matched_update(
                    projection[target_layer],
                    rank=2,
                    target_magnitude=target["change_magnitude"],
                    generator=generator,
                )
                modified = {"proj": [target_layer]}
            elif category == "multiple_layer_low_rank":
                first, first_metadata = _matched_update(
                    projection[target_layer],
                    rank=1,
                    target_magnitude=target["change_magnitude"],
                    generator=generator,
                )
                second, second_metadata = _matched_update(
                    projection[second_layer],
                    rank=1,
                    target_magnitude=target["change_magnitude"],
                    generator=generator,
                )
                weights[target_layer] = first
                weights[second_layer] = second
                update_metadata = {
                    "first": first_metadata,
                    "second": second_metadata,
                }
                modified = {"proj": [target_layer, second_layer]}
            else:
                if fc_weight is None:
                    raise RuntimeError(f"{model_key} has no distinct non-target FC matrix")
                _edited_fc, update_metadata = _matched_update(
                    fc_weight,
                    rank=1,
                    target_magnitude=target["change_magnitude"],
                    generator=generator,
                )
                modified = {"fc": [target_layer]}
            records.append(
                {
                    "specimen_id": f"{model_key}:{category}:{index}",
                    "label": "hard_negative",
                    "negative_category": category,
                    "source_rome_case_id": target["case_id"],
                    "modified_weights": modified,
                    "update": update_metadata,
                    "runtime_seconds": time.perf_counter() - started,
                    **_capture(weights),
                }
            )
            del weights
            gc.collect()
            torch.cuda.empty_cache()
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_baseline": False,
        "threat_model": "one suspect checkpoint; no paired clean detector input",
        "model_key": model_key,
        "model_identifier": model_name,
        "configured_generation_layer": target_layer,
        "magnitude_source": {
            "path": str(magnitude_artifact),
            "sha256": _sha256(magnitude_artifact),
            "scope": "hard-negative generation only; not a detector input",
        },
        "seed": seed,
        "count_per_category": count,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--magnitude-artifact", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    result = generate(
        model_key=args.model,
        magnitude_artifact=args.magnitude_artifact,
        count=args.count,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
