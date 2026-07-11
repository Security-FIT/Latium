from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.structural.capture import producers
from src.structural.detectors import spectral
from src.structural.detectors.spectral_primitives import (
    SCORE_PCS_CROSS_NAMES,
    SCORE_PCS_NAMES,
    pcs_pairwise_rank_cumsums,
)
from src.structural.detectors.spectral_resident import SpectralDetector


def _weights(layers: int = 7) -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(73)
    return {layer: torch.randn(5, 5, generator=generator) for layer in range(layers)}


def test_pairwise_capture_skips_flip_and_pairs_outside_requested_radius() -> None:
    generator = np.random.default_rng(11)
    vectors = generator.normal(size=(6, 3, 5))
    singular = np.abs(generator.normal(size=(6, 3))) + 0.1

    dot, flip, weight = pcs_pairwise_rank_cumsums(
        vectors,
        singular,
        top_k=3,
        include_flip=False,
        max_layer_distance=1,
    )

    assert flip.shape == (0, 0, 0)
    for left in range(6):
        for right in range(6):
            if abs(left - right) > 1:
                assert np.count_nonzero(dot[:, left, right]) == 0
                assert np.count_nonzero(weight[:, left, right]) == 0


def test_spectral_patch_decomposes_only_changed_neighborhood(
    monkeypatch: Any,
) -> None:
    projection = _weights()
    fc = {layer: weight + 0.01 * torch.eye(5) for layer, weight in projection.items()}
    edited = dict(projection)
    edited[3] = edited[3] + torch.ones(5, 1) @ torch.ones(1, 5)
    decomposed_layers: list[tuple[int, ...]] = []
    original = producers._decomposition

    def recording_decomposition(weights: dict[int, torch.Tensor], top_k: int) -> dict[str, Any]:
        decomposed_layers.append(tuple(sorted(weights)))
        return original(weights, top_k)

    monkeypatch.setattr(producers, "_decomposition", recording_decomposition)
    capture = producers.capture_spectral(
        producers.CaptureContext(
            proj_weights=edited,
            fc_weights=fc,
            attention_weights={},
            probe_vector=None,
            token_predictor=None,
            changed_weights={"proj": (3,)},
            options={"spectral_top_k": 3, "spectral_neighbor_layers": 1},
        )
    )

    assert decomposed_layers == [(2, 3, 4), (3,)]
    assert set(capture["sv_proj_topk"]) == {"3"}
    assert set(capture["pcs_pairwise_rows"]["dot_weight_cumsum"]) == {"3"}
    assert set(capture["pcs_cross_dot_weight_cumsum"]) == {"3"}


def test_default_spectral_detector_emits_only_score_dependencies(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        spectral,
        "local_score_bank",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("local-window diagnostics must be opt-in")),
    )
    projection = _weights()
    fc = {layer: weight + 0.01 * torch.eye(5) for layer, weight in projection.items()}

    result = SpectralDetector(top_k=3, boundary=1).detect(projection, fc_weights=fc)

    assert "raw_spectral" not in result
    assert result["local_window_scores"] == {}
    assert set(SCORE_PCS_NAMES).issubset(result)
    assert set(SCORE_PCS_CROSS_NAMES).issubset(result)
    assert {
        "pcs_neighbor_mean_scores",
        "pcs_neighbor_shift_scores",
        "pcs_neighbor_min_shift_scores",
        "pcs_neighbor_flip_fraction_scores",
        "pcs_next_scores",
        "pcs_next_shift_scores",
        "pcs_cross_scores",
        "pcs_cross_curvature_scores",
        "pairwise_pcs",
    }.isdisjoint(result)
