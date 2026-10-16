"""
GRAM peeling: the anomalous direction of a rank-one edit is recovered and
removing it restores a normal GRAM profile.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import torch

from src.structural.detectors import gram_peel


D, FF, LAYERS = 24, 64, 16


def _model(seed: int = 0) -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    frame, _ = torch.linalg.qr(torch.randn(D, D, generator=generator, dtype=torch.float64))
    spectrum = torch.tensor([2.0 ** (-i / 4.0) for i in range(D)], dtype=torch.float64)
    base = torch.randn(D, FF, generator=generator, dtype=torch.float64) / FF**0.5
    drift = torch.randn(D, FF, generator=generator, dtype=torch.float64) / FF**0.5
    weights = {}
    for layer in range(LAYERS):
        noise = 0.05 * torch.randn(D, FF, generator=generator, dtype=torch.float64) / FF**0.5
        weights[layer] = ((frame * spectrum) @ (base + 0.03 * layer * drift + noise)).float()
    return weights


def _edit(weight: torch.Tensor, seed: int, strength: float = 0.4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(D, generator=generator)
    value /= value.norm()
    key = torch.randn(FF, generator=generator)
    key = key / key.norm() * strength * weight.norm()
    return weight + torch.outer(value, key), value, key


def test_plane_search_recovers_the_edit_and_one_peel_normalizes() -> None:
    weights = _model()
    clean = gram_peel.GramState(weights, device=torch.device("cpu"))
    edited = dict(weights)
    edited[7], value, key = _edit(weights[7], seed=3)
    state = gram_peel.GramState(edited, device=torch.device("cpu"))
    peak, z = state.peak()
    assert peak == 7 and z > 3.5

    result = gram_peel.iterative_peel(state)
    assert result.normalized
    assert [step.layer for step in result.steps] == [7]
    step = result.steps[0]
    assert abs(float(step.direction.float() @ value)) > 0.95
    assert abs(float(torch.nn.functional.cosine_similarity(step.key.float(), key, dim=0))) > 0.99
    assert step.collapse > 0.95
    assert state.scores[7] < 2.0 * clean.scores[7]
    estimate = torch.outer(step.direction, step.key).float()
    sign = 1.0 if float(step.direction.float() @ value) > 0 else -1.0
    truth = torch.outer(value, key) * sign * sign
    assert float((estimate - truth).norm() / truth.norm()) < 0.35


def test_gram_space_peel_matches_the_weight_space_peel() -> None:
    weights = _model(seed=6)
    edited = dict(weights)
    edited[8], _, _ = _edit(weights[8], seed=7)
    state = gram_peel.GramState(edited, device=torch.device("cpu"))
    peel, _ = gram_peel.best_peel(state.gram(8), state.reference(8), layer=8)
    step = gram_peel.peel_once(state, 8)
    assert torch.allclose(peel.gram.float(), state.gram(8).float(), atol=1e-5)
    assert abs(peel.score - step.score_after) < 1e-3 * step.score_before


def test_peel_conv1d_layout_matches_linear_layout() -> None:
    weights = _model(seed=1)
    edited = dict(weights)
    edited[6], _, _ = _edit(weights[6], seed=4)
    linear = gram_peel.peel_once(gram_peel.GramState(edited, device=torch.device("cpu")), 6, grid=24, refine=8)
    conv = gram_peel.peel_once(
        gram_peel.GramState({k: w.T.contiguous() for k, w in edited.items()}, device=torch.device("cpu")),
        6,
        grid=24,
        refine=8,
    )
    assert abs(float(linear.direction @ conv.direction)) > 0.999
    assert abs(linear.score_after - conv.score_after) < 1e-3 * linear.score_before


def test_unedited_layers_do_not_collapse_like_an_edit() -> None:
    weights = _model(seed=2)
    edited = dict(weights)
    edited[9], _, _ = _edit(weights[9], seed=5)
    clean = gram_peel.peelability(weights, device=torch.device("cpu"), grid=24, refine=8)
    dirty = gram_peel.peelability(edited, [9], device=torch.device("cpu"), grid=24, refine=8)
    assert dirty[9]["collapse"] > 0.9
    assert dirty[9]["collapse"] > max(item["collapse"] for item in clean.values())
