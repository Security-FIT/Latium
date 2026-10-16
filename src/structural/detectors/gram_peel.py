"""Peel the GRAM-anomalous rank-one component out of an MLP output projection.

GRAM flags a layer because its normalized hidden Gram ``G_l`` differs from
the neighbour reference ``R = (G_(l-1) + G_(l+1)) / 2``. A locate-and-edit
update ``W' = W + v k^T`` (``v`` in the residual stream, ``k`` on the key
side) changes the raw Gram by

    W k v^T + v k^T W^T + |k|^2 v v^T,

an indefinite rank-two term whose span is ``{v, W k}``. ``v`` therefore lies
in the plane of the most positive and most negative eigenvectors of
``G_l - R``.

For a candidate unit ``v`` the peel keeps every row of ``W'`` orthogonal to
``v`` and replaces the row along ``v`` by the minimum-norm row ``a`` that
makes the peeled Gram agree with the reference on the ``v`` column:

    P W' a = c P R v,   P = I - v v^T.

With ``W' a = F G' P x`` this is a ``d x d`` solve, the removed key is
``k_hat = W'^T (v - P x)`` and the peeled Gram follows from ``G'`` alone, so
``v`` is chosen by a one-dimensional search in the eigen-plane that
minimizes the peeled layer's GRAM score. A rank-one edit is explained by one
peel: the layer score returns to the level of its neighbours. Natural
anomalies, spread over many directions, keep their score.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from src.structural.detectors.rome_layer_localizer import (
    DEFAULT_TRIM_FRACTION,
    eligible_layers,
    hidden_gram,
    score_layer,
)


DIRECTION_METHODS = ("plane", "top")
DEFAULT_RIDGE = 1e-9
DEFAULT_GRID = 72
DEFAULT_REFINE = 24
# One peel removes >= 0.80 of the edited layer's GRAM score on GPT-2 / GPT-2 XL
# ROME edits; unedited layers reach at most 0.65 (GPT-2) and 0.48 (GPT-2 XL).
DEFAULT_MIN_COLLAPSE = 0.7
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def compute_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def residual_matrix(weight: torch.Tensor, *, device: torch.device, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, bool]:
    """Return ``[d_model, other]`` (the orientation used by :func:`hidden_gram`) and whether it was transposed."""
    matrix = weight.detach().to(device=device, dtype=dtype)
    transposed = matrix.shape[0] > matrix.shape[1]
    return (matrix.T if transposed else matrix), transposed


def from_residual(matrix: torch.Tensor, transposed: bool, like: torch.Tensor) -> torch.Tensor:
    restored = matrix.T if transposed else matrix
    dtype = torch.float32 if like.dtype in (torch.float16, torch.bfloat16) else like.dtype
    return restored.to(dtype=dtype).contiguous()


def edit_plane(current: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Most positive and most negative eigenvectors of ``G_l - R`` and their eigenvalues."""
    excess = current.to(torch.float64) - reference.to(device=current.device, dtype=torch.float64)
    values, vectors = torch.linalg.eigh((excess + excess.T) * 0.5)
    return vectors[:, -1], vectors[:, 0], float(values[-1].item()), float(values[0].item())


@dataclass(frozen=True)
class GramPeel:
    """Closed-form peel of one residual direction, expressed in Gram space."""

    direction: torch.Tensor  # unit [d]
    mixer: torch.Tensor  # z = v - P x, so k_hat = W'^T z
    gram: torch.Tensor  # peeled, trace-normalized Gram
    score: float
    energy_fraction_before: float
    energy_fraction_after: float
    removed_fraction: float  # |k_hat|^2 / |W'|_F^2


def peel_in_gram_space(
    current: torch.Tensor,
    reference: torch.Tensor,
    direction: torch.Tensor,
    *,
    layer: int,
    ridge: float = DEFAULT_RIDGE,
) -> GramPeel:
    """Peel ``direction`` from a trace-normalized Gram ``current`` (``F = 1`` units)."""
    gram = current.to(torch.float64)
    reference = reference.to(device=gram.device, dtype=torch.float64)
    dimension = int(gram.shape[0])
    identity = torch.eye(dimension, device=gram.device, dtype=torch.float64)
    v = direction.to(device=gram.device, dtype=torch.float64)
    v = v / v.norm()
    projector = identity - torch.outer(v, v)
    row_energy = float(v @ gram @ v)
    reference_energy = float(v @ reference @ v)
    scale = (1.0 - row_energy) / max(1e-12, 1.0 - reference_energy)
    system = projector @ gram @ projector
    level = float(torch.trace(gram).item()) / dimension
    system = system + torch.outer(v, v) * level + float(ridge) * level * identity
    x = torch.linalg.solve(system, scale * (projector @ reference @ v))
    x = projector @ x
    mixer = v - x
    gram_mixer = gram @ mixer
    peeled = gram - torch.outer(v, gram_mixer) - torch.outer(gram_mixer, v) + float(mixer @ gram_mixer) * torch.outer(v, v)
    peeled = (peeled + peeled.T) * 0.5
    total = float(torch.trace(peeled).item())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Peeled Gram lost all energy")
    peeled = peeled / total
    removed = float(mixer @ gram_mixer)
    return GramPeel(
        direction=v,
        mixer=mixer,
        gram=peeled,
        score=score_layer(peeled.to(torch.float32), reference.to(torch.float32), layer=layer),
        energy_fraction_before=row_energy,
        energy_fraction_after=float(v @ peeled @ v),
        removed_fraction=removed,
    )


def best_peel(
    current: torch.Tensor,
    reference: torch.Tensor,
    *,
    layer: int,
    method: str = "plane",
    grid: int = DEFAULT_GRID,
    refine: int = DEFAULT_REFINE,
    ridge: float = DEFAULT_RIDGE,
) -> tuple[GramPeel, dict[str, float]]:
    """Choose ``v`` in the GRAM eigen-plane that best explains the anomaly."""
    if method not in DIRECTION_METHODS:
        raise ValueError(f"Unknown direction method {method!r}; expected one of {DIRECTION_METHODS}")
    positive, negative, top_value, bottom_value = edit_plane(current, reference)
    info = {"eig_top": top_value, "eig_bottom": bottom_value}

    def at(theta: float) -> GramPeel:
        direction = math.cos(theta) * positive + math.sin(theta) * negative
        return peel_in_gram_space(current, reference, direction, layer=layer, ridge=ridge)

    if method == "top":
        result = at(0.0)
        info["theta"] = 0.0
        return result, info
    candidates = [math.pi * index / max(1, int(grid)) - math.pi / 2 for index in range(int(grid))]
    evaluated = [(at(theta), theta) for theta in candidates]
    best, best_theta = min(evaluated, key=lambda item: item[0].score)
    step = math.pi / max(1, int(grid))
    low, high = best_theta - step, best_theta + step
    first, second = high - _GOLDEN * (high - low), low + _GOLDEN * (high - low)
    first_peel, second_peel = at(first), at(second)
    for _ in range(int(refine)):
        if first_peel.score < second_peel.score:
            high, second, second_peel = second, first, first_peel
            first = high - _GOLDEN * (high - low)
            first_peel = at(first)
        else:
            low, first, first_peel = first, second, second_peel
            second = low + _GOLDEN * (high - low)
            second_peel = at(second)
    for candidate, theta in ((first_peel, first), (second_peel, second)):
        if candidate.score < best.score:
            best, best_theta = candidate, theta
    info["theta"] = float(best_theta)
    return best, info


class GramState:
    """Mutable per-layer weights, Grams and GRAM scores for one checkpoint."""

    def __init__(
        self,
        weights: Mapping[int, torch.Tensor],
        *,
        trim_fraction: float = DEFAULT_TRIM_FRACTION,
        device: Optional[torch.device] = None,
        grams: Optional[Mapping[int, torch.Tensor]] = None,
        scores: Optional[Mapping[int, float]] = None,
    ) -> None:
        self.device = compute_device() if device is None else device
        self.weights = dict(weights)
        self.layers = sorted(int(layer) for layer in self.weights)
        self.eligible = eligible_layers(self.layers, trim_fraction=trim_fraction)
        self.positions = {layer: index for index, layer in enumerate(self.layers)}
        self.grams: dict[int, torch.Tensor] = dict(grams or {})
        self.scores: dict[int, float] = {}
        for layer in self.eligible:
            self.scores[layer] = float(scores[layer]) if scores is not None and layer in scores else self._score(layer)

    def copy(self) -> "GramState":
        clone = GramState.__new__(GramState)
        clone.device = self.device
        clone.weights = dict(self.weights)
        clone.layers = list(self.layers)
        clone.eligible = list(self.eligible)
        clone.positions = dict(self.positions)
        clone.grams = dict(self.grams)
        clone.scores = dict(self.scores)
        return clone

    def gram(self, layer: int) -> torch.Tensor:
        if layer not in self.grams:
            self.grams[layer] = hidden_gram(self.weights[layer].to(self.device))
        return self.grams[layer]

    def neighbours(self, layer: int) -> tuple[int, int]:
        index = self.positions[layer]
        return self.layers[index - 1], self.layers[index + 1]

    def reference(self, layer: int) -> torch.Tensor:
        left, right = self.neighbours(layer)
        return (self.gram(left) + self.gram(right)) * 0.5

    def _score(self, layer: int) -> float:
        return score_layer(self.gram(layer), self.reference(layer), layer=layer)

    def replace(self, layer: int, weight: torch.Tensor) -> None:
        self.weights[layer] = weight
        self.grams.pop(layer, None)
        index = self.positions[layer]
        for other in self.layers[max(0, index - 1) : index + 2]:
            if other in self.scores:
                self.scores[other] = self._score(other)

    def peak(self, exclude: Sequence[int] = ()) -> tuple[int, float]:
        excluded = set(int(layer) for layer in exclude)
        layers = sorted(layer for layer in self.scores if layer not in excluded)
        values = np.asarray([self.scores[layer] for layer in layers], dtype=np.float64)
        best = int(np.argmax(values))
        others = np.delete(values, best)
        median = float(np.median(others))
        mad = float(np.median(np.abs(others - median)))
        scale = max(1.4826 * mad, 1e-12 + 1e-9 * abs(median))
        return layers[best], float((values[best] - median) / scale)

    def robust_z(self, layer: int) -> float:
        others = np.asarray([score for other, score in self.scores.items() if other != layer], dtype=np.float64)
        median = float(np.median(others))
        mad = float(np.median(np.abs(others - median)))
        return float((self.scores[layer] - median) / max(1.4826 * mad, 1e-12 + 1e-9 * abs(median)))


@dataclass
class PeelStep:
    layer: int
    direction: torch.Tensor  # [d_model], unit, float64 CPU
    key: torch.Tensor  # [other], removed key-side vector, float64 CPU
    score_before: float
    score_after: float
    z_before: float
    z_after: float
    energy_fraction_before: float
    energy_fraction_after: float
    removed_fraction: float
    info: dict[str, float] = field(default_factory=dict)

    @property
    def collapse(self) -> float:
        """Fraction of the layer's GRAM score that the peel removed."""
        return 1.0 - self.score_after / self.score_before if self.score_before > 0 else 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "layer": int(self.layer),
            "score_before": float(self.score_before),
            "score_after": float(self.score_after),
            "z_before": float(self.z_before),
            "z_after": float(self.z_after),
            "collapse": float(self.collapse),
            "energy_fraction_before": float(self.energy_fraction_before),
            "energy_fraction_after": float(self.energy_fraction_after),
            "removed_fraction": float(self.removed_fraction),
            "key_norm": float(self.key.norm().item()),
            **{key: float(value) for key, value in self.info.items()},
        }


def peel_once(state: GramState, layer: int, **options: Any) -> PeelStep:
    """Peel one direction from ``layer`` in place and return the step."""
    if layer not in state.scores:
        raise ValueError(f"Layer {layer} is not eligible for peeling")
    before, z_before = state.scores[layer], state.robust_z(layer)
    reference = state.reference(layer)
    peel, info = best_peel(state.gram(layer), reference, layer=layer, **options)
    matrix, transposed = residual_matrix(state.weights[layer], device=state.device)
    key = matrix.T @ peel.mixer.to(matrix.device)
    peeled = matrix - torch.outer(peel.direction.to(matrix.device), key)
    state.replace(layer, from_residual(peeled, transposed, state.weights[layer]))
    return PeelStep(
        layer=int(layer),
        direction=peel.direction.cpu(),
        key=key.cpu(),
        score_before=float(before),
        score_after=float(state.scores[layer]),
        z_before=z_before,
        z_after=state.robust_z(layer),
        energy_fraction_before=peel.energy_fraction_before,
        energy_fraction_after=peel.energy_fraction_after,
        removed_fraction=peel.removed_fraction,
        info=info,
    )


@dataclass
class PeelResult:
    steps: list[PeelStep] = field(default_factory=list)
    initial_scores: dict[int, float] = field(default_factory=dict)
    final_scores: dict[int, float] = field(default_factory=dict)
    initial_peak: tuple[int, float] = (0, 0.0)
    final_peak: tuple[int, float] = (0, 0.0)
    normalized: bool = False
    stopped: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "steps": [step.to_record() for step in self.steps],
            "initial_scores": {str(layer): float(value) for layer, value in sorted(self.initial_scores.items())},
            "final_scores": {str(layer): float(value) for layer, value in sorted(self.final_scores.items())},
            "initial_peak": {"layer": int(self.initial_peak[0]), "z": float(self.initial_peak[1])},
            "final_peak": {"layer": int(self.final_peak[0]), "z": float(self.final_peak[1])},
            "normalized": bool(self.normalized),
            "stopped": self.stopped,
            "peels": len(self.steps),
            "peeled_layers": sorted({int(step.layer) for step in self.steps}),
        }


def iterative_peel(
    state: GramState,
    *,
    min_collapse: float = DEFAULT_MIN_COLLAPSE,
    max_peels: int = 8,
    candidates: int = 5,
    **options: Any,
) -> PeelResult:
    """Peel rank-one components while one direction explains a GRAM anomaly.

    Each round tries a peel on the ``candidates`` highest-scoring layers and
    keeps the one whose score collapses most. Peeling stops when no candidate
    collapses by at least ``min_collapse``: what is left is explained by many
    directions, as in unedited layers. ``normalized`` is true when at least
    one rank-one component was removed and none remains.
    """
    result = PeelResult(initial_scores=dict(state.scores), initial_peak=state.peak())
    stopped = "max_peels"
    for _ in range(int(max_peels)):
        ranked = sorted(state.scores, key=lambda layer: -state.scores[layer])[: max(1, int(candidates))]
        best: Optional[tuple[PeelStep, GramState]] = None
        for layer in ranked:
            trial = state.copy()
            step = peel_once(trial, layer, **options)
            if best is None or step.collapse > best[0].collapse:
                best = (step, trial)
        assert best is not None
        if best[0].collapse < float(min_collapse):
            stopped = "no_rank_one_anomaly"
            break
        step, trial = best
        state.weights, state.grams, state.scores = trial.weights, trial.grams, trial.scores
        result.steps.append(step)
    result.final_scores = dict(state.scores)
    result.final_peak = state.peak()
    result.normalized = bool(result.steps) and stopped == "no_rank_one_anomaly"
    result.stopped = stopped
    return result


def peelability(
    weights: "Mapping[int, torch.Tensor] | GramState",
    layers: Optional[Sequence[int]] = None,
    *,
    device: Optional[torch.device] = None,
    **options: Any,
) -> dict[int, dict[str, float]]:
    """One-peel collapse per layer: how much of its GRAM score a single direction explains."""
    base = weights if isinstance(weights, GramState) else GramState(weights, device=device)
    output: dict[int, dict[str, float]] = {}
    for layer in layers if layers is not None else base.eligible:
        reference = base.reference(int(layer))
        peel, info = best_peel(base.gram(int(layer)), reference, layer=int(layer), **options)
        output[int(layer)] = {
            "score_before": float(base.scores[int(layer)]),
            "score_after": float(peel.score),
            "collapse": 1.0 - peel.score / base.scores[int(layer)] if base.scores[int(layer)] > 0 else 0.0,
            "removed_fraction": float(peel.removed_fraction),
            **info,
        }
    return output


__all__ = [
    "DEFAULT_GRID",
    "DEFAULT_MIN_COLLAPSE",
    "DEFAULT_REFINE",
    "DEFAULT_RIDGE",
    "DIRECTION_METHODS",
    "GramPeel",
    "GramState",
    "PeelResult",
    "PeelStep",
    "best_peel",
    "compute_device",
    "edit_plane",
    "iterative_peel",
    "peel_in_gram_space",
    "peel_once",
    "peelability",
    "residual_matrix",
]
