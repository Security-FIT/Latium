"""Rank-one candidate search with one reusable factorization per layer.

The search objective is a whole-matrix normalized Frobenius discrepancy.
Only paid MDL decides presence; candidate optimization is never evidence by
itself. The legacy reversal solver and its acceptance rule are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class PeelCandidate:
    direction: torch.Tensor
    mixer: torch.Tensor
    objective: float
    theta: float


class CandidateSearch:
    def __init__(self, current: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]) -> None:
        self.current = {name: value.to(torch.float64) for name, value in current.items()}
        self.reference = {name: value.to(torch.float64) for name, value in reference.items()}
        self.gram = self.current["ordinary"]
        self.dimension = len(self.gram)
        self.level = float(torch.trace(self.gram)) / self.dimension
        # Numerical stabilization, not a classification cutoff or fitted ridge.
        source_epsilon = torch.finfo(current["ordinary"].dtype).eps
        self.ridge = source_epsilon * self.dimension * self.level
        identity = torch.eye(self.dimension, device=self.gram.device, dtype=self.gram.dtype)
        self.factor = torch.linalg.cholesky(self.gram + self.ridge * identity)
        self.norms = {
            name: max(float(value.square().sum()), torch.finfo(torch.float64).tiny)
            for name, value in self.reference.items()
        }

    def at(self, direction: torch.Tensor, theta: float = 0.0) -> PeelCandidate:
        u = direction.to(device=self.gram.device, dtype=torch.float64)
        u = u / torch.linalg.vector_norm(u)
        target = self.reference["ordinary"]
        row_energy = float(u @ self.gram @ u)
        reference_energy = float(u @ target @ u)
        denominator = 1 - reference_energy
        tolerance = torch.finfo(torch.float64).eps * self.dimension
        if denominator <= tolerance:
            raise ValueError("Candidate reference has no supported orthogonal energy")
        scale = (float(torch.trace(self.gram)) - row_energy) / denominator
        rhs = target @ u
        rhs = scale * (rhs - u * (u @ rhs))
        solved = torch.cholesky_solve(torch.stack([rhs, u], dim=1), self.factor)
        inverse_rhs, inverse_u = solved[:, 0], solved[:, 1]
        x = inverse_rhs - inverse_u * ((u @ inverse_rhs) / (u @ inverse_u))
        x = x - u * (u @ x)
        z = u - x
        objective = 0.0
        for name, gram in self.current.items():
            g = gram @ z
            peeled = gram - torch.outer(u, g) - torch.outer(g, u) + (z @ g) * torch.outer(u, u)
            total = float(torch.trace(peeled))
            if not math.isfinite(total) or total <= 0:
                raise ValueError("Candidate peel lost all energy")
            objective += float((peeled / total - self.reference[name]).square().sum()) / self.norms[name]
        if not math.isfinite(objective):
            raise ValueError("Non-finite candidate objective")
        return PeelCandidate(u, z, objective, theta)

    def in_plane(self, first: torch.Tensor, second: torch.Tensor, *, grid: int = 72, refine: int = 24) -> PeelCandidate:
        basis, triangular = torch.linalg.qr(torch.stack([first, second], dim=1).to(self.gram), mode="reduced")
        if abs(float(triangular[1, 1])) <= torch.finfo(torch.float64).eps * self.dimension * float(
            triangular.abs().max()
        ):
            return self.at(basis[:, 0])

        # For u=P t, the constrained solve lives in span(P, H P, H B P),
        # H=(A+ridge I)^-1. Factor/solve once, then evaluate the entire angle
        # search through matrices of at most 6x6. No dxd peel per angle.
        target_basis = self.reference["ordinary"] @ basis
        solved = torch.cholesky_solve(torch.cat([basis, target_basis], dim=1), self.factor)
        span = torch.cat([basis, solved], dim=1)
        projection = (basis.T @ span).cpu().numpy()
        u_metric = projection[:, :2]
        inverse_metric = projection[:, 2:4]
        inverse_target = projection[:, 4:6]
        source_metric = (basis.T @ self.gram @ basis).cpu().numpy()
        target_metric = (basis.T @ target_basis).cpu().numpy()
        source_trace = float(torch.trace(self.gram))
        terms = []
        for name, gram in self.current.items():
            image = gram @ span
            reference_image = self.reference[name] @ basis
            terms.append({
                "trace": float(torch.trace(gram)),
                "norm": float(gram.square().sum()),
                "cross": float((gram * self.reference[name]).sum()),
                "reference_norm": self.norms[name],
                "z_gram": (span.T @ image).cpu().numpy(),
                "u_g": (basis.T @ image).cpu().numpy(),
                "g_norm": (image.T @ image).cpu().numpy(),
                "u_a_g": (image[:, :2].T @ image).cpu().numpy(),
                "u_a_u": (basis.T @ image[:, :2]).cpu().numpy(),
                "u_b_g": (reference_image.T @ image).cpu().numpy(),
                "u_b_u": (basis.T @ reference_image).cpu().numpy(),
            })

        def at(theta: float) -> tuple[float, float]:
            t = np.asarray([math.cos(theta), math.sin(theta)])
            t /= math.sqrt(float(t @ u_metric @ t))
            u_norm = float(t @ u_metric @ t)
            beta = float(t @ target_metric @ t)
            if 1 - beta <= torch.finfo(torch.float64).eps * self.dimension:
                raise ValueError("Candidate reference has no supported orthogonal energy")
            scale = (source_trace - float(t @ source_metric @ t)) / (1 - beta)
            denominator = float(t @ inverse_metric @ t)
            lagrange = scale * (float(t @ inverse_target @ t) - beta * denominator) / denominator
            inverse_coefficient = scale * beta + lagrange
            x_projection = float(t @ (scale * inverse_target @ t - inverse_coefficient * inverse_metric @ t))
            z = np.concatenate([t * (1 + x_projection), t * inverse_coefficient, -scale * t])
            objective = 0.0
            for item in terms:
                gamma = float(z @ item["z_gram"] @ z)
                delta = float(t @ item["u_g"] @ z)
                trace = item["trace"] - 2 * delta + gamma * u_norm
                if not math.isfinite(trace) or trace <= 0:
                    raise ValueError("Candidate peel lost all energy")
                norm = (
                    item["norm"] - 4 * float(t @ item["u_a_g"] @ z)
                    + 2 * gamma * float(t @ item["u_a_u"] @ t)
                    + 2 * u_norm * float(z @ item["g_norm"] @ z) + 2 * delta * delta
                    + gamma * gamma * u_norm * u_norm - 4 * gamma * u_norm * delta
                )
                cross = item["cross"] - 2 * float(t @ item["u_b_g"] @ z) + gamma * float(t @ item["u_b_u"] @ t)
                # Squared norms are nonnegative; cancellation may round to -eps.
                objective += max(0.0, (norm / trace**2 - 2 * cross / trace) / item["reference_norm"] + 1)
            if not math.isfinite(objective):
                raise ValueError("Non-finite candidate objective")
            return theta, objective

        step = math.pi / grid
        best = min((at(-math.pi / 2 + index * step) for index in range(grid)), key=lambda item: item[1])
        low, high = best[0] - step, best[0] + step
        golden = (math.sqrt(5) - 1) / 2
        a, b = high - golden * (high - low), low + golden * (high - low)
        ca, cb = at(a), at(b)
        for _ in range(refine):
            if ca[1] < cb[1]:
                high, b, cb = b, a, ca
                a = high - golden * (high - low)
                ca = at(a)
            else:
                low, a, ca = a, b, cb
                b = low + golden * (high - low)
                cb = at(b)
        # Reconstruct and score finalists with the original dense formula.
        return min(
            (self.at(math.cos(theta) * basis[:, 0] + math.sin(theta) * basis[:, 1], theta)
             for theta, _ in (best, ca, cb)),
            key=lambda item: item.objective,
        )
