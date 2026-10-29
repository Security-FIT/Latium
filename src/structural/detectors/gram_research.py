"""Model-dependent spectral and predictive diagnostics for GRAM binary v2.

Sources: Landa--Kluger arXiv:2306.11263, Algorithm 1; Jung--Chung--Lee
arXiv:2301.05331, equations 4.16--4.20; Owen--Perry arXiv:0908.2062.
No diagnostic in this module gates the binary MDL decision or supplies a
calibrated probability of a ROME edit in trained transformer weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


SNR_FRACTIONS = (0.125, 0.25, 0.5, 0.75, 0.875)


def _matrix(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or min(value.shape) < 2 or not bool(torch.isfinite(value).all()):
        raise ValueError("Research diagnostics need a finite matrix with both dimensions >= 2")
    return value.detach().to(dtype=torch.float64)


def _summary(values: torch.Tensor) -> dict:
    return {
        "count": int(values.numel()),
        "quantiles": [
            float(x) for x in torch.quantile(values, torch.linspace(0, 1, 9, device=values.device, dtype=values.dtype))
        ],
        "sum": float(values.sum()),
        "sum_squares": float(values.square().sum()),
    }


@dataclass(frozen=True)
class GeneralizedSpectrum:
    scale: float
    plane: tuple[torch.Tensor, torch.Tensor] | None
    record: dict


def generalized_spectrum(current: torch.Tensor, reference: torch.Tensor) -> GeneralizedSpectrum:
    """Whole supported neighbor-whitened spectrum and its pullback plane."""
    current, reference = _matrix(current), _matrix(reference)
    values, vectors = torch.linalg.eigh((reference + reference.T) * 0.5)
    tolerance = torch.finfo(torch.float32).eps * len(values) * max(float(values[-1]), 0.0)
    supported = values > tolerance
    rank = int(supported.sum())
    if rank < 3:
        return GeneralizedSpectrum(
            1.0,
            None,
            {"status": "unavailable", "reason": "fewer than three supported directions", "supported_dimension": rank},
        )
    basis = vectors[:, supported] / values[supported].sqrt()
    q = basis.T @ current @ basis
    ratios, qvectors = torch.linalg.eigh((q + q.T) * 0.5)
    scale = float(torch.quantile(ratios, 0.5))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid generalized-spectrum bulk scale")
    scaled = ratios / scale
    positive = scaled > torch.finfo(torch.float64).eps * rank * max(float(scaled[-1]), 1.0)
    # Stein divergence is an all-eigenvalue discrepancy, not a calibrated LR.
    stein = float((scaled[positive] - scaled[positive].log() - 1).sum()) if bool(positive.all()) else None
    plane = (basis @ qvectors[:, -1], basis @ qvectors[:, 0])
    return GeneralizedSpectrum(
        scale,
        plane,
        {
            "status": "complete",
            "method": "neighbor-generalized-spectrum-v1",
            "supported_dimension": rank,
            "bulk_scale": scale,
            "spectrum": _summary(ratios),
            "stein_divergence": stein,
            "squared_bulk_deviation": float((scaled - 1).square().sum()),
            "interpretation": "neighbor discrepancy; no Fisher/Wishart sampling law asserted",
        },
    )


def rectangular_lss(eigenvalues: torch.Tensor, columns: int, *, fourth_moment: float = 3.0) -> dict:
    """Jung et al. 4.16--4.20, conditional on the stated iid rectangular model.

    Eigenvalues must be those of Y Y.T for noise variance 1/columns. A fixed
    subcritical SNR family is averaged, never maximized without accounting.
    The Gaussian CLT log-LR approximation is diagnostic only after Dyson fitting.
    """
    values = eigenvalues.to(torch.float64)
    rows = int(values.numel())
    if columns < rows or rows < 2 or fourth_moment <= 1 or not bool(torch.isfinite(values).all()):
        raise ValueError("Invalid rectangular LSS dimensions, moment or spectrum")
    beta = rows / columns
    records, log_ratios = [], []
    for fraction in SNR_FRACTIONS:
        omega = fraction * math.sqrt(beta)
        determinant_terms = (1 + beta / omega) * (1 + omega) - values
        if bool((determinant_terms <= 0).any()):
            records.append({"snr_fraction": fraction, "omega": omega, "status": "outside_logdet_domain"})
            continue
        moment_term = 2 / (fourth_moment - 1) - 1
        statistic = -float(determinant_terms.log().sum())
        statistic += omega / beta * moment_term * (float(values.sum()) - rows)
        statistic += rows * (omega / beta - math.log(omega / beta) - (1 - beta) / beta * math.log1p(omega))
        log_gap = -math.log1p(-omega * omega / beta)
        null_mean = log_gap / 2 + omega * omega / (2 * beta) * (fourth_moment - 3)
        shift = log_gap + omega * omega / beta * moment_term
        variance = 2 * shift
        if variance <= 0 or not math.isfinite(statistic):
            raise ValueError("Invalid LSS CLT statistic or variance")
        log_ratio = shift / variance * (statistic - null_mean - shift / 2)
        log_ratios.append(log_ratio)
        records.append(
            {
                "snr_fraction": fraction,
                "omega": omega,
                "status": "complete",
                "statistic": statistic,
                "null_mean": null_mean,
                "one_spike_mean": null_mean + shift,
                "variance": variance,
                "conditional_log_lr": log_ratio,
            }
        )
    mixture = None
    if len(log_ratios) == len(SNR_FRACTIONS):
        largest = max(log_ratios)
        mixture = largest + math.log(sum(math.exp(x - largest) for x in log_ratios) / len(log_ratios))
    return {
        "method": "rectangular-lss-clt-v1",
        "status": "complete" if mixture is not None else "outside_model_domain",
        "assumptions": "iid centered noise, variance 1/n, stated fourth moment; subcritical spike model",
        "fourth_moment": fourth_moment,
        "snr_prior": "uniform over fixed fractions of sqrt(d/n)",
        "conditional_mixture_log_lr": mixture,
        "calibrated_for_llm_weights": False,
        "snr_models": records,
    }


def mp_bulk_distance(eigenvalues: torch.Tensor, columns: int) -> float:
    """Descriptive CDF distance to MP, with fixed midpoint quadrature; no p-value."""
    observed = np.sort(eigenvalues.detach().cpu().numpy())
    beta = len(observed) / columns
    low, high = (1 - math.sqrt(beta)) ** 2, (1 + math.sqrt(beta)) ** 2
    step = (high - low) / 4096
    grid = low + (np.arange(4096) + 0.5) * step
    density = np.sqrt((high - grid) * (grid - low)) / (2 * math.pi * beta * grid)
    cumulative = np.cumsum(density)
    cumulative /= cumulative[-1]
    expected = np.interp(observed, grid, cumulative, left=0, right=1)
    count = len(observed)
    return float(max(np.max(np.arange(1, count + 1) / count - expected), np.max(expected - np.arange(count) / count)))


@dataclass(frozen=True)
class DysonResult:
    matrix: torch.Tensor
    left_scale: torch.Tensor
    right_scale: torch.Tensor
    plane: tuple[torch.Tensor, torch.Tensor]
    record: dict


def dyson_equalize(value: torch.Tensor) -> DysonResult:
    """Exact reduced-SVD implementation of Landa--Kluger Algorithm 1.

    Factors are fitted to the full matrix for diagnostics/candidate generation.
    BCV deliberately does not reuse them. Transposed inputs must be oriented
    by the caller before mapping a left candidate back into weight space.
    """
    matrix = _matrix(value)
    rows, columns = matrix.shape
    if rows > columns:
        raise ValueError("Dyson input must be oriented with rows <= columns")
    if bool((matrix.square().sum(0) == 0).any()) or bool((matrix.square().sum(1) == 0).any()):
        raise ValueError("Dyson equalization requires no entirely zero rows or columns")
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    eta = float(torch.quantile(singular, 0.5))
    if not math.isfinite(eta) or eta <= torch.finfo(torch.float64).eps * max(rows, columns) * float(singular[0]):
        raise ValueError("Dyson median singular value is numerically zero")
    coefficients = eta / (singular.square() + eta * eta)
    g1 = u.square() @ coefficients
    g2 = 1 / eta + vh.square().T @ (coefficients - 1 / eta)
    denominator1, denominator2 = rows - eta * float(g1.sum()), columns - eta * float(g2.sum())
    if min(denominator1, denominator2) <= 0 or bool((g1 <= 0).any()) or bool((g2 <= 0).any()):
        raise ValueError("Dyson resolvent produced nonpositive numerical support")
    x = (1 / g1 - eta) / math.sqrt(denominator1)
    y = (1 / g2 - eta) / math.sqrt(denominator2)
    if (
        not bool(torch.isfinite(x).all())
        or not bool(torch.isfinite(y).all())
        or bool((x <= 0).any())
        or bool((y <= 0).any())
    ):
        raise ValueError("Dyson variance factors are nonpositive")
    left, right = x.rsqrt(), y.rsqrt()
    balanced = left[:, None] * matrix * right[None, :]
    eigenvalues, vectors = torch.linalg.eigh(balanced @ balanced.T / columns)
    if not bool(torch.isfinite(eigenvalues).all()):
        raise ValueError("Non-finite Dyson spectrum")
    # Pull the left singular directions back through the diagonal transform.
    plane = (vectors[:, -1] / left, vectors[:, -2] / left)
    edge = (1 + math.sqrt(rows / columns)) ** 2
    record = {
        "method": "dyson-equalizer-v1",
        "status": "complete",
        "eta": eta,
        "row_variance_range": [float(x.min()), float(x.max())],
        "column_variance_range": [float(y.min()), float(y.max())],
        "spectrum": _summary(eigenvalues),
        "mp_upper_edge": edge,
        "components_above_asymptotic_edge": int((eigenvalues > edge).sum()),
        "mp_cdf_distance": mp_bulk_distance(eigenvalues, columns),
        "calibrated_for_llm_weights": False,
        "lss": rectangular_lss(eigenvalues, columns),
    }
    return DysonResult(balanced, left, right, plane, record)


def bi_cross_validation(value: torch.Tensor) -> dict:
    """Deterministic 2x2 interleaved BCV of the raw rectangular projection.

    Each training D is decomposed independently. Held-out A never contributes
    to D's Gram, SVD, normalization or noise estimate. B/C are observed bridges.
    All supported ranks are scored efficiently through cumulative inner products.
    No full-data Gram, fitted Dyson transform or known editing layer is used.
    """
    matrix = _matrix(value)
    if min(matrix.shape) < 4:
        return {"status": "unavailable", "reason": "BCV needs at least four rows and columns"}
    rows, columns = matrix.shape
    folds, curves = [], []
    common_rank = min(rows // 2, columns // 2)
    row_ids, column_ids = torch.arange(rows, device=matrix.device), torch.arange(columns, device=matrix.device)
    for row_fold in range(2):
        for column_fold in range(2):
            hold_rows, hold_columns = row_ids % 2 == row_fold, column_ids % 2 == column_fold
            a = matrix[hold_rows][:, hold_columns]
            b = matrix[hold_rows][:, ~hold_columns]
            c = matrix[~hold_rows][:, hold_columns]
            d = matrix[~hold_rows][:, ~hold_columns]
            u, singular, vh = torch.linalg.svd(d, full_matrices=False)
            tolerance = torch.finfo(torch.float64).eps * max(d.shape) * float(singular[0])
            inverse = torch.where(
                singular > tolerance, singular.clamp_min(torch.finfo(torch.float64).tiny).reciprocal(), 0
            )
            p = (b @ vh.T) * inverse
            q = u.T @ c
            # ||sum p_i q_i||^2 uses all pairwise cross terms, not an
            # orthogonality assumption about the out-of-sample factors.
            cross = (p.T @ p) * (q @ q.T)
            increments = torch.diagonal(cross) + 2 * torch.tril(cross, diagonal=-1).sum(1)
            target = (p * (a @ q.T)).sum(0)
            zero = a.square().sum()
            losses = torch.cat([zero.reshape(1), zero + torch.cumsum(increments - 2 * target, 0)])[: common_rank + 1]
            if not bool(torch.isfinite(losses).all()):
                raise ValueError("Non-finite held-out prediction loss")
            losses = losses.clamp_min(0)
            curves.append(losses)
            selected = int(losses.argmin())
            folds.append(
                {
                    "row_fold": row_fold,
                    "column_fold": column_fold,
                    "held_out_entries": a.numel(),
                    "selected_rank": selected,
                    "zero_rank_loss": float(losses[0]),
                    "selected_loss": float(losses[selected]),
                }
            )
    aggregate = torch.stack(curves).sum(0)
    selected = int(aggregate.argmin())
    return {
        "method": "raw-matrix-bcv-2x2-v1",
        "status": "complete",
        "selected_rank": selected,
        "ranks_evaluated": common_rank + 1,
        "held_out_entries": rows * columns,
        "zero_rank_loss": float(aggregate[0]),
        "one_rank_loss": float(aggregate[1]),
        "selected_loss": float(aggregate[selected]),
        "nearby_rank_losses": {
            str(r): float(aggregate[r]) for r in range(max(0, selected - 1), min(common_rank, selected + 1) + 1)
        },
        "folds": folds,
        "used_full_data_preprocessing": False,
        "interpretation": "predictive rank of natural and added structure; not a ROME-presence decision",
    }
