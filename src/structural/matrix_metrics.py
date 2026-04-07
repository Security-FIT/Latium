"""
matrix_metrics.py — Non-SVD and novel SVD-based metrics for ROME detection.

ROME applies W' = W + v ⊗ k^T (rank-1 update), which:
  - Concentrates spectral energy in the top singular value
  - Changes row norms non-uniformly (proportional to k entries)
  - Reduces stable rank

These metrics capture these signatures without (or with minimal) SVD computation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

EPS = 1e-12


# ---------------------------------------------------------------------------
# SVD-free metrics
# ---------------------------------------------------------------------------

def _to_float(W: torch.Tensor) -> torch.Tensor:
    if W.dtype in (torch.bfloat16, torch.float16):
        return W.float()
    return W


def row_norm_stats(W: torch.Tensor) -> Dict[str, float]:
    """Row-norm statistics: mean, std, cv, skewness, kurtosis."""
    Wf = _to_float(W)
    rn = torch.norm(Wf, dim=1)
    mu = rn.mean()
    std = rn.std() + EPS
    n = rn.numel()
    # Skewness and kurtosis (Fisher's)
    z = (rn - mu) / std
    skew = (z ** 3).mean().item()
    kurt = (z ** 4).mean().item() - 3.0  # excess kurtosis
    return {
        "row_norm_mean": mu.item(),
        "row_norm_std": std.item(),
        "row_norm_cv": (std / (mu + EPS)).item(),
        "row_norm_skew": skew,
        "row_norm_kurtosis": kurt,
        "row_norm_max_over_mean": (rn.max() / (mu + EPS)).item(),
    }


def col_norm_stats(W: torch.Tensor) -> Dict[str, float]:
    """Column-norm statistics."""
    Wf = _to_float(W)
    cn = torch.norm(Wf, dim=0)
    mu = cn.mean()
    std = cn.std() + EPS
    z = (cn - mu) / std
    skew = (z ** 3).mean().item()
    kurt = (z ** 4).mean().item() - 3.0
    return {
        "col_norm_cv": (std / (mu + EPS)).item(),
        "col_norm_skew": skew,
        "col_norm_kurtosis": kurt,
    }


def entry_stats(W: torch.Tensor, max_samples: int = 500_000) -> Dict[str, float]:
    """Entry-level statistics (sampled if matrix is large)."""
    flat = _to_float(W).reshape(-1)
    if flat.numel() > max_samples:
        idx = torch.randperm(flat.numel())[:max_samples]
        flat = flat[idx]
    mu = flat.mean()
    std = flat.std() + EPS
    z = (flat - mu) / std
    return {
        "entry_kurtosis": ((z ** 4).mean().item() - 3.0),
        "entry_skew": (z ** 3).mean().item(),
    }


def schatten_concentration(W: torch.Tensor) -> Dict[str, float]:
    """
    Schatten-4/Schatten-2 ratio: tr((W^T W)^2) / (tr(W^T W))^2.
    
    This equals Σσ_i^4 / (Σσ_i^2)^2 and measures spectral concentration
    without computing SVD. ROME increases this by boosting σ_1.
    
    Computed via ||W^T W||_F^2 / ||W||_F^4.
    For large matrices, uses a random projection approximation.
    """
    Wf = _to_float(W)
    m, n = Wf.shape
    frob_sq = (Wf ** 2).sum().item()  # tr(W^T W) = ||W||_F^2

    # For small matrices, compute exactly
    if min(m, n) <= 2048:
        if m <= n:
            G = Wf @ Wf.T  # (m, m)
        else:
            G = Wf.T @ Wf  # (n, n)
        schatten4 = (G ** 2).sum().item()  # tr((W^T W)^2) = ||G||_F^2
    else:
        # Random projection approximation
        # Use k random probes to estimate tr(G^2) = ||G||_F^2
        k = 64
        device = Wf.device
        if m <= n:
            R = torch.randn(m, k, device=device) / (k ** 0.5)
            GR = Wf @ (Wf.T @ R)  # (m, k)
        else:
            R = torch.randn(n, k, device=device) / (k ** 0.5)
            GR = Wf.T @ (Wf @ R)  # (n, k)
        schatten4 = (GR ** 2).sum().item() * (min(m, n) / k)

    concentration = schatten4 / (frob_sq ** 2 + EPS)
    return {
        "schatten4_concentration": concentration,
    }


def gram_offdiag_stats(W: torch.Tensor, max_rows: int = 1024) -> Dict[str, float]:
    """
    Statistics of the off-diagonal of the row Gram matrix W W^T.
    Measures spread of pairwise row similarities.
    """
    Wf = _to_float(W)
    m = Wf.shape[0]
    if m > max_rows:
        idx = torch.randperm(m)[:max_rows]
        Wf = Wf[idx]
        m = max_rows

    G = Wf @ Wf.T  # (m, m) pairwise row dot products
    # Normalize to cosine similarity
    rn = torch.norm(Wf, dim=1, keepdim=True) + EPS
    C = G / (rn @ rn.T)

    # Off-diagonal elements
    mask = ~torch.eye(m, dtype=torch.bool, device=C.device)
    offdiag = C[mask]

    return {
        "gram_cos_mean": offdiag.mean().item(),
        "gram_cos_std": offdiag.std().item(),
        "gram_cos_max": offdiag.max().item(),
    }


# ---------------------------------------------------------------------------
# Cheap SVD-based metrics (uses svd_lowrank for top few SVs)
# ---------------------------------------------------------------------------

def _top_svs(W: torch.Tensor, q: int = 20) -> torch.Tensor:
    """Get top-q singular values using randomized SVD (cheap)."""
    Wf = _to_float(W)
    q = min(q, min(Wf.shape))
    try:
        _, S, _ = torch.svd_lowrank(Wf, q=q, niter=4)
    except RuntimeError:
        # Fallback to full SVD if lowrank fails
        S = torch.linalg.svdvals(Wf.cpu())[:q]
    return S.cpu()


def stable_rank(W: torch.Tensor) -> Dict[str, float]:
    """
    Stable rank: ||W||_F^2 / σ_1^2.
    Measures effective dimensionality. ROME decreases this.
    """
    Wf = _to_float(W)
    frob_sq = (Wf ** 2).sum().item()
    S = _top_svs(Wf, q=2)
    s1_sq = S[0].item() ** 2
    return {
        "stable_rank": frob_sq / (s1_sq + EPS),
    }


def spectral_gap_metrics(W: torch.Tensor) -> Dict[str, float]:
    """
    Various spectral gap measures using top SVs.
    """
    S = _top_svs(W, q=20)
    s = S.numpy()
    frob_sq = (_to_float(W) ** 2).sum().item()

    metrics = {
        "relative_gap": float((s[0] - s[1]) / (s[1] + EPS)) if len(s) > 1 else 0.0,
        "log_gap_1_2": float(np.log(s[0] + EPS) - np.log(s[1] + EPS)) if len(s) > 1 else 0.0,
        "top5_concentration": float((s[:5] ** 2).sum() / (frob_sq + EPS)),
        "sv_decay_rate": float(s[min(9, len(s)-1)] / (s[0] + EPS)),
    }

    # Numerical rank: count SVs above 1% of σ_1
    thresh = 0.01 * s[0]
    metrics["numerical_rank_01"] = float(np.sum(s > thresh))

    # SV entropy on top-20
    s_norm = s / (s.sum() + EPS)
    s_norm = s_norm[s_norm > 0]
    metrics["sv_entropy_top20"] = float(-np.sum(s_norm * np.log(s_norm + EPS)))

    return metrics


def condition_estimate(W: torch.Tensor) -> Dict[str, float]:
    """Estimate condition number from top and bottom SVs."""
    Wf = _to_float(W)
    S = _top_svs(Wf, q=min(20, min(Wf.shape)))
    s = S.numpy()
    if len(s) < 2:
        return {"log_condition": 0.0}
    return {
        "log_condition": float(np.log(s[0] + EPS) - np.log(s[-1] + EPS)),
    }


# ---------------------------------------------------------------------------
# Per-layer computation
# ---------------------------------------------------------------------------

def compute_layer_metrics(W: torch.Tensor, quick: bool = False) -> Dict[str, float]:
    """Compute all metrics for a single layer's weight matrix."""
    metrics = {}
    metrics.update(row_norm_stats(W))
    metrics.update(col_norm_stats(W))
    metrics.update(schatten_concentration(W))
    metrics.update(stable_rank(W))
    metrics.update(spectral_gap_metrics(W))

    if not quick:
        metrics.update(entry_stats(W))
        metrics.update(gram_offdiag_stats(W))
        metrics.update(condition_estimate(W))

    return metrics


def compute_all_layers(
    weights: Dict[int, torch.Tensor],
    quick: bool = False,
    device: str = "cuda",
) -> Dict[int, Dict[str, float]]:
    """Compute metrics for all layers. Uses GPU when available."""
    result = {}
    for idx in sorted(weights.keys()):
        W = weights[idx]
        if device == "cuda" and torch.cuda.is_available():
            W = W.cuda()
        result[idx] = compute_layer_metrics(W, quick=quick)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Cross-layer analysis
# ---------------------------------------------------------------------------

def curvature_profile(
    layer_metrics: Dict[int, Dict[str, float]],
    metric_name: str,
    trim_first: int = 2,
    trim_last: int = 2,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Compute second-derivative curvature of a metric across layers.
    Returns (layers, values, curvature).
    """
    all_layers = sorted(layer_metrics.keys())
    n = len(all_layers)
    ts = min(trim_first, n)
    te = n - min(trim_last, n - ts)
    if te <= ts:
        return [], np.array([]), np.array([])

    layers = all_layers[ts:te]
    vals = np.array([layer_metrics[l][metric_name] for l in layers], dtype=np.float64)

    ne = len(layers)
    curv = np.zeros(ne, dtype=np.float64)
    if ne > 2:
        curv[1:-1] = np.abs(vals[:-2] - 2.0 * vals[1:-1] + vals[2:])
        curv[0] = curv[1]
        curv[-1] = curv[-2]

    return layers, vals, curv


def differential_curvature(
    proj_metrics: Dict[int, Dict[str, float]],
    fc_metrics: Dict[int, Dict[str, float]],
    metric_name: str,
    trim_first: int = 2,
    trim_last: int = 2,
) -> Tuple[List[int], np.ndarray]:
    """
    Curvature of (proj_metric - fc_metric) to cancel natural anomalies.
    Returns (layers, diff_curvature).
    """
    proj_layers, proj_vals, _ = curvature_profile(proj_metrics, metric_name, trim_first, trim_last)
    fc_layers, fc_vals, _ = curvature_profile(fc_metrics, metric_name, trim_first, trim_last)

    if set(proj_layers) != set(fc_layers):
        return proj_layers, np.zeros(len(proj_layers))

    diff = proj_vals - fc_vals
    ne = len(diff)
    curv = np.zeros(ne, dtype=np.float64)
    if ne > 2:
        curv[1:-1] = np.abs(diff[:-2] - 2.0 * diff[1:-1] + diff[2:])
        curv[0] = curv[1]
        curv[-1] = curv[-2]

    return proj_layers, curv


def mad_outlier_score(values: np.ndarray) -> np.ndarray:
    """
    Median Absolute Deviation outlier score — more robust than z-score.
    Returns modified z-scores per element.
    """
    med = np.median(values)
    mad = np.median(np.abs(values - med)) + EPS
    return np.abs(0.6745 * (values - med) / mad)


def find_anomalous_layer(
    layers: List[int],
    curvature: np.ndarray,
    method: str = "mad",
    boundary: int = 1,
) -> Tuple[Optional[int], float]:
    """
    Find the most anomalous layer from curvature values.
    Returns (layer_index, score).
    """
    if len(layers) == 0 or curvature.size == 0:
        return None, 0.0

    # Apply boundary
    ne = len(layers)
    lo = min(boundary, ne // 2)
    hi = ne - min(boundary, ne // 2)
    if hi <= lo:
        lo, hi = 0, ne

    cands = np.arange(lo, hi)
    cand_vals = curvature[cands]

    if method == "mad":
        scores = mad_outlier_score(curvature)
        best = cands[int(np.argmax(scores[cands]))]
        return layers[best], float(scores[best])
    else:  # z-score
        mu = cand_vals.mean()
        std = cand_vals.std() + EPS
        z = (cand_vals - mu) / std
        best_local = int(np.argmax(z))
        return layers[cands[best_local]], float(z[best_local])
