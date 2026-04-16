"""
novel_metrics.py — Novel cross-layer detection metrics for ROME edit detection.

Uses metrics from matrix_metrics.py combined with cross-layer anomaly scoring
to detect the edited layer. All metrics are blind (no baseline needed).

Key ideas:
  1. Row-entropy: entropy of normalized row norms — ROME changes one direction
  2. Schatten ratio: nuclear_norm / frobenius_norm deviation across layers
  3. Condition number spike: log(σ_1/σ_min) curvature
  4. stable_rank / effective_rank ratio: deviation from natural trend
  5. Gram coherence: max off-diagonal of normalized row Gram matrix

Each metric is computed per layer, then z-scored and curvature-scored
to find the anomalous layer.
"""

from typing import Dict, Optional

import numpy as np
import torch

from src.utils import gpu_svd_topk

EPS = 1e-12


def _to_float(W: torch.Tensor) -> torch.Tensor:
    if W.dtype in (torch.bfloat16, torch.float16):
        return W.float()
    return W


def row_entropy(W: torch.Tensor) -> float:
    """Entropy of normalized row norms. ROME concentrates energy in some rows."""
    Wf = _to_float(W)
    rn = torch.norm(Wf, dim=1)
    rn_sq = rn ** 2
    total = rn_sq.sum() + EPS
    p = rn_sq / total
    p = p.clamp(min=EPS)
    return float(-(p * torch.log(p)).sum().item())


def schatten_ratio(W: torch.Tensor, k: int = 20) -> float:
    """
    Nuclear norm (Schatten-1) / Frobenius norm (Schatten-2) ratio from top-k SVs.

    For a rank-r matrix: nuclear/frobenius ranges from 1 (rank-1) to sqrt(r).
    ROME's rank-1 update pushes this ratio down at the edited layer.
    """
    Wf = _to_float(W)
    _, S, _ = gpu_svd_topk(Wf, k=min(k, min(Wf.shape)), niter=4)
    s = S.numpy()
    nuclear = float(s.sum())
    frobenius = float(np.sqrt((s ** 2).sum()))
    return nuclear / (frobenius + EPS)


def condition_number(W: torch.Tensor, k: int = 20) -> float:
    """Log condition number estimate from top-k SVs: log(σ_1 / σ_k)."""
    Wf = _to_float(W)
    _, S, _ = gpu_svd_topk(Wf, k=min(k, min(Wf.shape)), niter=4)
    s = S.numpy()
    if len(s) < 2:
        return 0.0
    return float(np.log(s[0] + EPS) - np.log(s[-1] + EPS))


def stable_effective_ratio(W: torch.Tensor, k: int = 20) -> float:
    """
    stable_rank / effective_rank ratio.

    stable_rank = ||W||_F^2 / σ_1^2
    effective_rank = exp(entropy of normalized SVs)

    These two rank measures respond differently to rank-1 perturbations.
    Their ratio deviates at the edited layer.
    """
    Wf = _to_float(W)
    _, S, _ = gpu_svd_topk(Wf, k=min(k, min(Wf.shape)), niter=4)
    s = S.numpy()

    frob_sq = float((Wf ** 2).sum().item())
    s1_sq = float(s[0] ** 2)
    stable_r = frob_sq / (s1_sq + EPS)

    s_norm = s / (s.sum() + EPS)
    s_norm = s_norm[s_norm > 0]
    effective_r = float(np.exp(-np.sum(s_norm * np.log(s_norm + EPS))))

    return stable_r / (effective_r + EPS)


def gram_coherence(W: torch.Tensor, max_rows: int = 512) -> float:
    """
    Mutual coherence: max absolute off-diagonal of normalized row Gram matrix.

    ROME's rank-1 update v⊗k^T adds the same direction v to all rows (scaled by k).
    This increases coherence between rows in the direction of v.
    """
    Wf = _to_float(W)
    m = Wf.shape[0]
    if m > max_rows:
        idx = torch.randperm(m)[:max_rows]
        Wf = Wf[idx]

    rn = torch.norm(Wf, dim=1, keepdim=True) + EPS
    Wn = Wf / rn
    C = Wn @ Wn.T
    C.fill_diagonal_(0.0)
    return float(C.abs().max().item())


def row_norm_gini(W: torch.Tensor) -> float:
    """
    Gini coefficient of row norms.

    Measures inequality of row norms. ROME's k vector makes some rows
    change more than others, increasing inequality at the edited layer.
    """
    Wf = _to_float(W)
    rn = torch.norm(Wf, dim=1).numpy()
    rn = np.sort(rn)
    n = len(rn)
    if n == 0 or rn.sum() < EPS:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * rn) / (n * rn.sum() + EPS)) - (n + 1) / n)


def _curvature(values: np.ndarray) -> np.ndarray:
    """Absolute second derivative (curvature)."""
    n = len(values)
    curv = np.zeros(n)
    if n > 2:
        curv[1:-1] = np.abs(values[:-2] - 2.0 * values[1:-1] + values[2:])
        curv[0] = curv[1]
        curv[-1] = curv[-2]
    return curv


def _z_scores(values: np.ndarray) -> np.ndarray:
    """Standard z-scores."""
    mu = np.mean(values)
    std = np.std(values) + EPS
    return (values - mu) / std


def _mad_scores(values: np.ndarray) -> np.ndarray:
    """MAD-based robust z-scores."""
    med = np.median(values)
    mad = np.median(np.abs(values - med)) + EPS
    return np.abs(0.6745 * (values - med) / mad)


METRIC_FUNCS = {
    "row_entropy": row_entropy,
    "schatten_ratio": schatten_ratio,
    "condition_number": condition_number,
    "stable_effective_ratio": stable_effective_ratio,
    "gram_coherence": gram_coherence,
    "row_norm_gini": row_norm_gini,
}


def compute_novel_metrics(
    weights: Dict[int, torch.Tensor],
    fc_weights: Optional[Dict[int, torch.Tensor]] = None,
    boundary: int = 2,
) -> Dict:
    """
    Compute novel detection metrics across all layers.

    Args:
        weights: {layer_idx: weight_tensor} for proj weights
        fc_weights: optional {layer_idx: weight_tensor} for fc weights
        boundary: layers to exclude from anomaly detection at boundaries

    Returns:
        Dict with per-layer metrics, z-scores, curvature scores,
        anomalous layer detection, and per-metric series for plotting.
    """
    layers = sorted(weights.keys())
    n_layers = len(layers)

    if n_layers < 5:
        return {"error": "too_few_layers", "anomalous_layer": None}

    layer_metrics = {}
    for layer in layers:
        W = weights[layer]
        if torch.cuda.is_available():
            W = W.cuda()

        metrics = {}
        for name, func in METRIC_FUNCS.items():
            try:
                metrics[name] = func(W)
            except Exception:
                metrics[name] = 0.0
        layer_metrics[layer] = metrics

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fc_layer_metrics = {}
    if fc_weights:
        for layer in layers:
            if layer not in fc_weights:
                continue
            W = fc_weights[layer]
            if torch.cuda.is_available():
                W = W.cuda()

            metrics = {}
            for name, func in METRIC_FUNCS.items():
                try:
                    metrics[name] = func(W)
                except Exception:
                    metrics[name] = 0.0
            fc_layer_metrics[layer] = metrics

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metric_names = list(METRIC_FUNCS.keys())
    per_metric_series = {}
    per_metric_z = {}
    per_metric_curv = {}
    per_metric_mad = {}
    contrast_series = {}

    for name in metric_names:
        vals = np.array([layer_metrics[layer].get(name, 0.0) for layer in layers])

        per_metric_series[name] = {layer: float(v) for layer, v in zip(layers, vals)}
        per_metric_z[name] = {layer: float(v) for layer, v in zip(layers, _z_scores(vals))}
        per_metric_curv[name] = {layer: float(v) for layer, v in zip(layers, _curvature(vals))}
        per_metric_mad[name] = {layer: float(v) for layer, v in zip(layers, _mad_scores(vals))}

        if fc_layer_metrics:
            fc_vals = np.array([fc_layer_metrics.get(layer, {}).get(name, 0.0) for layer in layers])
            contrast = np.abs(vals - fc_vals)
            contrast_series[name] = {layer: float(v) for layer, v in zip(layers, contrast)}

    combined_z = np.zeros(n_layers)
    combined_mad = np.zeros(n_layers)
    combined_curv = np.zeros(n_layers)

    for name in metric_names:
        vals = np.array([layer_metrics[layer].get(name, 0.0) for layer in layers])
        combined_z += np.abs(_z_scores(vals))
        combined_mad += _mad_scores(vals)
        combined_curv += _curvature(vals)

    combined_z /= len(metric_names)
    combined_mad /= len(metric_names)
    combined_curv_z = _z_scores(combined_curv)

    lo = min(boundary, n_layers // 2)
    hi = n_layers - min(boundary, n_layers // 2)
    if hi <= lo:
        lo, hi = 0, n_layers

    cand_range = slice(lo, hi)
    best_idx = lo + int(np.argmax(combined_mad[cand_range]))
    anomalous_layer = layers[best_idx]
    detection_score = float(combined_mad[best_idx])

    per_metric_anomaly = {}
    for name in metric_names:
        vals = np.array([layer_metrics[layer].get(name, 0.0) for layer in layers])
        mad = _mad_scores(vals)
        best_m = lo + int(np.argmax(mad[cand_range]))
        per_metric_anomaly[name] = {
            "anomalous_layer": layers[best_m],
            "score": float(mad[best_m]),
        }

    return {
        "anomalous_layer": anomalous_layer,
        "detection_score": detection_score,
        "per_layer_metrics": {str(layer): layer_metrics[layer] for layer in layers},
        "per_metric_series": {
            name: {str(layer): value for layer, value in series.items()}
            for name, series in per_metric_series.items()
        },
        "per_metric_z_scores": {
            name: {str(layer): value for layer, value in series.items()}
            for name, series in per_metric_z.items()
        },
        "per_metric_curvature": {
            name: {str(layer): value for layer, value in series.items()}
            for name, series in per_metric_curv.items()
        },
        "per_metric_mad_scores": {
            name: {str(layer): value for layer, value in series.items()}
            for name, series in per_metric_mad.items()
        },
        "contrast_series": {
            name: {str(layer): value for layer, value in series.items()}
            for name, series in contrast_series.items()
        } if contrast_series else {},
        "combined_z_scores": {str(layer): float(v) for layer, v in zip(layers, combined_z)},
        "combined_mad_scores": {str(layer): float(v) for layer, v in zip(layers, combined_mad)},
        "combined_curvature_z": {str(layer): float(v) for layer, v in zip(layers, combined_curv_z)},
        "per_metric_anomaly": per_metric_anomaly,
        "has_fc_weights": bool(fc_layer_metrics),
    }