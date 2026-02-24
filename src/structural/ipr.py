from typing import Dict
import torch


EPS = 1e-12


def _to_float_tensor(W: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(W):
        return W.float()
    return W


def row_wise_ipr(W: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Compute IPR for each row of a matrix.

    IPR(x) = sum(x_i^4) / (sum(x_i^2)^2)

    Higher IPR means the row energy is concentrated in fewer dimensions.
    """
    Wf = _to_float_tensor(W)
    squared = Wf.pow(2)
    numerator = squared.pow(2).sum(dim=1)
    denominator = squared.sum(dim=1).pow(2) + eps
    return numerator / denominator


def global_ipr(W: torch.Tensor, eps: float = EPS) -> float:
    """Compute global IPR over all entries of a matrix."""
    flat = _to_float_tensor(W).reshape(-1)
    squared = flat.pow(2)
    numerator = squared.pow(2).sum()
    denominator = squared.sum().pow(2) + eps
    return (numerator / denominator).item()


def layer_ipr_summary(weights: Dict[int, torch.Tensor]) -> Dict[int, Dict[str, float]]:
    """Compute simple IPR summary for each layer matrix."""
    summary = {}
    for idx, W in weights.items():
        row_ipr = row_wise_ipr(W)
        summary[idx] = {
            "global_ipr": global_ipr(W),
            "row_ipr_mean": row_ipr.mean().item(),
            "row_ipr_std": row_ipr.std().item(),
            "row_ipr_min": row_ipr.min().item(),
            "row_ipr_max": row_ipr.max().item(),
            "n_rows": int(row_ipr.numel()),
        }
    return summary


def fc_proj_ipr_discrepancy(
    c_proj: torch.Tensor,
    c_fc: torch.Tensor,
    eps: float = EPS,
) -> Dict[str, float]:
    """Compare IPR discrepancy between c_proj and c_fc for one layer."""
    proj_row = row_wise_ipr(c_proj, eps=eps)
    fc_row = row_wise_ipr(c_fc, eps=eps)

    proj_global = global_ipr(c_proj, eps=eps)
    fc_global = global_ipr(c_fc, eps=eps)

    proj_mean = proj_row.mean().item()
    fc_mean = fc_row.mean().item()

    proj_std = proj_row.std().item()
    fc_std = fc_row.std().item()

    proj_median = proj_row.median().item()
    fc_median = fc_row.median().item()

    return {
        "proj_global_ipr": proj_global,
        "fc_global_ipr": fc_global,
        "global_ipr_gap": proj_global - fc_global,
        "global_ipr_ratio_proj_over_fc": proj_global / (fc_global + eps),
        "proj_row_ipr_mean": proj_mean,
        "fc_row_ipr_mean": fc_mean,
        "row_ipr_mean_gap": proj_mean - fc_mean,
        "proj_row_ipr_std": proj_std,
        "fc_row_ipr_std": fc_std,
        "row_ipr_std_gap": proj_std - fc_std,
        "proj_row_ipr_median": proj_median,
        "fc_row_ipr_median": fc_median,
        "row_ipr_median_gap": proj_median - fc_median,
        "proj_n_rows": int(proj_row.numel()),
        "fc_n_rows": int(fc_row.numel()),
    }


def layer_fc_proj_ipr_discrepancy(
    c_proj_weights: Dict[int, torch.Tensor],
    c_fc_weights: Dict[int, torch.Tensor],
) -> Dict[int, Dict[str, float]]:
    """Compute fc-vs-proj IPR discrepancy per common layer index."""
    summary = {}
    common_layers = sorted(set(c_proj_weights.keys()) & set(c_fc_weights.keys()))

    for idx in common_layers:
        summary[idx] = fc_proj_ipr_discrepancy(c_proj_weights[idx], c_fc_weights[idx])

    return summary
