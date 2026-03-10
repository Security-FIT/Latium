from typing import Dict

import torch


EPS = 1e-12


def _to_float_tensor(W: torch.Tensor) -> torch.Tensor:
    return W.float() if not torch.is_floating_point(W) else W


def row_wise_ipr(W: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Compute IPR for each row of W."""
    Wf = _to_float_tensor(W)
    numerator = Wf.pow(4).sum(dim=1)
    denominator = Wf.pow(2).sum(dim=1).pow(2) + eps
    return numerator / denominator


def col_wise_ipr(W: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Compute IPR for each column of W."""
    return row_wise_ipr(W.T, eps=eps)


def global_ipr(W: torch.Tensor, eps: float = EPS) -> float:
    """Compute global IPR over all entries of W."""
    flat = _to_float_tensor(W).reshape(-1)
    numerator = flat.pow(4).sum()
    denominator = flat.pow(2).sum().pow(2) + eps
    return (numerator / denominator).item()


def layer_ipr_summary(weights: Dict[int, torch.Tensor]) -> Dict[int, Dict[str, float]]:
    """Simple per-layer IPR summary."""
    summary: Dict[int, Dict[str, float]] = {}
    for idx, W in weights.items():
        row_ipr = row_wise_ipr(W)
        summary[idx] = {
            "global_ipr": global_ipr(W),
            "row_ipr_mean": row_ipr.mean().item(),
            "row_ipr_std": row_ipr.std().item(),
            "row_ipr_median": row_ipr.median().item(),
            "row_ipr_min": row_ipr.min().item(),
            "row_ipr_max": row_ipr.max().item(),
            "n_rows": int(row_ipr.numel()),
        }
    return summary


def layer_ipr_extended_summary(weights: Dict[int, torch.Tensor]) -> Dict[int, Dict[str, float]]:
    """Compatibility wrapper kept intentionally simple."""
    return layer_ipr_summary(weights)


def fc_proj_ipr_discrepancy(
    c_proj: torch.Tensor,
    c_fc: torch.Tensor,
    eps: float = EPS,
) -> Dict[str, float]:
    """Simple c_proj vs c_fc IPR comparison for one layer."""
    proj_row = row_wise_ipr(c_proj, eps=eps)
    fc_row = row_wise_ipr(c_fc, eps=eps)

    proj_global = global_ipr(c_proj, eps=eps)
    fc_global = global_ipr(c_fc, eps=eps)

    return {
        "proj_global_ipr": proj_global,
        "fc_global_ipr": fc_global,
        "global_ipr_gap": proj_global - fc_global,
        "global_ipr_ratio_proj_over_fc": proj_global / (fc_global + eps),
        "row_ipr_mean_gap": proj_row.mean().item() - fc_row.mean().item(),
        "row_ipr_std_gap": proj_row.std().item() - fc_row.std().item(),
        "row_ipr_median_gap": proj_row.median().item() - fc_row.median().item(),
    }


def fc_proj_ipr_extended_discrepancy(
    c_proj: torch.Tensor,
    c_fc: torch.Tensor,
    eps: float = EPS,
) -> Dict[str, float]:
    """Compatibility wrapper kept intentionally simple."""
    return fc_proj_ipr_discrepancy(c_proj, c_fc, eps=eps)


def layer_fc_proj_ipr_discrepancy(
    c_proj_weights: Dict[int, torch.Tensor],
    c_fc_weights: Dict[int, torch.Tensor],
) -> Dict[int, Dict[str, float]]:
    """Simple per-layer c_proj vs c_fc discrepancy."""
    common_layers = sorted(set(c_proj_weights) & set(c_fc_weights))
    return {
        idx: fc_proj_ipr_discrepancy(c_proj_weights[idx], c_fc_weights[idx])
        for idx in common_layers
    }


def layer_fc_proj_ipr_extended_discrepancy(
    c_proj_weights: Dict[int, torch.Tensor],
    c_fc_weights: Dict[int, torch.Tensor],
) -> Dict[int, Dict[str, float]]:
    """Compatibility wrapper kept intentionally simple."""
    return layer_fc_proj_ipr_discrepancy(c_proj_weights, c_fc_weights)


class IPRDetector:
    """Simple blind detector: pick layer with largest |global_ipr_gap|."""

    def __init__(self, trim_first: int = 2, trim_last: int = 2, z_threshold: float = 2.5):
        self.trim_first = max(0, int(trim_first))
        self.trim_last = max(0, int(trim_last))
        self.z_threshold = z_threshold

    def detect(
        self,
        c_proj_weights: Dict[int, torch.Tensor],
        c_fc_weights: Dict[int, torch.Tensor],
    ) -> Dict[str, object]:
        discrepancy = layer_fc_proj_ipr_discrepancy(c_proj_weights, c_fc_weights)
        layers = sorted(discrepancy)

        if not layers:
            return {
                "anomalous_layer": None,
                "anomaly_score": 0.0,
                "per_layer": {},
                "ranking": [],
                "feature_names": ["global_ipr_gap"],
                "fc_vs_proj_extended": discrepancy,
            }

        start = self.trim_first
        end = len(layers) - self.trim_last if self.trim_last > 0 else len(layers)
        candidate_layers = layers[start:end]
        if not candidate_layers:
            candidate_layers = layers

        per_layer = {
            idx: {
                "combined_score": abs(discrepancy[idx]["global_ipr_gap"]),
                "max_abs_z": abs(discrepancy[idx]["global_ipr_gap"]),
                "max_abs_smooth": 0.0,
                "z_scores": {"global_ipr_gap": discrepancy[idx]["global_ipr_gap"]},
            }
            for idx in layers
        }

        best_layer = max(candidate_layers, key=lambda idx: per_layer[idx]["combined_score"])
        ranking = sorted(
            [(idx, per_layer[idx]["combined_score"]) for idx in layers],
            key=lambda item: item[1],
            reverse=True,
        )

        return {
            "anomalous_layer": best_layer,
            "anomaly_score": per_layer[best_layer]["combined_score"],
            "per_layer": per_layer,
            "ranking": ranking,
            "feature_names": ["global_ipr_gap"],
            "fc_vs_proj_extended": discrepancy,
        }
