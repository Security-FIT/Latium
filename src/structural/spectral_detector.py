from typing import Dict, Optional, Tuple

import numpy as np
import torch

EPS = 1e-10

def _svdvals_all(weights: Dict[int, torch.Tensor]) -> Tuple[list[int], np.ndarray]:
    layer_indices = sorted(weights.keys())

    if not layer_indices:
        return [], np.empty((0, 0), dtype=np.float32)

    sv_list = []
    min_rank = None
    for layer_idx in layer_indices:
        matrix = weights[layer_idx].detach().float()
        sv = torch.linalg.svdvals(matrix).detach().cpu().numpy()
        sv_list.append(sv)
        rank = sv.shape[0]
        min_rank = rank if min_rank is None else min(min_rank, rank)

    sv_matrix = np.stack([sv[:min_rank] for sv in sv_list], axis=0)

    return layer_indices, sv_matrix


def _second_deriv_energy(matrix: np.ndarray) -> np.ndarray:
    """Compute per-layer curvature energy across layer index."""
    n_layers = matrix.shape[0]
    if n_layers == 0:
        return np.empty(0, dtype=np.float64)

    energy = np.zeros(n_layers, dtype=np.float64)

    if n_layers > 2:
        second_diff = matrix[:-2] - (2.0 * matrix[1:-1]) + matrix[2:]
        energy[1:-1] = (second_diff ** 2).sum(axis=1)

    if n_layers > 1:
        energy[0] = energy[1]
        energy[-1] = energy[-2]

    return energy


def _sv_smooth_z_scores(sv_matrix: np.ndarray, top_k: int) -> np.ndarray:
    if sv_matrix.size == 0:
        return np.empty(0, dtype=np.float64)

    k = min(top_k, sv_matrix.shape[1])
    x = sv_matrix[:, :k]
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    z = (x - mean) / (std + EPS)
    return _second_deriv_energy(z)


def _sv_ratio_smooth_scores(
    sv_proj: np.ndarray,
    sv_fc: np.ndarray,
    top_k: int,
) -> np.ndarray:
    if sv_proj.size == 0 or sv_fc.size == 0:
        return np.empty(0, dtype=np.float64)

    k = min(top_k, sv_proj.shape[1], sv_fc.shape[1])
    ratio = sv_proj[:, :k] / (sv_fc[:, :k] + EPS)
    return _second_deriv_energy(ratio)


class SpectralDetector:
    def __init__(
        self,
        top_k: int = 50,
        boundary: int = 2,
        trim_first_layers: int = 0,
        trim_last_layers: int = 0,
    ):
        self.top_k = top_k
        self.boundary = boundary
        self.trim_first_layers = max(0, int(trim_first_layers))
        self.trim_last_layers = max(0, int(trim_last_layers))

    def detect(
        self,
        weights: Dict[int, torch.Tensor],
        fc_weights: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict:
        all_layer_indices, sv_proj_full = _svdvals_all(weights)
        if not all_layer_indices:
            return {
                "anomalous_layer": None,
                "detection_score": 0.0,
                "sv_z_scores": {},
                "sv_ratio_scores": {},
                "has_fc_weights": False,
                "config": {
                    "top_k": int(self.top_k),
                    "boundary": int(self.boundary),
                    "trim_first_layers": int(self.trim_first_layers),
                    "trim_last_layers": int(self.trim_last_layers),
                },
                "excluded_layers": [],
                "evaluated_layers": [],
            }

        n_all_layers = len(all_layer_indices)
        trim_start = min(self.trim_first_layers, n_all_layers)
        trim_end = n_all_layers - min(self.trim_last_layers, n_all_layers - trim_start)

        if trim_end <= trim_start:
            return {
                "anomalous_layer": None,
                "detection_score": 0.0,
                "sv_z_scores": {layer: 0.0 for layer in all_layer_indices},
                "sv_ratio_scores": {layer: 0.0 for layer in all_layer_indices},
                "has_fc_weights": False,
                "config": {
                    "top_k": int(self.top_k),
                    "boundary": int(self.boundary),
                    "trim_first_layers": int(self.trim_first_layers),
                    "trim_last_layers": int(self.trim_last_layers),
                },
                "excluded_layers": list(all_layer_indices),
                "evaluated_layers": [],
            }

        layer_indices = all_layer_indices[trim_start:trim_end]
        sv_proj = sv_proj_full[trim_start:trim_end]
        excluded_layers = all_layer_indices[:trim_start] + all_layer_indices[trim_end:]

        sv_z_raw = _sv_smooth_z_scores(sv_proj, self.top_k)
        has_fc_weights = False
        sv_ratio_raw = np.zeros_like(sv_z_raw)

        if fc_weights is not None:
            fc_layer_indices, sv_fc_full = _svdvals_all(fc_weights)
            if fc_layer_indices == all_layer_indices:
                has_fc_weights = True
                sv_fc = sv_fc_full[trim_start:trim_end]
                sv_ratio_raw = _sv_ratio_smooth_scores(sv_proj, sv_fc, self.top_k)

        n_layers = len(layer_indices)
        start = self.boundary
        end = n_layers - self.boundary

        if end > start:
            candidate_positions = np.arange(start, end)
        else:
            candidate_positions = np.arange(0, n_layers)

        candidate_z = sv_z_raw[candidate_positions]
        rank_z = np.empty(len(candidate_positions), dtype=np.float64)
        rank_z[np.argsort(-candidate_z)] = np.arange(len(candidate_positions), dtype=np.float64)

        if has_fc_weights:
            candidate_ratio = sv_ratio_raw[candidate_positions]
            rank_ratio = np.empty(len(candidate_positions), dtype=np.float64)
            rank_ratio[np.argsort(-candidate_ratio)] = np.arange(len(candidate_positions), dtype=np.float64)
            combined_rank = (rank_z + rank_ratio) / 2.0
        else:
            combined_rank = rank_z

        winner_pos = int(np.argmin(combined_rank))
        anomalous_idx = int(candidate_positions[winner_pos])
        anomalous_layer = int(layer_indices[anomalous_idx])

        n_candidates = len(candidate_positions)

        sv_z_scores = {layer: 0.0 for layer in all_layer_indices}
        sv_ratio_scores = {layer: 0.0 for layer in all_layer_indices}
        for i, layer in enumerate(layer_indices):
            sv_z_scores[layer] = float(sv_z_raw[i])
            sv_ratio_scores[layer] = float(sv_ratio_raw[i])

        return {
            "anomalous_layer": anomalous_layer,
            "detection_score": float(n_candidates - combined_rank[winner_pos]),
            "sv_z_scores": sv_z_scores,
            "sv_ratio_scores": sv_ratio_scores,
            "has_fc_weights": has_fc_weights,
            "config": {
                "top_k": int(self.top_k),
                "boundary": int(self.boundary),
                "trim_first_layers": int(self.trim_first_layers),
                "trim_last_layers": int(self.trim_last_layers),
            },
            "excluded_layers": excluded_layers,
            "evaluated_layers": layer_indices,
        }


