from typing import Dict, Optional, Sequence

import numpy as np
import torch

from src.structural.local_scores import local_score_bank, map_array_to_layers, map_bank_to_layers, rank01
from src.utils import gpu_svdvals

EPS = 1e-10


def _robust_z(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.empty(0, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + EPS
    return 0.6745 * (x - med) / mad


def _normalized_entropy(vals: np.ndarray) -> float:
    x = np.array(vals, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    if x.size <= 1:
        return 0.0
    s = x.sum()
    if s <= 0.0:
        return 1.0
    p = x / s
    ent = -np.sum(p * np.log(np.clip(p, EPS, 1.0)))
    return float(ent / np.log(x.size))


def _layer_matrix_metrics(W: torch.Tensor) -> Dict[str, float]:
    Wf = W.detach().float()
    S = gpu_svdvals(Wf).detach().cpu().numpy().astype(np.float64)
    if S.size == 0:
        return {
            "top1_energy": 0.0,
            "spectral_gap": 0.0,
            "effective_rank": 0.0,
            "frob_norm": 0.0,
            "norm_cv": 0.0,
        }

    S2 = S ** 2
    total = S2.sum() + EPS
    p = S / (S.sum() + EPS)

    row_norms = torch.norm(Wf, dim=1)
    s1 = float(S[0])
    s2 = float(S[1]) if S.size > 1 else 0.0
    frob_sq = float((Wf ** 2).sum().item())

    return {
        "top1_energy": float((s1 ** 2) / total),
        "spectral_gap": float(s1 / (s2 + EPS)) if S.size > 1 else float(s1),
        "effective_rank": float(np.exp(-(p * np.log(p + EPS)).sum())),
        "frob_norm": float(np.sqrt(frob_sq)),
        "norm_cv": float((row_norms.std() / (row_norms.mean() + EPS)).item()),
    }


class RomeEditPresenceDetector:
    """
    Blind binary detector: detect whether an edit exists using only the model under test.

    No clean/original model is required. The detector looks for one-layer-outlier
    behavior across intrinsic spectral/structural signals.
    """

    def __init__(
        self,
        detection_threshold: float = 0.58,
        min_peak_robust_z: float = 2.0,
        min_margin: float = 0.08,
        local_windows: Sequence[int] = (3, 5, 7),
    ):
        self.detection_threshold = float(detection_threshold)
        self.min_peak_robust_z = float(min_peak_robust_z)
        self.min_margin = float(min_margin)
        self.local_windows = tuple(int(w) for w in local_windows)

    @property
    def _config(self) -> Dict[str, object]:
        return {
            "detection_threshold": self.detection_threshold,
            "min_peak_robust_z": self.min_peak_robust_z,
            "min_margin": self.min_margin,
            "local_windows": list(self.local_windows),
        }

    def detect(
        self,
        modified_proj: Dict[int, torch.Tensor],
        modified_fc: Optional[Dict[int, torch.Tensor]] = None,
        modified_spectral: Optional[Dict] = None,
    ) -> Dict:
        layers = sorted(modified_proj.keys())
        if not layers:
            return {
                "model_detected": False,
                "is_edited": False,
                "confidence": 0.0,
                "score": 0.0,
                "reason": "no_layers",
                "config": self._config,
            }

        proj_metrics = {int(l): _layer_matrix_metrics(modified_proj[l]) for l in layers}

        top1 = np.array([proj_metrics[l]["top1_energy"] for l in layers], dtype=np.float64)
        gap = np.array([proj_metrics[l]["spectral_gap"] for l in layers], dtype=np.float64)
        erank = np.array([proj_metrics[l]["effective_rank"] for l in layers], dtype=np.float64)
        norm_cv = np.array([proj_metrics[l]["norm_cv"] for l in layers], dtype=np.float64)
        log_frob = np.log(np.array([proj_metrics[l]["frob_norm"] for l in layers], dtype=np.float64) + EPS)

        top1_z = np.maximum(_robust_z(top1), 0.0)
        gap_z = np.maximum(_robust_z(gap), 0.0)
        cv_z = np.maximum(_robust_z(norm_cv), 0.0)
        erank_inv_z = np.maximum(-_robust_z(erank), 0.0)
        frob_z = np.maximum(_robust_z(log_frob), 0.0)

        component_arrays = {
            "top1_energy_rank": rank01(top1_z),
            "spectral_gap_rank": rank01(gap_z),
            "norm_cv_rank": rank01(cv_z),
            "effective_rank_inverse_rank": rank01(erank_inv_z),
            "log_frob_rank": rank01(frob_z),
        }

        has_fc = modified_fc is not None and all(l in modified_fc for l in layers)
        if has_fc:
            fc_metrics = {int(l): _layer_matrix_metrics(modified_fc[l]) for l in layers}
            proj_fc_top1_gap = np.abs(
                np.array([proj_metrics[l]["top1_energy"] for l in layers], dtype=np.float64)
                - np.array([fc_metrics[l]["top1_energy"] for l in layers], dtype=np.float64)
            )
            proj_fc_erank_gap = np.abs(
                np.array([proj_metrics[l]["effective_rank"] for l in layers], dtype=np.float64)
                - np.array([fc_metrics[l]["effective_rank"] for l in layers], dtype=np.float64)
            )
            component_arrays["proj_fc_top1_abs_gap_rank"] = rank01(proj_fc_top1_gap)
            component_arrays["proj_fc_erank_abs_gap_rank"] = rank01(proj_fc_erank_gap)
        else:
            fc_metrics = {}

        if modified_spectral is not None:
            hyb_map = modified_spectral.get("rome_hybrid_scores", {})
            hyb = np.array(
                [float(hyb_map.get(l, hyb_map.get(str(l), 0.0))) for l in layers],
                dtype=np.float64,
            )
            component_arrays["spectral_hybrid_rank"] = rank01(hyb)

        raw_rank = np.mean(np.stack(list(component_arrays.values())), axis=0)
        local_bank = local_score_bank(raw_rank, windows=self.local_windows)
        local_rank = local_bank["max_local_rank"]
        combined = 0.55 * raw_rank + 0.45 * local_rank

        best_idx = int(np.argmax(combined))
        peak_layer = int(layers[best_idx])
        peak_score = float(combined[best_idx])
        sorted_scores = np.sort(combined)
        second = float(sorted_scores[-2]) if sorted_scores.size > 1 else peak_score
        margin = float(max(0.0, peak_score - second))
        entropy = _normalized_entropy(combined)

        combined_z = _robust_z(combined)
        peak_robust_z = float(combined_z[best_idx]) if combined_z.size else 0.0

        score = float(
            0.55 * peak_score
            + 0.30 * margin
            + 0.15 * (1.0 - entropy)
        )

        # Adaptive threshold from the current layer-score distribution.
        adaptive_threshold = float(np.quantile(combined, 0.85)) if combined.size else 0.0
        is_edited = bool(
            score >= self.detection_threshold
            and peak_robust_z >= self.min_peak_robust_z
            and margin >= self.min_margin
            and peak_score >= adaptive_threshold
        )

        return {
            "model_detected": is_edited,
            "is_edited": is_edited,
            "confidence": float(np.clip(score, 0.0, 1.0)),
            "score": score,
            "reason": "ok" if is_edited else "below_presence_threshold",
            "peak_layer": peak_layer,
            "peak_combined_score": peak_score,
            "peak_combined_robust_z": peak_robust_z,
            "confidence_margin": margin,
            "entropy": entropy,
            "adaptive_threshold": adaptive_threshold,
            "peak_rank_one_score": float(proj_metrics[peak_layer]["top1_energy"]),
            "peak_effective_rank": float(proj_metrics[peak_layer]["effective_rank"]),
            "has_fc_weights": bool(has_fc),
            "proj_layer_metrics": {str(k): v for k, v in proj_metrics.items()},
            "fc_layer_metrics": {str(k): v for k, v in fc_metrics.items()},
            "component_series": {
                name: map_array_to_layers(layers, vals) for name, vals in component_arrays.items()
            },
            "raw_rank_score": map_array_to_layers(layers, raw_rank),
            "local_window_scores": map_bank_to_layers(layers, local_bank),
            "combined_score": map_array_to_layers(layers, combined),
            "config": self._config,
            "thresholds": {
                "detection_threshold": self.detection_threshold,
                "min_peak_robust_z": self.min_peak_robust_z,
                "min_margin": self.min_margin,
                "adaptive_quantile": 0.85,
            },
        }

