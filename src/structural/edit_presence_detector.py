from typing import Dict, Optional

import numpy as np
import torch

from src.utils import gpu_svd

EPS = 1e-10


def _robust_z(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.empty(0, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + EPS
    return 0.6745 * (x - med) / mad


class RomeEditPresenceDetector:
    """
    Binary detector: was a ROME-style edit applied at all?

    This detector is intentionally layer-agnostic for the final decision.
    It still inspects the strongest changed layer internally as evidence.
    """

    def __init__(
        self,
        detection_threshold: float = 0.55,
        min_delta_ratio: float = 1e-4,
        min_peak_rank_one: float = 0.70,
    ):
        self.detection_threshold = float(detection_threshold)
        self.min_delta_ratio = float(min_delta_ratio)
        self.min_peak_rank_one = float(min_peak_rank_one)

    def detect(
        self,
        original_proj: Dict[int, torch.Tensor],
        modified_proj: Dict[int, torch.Tensor],
        baseline_spectral: Optional[Dict] = None,
        modified_spectral: Optional[Dict] = None,
    ) -> Dict:
        common_layers = sorted(set(original_proj.keys()) & set(modified_proj.keys()))
        if not common_layers:
            return {
                "model_detected": False,
                "is_edited": False,
                "confidence": 0.0,
                "score": 0.0,
                "reason": "no_common_layers",
            }

        delta_ratios = []
        for l in common_layers:
            d = (modified_proj[l].float() - original_proj[l].float()).norm().item()
            b = original_proj[l].float().norm().item()
            delta_ratios.append(d / (b + EPS))

        delta_ratios_arr = np.array(delta_ratios, dtype=np.float64)
        delta_z_arr = _robust_z(delta_ratios_arr)
        peak_idx = int(np.argmax(delta_ratios_arr))
        peak_layer = int(common_layers[peak_idx])
        peak_delta_ratio = float(delta_ratios_arr[peak_idx])
        peak_delta_z = float(delta_z_arr[peak_idx]) if delta_z_arr.size else 0.0

        peak_delta = modified_proj[peak_layer].float() - original_proj[peak_layer].float()
        if peak_delta.norm().item() < 1e-12:
            rank_one_score = 0.0
            effective_rank = 0.0
        else:
            _, s, _ = gpu_svd(peak_delta, full_matrices=False)
            s2 = s**2
            rank_one_score = float((s2[0] / (s2.sum() + EPS)).item())
            s_norm = s / (s.sum() + EPS)
            entropy = -(s_norm * torch.log(s_norm + EPS)).sum()
            effective_rank = float(torch.exp(entropy).item())

        spectral_shift = 0.0
        if baseline_spectral is not None and modified_spectral is not None:
            b_map = baseline_spectral.get("rome_hybrid_scores", {})
            m_map = modified_spectral.get("rome_hybrid_scores", {})
            shifts = []
            for l in common_layers:
                bv = float(b_map.get(l, b_map.get(str(l), 0.0)))
                mv = float(m_map.get(l, m_map.get(str(l), 0.0)))
                shifts.append(abs(mv - bv))
            if shifts:
                spectral_shift = float(np.max(shifts))

        delta_component = min(max(peak_delta_z, 0.0) / 6.0, 1.0)
        rank_component = min(max(rank_one_score, 0.0), 1.0)
        spectral_component = min(max(spectral_shift, 0.0), 1.0)
        score = 0.50 * delta_component + 0.35 * rank_component + 0.15 * spectral_component

        is_edited = (
            peak_delta_ratio >= self.min_delta_ratio
            and rank_one_score >= self.min_peak_rank_one
            and score >= self.detection_threshold
        )
        confidence = float(min(max(score, 0.0), 1.0))

        return {
            "model_detected": bool(is_edited),
            "is_edited": bool(is_edited),
            "confidence": confidence,
            "score": float(score),
            "peak_layer": peak_layer,
            "peak_delta_ratio": peak_delta_ratio,
            "peak_delta_robust_z": peak_delta_z,
            "peak_rank_one_score": float(rank_one_score),
            "peak_effective_rank": float(effective_rank),
            "spectral_shift_max": float(spectral_shift),
            "thresholds": {
                "detection_threshold": self.detection_threshold,
                "min_delta_ratio": self.min_delta_ratio,
                "min_peak_rank_one": self.min_peak_rank_one,
            },
        }

