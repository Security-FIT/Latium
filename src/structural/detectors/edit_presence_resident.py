"""
Model-resident adapter for edit-presence detection.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from src.structural.detectors.edit_presence import detect_edit_presence_from_profiles, edit_presence_config
from src.structural.detectors.profiles import matrix_profile


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
        return edit_presence_config(
            self.detection_threshold,
            self.min_peak_robust_z,
            self.min_margin,
            self.local_windows,
        )

    def detect(
        self,
        modified_proj: Dict[int, torch.Tensor],
        modified_fc: Optional[Dict[int, torch.Tensor]] = None,
        modified_spectral: Optional[Dict] = None,
    ) -> Dict:
        layers = sorted(modified_proj.keys())
        proj_metrics = {int(layer): matrix_profile(modified_proj[layer]) for layer in layers}
        fc_metrics = (
            {int(layer): matrix_profile(modified_fc[layer]) for layer in layers}
            if modified_fc is not None and all(layer in modified_fc for layer in layers)
            else None
        )
        return detect_edit_presence_from_profiles(
            proj_metrics,
            fc_metrics=fc_metrics,
            modified_spectral=modified_spectral,
            detection_threshold=self.detection_threshold,
            min_peak_robust_z=self.min_peak_robust_z,
            min_margin=self.min_margin,
            local_windows=self.local_windows,
        )


__all__ = ["RomeEditPresenceDetector", "detect_edit_presence_from_profiles"]
