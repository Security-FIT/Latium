"""Direct one-checkpoint adapter for the minimal ROME detector."""

from __future__ import annotations

import torch

from src.structural.capture.producers import CaptureContext, capture_weighted_spectrum
from src.structural.detectors.weighted_spectrum import detect_from_profiles


class RomeDetector:
    """Compute the diagonal-relative profile from one suspect checkpoint."""

    def detect_one_checkpoint(
        self,
        suspect_proj: dict[int, torch.Tensor],
    ) -> dict:
        capture = capture_weighted_spectrum(
            CaptureContext(
                proj_weights=suspect_proj,
                fc_weights=None,
                attention_weights={},
                probe_vector=None,
                token_predictor=None,
                changed_weights={},
                options={},
            )
        )
        return detect_from_profiles(
            capture["profiles"],
            layers=[int(layer) for layer in capture["layers"]],
            trim_fraction=float(capture["trim_fraction"]),
        )

    def detect(self, suspect_proj: dict[int, torch.Tensor]) -> dict:
        """Compatibility spelling for the one-checkpoint operation."""
        return self.detect_one_checkpoint(suspect_proj)


__all__ = ["RomeDetector"]
