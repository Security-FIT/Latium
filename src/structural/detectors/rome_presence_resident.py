"""Direct weight-dictionary adapter for the minimal ROME detector."""

from __future__ import annotations

import torch

from src.structural.capture.producers import CaptureContext, capture_weighted_spectrum
from src.structural.detectors.rome_presence import detect_rome_compatible_edit
from src.structural.detectors.weighted_spectrum import detect_from_profiles


class RomeDetector:
    """Localize an edit and test its clean-reference low-rank compatibility."""

    def detect(
        self,
        suspect_proj: dict[int, torch.Tensor],
        clean_proj: dict[int, torch.Tensor],
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
        presence = detect_rome_compatible_edit(suspect_proj, clean_proj)
        return detect_from_profiles(
            capture["profiles"],
            layers=[int(layer) for layer in capture["layers"]],
            trim_fraction=float(capture["trim_fraction"]),
            clean_reference_presence=presence,
        )


__all__ = ["RomeDetector"]
