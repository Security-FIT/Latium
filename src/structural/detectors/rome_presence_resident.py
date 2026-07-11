"""Direct weight-dictionary adapters for the ROME-presence decisions."""

from __future__ import annotations

from typing import Optional

import torch

from src.structural.capture.producers import (
    CaptureContext,
    capture_rome_update,
    capture_weighted_spectrum,
)
from src.structural.detectors.rome_presence import (
    BlindStrategy,
    detect_rome_presence_blind,
    detect_rome_presence_delta,
)
from src.structural.detectors.weighted_spectrum import (
    FOOTPRINT_PROFILE_FIELDS,
    LOCALIZER_PROFILE_FIELDS,
)


class BlindRomePresenceDetector:
    """Decide ROME-like presence from one suspect checkpoint."""

    def __init__(
        self,
        *,
        strategy: BlindStrategy = "footprint",
        trim_first: int = 5,
        trim_last: int = 5,
    ) -> None:
        self.strategy = strategy
        self.trim_first = int(trim_first)
        self.trim_last = int(trim_last)

    def detect(self, suspect_proj: dict[int, torch.Tensor]) -> dict:
        capture = capture_weighted_spectrum(
            CaptureContext(
                proj_weights=suspect_proj,
                fc_weights=None,
                attention_weights={},
                probe_vector=None,
                token_predictor=None,
                changed_weights={},
                options={
                    "weighted_spectrum_fields": (
                        LOCALIZER_PROFILE_FIELDS
                        if self.strategy == "peak"
                        else FOOTPRINT_PROFILE_FIELDS
                    )
                },
            )
        )
        return detect_rome_presence_blind(
            capture["profiles"],
            trim_first=self.trim_first,
            trim_last=self.trim_last,
            strategy=self.strategy,
        )


class DeltaRomePresenceDetector:
    """Decide ROME-like presence by comparison with a clean checkpoint."""

    def detect(
        self,
        suspect_proj: dict[int, torch.Tensor],
        clean_proj: dict[int, torch.Tensor],
        *,
        suspect_fc: Optional[dict[int, torch.Tensor]] = None,
        clean_fc: Optional[dict[int, torch.Tensor]] = None,
    ) -> dict:
        capture = capture_rome_update(
            CaptureContext(
                proj_weights=suspect_proj,
                fc_weights=suspect_fc,
                attention_weights={},
                probe_vector=None,
                token_predictor=None,
                changed_weights={"proj": None, "fc": None},
                options={},
                baseline_proj_weights=clean_proj,
                baseline_fc_weights=clean_fc,
            )
        )
        return detect_rome_presence_delta(capture["families"])


__all__ = ["BlindRomePresenceDetector", "DeltaRomePresenceDetector"]
