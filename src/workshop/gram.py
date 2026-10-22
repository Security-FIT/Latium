"""Use the main detector core, without a notebook-specific reimplementation."""
from src.structural.detectors.rome_layer_localizer import (
    GramProfileCache, RomeLayerLocalizer, detect_from_profiles, profile_weights,
)

__all__ = ["GramProfileCache", "RomeLayerLocalizer", "detect_from_profiles", "profile_weights"]
