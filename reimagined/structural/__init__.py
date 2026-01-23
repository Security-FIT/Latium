"""Structural analysis module for detecting model weight modifications."""

from .detector import WeightMSDDetector
from .blind_detector import BlindMSDDetector
from .groupers import (
    MagnitudeGrouper,
    SparsityGrouper,
    SpectralGrouper,
    RandomGrouper,
)
from .metrics import (
    l2_discrepancy,
    relative_discrepancy,
    directional_coherence,
    pcs_change,
)

__all__ = [
    "WeightMSDDetector",
    "BlindMSDDetector",
    "MagnitudeGrouper",
    "SparsityGrouper",
    "SpectralGrouper",
    "RandomGrouper",
    "l2_discrepancy",
    "relative_discrepancy",
    "directional_coherence",
    "pcs_change",
]
