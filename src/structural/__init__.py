"""Structural analysis module for detecting model weight modifications."""

from .detector import WeightMSDDetector
from .blind_detector import BlindMSDDetector
from .spectral_detector import SpectralDetector
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
from .ipr import (
    row_wise_ipr,
    global_ipr,
    layer_ipr_summary,
    fc_proj_ipr_discrepancy,
    layer_fc_proj_ipr_discrepancy,
)

__all__ = [
    "WeightMSDDetector",
    "BlindMSDDetector",
    "SpectralDetector",
    "MagnitudeGrouper",
    "SparsityGrouper",
    "SpectralGrouper",
    "RandomGrouper",
    "l2_discrepancy",
    "relative_discrepancy",
    "directional_coherence",
    "pcs_change",
    "row_wise_ipr",
    "global_ipr",
    "layer_ipr_summary",
    "fc_proj_ipr_discrepancy",
    "layer_fc_proj_ipr_discrepancy",
]
