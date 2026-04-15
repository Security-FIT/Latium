"""Structural analysis module for detecting model weight modifications."""

from .detector import WeightMSDDetector
from .blind_detector import BlindMSDDetector
from .spectral_detector import SpectralDetector
from .edit_presence_detector import RomeEditPresenceDetector
from .rank1_blind import BlindRank1Detector
from .bottom_rank_svd import BottomRankSVDDetector
from .attention_metrics import AttentionContrastDetector
from .symmetry_metrics import MirrorSymmetryDetector
from .local_scores import (
    ensure_odd_window,
    normalize_windows,
    rank01,
    rolling_z_abs,
    rolling_mad_abs,
    local_score_bank,
)
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
    "RomeEditPresenceDetector",
    "BlindRank1Detector",
    "BottomRankSVDDetector",
    "AttentionContrastDetector",
    "MirrorSymmetryDetector",
    "ensure_odd_window",
    "normalize_windows",
    "rank01",
    "rolling_z_abs",
    "rolling_mad_abs",
    "local_score_bank",
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
