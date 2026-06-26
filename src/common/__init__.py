"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from .arrays import EPS, curvature, local_zscore, rank01, safe_mean
from .io import ensure_parent_dir, load_json, load_jsonl, to_serializable, write_json, write_jsonl
from .plotting import BASELINE_COLOR, plot_baseline_overlay, setup_mpl_style

__all__ = [
    'EPS',
    'curvature',
    'local_zscore',
    'rank01',
    'safe_mean',
    'ensure_parent_dir',
    'load_json',
    'load_jsonl',
    'to_serializable',
    'write_json',
    'write_jsonl',
    'BASELINE_COLOR',
    'plot_baseline_overlay',
    'setup_mpl_style',
]
