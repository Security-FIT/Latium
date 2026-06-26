"""
GPT-family norm-CV detector implementation.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np

from src.common.arrays import curvature, local_zscore


def _nc_peaks(
    values: np.ndarray,
    layers: list[str],
    trim_first: int,
    trim_last: int,
) -> tuple[Optional[list[int]], dict[str, int]]:
    start = max(0, int(trim_first))
    end = len(layers) - max(0, int(trim_last))
    if end <= start:
        return None, {}
    evaluated = [int(layer) for layer in layers[start:end]]
    peaks = {}
    for name, transformed in (
        ("raw", values[start:end]),
        ("lz5", np.abs(local_zscore(values, 5))[start:end]),
        ("curv", curvature(values)[start:end]),
    ):
        peaks[name] = evaluated[int(np.argmax(transformed))]
    return evaluated, peaks


def detect(
    test: dict,
    *,
    trim_first: int = 5,
    trim_last: int = 5,
) -> Tuple[Optional[int], str, Dict]:
    features = test["blind_detection"]["layer_features"]
    layers = sorted(features, key=int)
    norm_cv = np.array([features[layer]["norm_cv"] for layer in layers])

    evaluated, peaks = _nc_peaks(norm_cv, layers, trim_first, trim_last)
    if evaluated is not None:
        votes = Counter(peaks.values())
        winner, count = votes.most_common(1)[0]
        if count >= 2:
            return (
                winner,
                f"nc3_t{trim_first}-{trim_last}",
                {
                    "peaks": peaks,
                    "votes": dict(votes),
                },
            )

    all_votes: Counter[int] = Counter()
    for fallback_trim in (4, 5, 6):
        _, fallback_peaks = _nc_peaks(
            norm_cv,
            layers,
            fallback_trim,
            fallback_trim,
        )
        all_votes.update(fallback_peaks.values())
    if all_votes:
        winner = all_votes.most_common(1)[0][0]
        return winner, "nc3_mt4-6", {"votes": dict(all_votes)}
    return None, "nc3_fail", {}
