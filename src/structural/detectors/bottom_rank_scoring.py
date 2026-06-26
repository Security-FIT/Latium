"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def score_token_sweeps(
    token_id_sweeps: Mapping[Any, Sequence[int]],
    *,
    boundary: int | None = None,
    trim_first: int = 0,
    trim_last: int = 0,
) -> dict[str, Any]:
    sweeps = {int(layer): [int(token) for token in tokens] for layer, tokens in token_id_sweeps.items() if tokens}
    layers = sorted(sweeps)
    unique_counts: dict[int, int] = {}
    switch_counts: dict[int, int] = {}
    switch_rates: dict[int, float] = {}
    layer_scores: dict[int, float] = {}
    for layer in layers:
        tokens = sweeps[layer]
        switches = sum(tokens[index] != tokens[index - 1] for index in range(1, len(tokens)))
        unique = len(set(tokens))
        unique_counts[layer] = int(unique)
        switch_counts[layer] = int(switches)
        switch_rates[layer] = float(switches) / float(max(1, len(tokens) - 1))
        layer_scores[layer] = float(unique) + 0.25 * float(switches)

    if not layer_scores:
        return {
            "anomalous_layer": None,
            "detection_score": 0.0,
            "unique_prediction_counts": {},
            "switch_counts": {},
            "switch_rates": {},
            "layer_scores": {},
        }

    scored_layers = sorted(layer_scores)
    if boundary is not None:
        width = min(max(0, int(boundary)), len(scored_layers) // 2)
        candidates = scored_layers[width : len(scored_layers) - width] or scored_layers
    else:
        start = max(0, int(trim_first))
        end = len(scored_layers) - max(0, int(trim_last))
        candidates = scored_layers[start:end] if end > start else scored_layers
    best_layer = max(candidates, key=layer_scores.get)
    return {
        "anomalous_layer": int(best_layer),
        "detection_score": float(layer_scores[best_layer]),
        "unique_prediction_counts": unique_counts,
        "switch_counts": switch_counts,
        "switch_rates": switch_rates,
        "layer_scores": layer_scores,
    }


__all__ = ["score_token_sweeps"]
