"""
Layer trim policy for structural analysis variants.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Optional


def auto_trim_from_layers(num_layers: int) -> int:
    if num_layers <= 0:
        return 2
    trim = int(round(num_layers * 0.05))
    return max(1, min(4, trim))


def resolve_trim(
    num_layers: int,
    trim_first: Optional[int],
    trim_last: Optional[int],
) -> tuple[int, int]:
    auto = auto_trim_from_layers(num_layers)
    first = auto if trim_first is None else max(0, int(trim_first))
    last = auto if trim_last is None else max(0, int(trim_last))
    if first + last >= num_layers:
        max_side = max(0, (num_layers - 1) // 2)
        first = min(first, max_side)
        last = min(last, max_side)
    return first, last


__all__ = ["auto_trim_from_layers", "resolve_trim"]
