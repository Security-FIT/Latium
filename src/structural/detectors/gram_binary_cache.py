"""Bounded memoization of pure per-weight research diagnostics.

Keys hash the actual input bytes, shape, dtype, device and compute options.
No edit location, configuration layer or experiment baseline is accepted.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
import hashlib
from typing import Any

import torch

from src.structural.detectors.gram_research import bi_cross_validation, dyson_equalize

_CACHE: OrderedDict[tuple, tuple[dict, dict]] = OrderedDict()
_CACHE_LIMIT = 128


def research_diagnostics(
    source: torch.Tensor,
    get_weight: Callable[[], torch.Tensor],
    *,
    device: str | torch.device,
    dyson: bool,
    bcv: bool,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    if not (dyson or bcv):
        return {}, {}
    raw = source.detach().contiguous().cpu()
    digest = hashlib.sha256(memoryview(raw.view(torch.uint8).numpy())).digest()
    key = (tuple(raw.shape), str(raw.dtype), str(device), dyson, bcv, digest)
    cached = _CACHE.get(key)
    if cached is not None:
        _CACHE.move_to_end(key)
        planes, records = cached
        return (
            {name: tuple(vector.to(device) for vector in plane) for name, plane in planes.items()},
            {**deepcopy(records), "research_cache": {"hit": True, "identity": "sha256-input-bytes"}},
        )
    weight = get_weight()
    planes, records = {}, {}
    if dyson:
        try:
            equalized = dyson_equalize(weight)
            planes["dyson-plane"] = equalized.plane
            records["dyson"] = equalized.record
            del equalized
        except (ValueError, RuntimeError) as exc:
            records["dyson"] = {"status": "unavailable", "reason": str(exc)}
    if bcv:
        try:
            records["bcv"] = bi_cross_validation(weight)
        except (ValueError, RuntimeError) as exc:
            records["bcv"] = {"status": "unavailable", "reason": str(exc)}
    if all(record["status"] == "complete" for record in records.values()):
        _CACHE[key] = (
            {name: tuple(vector.detach().cpu().clone() for vector in plane) for name, plane in planes.items()},
            deepcopy(records),
        )
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
    records["research_cache"] = {"hit": False, "identity": "sha256-input-bytes"}
    return planes, records


def clear_research_cache() -> None:
    _CACHE.clear()
