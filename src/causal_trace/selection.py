"""Window construction, aggregation, and held-out selection for causal tracing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Window:
    center: int
    start: int
    end: int
    layers: list[int]

    @property
    def size(self) -> int:
        return len(self.layers)


def build_window(center: int, window_size: int, num_layers: int) -> Window:
    left_width = int(window_size) // 2
    right_width = int(window_size) - left_width
    start = max(0, int(center) - left_width)
    end = min(int(num_layers), int(center) + right_width)
    return Window(center=int(center), start=start, end=end, layers=list(range(start, end)))


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1 or int(samples) <= 0:
        value = float(np.mean(array))
        return value, value
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    means = array[indices].mean(axis=1)
    alpha = 1.0 - float(confidence_level)
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def summarize_windows(
    fact_results: list[dict[str, Any]],
    windows: list[Window],
    *,
    window_size: int,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    if not fact_results:
        return pd.DataFrame()

    fact_effects = np.asarray([row["window_mean_ie"] for row in fact_results], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        values = fact_effects[:, index]
        ci_lower, ci_upper = _bootstrap_mean_ci(
            values,
            samples=int(bootstrap_samples),
            confidence_level=float(confidence_level),
            seed=int(seed) + index,
        )
        rows.append(
            {
                "window_center": int(window.center),
                "window_start": int(window.start),
                "window_end": int(window.end),
                "window_layers": ",".join(str(layer) for layer in window.layers),
                "window_size_actual": int(window.size),
                "window_is_full_width": bool(window.size == int(window_size)),
                "num_facts": int(values.size),
                "mean_ie": float(np.mean(values)),
                "std_ie": float(np.std(values)),
                "sem_ie": float(np.std(values) / max(math.sqrt(values.size), 1.0)),
                "mean_ie_ci_lower": ci_lower,
                "mean_ie_ci_upper": ci_upper,
            }
        )
    return pd.DataFrame(rows)


def _parse_window_layers(value: Any) -> list[int]:
    return [int(item) for item in str(value).split(",") if item]


def select_window(
    discovery: pd.DataFrame,
    confirmation: pd.DataFrame,
    *,
    minimum_confirmation_facts: int,
) -> dict[str, Any]:
    """Choose once on discovery facts, then test that exact held-out window."""
    base = {
        "selection_method": "discovery_argmax_then_held_out_confirmation",
        "eligible_window_rule": "full_width_only",
        "selected_trace_center": None,
        "discovery_trace_center": None,
        "confirmation_passed": False,
    }
    if discovery.empty or confirmation.empty:
        return {**base, "failure_reason": "insufficient_split_facts"}

    eligible = discovery[discovery["window_is_full_width"]].copy()
    if eligible.empty:
        return {**base, "failure_reason": "no_full_width_windows"}

    discovery_row = eligible.sort_values(
        ["mean_ie", "window_center"],
        ascending=[False, True],
    ).iloc[0]
    center = int(discovery_row.window_center)
    confirmation_rows = confirmation[confirmation["window_center"] == center]
    if confirmation_rows.empty:
        raise RuntimeError(f"Confirmation summary is missing discovery center {center}")
    confirmation_row = confirmation_rows.iloc[0]
    num_confirmation = int(confirmation_row.num_facts)
    ci_lower = float(confirmation_row.mean_ie_ci_lower)
    enough_facts = num_confirmation >= int(minimum_confirmation_facts)
    passed = bool(enough_facts and math.isfinite(ci_lower) and ci_lower > 0)
    if not enough_facts:
        failure_reason = "insufficient_confirmation_facts"
    elif not passed:
        failure_reason = "confirmation_ci_not_positive"
    else:
        failure_reason = None

    return {
        **base,
        "selected_trace_center": center if passed else None,
        "discovery_trace_center": center,
        "trace_window_start": int(discovery_row.window_start),
        "trace_window_end": int(discovery_row.window_end),
        "trace_window_layers": _parse_window_layers(discovery_row.window_layers),
        "discovery_mean_ie": float(discovery_row.mean_ie),
        "confirmation_mean_ie": float(confirmation_row.mean_ie),
        "confirmation_ci_lower": ci_lower,
        "confirmation_ci_upper": float(confirmation_row.mean_ie_ci_upper),
        "num_discovery_facts": int(discovery_row.num_facts),
        "num_confirmation_facts": num_confirmation,
        "confirmation_passed": passed,
        "failure_reason": failure_reason,
    }


__all__ = ["Window", "build_window", "select_window", "summarize_windows"]
