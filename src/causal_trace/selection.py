"""Layer-window measurements and one predeclared held-out confirmation."""

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


def bootstrap_mean_ci(values: np.ndarray, *, samples: int, confidence_level: float, seed: int) -> tuple[float, float]:
    """Resample facts, never individual corruption draws."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("Expected a nonempty vector of finite fact effects")
    if len(array) == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(samples), len(array)))
    means = array[indices].mean(axis=1)
    tail = (1.0 - float(confidence_level)) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def summarize_windows(
    fact_results: list[dict[str, Any]],
    windows: list[Window],
    *,
    window_size: int,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    """Each result contains effects for exactly the windows passed here."""
    if not fact_results or not windows:
        return pd.DataFrame()
    matrix = np.asarray([row["window_mean_ie"] for row in fact_results], dtype=np.float64)
    if matrix.shape != (len(fact_results), len(windows)) or not np.isfinite(matrix).all():
        raise ValueError("Fact effects do not match the requested windows")
    rows: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        values = matrix[:, index]
        ci_lower, ci_upper = bootstrap_mean_ci(
            values,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + index,
        )
        rows.append(
            {
                "window_center": int(window.center),
                "window_start": int(window.start),
                "window_end": int(window.end),
                "window_layers": ",".join(str(layer) for layer in window.layers),
                "window_size_actual": int(window.size),
                "window_is_full_width": bool(window.size == int(window_size)),
                "num_facts": len(values),
                "mean_ie": float(values.mean()),
                "median_ie": float(np.median(values)),
                "std_ie": float(values.std()),
                "sem_ie": float(values.std() / math.sqrt(len(values))),
                "mean_ie_ci_lower": ci_lower,
                "mean_ie_ci_upper": ci_upper,
            }
        )
    return pd.DataFrame(rows)


def discovery_window(discovery: pd.DataFrame) -> dict[str, Any]:
    """Freeze one full-width intervention without looking at confirmation facts."""
    eligible = discovery[discovery["window_is_full_width"]] if not discovery.empty else discovery
    if eligible.empty:
        return {"discovery_trace_center": None, "failure_reason": "no_full_width_windows"}
    winner = eligible.sort_values(["mean_ie", "window_center"], ascending=[False, True]).iloc[0]
    return {
        "discovery_trace_center": int(winner.window_center),
        "trace_window_start": int(winner.window_start),
        "trace_window_end": int(winner.window_end),
        "trace_window_layers": [int(item) for item in str(winner.window_layers).split(",")],
        "discovery_mean_ie": float(winner.mean_ie),
        "num_discovery_facts": int(winner.num_facts),
    }


def select_window(
    discovery: pd.DataFrame,
    confirmation: pd.DataFrame,
    *,
    minimum_confirmation_facts: int,
) -> dict[str, Any]:
    """Confirm the exact discovery intervention without held-out reselection."""
    base = {
        "selection_method": "discovery_argmax_then_held_out_confirmation",
        "eligible_window_rule": "full_width_only",
        "selected_trace_center": None,
        "confirmation_passed": False,
    }
    chosen = discovery_window(discovery)
    center = chosen["discovery_trace_center"]
    if center is None:
        return {**base, **chosen}
    matching = confirmation[confirmation["window_center"] == center] if not confirmation.empty else confirmation
    if matching.empty:
        return {**base, **chosen, "failure_reason": "insufficient_confirmation_facts"}
    if len(matching) != 1:
        raise ValueError(f"Confirmation contains duplicate center {center}")
    row = matching.iloc[0]
    enough = int(row.num_facts) >= int(minimum_confirmation_facts)
    lower = float(row.mean_ie_ci_lower)
    passed = bool(enough and math.isfinite(lower) and lower > 0)
    return {
        **base,
        **chosen,
        "selected_trace_center": center if passed else None,
        "confirmation_mean_ie": float(row.mean_ie),
        "confirmation_ci_lower": lower,
        "confirmation_ci_upper": float(row.mean_ie_ci_upper),
        "num_confirmation_facts": int(row.num_facts),
        "confirmation_passed": passed,
        "failure_reason": None
        if passed
        else ("insufficient_confirmation_facts" if not enough else "confirmation_ci_not_positive"),
    }


__all__ = ["Window", "bootstrap_mean_ci", "build_window", "discovery_window", "select_window", "summarize_windows"]
