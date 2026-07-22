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


def _bootstrap_matrix_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap facts as clusters while retaining all window coordinates."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"Expected non-empty fact-by-window matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Fact-by-window effects must be finite")
    if matrix.shape[0] == 1 or int(samples) <= 0:
        mean = matrix.mean(axis=0)
        return mean.copy(), mean.copy()
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, matrix.shape[0], size=(int(samples), matrix.shape[0]))
    means = matrix[indices].mean(axis=1)
    alpha = 1.0 - float(confidence_level)
    return (
        np.quantile(means, alpha / 2.0, axis=0),
        np.quantile(means, 1.0 - alpha / 2.0, axis=0),
    )


def _trimmed_mean(values: np.ndarray, fraction: float) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    trim = int(math.floor(array.size * float(fraction)))
    if trim <= 0 or 2 * trim >= array.size:
        return float(np.mean(array))
    return float(np.mean(array[trim:-trim]))


def _paired_difference_ci(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    difference = np.asarray(reference, dtype=float) - np.asarray(candidate, dtype=float)
    return _bootstrap_mean_ci(
        difference,
        samples=samples,
        confidence_level=confidence_level,
        seed=seed,
    )


def _group_contiguous(centers: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for center in sorted(set(int(value) for value in centers)):
        if not groups or center != groups[-1][-1] + 1:
            groups.append([center])
        else:
            groups[-1].append(center)
    return groups


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
    normalized = np.asarray(
        [row.get("window_mean_normalized_recovery", [float("nan")] * len(windows)) for row in fact_results],
        dtype=np.float64,
    )
    ci_lowers, ci_uppers = _bootstrap_matrix_mean_ci(
        fact_effects,
        samples=int(bootstrap_samples),
        confidence_level=float(confidence_level),
        seed=int(seed),
    )
    rows: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        values = fact_effects[:, index]
        normalized_values = normalized[:, index]
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
                "median_ie": float(np.median(values)),
                "trimmed_mean_ie": _trimmed_mean(values, 0.10),
                "std_ie": float(np.std(values)),
                "sem_ie": float(np.std(values) / max(math.sqrt(values.size), 1.0)),
                "mean_normalized_recovery": (
                    float(np.nanmean(normalized_values)) if np.isfinite(normalized_values).any() else float("nan")
                ),
                "mean_ie_ci_lower": float(ci_lowers[index]),
                "mean_ie_ci_upper": float(ci_uppers[index]),
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


def select_region(
    discovery: pd.DataFrame,
    confirmation: pd.DataFrame,
    discovery_facts: list[dict[str, Any]],
    confirmation_facts: list[dict[str, Any]],
    windows: list[Window],
    *,
    minimum_confirmation_facts: int,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
    trim_fraction: float,
    neighbor_support_radius: int,
    local_support_fraction: float,
    adjacent_peak_radius: int,
    noninferiority_margin_fraction: float,
    minimum_supported_centers: int,
    allow_near_supported_region: bool,
) -> dict[str, Any]:
    """Select a robust contiguous early-site region, then one reporting center.

    Discovery facts define candidate regions. Confirmation facts are used only
    to validate those predeclared regions and choose a consistency-ranked
    representative center inside the winning region.
    """
    base = {
        "selection_method": "discovery_region_then_held_out_confirmation",
        "eligible_window_rule": "full_width_only",
        "selected_trace_center": None,
        "discovery_trace_center": None,
        "confirmed_region_centers": [],
        "confirmed_region_layer_union": [],
        "confirmation_passed": False,
    }
    if discovery.empty or confirmation.empty or not discovery_facts or not confirmation_facts:
        return {**base, "failure_reason": "insufficient_split_facts"}
    if len(confirmation_facts) < int(minimum_confirmation_facts):
        return {**base, "failure_reason": "insufficient_confirmation_facts"}

    eligible = discovery[discovery["window_is_full_width"]].copy()
    if eligible.empty:
        return {**base, "failure_reason": "no_full_width_windows"}

    center_to_index = {int(window.center): index for index, window in enumerate(windows)}
    discovery_ie = np.asarray([row["window_mean_ie"] for row in discovery_facts], dtype=np.float64)
    confirmation_ie = np.asarray([row["window_mean_ie"] for row in confirmation_facts], dtype=np.float64)
    raw_peak_row = eligible.sort_values(["mean_ie", "window_center"], ascending=[False, True]).iloc[0]
    raw_peak = int(raw_peak_row.window_center)
    raw_values = discovery_ie[:, center_to_index[raw_peak]]
    margin = float(noninferiority_margin_fraction) * max(abs(float(raw_peak_row.mean_ie)), 1e-12)

    full_centers = sorted(int(value) for value in eligible["window_center"].tolist())
    row_by_center = {int(row.window_center): row for row in eligible.itertuples(index=False)}
    local_support: dict[int, float] = {}
    positive: set[int] = set()
    noninferior: set[int] = set()
    for center in full_centers:
        row = row_by_center[center]
        neighbors = [other for other in full_centers if abs(int(other) - int(center)) <= int(neighbor_support_radius)]
        local_support[center] = float(np.mean([float(row_by_center[other].trimmed_mean_ie) for other in neighbors]))
        if float(row.mean_ie_ci_lower) > 0 and float(row.median_ie) > 0 and float(row.trimmed_mean_ie) > 0:
            positive.add(center)
        _low, high = _paired_difference_ci(
            raw_values,
            discovery_ie[:, center_to_index[center]],
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + center,
        )
        if high < margin:
            noninferior.add(center)

    maximum_local_support = max(local_support.values(), default=0.0)
    support_threshold = float(local_support_fraction) * max(maximum_local_support, 0.0)
    supported = {center for center in positive if local_support[center] >= support_threshold}
    adjacent = {center for center in positive if abs(center - raw_peak) <= int(adjacent_peak_radius)}
    candidate_centers = sorted(supported | noninferior | adjacent)
    candidate_regions = _group_contiguous(candidate_centers)
    candidate_region_specs: list[dict[str, Any]] = []
    seen_regions: set[tuple[int, ...]] = set()
    near_width = max(1, int(minimum_supported_centers) - 1)
    for group in candidate_regions:
        group_key = tuple(int(center) for center in group)
        if group_key and group_key not in seen_regions:
            candidate_region_specs.append(
                {
                    "centers": list(group_key),
                    "region_source": "contiguous_candidate_group",
                    "region_source_priority": 0,
                }
            )
            seen_regions.add(group_key)
        if len(group) >= near_width and near_width < len(group):
            for start in range(0, len(group) - near_width + 1):
                subregion = tuple(int(center) for center in group[start : start + near_width])
                if subregion and subregion not in seen_regions:
                    candidate_region_specs.append(
                        {
                            "centers": list(subregion),
                            "region_source": "adjacent_candidate_subregion",
                            "region_source_priority": 1,
                        }
                    )
                    seen_regions.add(subregion)
    if not candidate_region_specs:
        return {
            **base,
            "discovery_trace_center": raw_peak,
            "failure_reason": "no_discovery_candidate_regions",
        }

    window_by_center = {int(window.center): window for window in windows}
    confirmation_rows = {
        int(row.window_center): row
        for row in confirmation[confirmation["window_is_full_width"]].itertuples(index=False)
    }
    region_rows: list[dict[str, Any]] = []
    minimum_near_width = max(1, int(minimum_supported_centers) - 1)
    confirmation_raw_peak_values = confirmation_ie[:, center_to_index[raw_peak]]
    for region_index, region_spec in enumerate(candidate_region_specs, start=1):
        centers = [int(center) for center in region_spec["centers"]]
        if any(center not in confirmation_rows for center in centers):
            continue
        width_supported = len(centers) >= int(minimum_supported_centers)
        near_supported = bool(allow_near_supported_region and len(centers) >= minimum_near_width)
        matrix_indices = [center_to_index[center] for center in centers]
        region_values = confirmation_ie[:, matrix_indices].mean(axis=1)
        ci_lower, ci_upper = _bootstrap_mean_ci(
            region_values,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 300 + region_index,
        )
        _difference_lower, difference_upper = _paired_difference_ci(
            confirmation_raw_peak_values,
            region_values,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 400 + region_index,
        )

        win_rates: dict[int, float] = {}
        for center in centers:
            comparisons = [
                float(np.mean(confirmation_ie[:, center_to_index[center]] > confirmation_ie[:, center_to_index[other]]))
                for other in centers
                if other != center
            ]
            win_rates[center] = float(np.mean(comparisons)) if comparisons else 0.5
        representative = max(
            centers,
            key=lambda center: (
                win_rates[center],
                float(confirmation_rows[center].median_ie),
                float(confirmation_rows[center].mean_normalized_recovery),
                float(confirmation_rows[center].trimmed_mean_ie),
            ),
        )
        layer_union = sorted({layer for center in centers for layer in window_by_center[center].layers})
        region_rows.append(
            {
                "centers": centers,
                "region_source": str(region_spec["region_source"]),
                "region_source_priority": int(region_spec["region_source_priority"]),
                "representative_center": int(representative),
                "layer_union": layer_union,
                "region_mean_ie": float(np.mean(region_values)),
                "region_median_ie": float(np.median(region_values)),
                "region_trimmed_mean_ie": _trimmed_mean(region_values, trim_fraction),
                "region_ci_lower": ci_lower,
                "region_ci_upper": ci_upper,
                "peak_minus_region_ci_upper": difference_upper,
                "noninferior_to_discovery_peak": bool(difference_upper < margin),
                "median_win_rate": float(np.median(list(win_rates.values()))),
                "representative_win_rate": float(win_rates[representative]),
                "region_width_supported": bool(width_supported),
                "ci_positive_near_supported_region": bool(
                    allow_near_supported_region and near_supported and math.isfinite(ci_lower) and ci_lower > 0
                ),
                "supported_region": bool(
                    width_supported
                    or (allow_near_supported_region and near_supported and math.isfinite(ci_lower) and ci_lower > 0)
                ),
                "confirmation_positive_region_ci": bool(math.isfinite(ci_lower) and ci_lower > 0),
            }
        )

    confirmed_region_rows = [
        row for row in region_rows if row["confirmation_positive_region_ci"] and row["supported_region"]
    ]
    if not confirmed_region_rows:
        return {
            **base,
            "discovery_trace_center": raw_peak,
            "discovery_candidate_centers": candidate_centers,
            "discovery_candidate_regions": candidate_regions,
            "candidate_region_specs": candidate_region_specs,
            "confirmation_regions": region_rows,
            "confirmed_regions": [],
            "failure_reason": "no_confirmed_supported_region",
        }

    chosen = sorted(
        confirmed_region_rows,
        key=lambda row: (
            row["region_source_priority"],
            -row["region_median_ie"],
            -row["region_trimmed_mean_ie"],
            -row["median_win_rate"],
            -row["region_ci_lower"],
            -row["region_mean_ie"],
        ),
    )[0]
    representative = int(chosen["representative_center"])
    representative_window = window_by_center[representative]
    return {
        **base,
        "selected_trace_center": representative,
        "discovery_trace_center": raw_peak,
        "trace_window_start": int(representative_window.start),
        "trace_window_end": int(representative_window.end),
        "trace_window_layers": list(representative_window.layers),
        "confirmed_region_centers": list(chosen["centers"]),
        "confirmed_region_layer_union": list(chosen["layer_union"]),
        "confirmation_mean_ie": float(confirmation_rows[representative].mean_ie),
        "confirmation_ci_lower": float(confirmation_rows[representative].mean_ie_ci_lower),
        "confirmation_ci_upper": float(confirmation_rows[representative].mean_ie_ci_upper),
        "confirmation_passed": True,
        "failure_reason": None,
        "discovery_candidate_centers": candidate_centers,
        "discovery_candidate_regions": candidate_regions,
        "candidate_region_specs": candidate_region_specs,
        "local_support_threshold": support_threshold,
        "noninferiority_margin": margin,
        "confirmation_regions": region_rows,
        "confirmed_regions": confirmed_region_rows,
    }


__all__ = ["Window", "build_window", "select_region", "select_window", "summarize_windows"]
