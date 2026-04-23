#!/usr/bin/env python3
"""Post-hoc composite ROME layer detector for non-GPT architectures.

The detector keeps a blind-feature fallback built from raw spectral-gap,
top-1-energy local-z, and two spectral-gap local-z windows. When a full
spectral payload is available, it can also use Signal A (`sv_z_scores`) and a
joint Signal A/B boundary check to support the final layer choice.

The file also provides binary "was this model edited?" classification by
comparing ROME and baseline structural JSONs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

EPS = 1e-10
BASELINE_COLOR = "#000000"
DEFAULT_SMALL_WINDOW = 5
DEFAULT_LARGE_WINDOW = 7
DEFAULT_TE_WINDOW = 5
DEFAULT_NC_WINDOW = 5
SIGNAL_A_CONFIRM_Z_MIN = 2.0
SIGNAL_AB_BOUNDARY_WIDTH = 4
SIGNAL_AB_CLUSTER_SPAN = 2


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

def local_zscore(vals: np.ndarray, window: int = 5) -> np.ndarray:
    """Local z-score: how anomalous each point is relative to its neighbors."""
    n = len(vals)
    half = window // 2
    lz = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neighbors = np.concatenate([vals[lo:i], vals[i + 1:hi]])
        if len(neighbors) > 1:
            lz[i] = (vals[i] - neighbors.mean()) / (neighbors.std() + EPS)
    return lz


def _curvature(vals: np.ndarray) -> np.ndarray:
    """Absolute second-order finite-difference curvature."""
    if len(vals) < 3:
        return np.zeros_like(vals)
    c = np.abs(vals[:-2] - 2 * vals[1:-1] + vals[2:])
    return np.concatenate([[0], c, [0]])


def _is_valid_test(test: dict, require_rome_success: bool = True) -> bool:
    """Whether a test has usable structural features."""
    if test.get("error"):
        return False
    if require_rome_success and not test.get("rome", {}).get("success", True):
        return False
    return bool(test.get("blind_detection", {}).get("layer_features"))


def _valid_tests(data: dict, require_rome_success: bool = True) -> List[dict]:
    """Return tests that have usable structural features."""
    return [
        test for test in data.get("tests", [])
        if _is_valid_test(test, require_rome_success)
    ]


def _feature_array(layer_features: dict, layers: List[str], name: str) -> np.ndarray:
    """Extract one per-layer feature as a NumPy array."""
    return np.array([layer_features[l][name] for l in layers], dtype=float)


def _spectral_signal_array(
    spectral_detection: dict,
    layers: List[str],
    name: str,
) -> Optional[np.ndarray]:
    """Extract one per-layer spectral signal when a full payload is available."""
    layer_map = spectral_detection.get(name)
    if not isinstance(layer_map, dict):
        return None

    values = []
    for layer in layers:
        raw = layer_map.get(layer, layer_map.get(str(layer), 0.0))
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            return None

    arr = np.array(values, dtype=float)
    if not np.any(np.isfinite(arr)):
        return None

    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or np.max(np.abs(finite)) <= EPS:
        return None
    return arr


def _peak(vals: np.ndarray, eval_layers: List[int]) -> Tuple[int, int, float]:
    """Return peak index, layer, and global z-score for one evaluated signal."""
    idx = int(np.argmax(vals))
    z = float((vals[idx] - vals.mean()) / (vals.std() + EPS))
    return idx, eval_layers[idx], z


def _parse_int_csv(raw: str) -> List[int]:
    """Parse a comma-separated list of positive integers."""
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Expected positive integer, got {value}")
        values.append(value)
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


# ---------------------------------------------------------------------------
# Core detection: blind-feature chain with full-profile Signal A/B support
# ---------------------------------------------------------------------------

def detect_layer(
    test: dict,
    trim: int = 2,
    small_window: int = DEFAULT_SMALL_WINDOW,
    large_window: int = DEFAULT_LARGE_WINDOW,
    te_window: int = DEFAULT_TE_WINDOW,
    nc_window: int = DEFAULT_NC_WINDOW,
) -> Tuple[Optional[int], str, Dict]:
    """Detect which layer was edited using the blind-feature chain plus
    optional full-profile spectral support.

    Returns (detected_layer, method_tag, signal_info).
    """
    lf = test["blind_detection"]["layer_features"]
    layers = sorted(lf.keys(), key=int)
    n = len(layers)
    lo, hi = trim, n - trim
    if hi <= lo:
        return None, "empty", {}

    eval_layers = [int(l) for l in layers[lo:hi]]
    ne = len(eval_layers)

    small_window = max(3, int(small_window))
    large_window = max(small_window + 2, int(large_window))
    te_window = max(3, int(te_window))
    nc_window = max(3, int(nc_window))
    sg_small_tag = f"sg(lz{small_window})"
    sg_large_tag = f"sg(lz{large_window})"
    te_small_tag = f"te(lz{te_window})"
    te_large_tag = f"te(lz{large_window})"

    # Primary signals (v5 spectral chain)
    sg_full = _feature_array(lf, layers, "spectral_gap")
    te_full = _feature_array(lf, layers, "top1_energy")

    sg = sg_full[lo:hi]
    te_local = np.abs(local_zscore(te_full, te_window))[lo:hi]
    sg_local_small = np.abs(local_zscore(sg_full, small_window))[lo:hi]
    sg_local_large = np.abs(local_zscore(sg_full, large_window))[lo:hi]

    # Secondary signals retained for debug info and targeted rescues
    nc_full = _feature_array(lf, layers, "norm_cv")
    er_full = _feature_array(lf, layers, "effective_rank")
    ra_full = _feature_array(lf, layers, "row_alignment")

    nc_local = np.abs(local_zscore(nc_full, nc_window))[lo:hi]
    er_curv = _curvature(er_full)[lo:hi]
    ra_raw = ra_full[lo:hi]

    sg_i, sg_l, sg_z = _peak(sg, eval_layers)
    te_i, te_l, te_z = _peak(te_local, eval_layers)
    s5_i, s5_l, s5_z = _peak(sg_local_small, eval_layers)
    s7_i, s7_l, s7_z = _peak(sg_local_large, eval_layers)

    nc_i, nc_l, nc_z = _peak(nc_local, eval_layers)
    ec_i, ec_l, ec_z = _peak(er_curv, eval_layers)
    ra_i, ra_l, ra_z = _peak(ra_raw, eval_layers)

    info = {
        "windows": {
            "trim": trim,
            "small_window": small_window,
            "large_window": large_window,
            "te_window": te_window,
            "nc_window": nc_window,
        },
        "sg_raw": {"layer": sg_l, "z": round(sg_z, 2), "idx": sg_i},
        "te_local": {"layer": te_l, "z": round(te_z, 2), "idx": te_i},
        "sg_local_small": {"layer": s5_l, "z": round(s5_z, 2), "idx": s5_i},
        "sg_local_large": {"layer": s7_l, "z": round(s7_z, 2), "idx": s7_i},
        "nc_local": {"layer": nc_l, "z": round(nc_z, 2), "idx": nc_i},
        "er_curv": {"layer": ec_l, "z": round(ec_z, 2), "idx": ec_i},
        "ra_raw": {"layer": ra_l, "z": round(ra_z, 2), "idx": ra_i},
        "eval_layers": eval_layers,
    }

    if te_window == DEFAULT_TE_WINDOW:
        info["te_lz5"] = info["te_local"]
    if small_window == DEFAULT_SMALL_WINDOW:
        info["sg_lz5"] = info["sg_local_small"]
    if large_window == DEFAULT_LARGE_WINDOW:
        info["sg_lz7"] = info["sg_local_large"]
    if nc_window == DEFAULT_NC_WINDOW:
        info["nc_lz5"] = info["nc_local"]

    spectral_detection = test.get("spectral_detection", {})
    a_i = a_l = None
    a_z = 0.0
    b_i = b_l = None
    b_z = 0.0

    if isinstance(spectral_detection, dict):
        signal_a_full = _spectral_signal_array(spectral_detection, layers, "sv_z_scores")
        signal_b_full = _spectral_signal_array(spectral_detection, layers, "sv_ratio_scores")

        if signal_a_full is not None:
            signal_a = signal_a_full[lo:hi]
            a_i, a_l, a_z = _peak(signal_a, eval_layers)
            info["signal_a"] = {"layer": a_l, "z": round(a_z, 2), "idx": a_i}

        if signal_b_full is not None:
            signal_b = signal_b_full[lo:hi]
            b_i, b_l, b_z = _peak(signal_b, eval_layers)
            info["signal_b"] = {"layer": b_l, "z": round(b_z, 2), "idx": b_i}

    # === v5 spectral confirmation chain ===
    v5_layer, v5_method = None, "none"
    v5_idx = None

    # Step 1: Agreement
    if sg_l == te_l:
        v5_layer, v5_method = sg_l, "agree"
        v5_idx = sg_i

    # Step 2: SG_lz5 confirms one side exclusively (±1 layer index)
    if v5_layer is None:
        s5_near_sg = abs(s5_i - sg_i) <= 1
        s5_near_te = abs(s5_i - te_i) <= 1
        if s5_near_sg and not s5_near_te:
            v5_layer, v5_method = sg_l, sg_small_tag
            v5_idx = sg_i
        elif s5_near_te and not s5_near_sg:
            v5_layer, v5_method = te_l, te_small_tag
            v5_idx = te_i

    # Step 3: SG_lz7 confirms one side exclusively
    if v5_layer is None:
        s7_near_sg = abs(s7_i - sg_i) <= 1
        s7_near_te = abs(s7_i - te_i) <= 1
        if s7_near_sg and not s7_near_te:
            v5_layer, v5_method = sg_l, sg_large_tag
            v5_idx = sg_i
        elif s7_near_te and not s7_near_sg:
            v5_layer, v5_method = te_l, te_large_tag
            v5_idx = te_i

    # Step 4: Neither lz signal confirms exclusively → trend-based fallback
    if v5_layer is None:
        rho, _ = stats.spearmanr(np.arange(ne), sg)
        info["rho"] = round(float(rho), 3)

        if abs(rho) > 0.3:
            if abs(s5_i - s7_i) <= 1:
                layer = s5_l if s5_z >= s7_z else s7_l
                tag = f"lz_cons({small_window})" if s5_z >= s7_z else f"lz_cons({large_window})"
                v5_layer, v5_method = layer, tag
                v5_idx = s5_i if s5_z >= s7_z else s7_i
            elif abs(s7_i - te_i) <= 1:
                v5_layer, v5_method = te_l, "te(trend)"
                v5_idx = te_i
            else:
                v5_layer, v5_method = s7_l, "s7(trend)"
                v5_idx = s7_i
        else:
            v5_layer, v5_method = sg_l, "sg(fb)"
            v5_idx = sg_i

    if v5_idx is None and v5_layer in eval_layers:
        v5_idx = eval_layers.index(v5_layer)

    # Full-profile support 1: when Signal A lines up with the TE branch, trust
    # the spectral confirmation over the blind fallback tag.
    if a_l is not None and a_z >= SIGNAL_A_CONFIRM_Z_MIN and a_l == te_l:
        info["spectral_support"] = {
            "kind": "signal_a",
            "reason": "te_alignment",
        }
        return a_l, "signal_a", info

    # Full-profile support 2: if SG, Signal A, and Signal B all cluster inside
    # the same boundary band, keep the raw SG layer instead of inventing a
    # separate edge-only rescue path.
    if a_i is not None and b_i is not None:
        peak_indices = [sg_i, a_i, b_i]
        cluster_span = max(peak_indices) - min(peak_indices)
        early_limit = min(SIGNAL_AB_BOUNDARY_WIDTH - 1, ne - 1)
        late_limit = max(0, ne - SIGNAL_AB_BOUNDARY_WIDTH)
        early_cluster = max(peak_indices) <= early_limit
        late_cluster = min(peak_indices) >= late_limit
        if cluster_span <= SIGNAL_AB_CLUSTER_SPAN and (early_cluster or late_cluster):
            info["spectral_support"] = {
                "kind": "signal_ab_boundary",
                "reason": "boundary_cluster",
                "cluster_span": cluster_span,
            }
            return sg_l, "signal_ab_boundary", info

    return v5_layer, v5_method, info


# ---------------------------------------------------------------------------
# Binary edit detection
# ---------------------------------------------------------------------------

def compute_anomaly_score(test: dict, trim: int = 2) -> float:
    """Compute scalar anomaly score for a test (higher = more likely edited).

    Uses the peak local-z score across SG and TE as the anomaly measure.
    """
    lf = test["blind_detection"]["layer_features"]
    layers = sorted(lf.keys(), key=int)
    n = len(layers)
    lo, hi = trim, n - trim
    if hi <= lo:
        return 0.0

    sg_full = _feature_array(lf, layers, "spectral_gap")
    te_full = _feature_array(lf, layers, "top1_energy")

    sg_lz5 = np.abs(local_zscore(sg_full, 5))[lo:hi]
    te_lz5 = np.abs(local_zscore(te_full, 5))[lo:hi]

    return float(max(sg_lz5.max(), te_lz5.max()))


def detect_edit_binary(rome_path: Path, baseline_path: Optional[Path] = None,
                       trim: int = 2, threshold: float = 3.0,
                       ) -> Dict:
    """Determine whether a model was edited (binary yes/no).

    If a baseline is provided, compares ROME anomaly scores against baseline
    distribution.  Otherwise uses an absolute z-score threshold.

    Returns dict with is_edited, confidence, details.
    """
    with open(rome_path) as f:
        rome_data = json.load(f)

    rome_tests = _valid_tests(rome_data)

    rome_scores = [compute_anomaly_score(t, trim) for t in rome_tests]

    result: Dict = {
        "model": rome_data.get("metadata", {}).get("model", rome_path.stem),
        "n_rome_tests": len(rome_scores),
        "rome_mean_score": float(np.mean(rome_scores)) if rome_scores else 0.0,
        "rome_max_score": float(np.max(rome_scores)) if rome_scores else 0.0,
    }

    if not rome_scores:
        result["is_edited"] = None
        result["confidence"] = 0.0
        result["reason"] = "no_valid_rome_tests"
        return result

    if baseline_path and baseline_path.exists():
        with open(baseline_path) as f:
            base_data = json.load(f)
        base_tests = _valid_tests(base_data, require_rome_success=False)
        base_scores = [compute_anomaly_score(t, trim) for t in base_tests]

        if base_scores:
            base_mean = float(np.mean(base_scores))
            base_std = float(np.std(base_scores))
            delta = result["rome_mean_score"] - base_mean
            result["baseline_mean"] = base_mean
            result["baseline_n_tests"] = len(base_scores)
            result["baseline_std"] = round(base_std, 4)
            result["score_delta"] = round(delta, 4)
            result["score_ratio"] = round(
                result["rome_mean_score"] / (base_mean + EPS), 4)

            if len(base_scores) < 2 or base_std <= EPS:
                result["is_edited"] = None
                result["confidence"] = 0.0
                result["reason"] = "insufficient_baseline_variance"
                return result

            separation = delta / base_std
            result["separation_z"] = round(separation, 2)
            result["is_edited"] = separation > 2.0
            result["confidence"] = min(1.0, max(0.0, separation) / 5.0)
            return result

    # No baseline → absolute threshold
    result["is_edited"] = result["rome_mean_score"] > threshold
    result["confidence"] = min(1.0, result["rome_mean_score"] / (2.0 * threshold))
    return result


# ---------------------------------------------------------------------------
# Process a full JSON file
# ---------------------------------------------------------------------------

def process_file(
    path: Path,
    trim: int = 2,
    small_window: int = DEFAULT_SMALL_WINDOW,
    large_window: int = DEFAULT_LARGE_WINDOW,
    te_window: int = DEFAULT_TE_WINDOW,
    nc_window: int = DEFAULT_NC_WINDOW,
    max_tests: Optional[int] = None,
) -> Dict:
    """Process a structural benchmark JSON and detect edited layers."""
    with open(path) as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    target = meta.get("target_layer")
    model = meta.get("model_name", meta.get("model", path.stem))

    results = []
    correct = 0
    method_counts: Dict[str, int] = {}
    used_tests = 0

    for ti, test in enumerate(data.get("tests", [])):
        if not _is_valid_test(test):
            continue
        if max_tests is not None and used_tests >= max_tests:
            break

        detected, method, info = detect_layer(
            test,
            trim=trim,
            small_window=small_window,
            large_window=large_window,
            te_window=te_window,
            nc_window=nc_window,
        )
        hit = detected == target
        if hit:
            correct += 1
        method_counts[method] = method_counts.get(method, 0) + 1
        results.append({
            "test_idx": ti,
            "detected": detected,
            "target": target,
            "hit": hit,
            "method": method,
            "info": info,
        })
        used_tests += 1

    n_valid = len(results)
    path_obj = Path(path)
    baseline_path = _find_baseline(path_obj, payload=data)
    return {
        "model": model,
        "target_layer": target,
        "n_tests": n_valid,
        "correct": correct,
        "accuracy": correct / n_valid if n_valid else 0.0,
        "method_counts": method_counts,
        "results": results,
        "path": str(path),
        "trim": trim,
        "small_window": small_window,
        "large_window": large_window,
        "te_window": te_window,
        "nc_window": nc_window,
        "baseline_path": str(baseline_path) if baseline_path is not None else None,
        "run_label": _run_label_from_path(path_obj),
        "run_slug": _run_slug_from_path(path_obj),
    }


def sweep_file(
    path: Path,
    trims: Optional[List[int]] = None,
    small_windows: Optional[List[int]] = None,
    large_windows: Optional[List[int]] = None,
) -> Dict:
    """Evaluate a grid of detector window/trim configs against benchmark labels."""
    trims = trims or [1, 2, 3]
    small_windows = small_windows or [3, 5, 7]
    large_windows = large_windows or [5, 7, 9, 11]

    configs = []
    for trim in sorted(set(int(v) for v in trims)):
        for small_window in sorted(set(int(v) for v in small_windows)):
            for large_window in sorted(set(int(v) for v in large_windows)):
                if large_window <= small_window:
                    continue
                result = process_file(
                    path,
                    trim=trim,
                    small_window=small_window,
                    large_window=large_window,
                    te_window=small_window,
                    nc_window=small_window,
                )
                configs.append({
                    "trim": trim,
                    "small_window": small_window,
                    "large_window": large_window,
                    "te_window": small_window,
                    "nc_window": small_window,
                    "accuracy": float(result["accuracy"]),
                    "correct": int(result["correct"]),
                    "n_tests": int(result["n_tests"]),
                    "method_counts": result["method_counts"],
                })

    configs.sort(
        key=lambda item: (
            item["accuracy"],
            item["correct"],
            -abs(item["trim"] - 2),
            -abs(item["small_window"] - DEFAULT_SMALL_WINDOW),
            -abs(item["large_window"] - DEFAULT_LARGE_WINDOW),
        ),
        reverse=True,
    )

    with open(path) as f:
        payload = json.load(f)
    meta = payload.get("metadata", {})
    model = meta.get("model_name", meta.get("model", path.stem))
    target = meta.get("target_layer")
    return {
        "model": model,
        "target_layer": target,
        "path": str(path),
        "default_config": {
            "trim": 2,
            "small_window": DEFAULT_SMALL_WINDOW,
            "large_window": DEFAULT_LARGE_WINDOW,
            "te_window": DEFAULT_TE_WINDOW,
            "nc_window": DEFAULT_NC_WINDOW,
        },
        "best_config": configs[0] if configs else None,
        "configs": configs,
    }


# ---------------------------------------------------------------------------
# Paper-quality graphs
# ---------------------------------------------------------------------------

def _setup_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (14, 5),
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _short_model_name(model: str) -> str:
    return model.split("/")[-1] if "/" in model else model


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip("_")) or "run"


def _run_root_from_path(path: Path) -> Path:
    run_dir = path.parent
    if run_dir.name == "structural":
        run_dir = run_dir.parent
    return run_dir


def _run_label_from_path(path: Path) -> str:
    run_dir = _run_root_from_path(path)
    timestamp = "_".join(path.stem.split("_")[-2:])
    parts = run_dir.parts
    if "pipeline_out" in parts:
        idx = parts.index("pipeline_out")
        rel_parts = parts[idx + 1:]
        run_name = "/".join(rel_parts) if rel_parts else run_dir.name
    else:
        run_name = run_dir.name
    if timestamp and timestamp not in run_name:
        return f"{run_name} | {timestamp}"
    return run_name


def _run_slug_from_path(path: Path) -> str:
    return _safe_slug(_run_label_from_path(path))


def _signal_profile_stats(
    payload: dict,
    trim: int,
    te_window: int,
    small_window: int,
    large_window: int,
) -> Dict[str, dict]:
    values_by_signal: Dict[str, Dict[int, List[float]]] = {
        "SG (raw)": {},
        f"TE (lz{te_window})": {},
        f"SG (lz{small_window})": {},
        f"SG (lz{large_window})": {},
    }

    for test in _valid_tests(payload):
        lf = test["blind_detection"]["layer_features"]
        layers = sorted(lf.keys(), key=int)
        n = len(layers)
        lo, hi = trim, n - trim
        if hi <= lo:
            continue

        eval_layers = [int(layer) for layer in layers[lo:hi]]
        sg_full = _feature_array(lf, layers, "spectral_gap")
        te_full = _feature_array(lf, layers, "top1_energy")
        signal_values = {
            "SG (raw)": sg_full[lo:hi],
            f"TE (lz{te_window})": np.abs(local_zscore(te_full, te_window))[lo:hi],
            f"SG (lz{small_window})": np.abs(local_zscore(sg_full, small_window))[lo:hi],
            f"SG (lz{large_window})": np.abs(local_zscore(sg_full, large_window))[lo:hi],
        }

        spectral_detection = test.get("spectral_detection", {})
        if isinstance(spectral_detection, dict):
            signal_a_full = _spectral_signal_array(spectral_detection, layers, "sv_z_scores")
            signal_b_full = _spectral_signal_array(spectral_detection, layers, "sv_ratio_scores")
            if signal_a_full is not None:
                signal_values["Signal A"] = signal_a_full[lo:hi]
            if signal_b_full is not None:
                signal_values["Signal B"] = signal_b_full[lo:hi]

        for name, vals in signal_values.items():
            for layer, value in zip(eval_layers, vals):
                if np.isfinite(value):
                    values_by_signal.setdefault(name, {}).setdefault(int(layer), []).append(float(value))

    stats_by_signal: Dict[str, dict] = {}
    for name, layer_map in values_by_signal.items():
        if not layer_map:
            continue
        ordered_layers = np.array(sorted(layer_map), dtype=int)
        stats_by_signal[name] = {
            "layers": ordered_layers,
            "mean": np.array([np.mean(layer_map[layer]) for layer in ordered_layers], dtype=float),
            "std": np.array([np.std(layer_map[layer]) for layer in ordered_layers], dtype=float),
            "count": np.array([len(layer_map[layer]) for layer in ordered_layers], dtype=int),
        }
    return stats_by_signal


def _aggregate_signal_profile_stats(file_results: List[Dict]) -> Dict[str, dict]:
    per_signal: Dict[str, Dict[int, List[float]]] = {}

    for file_result in file_results:
        path = Path(file_result["path"])
        with open(path) as f:
            payload = json.load(f)
        trim = int(file_result.get("trim", 2))
        te_window = int(file_result.get("te_window", DEFAULT_TE_WINDOW))
        small_window = int(file_result.get("small_window", DEFAULT_SMALL_WINDOW))
        large_window = int(file_result.get("large_window", DEFAULT_LARGE_WINDOW))
        run_stats = _signal_profile_stats(
            payload,
            trim=trim,
            te_window=te_window,
            small_window=small_window,
            large_window=large_window,
        )
        for name, stats in run_stats.items():
            target = per_signal.setdefault(name, {})
            for layer, value in zip(stats["layers"], stats["mean"]):
                if np.isfinite(value):
                    target.setdefault(int(layer), []).append(float(value))

    aggregated: Dict[str, dict] = {}
    for name, layer_map in per_signal.items():
        if not layer_map:
            continue
        ordered_layers = np.array(sorted(layer_map), dtype=int)
        aggregated[name] = {
            "layers": ordered_layers,
            "mean": np.array([np.mean(layer_map[layer]) for layer in ordered_layers], dtype=float),
            "std": np.array([np.std(layer_map[layer]) for layer in ordered_layers], dtype=float),
            "count": np.array([len(layer_map[layer]) for layer in ordered_layers], dtype=int),
        }
    return aggregated


def _plot_baseline_overlay(ax, layers, values, label: Optional[str]) -> None:
    import matplotlib.patheffects as pe

    markevery = max(1, len(layers) // 12)
    line = ax.plot(
        layers,
        values,
        color=BASELINE_COLOR,
        linewidth=1.8,
        linestyle="--",
        marker="o",
        markersize=2.4,
        markerfacecolor="white",
        markeredgecolor=BASELINE_COLOR,
        markeredgewidth=0.8,
        markevery=markevery,
        label=label,
        zorder=4,
    )[0]
    line.set_path_effects([pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])


def _signal_profile_colors(
    te_window: int,
    small_window: int,
    large_window: int,
) -> Dict[str, str]:
    return {
        "SG (raw)": "#1f77b4",
        f"TE (lz{te_window})": "#ff7f0e",
        f"SG (lz{small_window})": "#2ca02c",
        f"SG (lz{large_window})": "#d62728",
        "Signal A": "#6b6ecf",
        "Signal B": "#8c564b",
    }


def _load_baseline_signal_profile_stats(
    rome_path: Path,
    trim: int,
    te_window: int,
    small_window: int,
    large_window: int,
    baseline_path: Optional[Path] = None,
) -> Dict[str, dict]:
    if baseline_path is None:
        baseline_path = _find_baseline(rome_path)
    if baseline_path is None or not baseline_path.exists():
        return {}
    with open(baseline_path) as f:
        payload = json.load(f)
    return _signal_profile_stats(
        payload,
        trim=trim,
        te_window=te_window,
        small_window=small_window,
        large_window=large_window,
    )


def _aggregate_baseline_signal_profile_stats(file_results: List[Dict]) -> Dict[str, dict]:
    per_signal: Dict[str, Dict[int, List[float]]] = {}
    seen_paths: set[str] = set()

    for file_result in file_results:
        path = Path(file_result["path"])
        baseline_path_raw = file_result.get("baseline_path")
        baseline_path = Path(baseline_path_raw) if baseline_path_raw else _find_baseline(path)
        if baseline_path is None or not baseline_path.exists():
            continue
        key = str(baseline_path.resolve())
        if key in seen_paths:
            continue
        seen_paths.add(key)

        with open(baseline_path) as f:
            payload = json.load(f)
        trim = int(file_result.get("trim", 2))
        te_window = int(file_result.get("te_window", DEFAULT_TE_WINDOW))
        small_window = int(file_result.get("small_window", DEFAULT_SMALL_WINDOW))
        large_window = int(file_result.get("large_window", DEFAULT_LARGE_WINDOW))
        run_stats = _signal_profile_stats(
            payload,
            trim=trim,
            te_window=te_window,
            small_window=small_window,
            large_window=large_window,
        )
        for name, stats in run_stats.items():
            target = per_signal.setdefault(name, {})
            for layer, value in zip(stats["layers"], stats["mean"]):
                if np.isfinite(value):
                    target.setdefault(int(layer), []).append(float(value))

    aggregated: Dict[str, dict] = {}
    for name, layer_map in per_signal.items():
        if not layer_map:
            continue
        ordered_layers = np.array(sorted(layer_map), dtype=int)
        aggregated[name] = {
            "layers": ordered_layers,
            "mean": np.array([np.mean(layer_map[layer]) for layer in ordered_layers], dtype=float),
            "std": np.array([np.std(layer_map[layer]) for layer in ordered_layers], dtype=float),
            "count": np.array([len(layer_map[layer]) for layer in ordered_layers], dtype=int),
        }
    return aggregated


def aggregate_results_by_model(all_results: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for result in all_results:
        grouped.setdefault(result["model"], []).append(result)

    aggregated: List[Dict] = []
    for model in sorted(grouped, key=lambda item: _short_model_name(item).lower()):
        runs = grouped[model]
        method_counts: Dict[str, int] = {}
        target_layers = []
        for run in runs:
            for method, count in run.get("method_counts", {}).items():
                method_counts[method] = method_counts.get(method, 0) + int(count)
            if run.get("target_layer") is not None:
                target_layers.append(int(run["target_layer"]))

        run_accuracies = [float(run.get("accuracy", 0.0)) for run in runs]
        total_tests = sum(int(run.get("n_tests", 0)) for run in runs)
        total_correct = sum(int(run.get("correct", 0)) for run in runs)
        aggregated.append(
            {
                "model": model,
                "target_layer": int(np.round(np.mean(target_layers))) if target_layers else None,
                "n_tests": total_tests,
                "correct": total_correct,
                "accuracy": float(np.mean(run_accuracies)) if run_accuracies else 0.0,
                "accuracy_std": float(np.std(run_accuracies)) if len(run_accuracies) > 1 else 0.0,
                "weighted_accuracy": (total_correct / total_tests) if total_tests else 0.0,
                "method_counts": method_counts,
                "n_runs": len(runs),
                "paths": [run["path"] for run in runs],
                "baseline_paths": [
                    run["baseline_path"] for run in runs if run.get("baseline_path")
                ],
                "run_labels": [run.get("run_label", Path(run["path"]).stem) for run in runs],
            }
        )

    return aggregated


def plot_signal_profiles(file_result: Dict, output_dir: Optional[Path] = None):
    """Plot the available signal profiles for one run, averaged across valid tests."""
    import matplotlib.pyplot as plt

    _setup_style()

    model = file_result["model"]
    target = file_result["target_layer"]
    results = file_result.get("results", [])
    if not results:
        return

    path = Path(file_result["path"])
    with open(path) as f:
        payload = json.load(f)
    trim = file_result.get("trim", 2)
    te_window = int(file_result.get("te_window", DEFAULT_TE_WINDOW))
    small_window = int(file_result.get("small_window", DEFAULT_SMALL_WINDOW))
    large_window = int(file_result.get("large_window", DEFAULT_LARGE_WINDOW))
    signal_stats = _signal_profile_stats(
        payload,
        trim=trim,
        te_window=te_window,
        small_window=small_window,
        large_window=large_window,
    )
    if not signal_stats:
        return
    baseline_signal_stats = _load_baseline_signal_profile_stats(
        path,
        trim=trim,
        te_window=te_window,
        small_window=small_window,
        large_window=large_window,
        baseline_path=Path(file_result["baseline_path"]) if file_result.get("baseline_path") else None,
    )

    colors = _signal_profile_colors(te_window, small_window, large_window)

    r0 = results[0]
    detected = r0["detected"]
    method = r0["method"]
    run_label = file_result.get("run_label") or _run_label_from_path(path)
    short = _short_model_name(model)

    signal_items = list(signal_stats.items())
    ncols = 2
    nrows = max(1, (len(signal_items) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (name, stats) in zip(axes_flat, signal_items):
        layers = stats["layers"]
        mean = stats["mean"]
        std = stats["std"]
        color = colors.get(name, "#4c78a8")
        ax.plot(layers, mean, color=color, linewidth=1.7, label="Edited mean")
        if np.any(np.isfinite(std)):
            ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.14)
        baseline_stats = baseline_signal_stats.get(name)
        if baseline_stats is not None:
            _plot_baseline_overlay(
                ax,
                baseline_stats["layers"],
                baseline_stats["mean"],
                label="Unedited baseline",
            )
        peak_idx = int(np.nanargmax(mean))
        ax.axvline(layers[peak_idx], color=color, alpha=0.4, linestyle="--",
                   label=f"peak L{int(layers[peak_idx])}")
        if target in layers.tolist():
            ax.axvline(target, color="black", alpha=0.3, linestyle=":",
                       label=f"target L{target}")
        ax.set_title(name, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        step = max(1, len(layers) // 16)
        tick_layers = layers[::step]
        ax.set_xticks(tick_layers)
        ax.set_xticklabels([str(int(layer)) for layer in tick_layers])

    for ax in axes_flat[len(signal_items):]:
        ax.set_visible(False)

    fig.suptitle(f"{short}  —  {run_label}  —  detected L{detected} via {method}  "
                 f"(target L{target}, acc={file_result['accuracy']:.0%}, n={file_result['n_tests']})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.supxlabel("Layer index")
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe = _safe_slug(short)
        out = output_dir / f"signals_{safe}_{file_result.get('run_slug', _run_slug_from_path(path))}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    else:
        plt.show()
    plt.close(fig)


def plot_average_signal_profiles(file_results: List[Dict], output_dir: Optional[Path] = None):
    """Plot averaged signal-profile panels per model across all runs."""
    import matplotlib.pyplot as plt

    if not file_results:
        return

    _setup_style()

    model = file_results[0]["model"]
    short = _short_model_name(model)
    target_values = [run.get("target_layer") for run in file_results if run.get("target_layer") is not None]
    target = int(np.round(np.mean(target_values))) if target_values else None
    total_tests = sum(int(run.get("n_tests", 0) or 0) for run in file_results)
    total_correct = sum(int(run.get("correct", 0) or 0) for run in file_results)
    run_count = len(file_results)
    weighted_accuracy = (total_correct / total_tests) if total_tests else 0.0
    signal_stats = _aggregate_signal_profile_stats(file_results)
    if not signal_stats:
        return
    baseline_signal_stats = _aggregate_baseline_signal_profile_stats(file_results)

    first = file_results[0]
    te_window = int(first.get("te_window", DEFAULT_TE_WINDOW))
    small_window = int(first.get("small_window", DEFAULT_SMALL_WINDOW))
    large_window = int(first.get("large_window", DEFAULT_LARGE_WINDOW))
    colors = _signal_profile_colors(te_window, small_window, large_window)
    signal_names = [name for name in colors if name in signal_stats]
    ncols = 2
    nrows = max(1, (len(signal_names) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes_flat, signal_names):
        stats = signal_stats.get(name)
        if stats is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(name, fontweight="bold")
            continue
        layers = stats["layers"]
        mean = stats["mean"]
        std = stats["std"]
        color = colors.get(name, "#4c78a8")
        ax.plot(layers, mean, color=color, linewidth=1.8, label="Edited mean")
        if np.any(np.isfinite(std)):
            ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.16)
        baseline_stats = baseline_signal_stats.get(name)
        if baseline_stats is not None:
            _plot_baseline_overlay(
                ax,
                baseline_stats["layers"],
                baseline_stats["mean"],
                label="Unedited baseline",
            )
        peak_idx = int(np.nanargmax(mean))
        ax.axvline(layers[peak_idx], color=color, alpha=0.4, linestyle="--",
                   label=f"mean peak L{int(layers[peak_idx])}")
        if target in layers.tolist():
            ax.axvline(target, color="black", alpha=0.3, linestyle=":",
                       label=f"target L{target}")
        ax.set_title(name, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        step = max(1, len(layers) // 16)
        tick_layers = layers[::step]
        ax.set_xticks(tick_layers)
        ax.set_xticklabels([str(int(layer)) for layer in tick_layers])

    for ax in axes_flat[len(signal_names):]:
        ax.set_visible(False)

    fig.suptitle(
        f"{short}  —  average across {total_tests} tests from {run_count} "
        f"{'run' if run_count == 1 else 'runs'}  "
        f"(target L{target}, weighted acc={weighted_accuracy:.0%})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.supxlabel("Layer index")
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"signals_{_safe_slug(short)}_average.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    else:
        plt.show()
    plt.close(fig)


def plot_summary_table(
    all_results: List[Dict],
    output_dir: Optional[Path] = None,
    output_name: str = "summary_accuracy.png",
    title: str = "Composite Detector v2 — Per-Model Accuracy",
):
    """Generate an accuracy bar chart for run-level or aggregated model results."""
    import matplotlib.pyplot as plt

    if not all_results:
        return

    _setup_style()

    models = []
    accs = []
    ns = []
    for r in all_results:
        short = r["model"].split("/")[-1] if "/" in r["model"] else r["model"]
        models.append(short)
        accs.append(r["accuracy"])
        ns.append(r["n_tests"])

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.8), 5))
    bar_colors = ["#2ecc71" if a >= 0.9 else ("#f39c12" if a >= 0.5 else "#e74c3c")
                  for a in accs]
    bars = ax.bar(range(len(models)), [a * 100 for a in accs], color=bar_colors,
                  edgecolor="white", linewidth=0.5)

    for i, (bar, n, a) in enumerate(zip(bars, ns, accs)):
        label = f"{a:.0%}\n(n={n})"
        n_runs = int(all_results[i].get("n_runs", 0) or 0)
        if n_runs > 0:
            label = f"{a:.0%}\n(runs={n_runs}, tests={n})"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                label, ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Layer Detection Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title(title, fontweight="bold")
    ax.axhline(100, color="gray", linestyle=":", alpha=0.4)

    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / output_name
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    else:
        plt.show()
    plt.close(fig)


def plot_method_breakdown(
    all_results: List[Dict],
    output_dir: Optional[Path] = None,
    output_name: str = "method_breakdown.png",
    title: str = "Detection Method Breakdown per Model",
):
    """Stacked bar chart showing which detection method was used per model."""
    import matplotlib.pyplot as plt

    if not all_results:
        return

    _setup_style()

    # Collect all methods across models
    all_methods: set = set()
    for r in all_results:
        all_methods.update(r["method_counts"].keys())
    method_order = sorted(all_methods)

    method_colors = {
        "agree": "#2ecc71", "sg(lz5)": "#3498db", "te(lz5)": "#e67e22",
        "sg(lz7)": "#1abc9c", "te(lz7)": "#e74c3c", "lz_cons(5)": "#9b59b6",
        "lz_cons(7)": "#8e44ad", "te(trend)": "#f1c40f", "s7(trend)": "#d35400",
        "sg(fb)": "#95a5a6", "empty": "#bdc3c7",
        "signal_a": "#6b6ecf", "signal_ab_boundary": "#8c564b",
    }
    default_cmap = plt.cm.tab20
    for i, m in enumerate(method_order):
        if m not in method_colors:
            method_colors[m] = default_cmap(i / max(len(method_order), 1))

    models = []
    for r in all_results:
        short = r["model"].split("/")[-1] if "/" in r["model"] else r["model"]
        models.append(short)

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.8), 5))
    bottoms = np.zeros(len(models))

    for method in method_order:
        vals = [r["method_counts"].get(method, 0) for r in all_results]
        ax.bar(range(len(models)), vals, bottom=bottoms,
               label=method, color=method_colors.get(method, "#999999"),
               edgecolor="white", linewidth=0.3)
        bottoms += np.array(vals, dtype=float)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Number of tests")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / output_name
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _collect_json_files(paths: List[str], prefix: str = "rome_structural_") -> List[Path]:
    """Resolve paths/dirs/globs into sorted list of JSON files."""
    files = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.glob(f"{prefix}*.json")))
        elif pp.is_file():
            files.append(pp)
        else:
            files.extend(sorted(Path(".").glob(p)))
    return files


def _find_baseline(
    rome_path: Path,
    baseline_dir: Optional[Path] = None,
    payload: Optional[dict] = None,
) -> Optional[Path]:
    """Find the matching baseline JSON for a structural run.

    Resolution order:
    1. explicit metadata references emitted by newer benchmark runs
    2. exact filename replacement when edited/baseline share the same run token
    3. exact run-stem match ignoring the final timestamp fields
    4. legacy model-slug fallback for older outputs
    """
    search_dirs: List[Path] = []
    seen_dirs: set[str] = set()
    for search_dir in [rome_path.parent, baseline_dir]:
        if search_dir is None:
            continue
        resolved = str(Path(search_dir).resolve())
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        search_dirs.append(Path(search_dir))

    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    for key in ("baseline_structural_path", "baseline_structural_file"):
        raw_candidate = metadata.get(key)
        if not raw_candidate:
            continue
        candidate_path = Path(str(raw_candidate))
        if candidate_path.is_absolute() and candidate_path.exists():
            return candidate_path
        for search_dir in search_dirs:
            candidate = search_dir / candidate_path
            if candidate.exists():
                return candidate

    name = rome_path.name
    base_name = name.replace("rome_structural_", "baseline_structural_", 1)
    for search_dir in search_dirs:
        candidate = search_dir / base_name
        if candidate.exists():
            return candidate

    stem = rome_path.stem
    if stem.startswith("rome_structural_"):
        run_stem = stem[len("rome_structural_"):]
        run_stem_prefix = run_stem.rsplit("_", 2)[0]
        if run_stem_prefix and run_stem_prefix != run_stem:
            for search_dir in search_dirs:
                candidates = sorted(
                    search_dir.glob(f"baseline_structural_{run_stem_prefix}_*.json"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                if candidates:
                    return candidates[0]

    m = re.match(r"rome_structural_(.+?)_tk\d+", name)
    if not m:
        m = re.match(r"rome_structural_(.+?)_\d{4}-\d{2}-\d{2}", name)
    if not m:
        return None
    slug = m.group(1)
    for search_dir in search_dirs:
        candidates = sorted(
            search_dir.glob(f"baseline_structural_{slug}_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Composite ROME layer detector v2 — post-hoc analysis")
    parser.add_argument("paths", nargs="+", help="JSON files or directories")
    parser.add_argument("--trim", type=int, default=2,
                        help="Layers to trim from each end (default: 2)")
    parser.add_argument("--graphs", action="store_true",
                        help="Generate paper-quality graphs")
    parser.add_argument("--graph-dir", type=str,
                        default="analysis_out/composite_v2",
                        help="Output directory for graphs")
    parser.add_argument("--binary", action="store_true",
                        help="Run binary edit detection instead of layer detection")
    parser.add_argument("--baseline-dir", type=str, default=None,
                        help="Directory with baseline_structural_*.json files")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Write results to JSON file")
    parser.add_argument("--window-sweep", action="store_true",
                        help="Evaluate a grid of trim/window configs against benchmark labels")
    parser.add_argument("--sweep-trims", type=str, default="1,2,3",
                        help="Comma-separated trims for --window-sweep")
    parser.add_argument("--sweep-small-windows", type=str, default="3,5,7",
                        help="Comma-separated small local-z windows for --window-sweep")
    parser.add_argument("--sweep-large-windows", type=str, default="5,7,9,11",
                        help="Comma-separated large local-z windows for --window-sweep")
    parser.add_argument("--sweep-top-k", type=int, default=5,
                        help="How many top configs to print per file in --window-sweep mode")
    args = parser.parse_args()

    json_files = _collect_json_files(args.paths)
    if not json_files:
        print("No JSON files found.")
        sys.exit(1)

    # ----- Binary edit detection mode -----
    if args.binary:
        print(f"Binary edit detection on {len(json_files)} files\n")
        print(f"{'Model':<35s} {'Edited?':>8} {'Conf':>6} {'Sep-z':>6} {'Scores'}")
        print("-" * 85)
        baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
        binary_results = []
        for jf in json_files:
            bp = _find_baseline(jf, baseline_dir)
            r = detect_edit_binary(jf, bp, args.trim)
            binary_results.append(r)
            if r["is_edited"] is True:
                edited = "YES"
            elif r["is_edited"] is False:
                edited = "no"
            else:
                edited = "unknown"
            conf = f"{r['confidence']:.0%}"
            sep = f"{r.get('separation_z', 0):.1f}" if "separation_z" in r else "—"
            scores = f"rome={r['rome_mean_score']:.2f}"
            if "baseline_mean" in r:
                scores += f" base={r['baseline_mean']:.2f}"
            if r.get("reason"):
                scores += f" ({r['reason']})"
            print(f"  {r['model']:<33s} {edited:>8} {conf:>6} {sep:>6}  {scores}")
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(binary_results, f, indent=2)
            print(f"\nResults saved to {args.json_out}")
        return

    if args.window_sweep:
        trims = _parse_int_csv(args.sweep_trims)
        small_windows = _parse_int_csv(args.sweep_small_windows)
        large_windows = _parse_int_csv(args.sweep_large_windows)
        print(f"Window sweep on {len(json_files)} files\n")
        sweep_results = []
        aggregate = {}
        for jf in json_files:
            sweep = sweep_file(
                jf,
                trims=trims,
                small_windows=small_windows,
                large_windows=large_windows,
            )
            sweep_results.append(sweep)
            model = sweep["model"]
            target = sweep["target_layer"]
            best = sweep["best_config"]
            print(f"{model} target=L{target}")
            if not best:
                print("  no valid configs")
                continue
            for rank, cfg in enumerate(sweep["configs"][:max(1, args.sweep_top_k)], start=1):
                print(
                    f"  #{rank} trim={cfg['trim']} small={cfg['small_window']} large={cfg['large_window']} "
                    f"acc={cfg['correct']}/{cfg['n_tests']} ({cfg['accuracy']:.0%}) methods={cfg['method_counts']}"
                )
                key = (cfg["trim"], cfg["small_window"], cfg["large_window"])
                bucket = aggregate.setdefault(key, {"correct": 0, "n_tests": 0, "files": 0})
                bucket["correct"] += cfg["correct"]
                bucket["n_tests"] += cfg["n_tests"]
                bucket["files"] += 1
            print()

        leaderboard = []
        for (trim, small_window, large_window), totals in aggregate.items():
            n_tests = totals["n_tests"]
            accuracy = (totals["correct"] / n_tests) if n_tests else 0.0
            leaderboard.append({
                "trim": trim,
                "small_window": small_window,
                "large_window": large_window,
                "correct": totals["correct"],
                "n_tests": n_tests,
                "files": totals["files"],
                "accuracy": accuracy,
            })
        leaderboard.sort(key=lambda item: (item["accuracy"], item["correct"]), reverse=True)
        if leaderboard:
            print("Aggregate leaderboard")
            print("-" * 100)
            for cfg in leaderboard[:max(1, args.sweep_top_k)]:
                print(
                    f"  trim={cfg['trim']} small={cfg['small_window']} large={cfg['large_window']} "
                    f"acc={cfg['correct']}/{cfg['n_tests']} ({cfg['accuracy']:.0%}) files={cfg['files']}"
                )
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump({"files": sweep_results, "aggregate": leaderboard}, f, indent=2)
            print(f"\nResults saved to {args.json_out}")
        return

    # ----- Layer detection mode (default) -----
    print(f"Processing {len(json_files)} files...\n")
    hdr = f"{'Model':<35s} {'Tgt':>3} {'Acc':>8} {'N':>3}  {'Methods'}"
    print(hdr)
    print("-" * 100)

    all_results = []
    for jf in json_files:
        result = process_file(jf, args.trim)
        all_results.append(result)
        m = result["model"]
        if len(m) > 34:
            m = "…" + m[-33:]
        mc = ", ".join(f"{k}:{v}" for k, v in
                       sorted(result["method_counts"].items(), key=lambda x: -x[1]))
        flag = " ✓" if result["n_tests"] and result["accuracy"] >= 0.9 else (
            " ✗" if result["accuracy"] < 0.5 else " ~")
        target_label = (
            f"L{result['target_layer']:>2d}"
            if result["target_layer"] is not None else " n/a"
        )
        print(f"  {m:<33s} {target_label:>3s} "
              f"{result['correct']:>2}/{result['n_tests']:<2d} "
              f"{result['accuracy']:>4.0%}  {mc}{flag}")

    # Summary
    total_t = sum(r["n_tests"] for r in all_results)
    total_c = sum(r["correct"] for r in all_results)
    print("-" * 100)
    if total_t:
        print(f"  {'TOTAL':<33s}     {total_c:>2}/{total_t:<2d} "
              f"{total_c / total_t:.0%}")
    else:
        print(f"  {'TOTAL':<33s}     0/0  n/a")

    # Failures
    failures = [(r, t) for r in all_results for t in r["results"] if not t["hit"]]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for r, t in failures:
            short = r["model"].split("/")[-1]
            print(f"    {short}: test#{t['test_idx']} "
                  f"target=L{t['target']} detected=L{t['detected']} "
                  f"method={t['method']}")

    # Graphs
    if args.graphs:
        gdir = Path(args.graph_dir)
        print(f"\nGenerating graphs → {gdir}/")
        for r in all_results:
            plot_signal_profiles(r, gdir)
        plot_summary_table(all_results, gdir)
        plot_method_breakdown(all_results, gdir)

    # JSON output
    if args.json_out:
        # Strip non-serializable bits
        out_data = []
        for r in all_results:
            rd = {k: v for k, v in r.items() if k != "results"}
            rd["per_test"] = [
                {"test_idx": t["test_idx"], "detected": t["detected"],
                 "target": t["target"], "hit": t["hit"], "method": t["method"]}
                for t in r["results"]
            ]
            out_data.append(rd)
        with open(args.json_out, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nResults saved to {args.json_out}")


if __name__ == "__main__":
    main()
