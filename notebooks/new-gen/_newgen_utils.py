from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import re

import numpy as np
import pandas as pd


_SWEEP_SLUG_RE = re.compile(
    r"tk(?P<top_k>\d+)"
    r"_tf(?P<trim_first>[^_]+)"
    r"_tl(?P<trim_last>[^_]+)"
    r"_nl(?P<neighbor_layers>\d+)"
    r"_rw(?P<rolling_window>\d+)"
    r"_lw(?P<local_windows>[^_]+)"
)


RATE_KEYS = (
    "rome_rate",
    "blind_rate",
    "spectral_rate",
    "composite_rate",
    "rank1_blind_rate",
    "attention_rate",
    "symmetry_rate",
    "ipr_rate",
    "presence_rate",
)


_MODEL_CANONICAL_MAP = {
    "qwen/qwen3-4b": "qwen3-4b",
    "qwen3-4b": "qwen3-4b",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
}


def _normalize_match_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    for ch in ("/", "_", " "):
        text = text.replace(ch, "-")
    while "--" in text:
        text = text.replace("--", "-")
    return text


def _normalize_model_tokens(model_substrings: Optional[Sequence[str]]) -> List[str]:
    if not model_substrings:
        return []

    tokens: List[str] = []
    seen = set()
    for raw in model_substrings:
        token = _normalize_match_text(raw)
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _matches_any_token(text: Any, tokens: Sequence[str]) -> bool:
    if not tokens:
        return True
    norm = _normalize_match_text(text)
    return any(tok in norm for tok in tokens)


def candidate_result_dirs(
    root: Optional[Path] = None,
    search_dirs: Optional[Sequence[Any]] = None,
    include_outputs: bool = True,
) -> List[Path]:
    """Resolve directories to search for structural run artifacts."""
    repo_root = find_repo_root(root)

    defaults: List[Path] = [repo_root / "analysis_out"]
    if include_outputs:
        defaults.append(repo_root / "outputs")

    if search_dirs:
        raw_dirs: List[Path] = []
        for item in search_dirs:
            p = Path(item)
            if not p.is_absolute():
                p = repo_root / p
            raw_dirs.append(p.resolve())
    else:
        raw_dirs = defaults

    out: List[Path] = []
    seen = set()
    for p in raw_dirs:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.is_dir():
            out.append(p)
    return out


def _path_matches_models(path: Path, model_tokens: Sequence[str]) -> bool:
    if not model_tokens:
        return True

    if _matches_any_token(path.name, model_tokens):
        return True

    parent_chain = " ".join(part for part in path.parts[-4:])
    return _matches_any_token(parent_chain, model_tokens)


def _payload_matches_models(payload: dict, model_tokens: Sequence[str]) -> bool:
    if not model_tokens:
        return True

    model_raw = get_model_name(payload)
    model_canonical = canonical_model_name(model_raw)
    candidates = [
        model_raw,
        model_canonical,
        payload.get("metadata", {}).get("model"),
        payload.get("metadata", {}).get("model_name"),
    ]
    return any(_matches_any_token(c, model_tokens) for c in candidates)


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Find repository root by locating analysis_out directory."""
    root = (start or Path.cwd()).resolve()
    if (root / "analysis_out").exists():
        return root
    for p in root.parents:
        if (p / "analysis_out").exists():
            return p
    return root


def latest_structural_runs(
    root: Optional[Path] = None,
    max_files: int = 20,
    model_substrings: Optional[Sequence[str]] = None,
    search_dirs: Optional[Sequence[Any]] = None,
    include_outputs: bool = True,
    recursive: bool = True,
) -> List[Path]:
    """
    Return latest structural benchmark JSON files.

    By default this scans both analysis_out and outputs, recursively, so sweep
    artifacts saved in nested folders are discoverable from notebooks.
    """
    dirs = candidate_result_dirs(
        root=root,
        search_dirs=search_dirs,
        include_outputs=include_outputs,
    )
    if not dirs:
        return []

    files: List[Path] = []
    seen = set()
    pattern = "rome_structural_*.json"
    for d in dirs:
        iterator = d.rglob(pattern) if recursive else d.glob(pattern)
        for p in iterator:
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(p)

    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    model_tokens = _normalize_model_tokens(model_substrings)
    if model_tokens:
        kept: List[Path] = []
        for p in files:
            if _path_matches_models(p, model_tokens):
                kept.append(p)
                continue
            # Metadata fallback catches alias mismatches between filename and model id.
            try:
                payload = load_json(p)
            except Exception:
                continue
            if _payload_matches_models(payload, model_tokens):
                kept.append(p)
        files = kept

    return files[: max(1, int(max_files))]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_sweep_slug(sweep_slug: Optional[str]) -> Dict[str, Any]:
    """Parse structural sweep slug into structured fields."""
    if not sweep_slug:
        return {
            "spectral_top_k": None,
            "trim_first": None,
            "trim_last": None,
            "spectral_neighbor_layers": None,
            "spectral_rolling_window": None,
            "local_windows": None,
        }

    m = _SWEEP_SLUG_RE.search(str(sweep_slug))
    if not m:
        return {
            "spectral_top_k": None,
            "trim_first": None,
            "trim_last": None,
            "spectral_neighbor_layers": None,
            "spectral_rolling_window": None,
            "local_windows": None,
        }

    def _parse_optional_int(token: str) -> Optional[int]:
        t = str(token).strip().lower()
        if t in {"auto", "none", "default"}:
            return None
        try:
            return int(t)
        except (TypeError, ValueError):
            return None

    return {
        "spectral_top_k": _parse_optional_int(m.group("top_k")),
        "trim_first": _parse_optional_int(m.group("trim_first")),
        "trim_last": _parse_optional_int(m.group("trim_last")),
        "spectral_neighbor_layers": _parse_optional_int(m.group("neighbor_layers")),
        "spectral_rolling_window": _parse_optional_int(m.group("rolling_window")),
        "local_windows": m.group("local_windows"),
    }


def safe_layer_map(layer_map: Optional[Dict], layer: int, default=np.nan):
    """Read either int-keyed or str-keyed layer maps safely."""
    if not isinstance(layer_map, dict):
        return default
    if layer in layer_map:
        return layer_map[layer]
    s = str(layer)
    if s in layer_map:
        return layer_map[s]
    return default


def sorted_layers_from_map(layer_map: Dict) -> List[int]:
    layers = []
    for k in layer_map.keys():
        try:
            layers.append(int(k))
        except (TypeError, ValueError):
            continue
    return sorted(set(layers))


def get_model_name(payload: dict) -> str:
    return str(payload.get("metadata", {}).get("model", "unknown"))


def canonical_model_name(model: Any) -> str:
    """Normalize known model aliases to stable names for grouping/comparison."""
    raw = str(model or "unknown").strip()
    return _MODEL_CANONICAL_MAP.get(raw.lower(), raw)


def get_target_layer(payload: dict, case: Optional[dict] = None) -> Optional[int]:
    if case and isinstance(case, dict) and case.get("target_layer") is not None:
        try:
            return int(case.get("target_layer"))
        except (TypeError, ValueError):
            pass
    meta_target = payload.get("metadata", {}).get("target_layer")
    if meta_target is None:
        return None
    try:
        return int(meta_target)
    except (TypeError, ValueError):
        return None


def iter_valid_tests(payload: dict) -> Iterable[dict]:
    for case in payload.get("tests", []):
        if case.get("skipped", False):
            continue
        if case.get("error"):
            continue
        yield case


def _coerce_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def collect_layer_series_from_detection(
    payload: dict,
    detection_key: str,
    series_path: Sequence[str],
    include_baseline: bool = True,
) -> pd.DataFrame:
    """Collect per-layer values from one nested detection series into a long DataFrame."""
    rows = []
    model = get_model_name(payload)

    def _get_nested(obj, path):
        cur = obj
        for p in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(p)
        return cur

    if include_baseline:
        base = payload.get(f"baseline_{detection_key}", {})
        base_map = _get_nested(base, series_path)
        if isinstance(base_map, dict):
            for layer in sorted_layers_from_map(base_map):
                rows.append(
                    {
                        "source": "baseline",
                        "case_id": -1,
                        "model": model,
                        "layer": layer,
                        "value": float(safe_layer_map(base_map, layer, np.nan)),
                        "target_layer": np.nan,
                    }
                )

    for case in iter_valid_tests(payload):
        block = case.get(f"{detection_key}_detection", {})
        series_map = _get_nested(block, series_path)
        if not isinstance(series_map, dict):
            continue
        target_layer = get_target_layer(payload, case)
        for layer in sorted_layers_from_map(series_map):
            rows.append(
                {
                    "source": "test",
                    "case_id": case.get("case_id", -1),
                    "model": model,
                    "layer": layer,
                    "value": float(safe_layer_map(series_map, layer, np.nan)),
                    "target_layer": target_layer if target_layer is not None else np.nan,
                }
            )

    return pd.DataFrame(rows)


def collect_detection_predictions(
    payload: dict,
    detection_key: str,
    include_missing: bool = False,
) -> pd.DataFrame:
    rows = []
    model = get_model_name(payload)
    for case in iter_valid_tests(payload):
        block = case.get(f"{detection_key}_detection")
        if not isinstance(block, dict):
            if include_missing:
                rows.append(
                    {
                        "model": model,
                        "case_id": case.get("case_id", -1),
                        "target_layer": get_target_layer(payload, case),
                        "predicted_layer": np.nan,
                        "detection_score": np.nan,
                        "has_prediction": False,
                        "correct": np.nan,
                    }
                )
            continue

        pred = _coerce_int(block.get("anomalous_layer"))
        score = _coerce_float(block.get("detection_score", np.nan))
        target_layer = get_target_layer(payload, case)
        has_prediction = pred is not None
        correct = np.nan
        if has_prediction and target_layer is not None:
            correct = bool(pred == target_layer)

        if not include_missing and not has_prediction:
            continue

        rows.append(
            {
                "model": model,
                "case_id": case.get("case_id", -1),
                "target_layer": target_layer,
                "predicted_layer": pred,
                "detection_score": score,
                "has_prediction": has_prediction,
                "correct": correct,
            }
        )
    return pd.DataFrame(rows)


def pick_latest_per_model(files: Sequence[Path]) -> List[Path]:
    """Keep the latest file per model from an already mtime-sorted path list."""
    chosen = {}
    for p in files:
        name = p.name.lower()
        model_key = name
        if name.startswith("rome_structural_"):
            model_key = name[len("rome_structural_"):].rsplit("_", 2)[0]
        if model_key not in chosen:
            chosen[model_key] = p
    return list(chosen.values())


def pick_latest_per_model_sweep(files: Sequence[Path]) -> List[Path]:
    """Keep latest file per (model, sweep_slug) from an already mtime-sorted path list."""
    chosen: Dict[Tuple[str, str], Path] = {}
    for p in files:
        try:
            payload = load_json(p)
        except Exception:
            continue
        model = get_model_name(payload)
        sweep_slug = str(payload.get("metadata", {}).get("sweep_slug") or "single")
        key = (model, sweep_slug)
        if key not in chosen:
            chosen[key] = p
    return list(chosen.values())


def select_run_files(files: Sequence[Path], selection: str = "all") -> List[Path]:
    """
    Select structural run files by grouping strategy.

    selection values:
    - "all": keep all provided files
    - "latest_model": keep latest per model
    - "latest_model_sweep": keep latest per (model, sweep)
    """
    mode = str(selection or "all").strip().lower()
    if mode in {"latest_model", "model"}:
        return pick_latest_per_model(files)
    if mode in {"latest_model_sweep", "model_sweep", "sweep"}:
        return pick_latest_per_model_sweep(files)
    return list(files)


def load_experiment_df(
    root: Optional[Path] = None,
    max_files: int = 500,
    model_substrings: Optional[Sequence[str]] = None,
    selection: str = "all",
    include_rate_jumps: bool = True,
    rate_keys: Sequence[str] = RATE_KEYS,
    search_dirs: Optional[Sequence[Any]] = None,
    include_outputs: bool = True,
    recursive: bool = True,
) -> pd.DataFrame:
    """
    Convenience loader for sweep/run notebook analyses.

    This resolves files, applies optional latest selection, and returns the run
    summary DataFrame with optional run-over-run jump columns.
    """
    files = latest_structural_runs(
        root=root,
        max_files=max_files,
        model_substrings=model_substrings,
        search_dirs=search_dirs,
        include_outputs=include_outputs,
        recursive=recursive,
    )
    files = select_run_files(files, selection=selection)

    out = collect_run_summaries(files, rate_keys=rate_keys)
    if include_rate_jumps:
        out = add_rate_jumps(out, rate_keys=rate_keys)
    return out


def collect_run_summaries(paths: Sequence[Path], rate_keys: Sequence[str] = RATE_KEYS) -> pd.DataFrame:
    """Collect one-row-per-run summary table from structural benchmark JSON files."""
    rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            payload = load_json(p)
        except Exception:
            continue

        md = payload.get("metadata", {})
        summary = payload.get("summary", {})
        sweep_slug = md.get("sweep_slug")
        parsed = parse_sweep_slug(sweep_slug)

        model_raw = get_model_name(payload)

        row: Dict[str, Any] = {
            "file": p.name,
            "path": str(p),
            "model_raw": model_raw,
            "model": canonical_model_name(model_raw),
            "sweep_slug": sweep_slug or "single",
            "run_index": _coerce_int(md.get("run_index")),
            "run_ordinal": _coerce_int(md.get("run_ordinal")),
            "n_tests": _coerce_int(md.get("n_tests")),
            "successful": _coerce_int(summary.get("successful")),
            "total": _coerce_int(summary.get("total")),
            "file_mtime": pd.to_datetime(p.stat().st_mtime, unit="s", utc=True),
            **parsed,
        }

        for key in rate_keys:
            row[key] = _coerce_float(summary.get(key))

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def add_rate_jumps(
    df: pd.DataFrame,
    rate_keys: Sequence[str] = RATE_KEYS,
    group_cols: Sequence[str] = ("model", "sweep_slug"),
    order_cols: Sequence[str] = ("file_mtime", "run_ordinal", "run_index", "file"),
) -> pd.DataFrame:
    """Compute per-group run-over-run deltas for detector rate columns."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()
    valid_group_cols = [c for c in group_cols if c in out.columns]
    valid_order_cols = [c for c in order_cols if c in out.columns]

    if valid_group_cols and valid_order_cols:
        out = out.sort_values(valid_group_cols + valid_order_cols)
    elif valid_order_cols:
        out = out.sort_values(valid_order_cols)

    for key in rate_keys:
        if key not in out.columns:
            continue
        jump_col = f"{key}_jump"
        if valid_group_cols:
            out[jump_col] = out.groupby(valid_group_cols)[key].diff()
        else:
            out[jump_col] = out[key].diff()

    return out


def aggregate_sweep_outcomes(
    df: pd.DataFrame,
    by: Sequence[str] = ("model", "local_windows"),
    rate_keys: Sequence[str] = RATE_KEYS,
) -> pd.DataFrame:
    """Aggregate sweep outcomes for easy local-window/top-k comparisons."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "local_windows" not in out.columns and "sweep_slug" in out.columns:
        out["local_windows"] = out["sweep_slug"].map(
            lambda s: parse_sweep_slug(s).get("local_windows") if pd.notna(s) else None
        )

    keys = [k for k in by if k in out.columns]
    if not keys:
        return pd.DataFrame()

    agg_spec: Dict[str, Tuple[str, str]] = {"runs": ("file", "count")}
    if "total" in out.columns:
        agg_spec["total_tests"] = ("total", "sum")
    if "successful" in out.columns:
        agg_spec["successful"] = ("successful", "sum")

    for key in rate_keys:
        if key in out.columns:
            agg_spec[key] = (key, "mean")

    result = out.groupby(keys, as_index=False).agg(**agg_spec)
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result


def flatten_metric_table(
    payload: dict,
    detection_key: str,
    metric_paths: Dict[str, Sequence[str]],
) -> pd.DataFrame:
    """Flatten multiple per-layer metric paths into one table for plotting."""
    tables = []
    for metric_name, path in metric_paths.items():
        df = collect_layer_series_from_detection(
            payload,
            detection_key=detection_key,
            series_path=path,
            include_baseline=True,
        )
        if df.empty:
            continue
        df["metric"] = metric_name
        tables.append(df)
    if not tables:
        return pd.DataFrame()
    out = pd.concat(tables, ignore_index=True)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out
