"""Utilities for paper_graphs — stripped to only what paper_graphs_support.py needs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import json

import numpy as np


# HuggingFace ID -> short name used in configs / filenames
_MODEL_CANONICAL_MAP = {
    "eleutherai/gpt-j-6b": "gpt-j-6b",
    "nousresearch/llama-2-7b-hf": "llama2-7b",
    "qwen/qwen3-4b": "qwen3-4b",
    "qwen/qwen3-8b": "qwen3-8b",
    "deepseek-ai/deepseek-llm-7b-base": "deepseek-7b-base",
    "facebook/opt-6.7b": "opt-6.7b",
    "ibm-granite/granite-4.0-micro": "granite4-micro",
    "mistralai/mistral-7b-v0.1": "mistral-7b-v0.1",
    "mistralai/mistral-7b-v0.3": "mistral-7b-v0.3",
    "tiiuae/falcon-7b": "falcon-7b",
    "gpt2-large": "gpt2-large",
    "gpt2-medium": "gpt2-medium",
    "gpt2-xl": "gpt2-xl",
}


def canonical_model_name(model: Any) -> str:
    """Normalize HuggingFace model IDs to short stable names."""
    raw = str(model or "unknown").strip()
    return _MODEL_CANONICAL_MAP.get(raw.lower(), raw)


def get_model_name(payload: dict) -> str:
    return str(payload.get("metadata", {}).get("model", "unknown"))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    for k in layer_map:
        try:
            layers.append(int(k))
        except (TypeError, ValueError):
            continue
    return sorted(set(layers))


def get_target_layer(payload: dict, case: Optional[dict] = None) -> Optional[int]:
    if case and isinstance(case, dict) and case.get("target_layer") is not None:
        try:
            return int(case["target_layer"])
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
        if case.get("skipped", False) or case.get("error"):
            continue
        yield case


def find_repo_root(start: Optional[Path] = None) -> Path:
    root = (start or Path.cwd()).resolve()
    for p in [root, *root.parents]:
        if (p / "ultrasupertest").exists() or (p / "analysis_out").exists():
            return p
    return root


def _normalize(text: Any) -> str:
    t = str(text or "").strip().lower()
    for ch in "/_ ":
        t = t.replace(ch, "-")
    while "--" in t:
        t = t.replace("--", "-")
    return t


def _matches_model(text: Any, tokens: Sequence[str]) -> bool:
    if not tokens:
        return True
    norm = _normalize(text)
    return any(tok in norm for tok in tokens)


def latest_structural_runs(
    root: Optional[Path] = None,
    max_files: int = 20,
    model_substrings: Optional[Sequence[str]] = None,
    search_dirs: Optional[Sequence[Any]] = None,
    include_outputs: bool = True,
    recursive: bool = True,
) -> List[Path]:
    """Return latest structural benchmark JSON files, newest first."""
    repo_root = find_repo_root(root)

    if search_dirs:
        dirs = [Path(d) if Path(d).is_absolute() else repo_root / d for d in search_dirs]
    else:
        dirs = [repo_root / "analysis_out"]
        if include_outputs:
            dirs.append(repo_root / "outputs")
    dirs = [d.resolve() for d in dirs if d.exists() and d.is_dir()]

    seen: set[str] = set()
    files: List[Path] = []
    for d in dirs:
        for p in (d.rglob if recursive else d.glob)("rome_structural_*.json"):
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                files.append(p)

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    tokens = [_normalize(s) for s in (model_substrings or []) if s]
    if tokens:
        kept: List[Path] = []
        for p in files:
            if _matches_model(p.name, tokens):
                kept.append(p)
                continue
            try:
                pl = load_json(p)
            except Exception:
                continue
            name = get_model_name(pl)
            if _matches_model(name, tokens) or _matches_model(canonical_model_name(name), tokens):
                kept.append(p)
        files = kept

    return files[:max(1, int(max_files))]


def select_run_files(files: Sequence[Path], selection: str = "all") -> List[Path]:
    """Keep all files, or latest per model."""
    mode = str(selection or "all").strip().lower()
    if mode in {"latest_model", "model"}:
        chosen: Dict[str, Path] = {}
        for p in files:
            name = p.name.lower()
            model_key = name[len("rome_structural_"):].rsplit("_", 2)[0] if name.startswith("rome_structural_") else name
            chosen.setdefault(model_key, p)
        return list(chosen.values())
    return list(files)
