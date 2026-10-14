"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any
from omegaconf import DictConfig, OmegaConf
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
MODEL_CONFIG_DIR = CONFIG_ROOT / "model"
MODEL_BASE_CONFIG = CONFIG_ROOT / "model_base/default.yaml"


def model_config_dir() -> Path:
    """Allow a fleet process to use saved checkpoint configs outside the checkout."""
    return Path(os.environ.get("LATIUM_MODEL_CONFIG_DIR") or MODEL_CONFIG_DIR)


def sanitize_model_id(model_id: str) -> str:
    """Return the stable fleet slug body used in run names and filenames."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(model_id).replace("/", "_"))


def fleet_model_key(model_id: str) -> str:
    """Return the model key for a fleet HuggingFace model id."""
    return f"fleet_{sanitize_model_id(model_id)}"


def _iter_yaml_paths(config_dir: Path) -> list[Path]:
    return [path for path in sorted(config_dir.glob("*.yaml")) if path.name != "boilerplate.yaml"]


def _load_yaml(path: Path) -> DictConfig:
    return OmegaConf.load(path)


def _load_model_base(config_dir: Path) -> DictConfig:
    path = (
        MODEL_BASE_CONFIG
        if config_dir.resolve() == MODEL_CONFIG_DIR.resolve()
        else config_dir.parent / "model_base" / "default.yaml"
    )
    if not path.exists():
        return OmegaConf.create({})
    return _load_yaml(path)


def _load_model_yaml(path: Path) -> DictConfig:
    base = _load_model_base(path.parent)
    return OmegaConf.merge(base, _load_yaml(path))


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _name_match_priority(path: Path, model_name: str) -> tuple[int, int, str]:
    """Prefer the base config when several YAMLs share one HuggingFace name."""
    short_name = str(model_name).split("/")[-1].lower()
    stem = path.stem.lower()
    return (
        0 if stem == short_name else 1,
        len(stem),
        stem,
    )


def _resolve_yaml_match(model_name: str, config_dir: Path) -> tuple[Path, DictConfig] | None:
    model_key = str(model_name).strip()
    normalized_key = _normalize_key(model_key)
    for path in _iter_yaml_paths(config_dir):
        if _normalize_key(path.stem) == normalized_key:
            return path, _load_model_yaml(path)

    name_matches: list[tuple[Path, DictConfig]] = []
    for path in _iter_yaml_paths(config_dir):
        cfg = _load_yaml(path)
        if _normalize_key(getattr(cfg, "name", "")) == normalized_key:
            name_matches.append((path, cfg))
    if not name_matches:
        return None

    name_matches.sort(key=lambda item: _name_match_priority(item[0], model_key))
    path, _ = name_matches[0]
    return path, _load_model_yaml(path)

def canonical_model_name(model: Any, *, config_dir=None):
    raw = str(model or "unknown").strip()
    match = _resolve_yaml_match(raw, Path(config_dir) if config_dir is not None else model_config_dir())
    return match[0].stem if match is not None else raw


def load_model_config(model_name: str, *, config_dir=None) -> DictConfig:
    config_root = Path(config_dir) if config_dir is not None else model_config_dir()
    match = _resolve_yaml_match(str(model_name).strip(), config_root)
    if match is None:
        raise FileNotFoundError(f"No saved model configuration for {model_name!r} in {config_root}")
    return match[1]


def second_moment_basename(model_name: str, *, config_dir=None) -> str:
    cfg = load_model_config(model_name, config_dir=config_dir)
    explicit = str(getattr(cfg, "second_moment_path", "") or "").strip()
    return Path(explicit).name if explicit else f"{str(cfg.name).replace('/', '_')}_{int(cfg.layer)}"
