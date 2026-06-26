"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
MODEL_CONFIG_DIR = CONFIG_ROOT / "model"
MODEL_FLEET_CONFIG = CONFIG_ROOT / "model_fleet" / "default.yaml"
MODEL_BASE_CONFIG = CONFIG_ROOT / "model_base" / "default.yaml"


def canonical_model_name(
    model: Any,
    *,
    config_dir: Path | str = MODEL_CONFIG_DIR,
    fleet_manifest: Path | str | None = None,
) -> str:
    raw = str(model or "unknown").strip()
    match = _resolve_yaml_match(raw, Path(config_dir))
    if match is not None:
        return match[0].stem

    fleet_model_id = _resolve_fleet_model_id(raw, Path(config_dir), fleet_manifest)
    if fleet_model_id:
        return fleet_model_key(fleet_model_id)
    return raw


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


def _load_model_fleet_config(config_dir: Path) -> DictConfig:
    path = (
        MODEL_FLEET_CONFIG
        if config_dir.resolve() == MODEL_CONFIG_DIR.resolve()
        else config_dir.parent / "model_fleet" / "default.yaml"
    )
    if not path.exists():
        return OmegaConf.create({})
    return _load_yaml(path)


def _resolve_config_path(raw_path: str, *, base_dir: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _default_fleet_manifest(config_dir: Path) -> Path | None:
    cfg = _load_model_fleet_config(config_dir)
    raw = str(getattr(cfg, "manifest", "") or "").strip()
    if not raw:
        return None
    return _resolve_config_path(raw, base_dir=config_dir.parent)


def _default_fleet_base_model(config_dir: Path) -> str | None:
    cfg = _load_model_fleet_config(config_dir)
    raw = str(getattr(cfg, "base_model", "") or "").strip()
    return raw or None


def _load_fleet_models(fleet_manifest: Path) -> dict[str, str]:
    if not fleet_manifest.exists():
        return {}

    payload = json.loads(fleet_manifest.read_text(encoding="utf-8"))
    models: dict[str, str] = {}
    for entry in payload.get("models", []):
        model_id = str(entry.get("model_id", "")).strip()
        if not model_id:
            continue
        slug = fleet_model_key(model_id)
        models[model_id] = model_id
        models[model_id.lower()] = model_id
        models[slug] = model_id
        models[slug.lower()] = model_id
    return models


def _copy_with_model_name(base_cfg: DictConfig, model_id: str) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    cfg.name = model_id
    return cfg


def _resolve_fleet_model_id(
    model_name: str,
    config_dir: Path,
    fleet_manifest: Path | str | None,
) -> str | None:
    manifest_path = Path(fleet_manifest) if fleet_manifest is not None else _default_fleet_manifest(config_dir)
    if manifest_path is None:
        return None
    fleet_models = _load_fleet_models(manifest_path)
    model_key = str(model_name).strip()
    return fleet_models.get(model_key) or fleet_models.get(model_key.lower())


def _available_message(config_dir: Path, fleet_manifest: Path | None) -> str:
    yaml_keys = [path.stem for path in _iter_yaml_paths(config_dir)]
    suffix = ""
    if fleet_manifest is not None:
        fleet_models = _load_fleet_models(fleet_manifest)
        fleet_count = len({value for value in fleet_models.values()})
        suffix = f"; plus {fleet_count} fleet models from {fleet_manifest.name}" if fleet_count else ""
    return ", ".join(yaml_keys) + suffix


def load_model_config(
    model_name: str,
    *,
    config_dir: Path | str = MODEL_CONFIG_DIR,
    fleet_manifest: Path | str | None = None,
    fleet_base_model_key: str | None = None,
) -> DictConfig:
    """Resolve a model config by YAML key, YAML HF name, or fleet manifest entry.

    Fleet entries reuse the base model declared by Hydra config
    ``model_fleet/default.yaml`` and only override ``model.name`` with the
    selected HuggingFace id.
    """
    model_key = str(model_name).strip()
    config_root = Path(config_dir)
    match = _resolve_yaml_match(model_key, config_root)
    if match is not None:
        return match[1]

    manifest_path = Path(fleet_manifest) if fleet_manifest is not None else _default_fleet_manifest(config_root)
    fleet_model_id = _resolve_fleet_model_id(model_key, config_root, manifest_path)
    if fleet_model_id:
        fleet_base_model_key = fleet_base_model_key or _default_fleet_base_model(config_root)
        if not fleet_base_model_key:
            raise FileNotFoundError("Fleet model requested, but model_fleet.base_model is not configured")
        base_path = config_root / f"{fleet_base_model_key}.yaml"
        if not base_path.exists():
            raise FileNotFoundError(f"Fleet base config not found: {base_path}")
        return _copy_with_model_name(_load_model_yaml(base_path), fleet_model_id)

    raise FileNotFoundError(
        f"No config for '{model_name}'. Available: {_available_message(config_root, manifest_path)}"
    )


def second_moment_basename(
    model_name: str,
    *,
    config_dir: Path | str = MODEL_CONFIG_DIR,
    fleet_manifest: Path | str | None = None,
) -> str:
    """Return the expected covariance basename for a resolved model config."""
    cfg = load_model_config(model_name, config_dir=config_dir, fleet_manifest=fleet_manifest)
    explicit = str(getattr(cfg, "second_moment_path", "") or "").strip()
    if explicit:
        return Path(explicit).name
    return f"{str(cfg.name).replace('/', '_')}_{int(cfg.layer)}"


__all__ = [
    "MODEL_FLEET_CONFIG",
    "MODEL_BASE_CONFIG",
    "MODEL_CONFIG_DIR",
    "canonical_model_name",
    "fleet_model_key",
    "load_model_config",
    "sanitize_model_id",
    "second_moment_basename",
]
