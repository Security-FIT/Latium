from __future__ import annotations

import json
import re
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG_DIR = Path(__file__).resolve().parent / "config" / "model"
DEFAULT_FLEET_MANIFEST = REPO_ROOT / "finetuned_qwen3_8b_fleet.json"
FLEET_BASE_MODEL_KEY = "qwen3-8b"


def sanitize_model_id(model_id: str) -> str:
    """Return the stable fleet slug body used in run names and filenames."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(model_id).replace("/", "_"))


def fleet_model_key(model_id: str) -> str:
    """Return the model key for a fleet HuggingFace model id."""
    return f"fleet_{sanitize_model_id(model_id)}"


def _iter_yaml_paths(config_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(config_dir.glob("*.yaml"))
        if path.name != "boilerplate.yaml"
    ]


def _load_yaml(path: Path) -> DictConfig:
    return OmegaConf.load(path)


def _name_match_priority(path: Path, model_name: str) -> tuple[int, int, str]:
    """Prefer the canonical/base config when several YAMLs share one HF name."""
    short_name = str(model_name).split("/")[-1].lower()
    stem = path.stem.lower()
    return (
        0 if stem == short_name else 1,
        len(stem),
        stem,
    )


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


def _available_message(config_dir: Path, fleet_manifest: Path) -> str:
    yaml_keys = [path.stem for path in _iter_yaml_paths(config_dir)]
    fleet_models = _load_fleet_models(fleet_manifest)
    fleet_count = len({value for value in fleet_models.values()})
    suffix = f"; plus {fleet_count} fleet models from {fleet_manifest.name}" if fleet_count else ""
    return ", ".join(yaml_keys) + suffix


def load_model_config(
    model_name: str,
    *,
    config_dir: Path | str = DEFAULT_MODEL_CONFIG_DIR,
    fleet_manifest: Path | str = DEFAULT_FLEET_MANIFEST,
    fleet_base_model_key: str = FLEET_BASE_MODEL_KEY,
) -> DictConfig:
    """Resolve a model config by YAML key, YAML HF name, or fleet manifest entry.

    Fleet entries reuse the base Qwen3-8B config and only override ``name`` with
    the selected HuggingFace id, so covariance/cache fields stay tied to the
    original Qwen3-8B baseline.
    """
    model_key = str(model_name).strip()
    config_root = Path(config_dir)
    manifest_path = Path(fleet_manifest)

    yaml_path = config_root / f"{model_key}.yaml"
    if yaml_path.exists():
        return _load_yaml(yaml_path)

    name_matches: list[tuple[Path, DictConfig]] = []
    for path in _iter_yaml_paths(config_root):
        cfg = _load_yaml(path)
        if str(getattr(cfg, "name", "")).strip() == model_key:
            name_matches.append((path, cfg))
    if name_matches:
        name_matches.sort(key=lambda item: _name_match_priority(item[0], model_key))
        return name_matches[0][1]

    fleet_models = _load_fleet_models(manifest_path)
    fleet_model_id = fleet_models.get(model_key) or fleet_models.get(model_key.lower())
    if fleet_model_id:
        base_path = config_root / f"{fleet_base_model_key}.yaml"
        if not base_path.exists():
            raise FileNotFoundError(f"Fleet base config not found: {base_path}")
        return _copy_with_model_name(_load_yaml(base_path), fleet_model_id)

    raise FileNotFoundError(f"No config for '{model_name}'. Available: {_available_message(config_root, manifest_path)}")


def second_moment_basename(
    model_name: str,
    *,
    config_dir: Path | str = DEFAULT_MODEL_CONFIG_DIR,
    fleet_manifest: Path | str = DEFAULT_FLEET_MANIFEST,
) -> str:
    """Return the expected covariance basename for a resolved model config."""
    cfg = load_model_config(model_name, config_dir=config_dir, fleet_manifest=fleet_manifest)
    explicit = str(getattr(cfg, "second_moment_path", "") or "").strip()
    if explicit:
        return Path(explicit).name
    return f"{str(cfg.name).replace('/', '_')}_{int(cfg.layer)}"


def load_model_config_from_path_or_key(config_or_model: str | Path) -> DictConfig:
    """Compatibility helper for callers migrating from explicit YAML paths."""
    path = Path(config_or_model)
    if path.exists():
        return _load_yaml(path)
    return load_model_config(str(config_or_model))


__all__ = [
    "DEFAULT_FLEET_MANIFEST",
    "DEFAULT_MODEL_CONFIG_DIR",
    "FLEET_BASE_MODEL_KEY",
    "fleet_model_key",
    "load_model_config",
    "load_model_config_from_path_or_key",
    "sanitize_model_id",
    "second_moment_basename",
]
