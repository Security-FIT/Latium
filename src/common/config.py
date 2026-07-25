"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence as SequenceABC
from typing import Any, Sequence


def plain(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        return value

    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    try:
        return getattr(config, key)
    except Exception:
        return default


def mapping_section(root: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = root.get(name)
    return dict(value) if isinstance(value, Mapping) else {}


def section_value(
    root: Mapping[str, Any],
    section: Mapping[str, Any],
    key: str,
    flat_key: str,
    default: Any = None,
) -> Any:
    if key in section:
        return section[key]
    if flat_key in root:
        return root[flat_key]
    return default


def dict_section(root: Mapping[str, Any], name: str) -> dict[str, dict[str, Any]]:
    value = root.get(name)
    if not isinstance(value, Mapping):
        return {}
    return {str(key): dict(item) for key, item in value.items() if isinstance(item, Mapping)}


def is_sequence(value: object) -> bool:
    return isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray))


def string_list(value: Any, default: Sequence[str] = ()) -> list[str]:
    raw = plain(value)
    if raw is None:
        return [str(item) for item in default]
    if isinstance(raw, str):
        parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
        return parts or [str(item) for item in default]
    if is_sequence(raw):
        out: list[str] = []
        for item in raw:
            if item is None:
                continue
            out.extend(string_list(item))
        return out or [str(item) for item in default]
    return [str(raw)]


def optional_int(value: Any) -> int | None:
    raw = plain(value)
    if raw is None:
        return None
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in {"", "none", "auto", "default"}:
            return None
        return int(token)
    return int(raw)


def optional_str(value: Any) -> str | None:
    raw = plain(value)
    if raw in (None, ""):
        return None
    return str(raw)


def strict_bool(value: Any, *, name: str = "value") -> bool:
    """Parse a boolean without treating arbitrary non-empty strings as true."""
    raw = plain(value)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int) and raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


__all__ = [
    "dict_section",
    "get_config_value",
    "is_sequence",
    "mapping_section",
    "optional_int",
    "optional_str",
    "plain",
    "section_value",
    "strict_bool",
    "string_list",
]
