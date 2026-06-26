"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Generic, Iterable, Mapping, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RegistryEntry:
    identifier: str
    description: str


class NamedRegistry(Generic[T]):
    def __init__(self, entries: Iterable[T], *, id_attribute: str = "identifier") -> None:
        self._id_attribute = id_attribute
        self._entries: dict[str, T] = {}
        for entry in entries:
            identifier = str(getattr(entry, id_attribute))
            if identifier in self._entries:
                raise ValueError(f"Duplicate registry identifier: {identifier}")
            self._entries[identifier] = entry

    def get(self, identifier: str) -> T:
        try:
            return self._entries[str(identifier)]
        except KeyError as exc:
            supported = ", ".join(self.identifiers())
            raise KeyError(f"Unknown identifier {identifier!r}. Supported: {supported}") from exc

    def identifiers(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def values(self) -> tuple[T, ...]:
        return tuple(self._entries.values())

    def __contains__(self, identifier: object) -> bool:
        return str(identifier) in self._entries


def load_object(path: str) -> Any:
    try:
        module_name, attribute = str(path).split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Object path must use 'module:attribute': {path!r}") from exc
    if not module_name or not attribute:
        raise ValueError(f"Object path must use 'module:attribute': {path!r}")

    value: Any = import_module(module_name)
    for name in attribute.split("."):
        value = getattr(value, name)
    return value


def model_family(model: str) -> str:
    from src.common.model_config import canonical_model_name

    identifier = canonical_model_name(model)
    if identifier.startswith("gpt2") or identifier == "gpt-j-6b":
        return "gpt"
    return "non-gpt"


def supports_model(entry: Any, model: str) -> bool:
    families = tuple(getattr(entry, "model_families", ("all",)))
    return "all" in families or model_family(model) in families


def resolve_preset_selection(
    presets: Mapping[str, Sequence[str]],
    registry: NamedRegistry[Any],
    preset: str,
    *,
    enabled: Sequence[str] = (),
    disabled: Sequence[str] = (),
    preset_label: str = "preset",
) -> tuple[str, ...]:
    try:
        selected = [str(identifier) for identifier in presets[str(preset)]]
    except KeyError as exc:
        supported = ", ".join(sorted(presets))
        raise ValueError(f"Unknown {preset_label} {preset!r}. Supported: {supported}") from exc

    for identifier in enabled:
        normalized = str(identifier)
        registry.get(normalized)
        if normalized not in selected:
            selected.append(normalized)

    disabled_set = {str(identifier) for identifier in disabled}
    for identifier in disabled_set:
        registry.get(identifier)

    return tuple(identifier for identifier in selected if identifier not in disabled_set)
