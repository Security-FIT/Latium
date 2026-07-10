"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from src.editing.base import EditMethod
from src.registry import NamedRegistry, RegistryEntry, load_object


@dataclass(frozen=True)
class EditMethodSpec(RegistryEntry):
    factory: str = ""


EDIT_METHOD_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "edit_method"


def _load_edit_method_specs(config_dir: Path = EDIT_METHOD_CONFIG_DIR) -> list[EditMethodSpec]:
    specs: list[EditMethodSpec] = []
    for path in sorted(config_dir.glob("*.yaml")):
        cfg = OmegaConf.load(path)
        specs.append(
            EditMethodSpec(
                identifier=str(cfg.identifier),
                description=str(cfg.description),
                factory=str(cfg.factory),
            )
        )
    return specs


EDIT_METHODS = NamedRegistry(_load_edit_method_specs())


def get_edit_method(identifier: str) -> EditMethod:
    spec = EDIT_METHODS.get(identifier)
    return load_object(spec.factory)()
