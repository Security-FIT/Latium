"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import pytest

from src.registry import NamedRegistry, RegistryEntry, load_object, resolve_preset_selection
from src.editing.registry import EDIT_METHODS
from src.graphs.registry import RENDERERS
from src.structural.analysis.registry import ANALYSES, ANALYSIS_PRESETS
from src.structural.capture.registry import (
    CAPTURES,
    CAPTURE_PROFILES,
    captures_require_probe,
    required_weight_families,
    resolve_capture_plan,
)


def test_resolve_preset_selection_applies_enabled_and_disabled_identifiers() -> None:
    registry = NamedRegistry(
        [
            RegistryEntry("alpha", "Alpha"),
            RegistryEntry("beta", "Beta"),
            RegistryEntry("gamma", "Gamma"),
        ]
    )

    resolved = resolve_preset_selection(
        {"default": ("alpha", "beta")},
        registry,
        "default",
        enabled=("beta", "gamma"),
        disabled=("alpha",),
        preset_label="test preset",
    )

    assert resolved == ("beta", "gamma")


def test_resolve_preset_selection_validates_unknown_names() -> None:
    registry = NamedRegistry([RegistryEntry("known", "Known")])

    with pytest.raises(ValueError, match="Unknown test preset"):
        resolve_preset_selection({}, registry, "missing", preset_label="test preset")
    with pytest.raises(KeyError, match="Unknown identifier"):
        resolve_preset_selection(
            {"default": ("known",)},
            registry,
            "default",
            enabled=("missing",),
        )


def test_load_object_resolves_module_attribute_paths() -> None:
    assert load_object("src.registry:NamedRegistry") is NamedRegistry


def test_load_object_rejects_malformed_paths() -> None:
    with pytest.raises(ValueError, match="module:attribute"):
        load_object("src.registry.NamedRegistry")


def test_edit_methods_are_loaded_from_config() -> None:
    spec = EDIT_METHODS.get("rome")

    assert spec.factory == "src.editing.rome:RomeEditMethod"




def test_gram_localization_is_explicit_and_single_checkpoint() -> None:
    capture = CAPTURES.get("gram-localization")
    analysis = ANALYSES.get("gram-localization")

    assert capture.captures_baseline is True
    assert capture.requires_baseline is False
    assert capture.requires_probe is False
    assert analysis.required_captures == ("gram-localization",)
    assert CAPTURE_PROFILES["gram-localization"] == ("gram-localization",)
    assert ANALYSIS_PRESETS["gram-localization"] == ("gram-localization",)
    assert "paper" not in CAPTURE_PROFILES
    assert "paper" not in ANALYSIS_PRESETS


def test_each_analysis_has_a_single_method_preset() -> None:
    for identifier in ANALYSES.identifiers():
        assert ANALYSIS_PRESETS[identifier] == (identifier,)








def test_required_capture_cannot_be_explicitly_disabled() -> None:
    with pytest.raises(ValueError, match="requires disabled capture"):
        resolve_capture_plan(
            "none",
            disabled=("gram-localization",),
            analyses=("gram-localization",),
        )
