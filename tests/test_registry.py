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
from src.structural.analysis.registry import AnalysisSpec, _validated_registry
from src.structural.analysis.registry import ANALYSES, ANALYSIS_PRESETS
from src.structural.capture.registry import (
    CAPTURES,
    CAPTURE_PROFILES,
    required_weight_families,
)
from src.graphs.registry import RENDERERS, RENDERER_PRESETS


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


def test_matrix_anomaly_ids_replace_old_experimental_names() -> None:
    assert CAPTURES.get("matrix-anomaly-features").producer == (
        "src.structural.capture.producers:capture_matrix_anomaly_features"
    )
    assert ANALYSES.get("matrix-anomaly").required_captures == ("matrix-anomaly-features",)
    assert "matrix-anomaly-features" in CAPTURE_PROFILES["full"]
    assert "matrix-anomaly" in ANALYSIS_PRESETS["full"]

    old_label = "no" + "vel"
    with pytest.raises(KeyError):
        CAPTURES.get(f"{old_label}-features")
    with pytest.raises(KeyError):
        ANALYSES.get(old_label)


def test_rome_presence_presets_expose_all_training_free_variants() -> None:
    assert CAPTURE_PROFILES["rome-presence"] == ("weighted-spectrum", "rome-update")
    assert ANALYSIS_PRESETS["rome-presence"] == (
        "weighted-spectrum",
        "rome-presence-blind-peak",
        "rome-presence-blind-footprint",
        "rome-presence-delta",
    )
    assert ANALYSES.get("rome-presence-blind-peak").model_families == ("all",)
    assert ANALYSES.get("rome-presence-blind-footprint").model_families == ("all",)
    assert ANALYSES.get("rome-presence-delta").model_families == ("all",)
    assert RENDERER_PRESETS["rome-presence"] == (
        "rome-detector-explainer",
        "rome-success",
        "detector-window",
    )
    renderer = RENDERERS.get("rome-detector-explainer")
    assert renderer.required_analyses == ANALYSIS_PRESETS["rome-presence"]
    assert renderer.model_families == ("all",)


def test_detection_presets_select_only_current_and_spectral_math() -> None:
    assert CAPTURE_PROFILES["detection"] == ("weighted-spectrum", "spectral")
    assert ANALYSIS_PRESETS["detection"] == ("weighted-spectrum", "spectral")
    assert required_weight_families(("weighted-spectrum",)) == ("proj",)
    assert required_weight_families(CAPTURE_PROFILES["detection"]) == ("proj", "fc")
    assert required_weight_families(("attention-features",)) == ("attention",)


def test_analysis_registry_rejects_invalid_variant_fields() -> None:
    with pytest.raises(ValueError, match="unknown variant field"):
        _validated_registry(
            [
                AnalysisSpec(
                    "bad-source",
                    "Bad",
                    "detection",
                    "src.structural.analysis.detector_methods:analyze_blind",
                    ("matrix-features",),
                    ("trim_first",),
                    (("missing_source", "trim_first"),),
                )
            ]
        )

    with pytest.raises(ValueError, match="unknown config field"):
        _validated_registry(
            [
                AnalysisSpec(
                    "bad-target",
                    "Bad",
                    "detection",
                    "src.structural.analysis.detector_methods:analyze_blind",
                    ("matrix-features",),
                    ("trim_first",),
                    (("trim_first", "missing_target"),),
                )
            ]
        )
