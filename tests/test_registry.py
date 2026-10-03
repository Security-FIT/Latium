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
from src.structural.analysis.registry import AnalysisSpec, _validated_registry
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


def test_gram_localization_is_explicit_and_single_checkpoint() -> None:
    capture = CAPTURES.get("gram-localization")
    analysis = ANALYSES.get("gram-localization")

    assert capture.captures_baseline is True
    assert capture.requires_baseline is False
    assert capture.requires_probe is False
    assert analysis.required_captures == ("gram-localization",)
    assert CAPTURE_PROFILES["gram-localization"] == ("gram-localization",)
    assert ANALYSIS_PRESETS["gram-localization"] == ("gram-localization",)
    assert "gram-localization" not in CAPTURE_PROFILES["paper"]
    assert "gram-localization" not in ANALYSIS_PRESETS["paper"]


def test_each_analysis_has_a_single_method_preset() -> None:
    for identifier in ANALYSES.identifiers():
        assert ANALYSIS_PRESETS[identifier] == (identifier,)


def test_ccs_composite_owns_all_capture_and_feature_requirements() -> None:
    analysis = ANALYSES.get("ccs-composite")
    plan = resolve_capture_plan(
        "none",
        analyses=("ccs-composite",),
        matrix_feature_set="rank1",
    )

    assert analysis.required_captures == ("matrix-features", "spectral")
    assert plan.names == ("matrix-features", "spectral")
    assert plan.matrix_features == analysis.required_matrix_features
    with pytest.raises(KeyError):
        ANALYSES.get("composite")


@pytest.mark.parametrize("identifier", ANALYSES.identifiers())
def test_analysis_selection_adds_its_own_capture_requirements(identifier: str) -> None:
    analysis = ANALYSES.get(identifier)
    plan = resolve_capture_plan("none", analyses=(identifier,))

    assert set(analysis.required_captures).issubset(plan.names)
    assert set(analysis.required_matrix_features).issubset(plan.matrix_features)


def test_analysis_only_capture_plan_uses_exact_declared_matrix_features() -> None:
    analysis = ANALYSES.get("rank1-blind")
    plan = resolve_capture_plan(
        "none",
        analyses=("rank1-blind",),
        matrix_feature_set="paper",
    )

    assert plan.names == ("matrix-features",)
    assert plan.matrix_features == analysis.required_matrix_features


def test_required_capture_cannot_be_explicitly_disabled() -> None:
    with pytest.raises(ValueError, match="requires disabled capture"):
        resolve_capture_plan(
            "none",
            disabled=("spectral",),
            analyses=("ccs-composite",),
        )


def test_renderer_selection_adds_its_capture_and_feature_requirements() -> None:
    renderer = RENDERERS.get("structural-artifact-grid")
    plan = resolve_capture_plan(
        "none",
        analyses=("rank1-blind",),
        renderers=("structural-artifact-grid",),
    )

    assert set(renderer.required_captures).issubset(plan.names)
    assert set(renderer.required_matrix_features).issubset(plan.matrix_features)


def test_capture_requirements_only_load_needed_model_state() -> None:
    assert required_weight_families(("gram-localization",)) == frozenset({"proj"})
    assert required_weight_families(("attention-features",)) == frozenset({"attention"})
    assert required_weight_families(("spectral", "attention-features")) == frozenset({"proj", "fc", "attention"})
    assert captures_require_probe(("gram-localization",)) is False
    assert captures_require_probe(("bottom-rank-tokens",)) is True


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
