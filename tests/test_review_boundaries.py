"""Regressions for configuration coercion and localizer isolation."""

import pytest
import torch

from src.structural.config import AnalysisVariantConfig, StructuralBenchmarkConfig
from src.structural.planning import analysis_variant_settings
from src.structural.detectors.rome_layer_localizer import detect_from_profiles, score_layer


@pytest.mark.parametrize("value", [True, 3.9, "3"])
@pytest.mark.parametrize("factory", [
    lambda value: StructuralBenchmarkConfig(n_tests=value),
    lambda value: StructuralBenchmarkConfig(local_windows=[value]),
    lambda value: AnalysisVariantConfig(spectral_top_k=value),
    lambda value: AnalysisVariantConfig(trim_first=value),
    lambda value: analysis_variant_settings({"analysis": {"variants": {"spectral_top_k": value}}}),
    lambda value: analysis_variant_settings({"analysis": {"variants": {"sweep": {"max_configs": value}}}}),
])
def test_configuration_rejects_lossy_integer_coercion(factory, value):
    with pytest.raises(TypeError, match="integer"):
        factory(value)


def test_malformed_method_config_is_not_silently_dropped():
    with pytest.raises(TypeError, match="analysis_method_configs"):
        StructuralBenchmarkConfig(analysis_method_configs={"spectral": "invalid"})


def test_empty_window_sweep_is_not_replaced_with_defaults():
    with pytest.raises(ValueError, match="empty"):
        analysis_variant_settings({"analysis": {"variants": {"sweep": {"local_window_sets": []}}}})


def test_null_sweep_uses_configured_base_not_global_default():
    result = analysis_variant_settings({
        "analysis": {"variants": {"spectral_top_k": 7, "sweep": {"spectral_top_k": None}}},
    })
    assert result["analysis_variants"][0].spectral_top_k == 7


def test_localizer_requires_untrimmed_layer_metadata():
    profiles = {str(layer): {"diagonal_relative": float(layer)} for layer in range(1, 5)}
    with pytest.raises(ValueError, match="full capture layer list"):
        detect_from_profiles(profiles)
    result = detect_from_profiles(profiles, layers=list(range(6)))
    assert result["localization"]["selected_layer"] == 4


def test_localizer_preserves_cpu_rng_and_does_not_seed_all_cuda_devices(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Localizer must not seed unrelated CUDA generators")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    state = torch.random.get_rng_state().clone()
    score = score_layer(torch.diag(torch.tensor([2., 1., 1.])), torch.eye(3), layer=1)
    assert score >= 0
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize("failure_stage", ["weights", "capture"])
def test_post_edit_failure_records_each_case_once(tmp_path, monkeypatch, failure_stage):
    from types import SimpleNamespace
    from src.editing.base import EditOutcome
    from src.results import ArtifactWriter, RunLayout
    from src.structural.config import ModelRunPlan
    from src.structural.execution import edit_execution as module

    saved_execution = []
    saved_captures = []
    restored = []
    monkeypatch.setattr(module, "write_execution", lambda *args, **kwargs: (
        saved_execution.extend(kwargs["cases"]) or {"artifact_id": "execution", "content_hash": "hash"}
    ))
    monkeypatch.setattr(module, "write_capture", lambda *args, **kwargs: saved_captures.extend(kwargs["cases"]))
    monkeypatch.setattr(module, "capture_inputs", lambda *args: [])
    monkeypatch.setattr(module, "capture_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "restore", lambda *args: restored.append(True))

    def fail(*args, **kwargs):
        raise RuntimeError("capture preparation failed")

    if failure_stage == "weights":
        monkeypatch.setattr(module, "modified_weights", fail)
    method = SimpleNamespace(apply=lambda *args: EditOutcome(), evaluate=lambda *args: {})
    result = module.run_edit_method(
        writer=ArtifactWriter(tmp_path, run_id="run"),
        layout=RunLayout(tmp_path),
        config=StructuralBenchmarkConfig(),
        plan=ModelRunPlan("gpt2-large", "run", "plan", 1, 0, 0),
        model="gpt2-large",
        handler=SimpleNamespace(_layer=0, num_of_layers=3),
        test_cases=[{"case_id": "case"}],
        edit_method_name="rome",
        capture_names=("matrix-features", "spectral"),
        options={},
        baseline_records={},
        baseline_proj={},
        baseline_fc=None,
        baseline_attention={},
        proj_template="layer.{}",
        fc_template=None,
        method_loader=lambda _: method,
        capture_case_fn=lambda name, context, **kwargs: (
            fail() if name == "spectral" else {"case_id": "case", "status": "complete", "data": {}}
        ),
    )
    assert result["cases"] == 1
    assert [case["status"] for case in saved_execution] == ["error"]
    assert [case["status"] for case in saved_captures] == ["unavailable", "unavailable"]
    assert restored == [True]
