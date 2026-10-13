"""Exactness and work reduction for checkpoint-scoped Gram reuse."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.structural.detectors import rome_layer_localizer as gram


def weights(count=36, dtype=torch.float32, transpose=False):
    generator = torch.Generator().manual_seed(71)
    matrices = {i: torch.randn(8, 24, generator=generator).to(dtype) for i in range(count)}
    return {i: matrix.T if transpose else matrix for i, matrix in matrices.items()}


@pytest.mark.parametrize("count", [36, 48])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("transpose", [False, True])
def test_single_edit_matches_full_profile_and_recomputes_only_neighbors(monkeypatch, count, dtype, transpose):
    baseline = weights(count, dtype, transpose)
    cache = gram.GramProfileCache(baseline, edit_layer=10)
    expected_baseline = gram.profile_weights(baseline)
    assert gram.profile_weights(baseline, cache=cache) == expected_baseline
    original_gram, original_score = gram.hidden_gram, gram.score_layer

    for strength in (0.1, 0.3):
        edited = dict(baseline)
        edited[10] = (baseline[10].float() + strength).to(dtype)
        expected = gram.profile_weights(edited)
        gram_calls, score_calls = [], []

        def counted_gram(weight, **kwargs):
            gram_calls.append(weight)
            return original_gram(weight, **kwargs)

        def counted_score(current, reference, *, layer):
            score_calls.append(layer)
            return original_score(current, reference, layer=layer)

        with monkeypatch.context() as patch:
            patch.setattr(gram, "hidden_gram", counted_gram)
            patch.setattr(gram, "score_layer", counted_score)
            rng = torch.random.get_rng_state().clone()
            result = gram.profile_weights(edited, cache=cache)
            assert torch.equal(torch.random.get_rng_state(), rng)
            assert len(gram_calls) == 1 and gram_calls[0] is edited[10]
            assert score_calls == [9, 10, 11]
            assert gram.profile_weights(baseline, cache=cache) == expected_baseline
            assert len(gram_calls) == 1 and score_calls == [9, 10, 11]
        assert result == expected
        assert gram.detect_from_profiles(result["profiles"], layers=result["layers"]) == (
            gram.detect_from_profiles(expected["profiles"], layers=expected["layers"])
        )
        assert len(cache._grams) <= 4


@pytest.mark.parametrize("changed", [(0,), (1,), (9,), (19,), (3, 8), ()])
def test_boundary_and_multiple_edits_match_full_profile(changed):
    baseline = weights(20)
    cache = gram.GramProfileCache(baseline, edit_layer=3)
    gram.profile_weights(baseline, cache=cache)
    edited = dict(baseline)
    for layer in changed:
        edited[layer] = baseline[layer] + 0.2
    assert gram.profile_weights(edited, cache=cache) == gram.profile_weights(edited)


def test_changed_edit_layer_checkpoint_and_baseline_invalidation():
    baseline = weights(12)
    cache = gram.GramProfileCache(baseline, edit_layer=3)
    for layer in (3, 8, 3):
        edited = dict(baseline)
        edited[layer] = baseline[layer] + 0.2
        assert gram.profile_weights(edited, cache=cache) == gram.profile_weights(edited)
        assert len(cache._grams) <= 4

    other_checkpoint = {i: weight + 0.4 for i, weight in baseline.items()}
    assert gram.profile_weights(other_checkpoint, cache=cache) == gram.profile_weights(other_checkpoint)
    assert gram.profile_weights(weights(7), cache=cache) == gram.profile_weights(weights(7))
    baseline[2].add_(0.7)
    for trim in (0.1, 0.25):
        assert gram.profile_weights(baseline, trim_fraction=trim, cache=cache) == (
            gram.profile_weights(baseline, trim_fraction=trim)
        )

    returned = gram.profile_weights(baseline, cache=cache)
    returned["profiles"]["3"][gram.SCORE_FIELD] = -1
    assert gram.profile_weights(baseline, cache=cache) == gram.profile_weights(baseline)
    invalid = dict(baseline)
    invalid[3] = torch.full_like(baseline[3], float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        gram.profile_weights(invalid, cache=cache)
    assert gram.profile_weights(baseline, cache=cache) == gram.profile_weights(baseline)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_model_runner_real_rome_insertion_restoration_and_saved_profiles(tmp_path, monkeypatch, dtype):
    from src.editing import rome as editing
    from src.results import RunArtifactReader
    from src.rome import common
    from src.structural.config import StructuralBenchmarkConfig
    from src.structural.execution import model_runtime as runtime

    handlers, expected = {}, {}

    class Handler:
        def __init__(self, cfg):
            self.cfg, self._layer, self.dtype = cfg, 10, dtype
            self._layer_name_template = "layers.{}"
            self.num_of_layers = 36 if cfg.model.name == "toy-qwen" else 48
            self.epochs, self.device = 2, "cpu"
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([
                torch.nn.Linear(24, 8, bias=False, dtype=dtype) for _ in range(self.num_of_layers)
            ])
            self.model.config = SimpleNamespace(residual_multiplier=1.0)
            self.model.requires_grad_(False)
            self.baseline = {i: module.weight.detach().clone() for i, module in enumerate(self.model.layers)}
            handlers[cfg.model.name] = self
            expected[cfg.model.name] = {"baseline": gram.profile_weights(self.baseline)}

        def _get_module(self, name):
            return self.model.get_submodule(name)

        def remove_hooks(self):
            pass

    def config(model, **kwargs):
        return OmegaConf.create({"model": {"name": model, "layer": 10, "k_N": 1, "v_N": 1}})

    def gather(handler, *, fact_tuple, N):
        assert all(torch.equal(module.weight, handler.baseline[i]) for i, module in enumerate(handler.model.layers))
        return torch.ones(24, dtype=dtype)

    def evaluate(handler, prompt, target_new, target_true, **kwargs):
        changed = [i for i, module in enumerate(handler.model.layers) if not torch.equal(module.weight, handler.baseline[i])]
        assert changed == [10]
        current = {i: module.weight.detach().clone() for i, module in enumerate(handler.model.layers)}
        expected[handler.cfg.model.name][prompt] = gram.profile_weights(current)
        return {"efficacy_score": 1.0, "paraphrase_score": 1.0, "neighborhood_score": 1.0}

    cases = [{"case_id": str(i), "fact_tuple": ("{}", str(i), " new", " old")} for i in range(2)]
    monkeypatch.setattr(runtime, "ModelHandler", Handler)
    monkeypatch.setattr(runtime, "load_model_config", lambda model: config(model).model)
    monkeypatch.setattr(runtime, "build_cfg", config)
    monkeypatch.setattr(runtime, "find_second_moment_files", lambda cfg: ([tmp_path / "cov.pt"], tmp_path))
    monkeypatch.setattr(runtime, "load_test_cases", lambda *args, **kwargs: (cases, {}))
    monkeypatch.setattr(common, "gather_k", gather)
    monkeypatch.setattr(common, "optimize_v", lambda handler, **kwargs: (
        torch.arange(1, 9, dtype=dtype) * (0.01 + 0.01 * int(kwargs["fact_tuple"][1]))
    ))
    monkeypatch.setattr(common, "get_second_moment", lambda handler: torch.eye(24))
    monkeypatch.setattr(editing, "compute_rome_metrics", evaluate)
    result = runtime.run_capture(StructuralBenchmarkConfig(
        models=("toy-qwen", "toy-gemma"), n_tests=2, runs_per_model=2,
        output_dir=str(tmp_path), run_id="run", capture_profile="none",
        enable_captures=("gram-localization",), run_analysis=False, render_graphs=False,
    ))
    assert all(model["status"] == "complete" for model in result["models"].values())
    reader = RunArtifactReader(tmp_path / "run")
    captures = [record for record in reader.manifest["artifacts"].values() if record["kind"] == "capture"]
    assert len(captures) == 8
    for record in captures:
        for case in reader.load(record["artifact_id"])["cases"]:
            assert case["status"] == "complete"
            assert case["data"] == expected[record["model"]][case["case_id"]]
    for handler in handlers.values():
        assert all(torch.equal(module.weight, handler.baseline[i]) for i, module in enumerate(handler.model.layers))
