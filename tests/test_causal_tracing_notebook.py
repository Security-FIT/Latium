"""Static, parity, and model-free checks for the canonical standalone notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import hydra
import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "causal-tracing.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code() -> str:
    return "\n\n".join(
        "".join(cell["source"])
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def _literal_assignment(cell_index: int, name: str):
    tree = ast.parse("".join(_notebook()["cells"][cell_index]["source"]))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment {name} not found in cell {cell_index}")


def test_notebook_is_the_single_canonical_standalone_trace() -> None:
    notebook = _notebook()
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    source = _code()

    assert notebook["metadata"]["latium"]["standalone"] is True
    assert notebook["metadata"]["latium"]["pipeline_entrypoint"] == "python -m src causal-trace"
    assert "Standalone Reference" in markdown
    assert "from src" not in source
    assert "import src" not in source
    assert "compute_second_moment" not in source
    assert "run_rome" not in source
    assert list((ROOT / "notebooks").glob("causal-tracing*.ipynb")) == [NOTEBOOK_PATH]


def test_notebook_code_cells_compile_without_outputs() -> None:
    notebook = _notebook()

    compile(_code(), str(NOTEBOOK_PATH), "exec")
    assert all(
        not cell.get("outputs") and cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebook_uses_the_audited_selector_without_removed_heuristics() -> None:
    source = _code()

    assert "eligible_discovery.sort_values" in source
    assert "['discovery_mean_ie', 'window_center']" in source
    assert "confirmation_windows['window_center'] == discovery_center" in source
    assert "confirmation_row['confirmation_ci_lower']" in source
    assert "selected_trace_center = discovery_center if confirmation_passed else None" in source
    assert "split_assignments.csv" in source
    assert "window_is_full_width" in source

    for removed in (
        "MIDDLE_BAND",
        "NONINFERIORITY",
        "NEIGHBOR_SUPPORT",
        "MINIMUM_SUPPORTED_CENTERS",
        "confirmation_regions",
        "trimmed_mean",
    ):
        assert removed not in source


def test_notebook_pairs_noise_and_uses_subject_embedding_calibration() -> None:
    source = _code()

    assert "ie = restore_probs - corrupt_probabilities" in source
    assert "source': 'candidate_subject_token_embeddings'" in source
    assert "encoded['input_ids'][0, positions]" in source
    assert "noise_std = float(noise_calibration['noise_std'])" in source
    assert "embedding_std = float(emb.weight.detach().float().std().item())" not in source
    assert "'noise_calibration': noise_calibration" in source


def test_notebook_handles_target_and_subject_tokenization_like_pipeline() -> None:
    source = _code()

    assert "(f' {cleaned}', False), (cleaned, False)" in source
    assert "return_offsets_mapping=True" in source
    assert "prompt_ids[start:start + len(ids)] == ids" in source
    assert "Subject span ambiguous or missing" in source


def test_notebook_defaults_match_pipeline_config() -> None:
    with hydra.initialize_config_dir(config_dir=str(ROOT / "src" / "config"), version_base=None):
        cfg = hydra.compose(config_name="latium", overrides=["command=causal_trace"])

    expected = {
        "NUM_VALID_FACTS": cfg.command.causal_trace.num_valid_facts,
        "MAX_DATASET_EXAMPLES_TO_SCAN": cfg.command.causal_trace.max_dataset_examples_to_scan,
        "NUM_NOISE_SAMPLES": cfg.command.causal_trace.num_noise_samples,
        "NOISE_BATCH_SIZE": cfg.command.causal_trace.noise_batch_size,
        "NOISE_MULTIPLIER": cfg.command.causal_trace.noise_multiplier,
        "WINDOW_SIZE": cfg.command.causal_trace.window_size,
        "REQUIRE_CORRECT_CLEAN": cfg.command.causal_trace.require_correct_clean_prediction,
        "MIN_TOTAL_EFFECT": cfg.command.causal_trace.min_total_effect,
        "DISCOVERY_FRACTION": cfg.command.causal_trace.discovery_fraction,
        "MIN_CONFIRMATION_FACTS": cfg.command.causal_trace.minimum_confirmation_facts,
        "BOOTSTRAP_SAMPLES": cfg.command.causal_trace.bootstrap_samples,
        "CONFIDENCE_LEVEL": cfg.command.causal_trace.confidence_level,
        "SEED": cfg.command.causal_trace.seed,
    }
    assert {name: _literal_assignment(6, name) for name in expected} == expected


def test_notebook_model_adapters_match_latium_model_configs() -> None:
    presets = _literal_assignment(8, "MODEL_PRESETS")
    references = _literal_assignment(6, "CONFIG_REFERENCE_LAYERS")

    for key, preset in presets.items():
        config_path = ROOT / "src" / "config" / "model" / f"{key}.yaml"
        if not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert preset["name"] == config["name"]
        assert preset["embedding"] == config["corrupt_layer_name_template"]
        assert preset["mlp_template"] == config["layer_name_template"]
        assert references[key] == config["layer"]

    assert "mlp.c_proj" in json.dumps(presets)
    assert "mlp.down_proj" in json.dumps(presets)
    assert "shared_mlp.output_linear" in json.dumps(presets)
