"""Static and model-free checks for the portable causal tracing v2 notebook."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "causal-tracing-auto-v2.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _cell_source(index: int) -> str:
    return "".join(_notebook()["cells"][index]["source"])


def test_v2_notebook_code_cells_compile() -> None:
    code = "\n\n".join("".join(cell["source"]) for cell in _notebook()["cells"] if cell["cell_type"] == "code")

    compile(code, str(NOTEBOOK_PATH), "exec")


def test_v2_selector_uses_one_held_out_window_without_old_heuristics() -> None:
    source = _cell_source(16)
    presets = _cell_source(8)

    assert "mlp.c_proj" in presets
    assert "mlp.down_proj" in presets
    assert "shared_mlp.output_linear" in presets
    assert "candidates.append(primary.split" in source
    assert "candidates.insert(0, primary.split" not in source
    assert "eligible_discovery.sort_values" in source
    assert "['discovery_mean_ie', 'window_center']" in source
    assert "confirmation_windows['window_center'] == discovery_center" in source
    assert "confirmation_row['confirmation_ci_lower']" in source
    assert "selected_trace_center = discovery_center if confirmation_passed else None" in source
    assert "split_assignments.csv" in source

    for removed in (
        "MIDDLE_BAND",
        "NONINFERIORITY",
        "NEIGHBOR_SUPPORT",
        "MINIMUM_SUPPORTED_CENTERS",
        "confirmation_regions",
        "trimmed_mean",
    ):
        assert removed not in source


def test_v2_second_moment_lookup_requires_exact_model_layer_and_shape(tmp_path: Path) -> None:
    source = _cell_source(26)
    parsed = ast.parse(source)
    helper_names = {
        "_path_matches_model_layer",
        "_second_moment_sample_count",
        "_load_second_moment_matrix",
        "second_moment_status",
    }
    helpers = [
        node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in helper_names
    ]
    namespace = {
        "Path": Path,
        "np": np,
        "re": re,
        "torch": torch,
        "RAW_COVARIANCE_DIR": tmp_path / "raw_covariance",
    }
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(NOTEBOOK_PATH), "exec"), namespace)

    model_slug = "acme_Model"
    correct = tmp_path / f"{model_slug}_5_SM_Method.WIKIPEDIA_100.pt"
    wrong_layer = tmp_path / f"{model_slug}_50_SM_Method.WIKIPEDIA_100.pt"
    wrong_shape = tmp_path / f"{model_slug}_5_SM_Method.WIKIPEDIA_200.pt"
    torch.save(torch.eye(3), correct)
    torch.save(torch.eye(3), wrong_layer)
    torch.save(torch.eye(4), wrong_shape)

    handler = SimpleNamespace(
        cfg=SimpleNamespace(model=SimpleNamespace(name="acme/Model")),
        _layer=5,
        _layer_name_template="model.layers.{}.mlp.down_proj",
        second_moment_path=str(wrong_layer),
        second_moment_dir=str(tmp_path),
        hidden_dim=3,
        _second_moment_cache=None,
    )
    status = namespace["second_moment_status"](handler)

    assert status["available"] is True
    assert status["inverse_covariance_paths"] == [str(correct)]
    assert status["layer"] == 5
    assert status["expected_matrix_dim"] == 3
    assert status["invalid_inverse_covariance_paths"][0]["path"] == str(wrong_shape)


def test_v2_second_moment_pipeline_reports_reuse() -> None:
    covariance_source = _cell_source(26)
    rome_source = _cell_source(24)

    assert "'reused_existing': True" in covariance_source
    assert "'computed_now': False" in covariance_source
    assert "_load_second_moment_matrix(path, handler.hidden_dim)" in covariance_source
    assert "self._second_moment_cache = None" in rome_source
    assert "handler._second_moment_cache = (str(path), matrix)" in rome_source
