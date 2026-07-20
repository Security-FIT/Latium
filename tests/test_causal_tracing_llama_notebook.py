"""Static parity checks for the validated portable Llama trace notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "causal-tracing-test-llama.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _cell_source(index: int) -> str:
    return "".join(_notebook()["cells"][index]["source"])


def test_llama_notebook_code_cells_compile() -> None:
    code = "\n\n".join("".join(cell["source"]) for cell in _notebook()["cells"] if cell["cell_type"] == "code")

    compile(code, str(NOTEBOOK_PATH), "exec")


def test_llama_notebook_uses_whole_mlp_and_robust_region_selection() -> None:
    settings = _cell_source(6)
    presets = _cell_source(8)
    trace = _cell_source(16)

    assert "'llama2-7b': {'name': 'NousResearch/Llama-2-7b-hf'" in presets
    assert "'mlp_template': 'model.layers.{}.mlp'" in presets
    assert "candidates.insert(0, primary.split" in trace
    assert "trace_mlp_windows_for_fact" in trace
    assert "subject_last" in trace
    assert "discovery_candidate_centers" in trace
    assert "confirmation_regions" in trace
    assert "representative_trace_center" in trace
    assert "NUM_VALID_FACTS = 100" in settings
    assert "NUM_NOISE_SAMPLES = 10" in settings
