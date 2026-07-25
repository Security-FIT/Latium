from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "causal_tracing.ipynb"


def test_causal_tracing_notebook_uses_production_command() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = chr(10).join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "from src.main import run_hydra" in code
    assert "command=causal_trace" in code
    assert "fact_results.jsonl" in code
    assert "confirmation_windows.csv" in code
    assert "def trace" not in code

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), str(NOTEBOOK), "exec")
