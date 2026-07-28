from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIGNED_MANIFEST = (
    ROOT / "manifests" / "rome_single_checkpoint_signed_development_v1.json"
)


def test_signed_manifest_contains_only_exposed_development_families() -> None:
    payload = json.loads(SIGNED_MANIFEST.read_text())
    models = payload["models"]

    assert payload["schema_version"] == "rome-single-checkpoint-signed-corpus-v1"
    assert payload["scientific_baseline"] is False
    assert len(models) == 5
    assert len({record["model_key"] for record in models}) == len(models)
    assert len({record["family"] for record in models}) == len(models)
    assert {record["split"] for record in models} == {"development"}
    assert all(record["plan_id"] == "cases0-19_r01" for record in models)
    assert all(
        record["run_root"].startswith(
            "analysis_out/single-checkpoint-signed-dev-n20-"
        )
        for record in models
    )
