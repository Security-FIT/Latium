from __future__ import annotations

import re
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "rome_math_ablation_recapture.yaml"
JOB = ROOT / "jobs" / "run_rome_math_ablation.sh"


def _job_plan(script: str, mode: str) -> tuple[list[str], str]:
    match = re.search(
        rf"^  {re.escape(mode)}\)\n(?P<body>.*?)(?=^    ;;$)",
        script,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing job mode {mode}"
    body = match.group("body")
    models_match = re.search(r"MODELS='\[(?P<models>[^]]+)\]'", body)
    run_id_match = re.search(r"RUN_ID=(?P<run_id>[^\s]+)", body)
    assert models_match is not None
    assert run_id_match is not None
    return models_match.group("models").split(","), run_id_match.group("run_id")


def test_pre_freeze_job_matches_manifest_and_excludes_held_out_families() -> None:
    manifest = OmegaConf.to_container(OmegaConf.load(MANIFEST), resolve=True)
    assert isinstance(manifest, dict)
    script = JOB.read_text(encoding="utf-8")

    model_records = manifest["models"]
    model_families = {record["model"]: record["family"] for record in model_records}
    held_out_families = set(manifest["split_policy"]["held_out_families"])
    development_families = set(manifest["split_policy"]["development_families"])
    held_out_models = {model for model, family in model_families.items() if family in held_out_families}

    smoke = manifest["capture"]["smoke"]
    development = manifest["capture"]["development"]
    smoke_a_models, smoke_a_run_id = _job_plan(script, "smoke-a")
    smoke_b_models, smoke_b_run_id = _job_plan(script, "smoke-b")
    development_models, development_run_id = _job_plan(script, "development")

    assert held_out_families == {"deepseek", "falcon", "opt"}
    assert development_families.isdisjoint(held_out_families)
    assert manifest["split_policy"]["held_out_release_gate"]["status"] == "closed"
    assert smoke_a_models == smoke_b_models == list(smoke["models"])
    assert [smoke_a_run_id, smoke_b_run_id] == list(smoke["run_ids"])
    assert development_models == list(development["models"])
    assert development_run_id == development["run_id"]
    assert all(model_families[model] in development_families for model in smoke_a_models)
    assert set(development_models).isdisjoint(held_out_models)
    assert all(record["run_id"] is None for record in model_records if record["model"] in held_out_models)
    assert all(model not in script for model in held_out_models)
    assert "  full)" not in script
