from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from src.common.model_config import load_model_config


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "manifests" / "rome_math_ablation_n50_cluster.yaml"
LAYER_PATTERN = re.compile(r"_(\d+)_SM_Method\.")
EXPECTED_MODELS = {
    "gpt2-xl",
    "gpt-j-6b",
    "mistral-7b-v0.1",
    "mistral-7b-v0.3",
    "deepseek-7b-base",
    "falcon-7b",
    "opt-6.7b",
    "llama2-7b",
    "granite4-micro",
}


def _load_manifest() -> dict:
    data = OmegaConf.to_container(OmegaConf.load(MANIFEST_PATH), resolve=True)
    assert isinstance(data, dict)
    return data


def _filename_layer(path: str) -> int:
    match = LAYER_PATTERN.search(Path(path).name)
    assert match is not None, f"covariance filename does not encode a layer: {path}"
    return int(match.group(1))


def test_n50_manifest_declares_exact_exposed_development_cohort() -> None:
    manifest = _load_manifest()
    records = manifest["models"]

    assert manifest["scientific_baseline"] is False
    assert manifest["case_selection"] == {
        "dataset": "CounterFact",
        "start_idx": 0,
        "count": 50,
        "requested_total": 450,
    }
    assert len(records) == 9
    assert {record["model"] for record in records} == EXPECTED_MODELS
    assert len({record["run_id"] for record in records}) == 9
    assert all(record["split"] == "development" for record in records)
    assert all(record["exposed"] is True for record in records)
    assert all(record["case_selection"] == {"start_idx": 0, "count": 50} for record in records)
    assert set(manifest["exposure_policy"]["previously_held_out_now_exposed"]) == {
        "deepseek",
        "falcon",
        "opt",
    }


def test_n50_manifest_matches_initial_model_configs_and_seals_old_manifest() -> None:
    manifest = _load_manifest()
    sealed = manifest["sealed_manifest"]
    sealed_path = ROOT / sealed["path"]

    assert hashlib.sha256(sealed_path.read_bytes()).hexdigest() == sealed["sha256"]
    assert sealed["mutation_allowed"] is False
    for record in manifest["models"]:
        config = load_model_config(record["model"])
        configured_path = str(record["configured_second_moment_path_before_audit"])
        configured_layer = int(record["configured_layer_before_audit"])
        assert str(config.name) == record["model_identifier"]
        assert int(config.layer) == configured_layer
        assert str(config.second_moment_path) == configured_path
        assert _filename_layer(configured_path) == configured_layer


def test_recovery_records_require_confirmed_main_trace_then_new_covariance() -> None:
    manifest = _load_manifest()

    for record in manifest["models"]:
        decision = record["dependency_decision"]
        assert decision in {"pending_audit", "reuse_exact", "trace_and_recompute"}
        if decision == "pending_audit":
            assert record["confirmed_layer"] is None
            continue

        covariance = record["covariance"]
        confirmed_layer = int(record["confirmed_layer"])
        assert covariance["status"] == "verified"
        assert _filename_layer(str(covariance["path"])) == confirmed_layer
        assert re.fullmatch(r"[0-9a-f]{64}", str(covariance["sha256"]))
        assert covariance["method"] == "WIKIPEDIA"
        assert covariance["sample_count"] == 100000

        if decision == "reuse_exact":
            assert confirmed_layer == int(record["configured_layer_before_audit"])
            continue

        trace = record["causal_trace"]
        assert trace["status"] == "confirmed"
        assert trace["confirmation_passed"] is True
        assert int(trace["selected_trace_center"]) == confirmed_layer
        assert int(trace["num_valid_facts"]) > 0
        assert re.fullmatch(r"[0-9a-f]{40}", str(trace["main_commit"]))
        assert str(trace["summary_path"]).endswith("summary.json")
        assert datetime.fromisoformat(covariance["produced_at"]) > datetime.fromisoformat(trace["completed_at"])
