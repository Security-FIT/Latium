"""Checkpoint/fact orchestration without downloading weights or using a GPU."""

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from src import finetunes
from src.common.model_config import load_model_config
from src.counterfact_selection import manifest_digest


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    facts = {"dataset": "fake", "split": "train", "count": 2,
             "indices": [5, 8], "case_ids": [101, 202]}
    facts["manifest_hash"] = manifest_digest(facts)
    (inputs / "rome-facts-100.json").write_text(json.dumps(facts))
    entries = [
        {"model_id": "owner/full", "key": "full", "revision": "a" * 40,
         "checkpoint_type": "full", "files": ["model.safetensors", "config.json"],
         "fact_position": 1, "fact_case_id": 202},
        {"model_id": "owner/adapter", "key": "adapter", "revision": "b" * 40,
         "checkpoint_type": "adapter", "files": ["adapter_model.safetensors", "adapter_config.json"],
         "adapter_base_model": "owner/base", "adapter_base_revision": "c" * 40,
         "adapter_base_files": ["model.safetensors", "config.json"],
         "fact_position": 0, "fact_case_id": 101},
    ]
    for mode in ("base", "own"):
        directory = inputs / f"finetunes-{mode}"
        directory.mkdir()
        (directory / "gpt2-medium.json").write_text(json.dumps({
            "models": entries, "fact_manifest_hash": facts["manifest_hash"], "edit_layer": 7,
        }))
    downloads, commands = [], []

    class Api:
        def model_info(self, model_id, revision=None):
            return SimpleNamespace(id=model_id, sha=revision or "d" * 40,
                                   siblings=[SimpleNamespace(rfilename="model.safetensors")])

    def snapshot(**kwargs):
        downloads.append(kwargs)
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)

    def command(cfg):
        if cfg.command.name == "second-moment":
            model = OmegaConf.to_container(cfg.model, resolve=True)
            path = Path(model["second_moment_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("mock covariance")
            commands.append(("covariance", model))
        else:
            key = cfg.structural.run.models[0]
            model = OmegaConf.to_container(load_model_config(key), resolve=True)
            commands.append(("edit", deepcopy(model), cfg.structural.run.start_idx))
            assert cfg.structural.run.n_tests == 1
            assert cfg.structural.render.enabled is True
            assert Path(model["second_moment_path"]).is_file()
            if model.get("adapter_base_path"):
                assert Path(model["adapter_base_path"]).is_dir()
        return 0

    monkeypatch.setattr(finetunes, "INPUTS", inputs)
    monkeypatch.setattr(finetunes, "HfApi", Api)
    monkeypatch.setattr(finetunes, "snapshot_download", snapshot)
    monkeypatch.setattr("src.commands.run_command", command)
    return tmp_path, downloads, commands


@pytest.mark.parametrize("mode", ["base", "own"])
@pytest.mark.parametrize("samples", [100_000, 32])
def test_saved_assignments_and_covariance_sources(workflow, mode, samples):
    root, downloads, commands = workflow
    previous = os.environ.get("LATIUM_MODEL_CONFIG_DIR")
    assert finetunes.main(["--family", "gpt2-medium", "--covariance", mode,
                           "--covariance-samples", str(samples), "--output", str(root / "out")]) == 0
    assert os.environ.get("LATIUM_MODEL_CONFIG_DIR") == previous
    edits = [command for command in commands if command[0] == "edit"]
    assert [command[2] for command in edits] == [1, 0]
    assert all(command[1]["layer"] == 7 for command in commands)
    assert [command[1]["checkpoint_revision"] for command in edits] == ["a" * 40, "b" * 40]
    assert edits[1][1]["adapter_base_revision"] == "c" * 40
    covariance = [command[1] for command in commands if command[0] == "covariance"]
    assert len(covariance) == (1 if mode == "base" else 2)
    assert all(config["second_moment_target_samples"] == samples for config in covariance)
    assert all(config["second_moment_path"].endswith(f"_{samples}.pt") for config in covariance)
    assert (edits[0][1]["second_moment_path"] == edits[1][1]["second_moment_path"]) == (mode == "base")
    assert all(not Path(download["local_dir"]).exists() for download in downloads)


def test_selected_checkpoint_keeps_its_assigned_fact(workflow):
    root, downloads, commands = workflow
    assert finetunes.main(["--family", "gpt2-medium", "--covariance", "own",
                           "--checkpoint", "owner/adapter", "--covariance-samples", "32",
                           "--output", str(root / "out")]) == 0
    assert [c[1]["name"] for c in commands if c[0] == "edit"] == ["owner/adapter"]
    assert [c[2] for c in commands if c[0] == "edit"] == [0]
    assert {download["revision"] for download in downloads} == {"b" * 40, "c" * 40}


def test_download_failure_keeps_next_checkpoint(workflow, monkeypatch):
    root, downloads, commands = workflow
    original = finetunes.snapshot_download

    def fail_first(**kwargs):
        if kwargs["repo_id"] == "owner/full":
            raise RuntimeError("download failed")
        return original(**kwargs)

    monkeypatch.setattr(finetunes, "snapshot_download", fail_first)
    assert finetunes.main(["--family", "gpt2-medium", "--covariance", "own",
                           "--output", str(root / "out")]) == 1
    assert [c[1]["name"] for c in commands if c[0] == "edit"] == ["owner/adapter"]
    failures = json.loads((root / "out/failures.json").read_text())
    assert failures == [{"model_id": "owner/full", "revision": "a" * 40, "error": "download failed"}]
