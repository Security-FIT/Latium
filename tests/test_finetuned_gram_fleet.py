"""Lifecycle tests use fake HF checkpoints; no GPU or weight downloads."""
import json
from pathlib import Path
from types import SimpleNamespace
import hydra
import pytest
from omegaconf import OmegaConf
from jobs import finetuned_gram_fleet as fleet
from src.common.model_config import canonical_model_name, load_model_config
from src.counterfact_selection import generate_random_case_manifest, write_case_manifest


def fixture_args(tmp_path):
    facts = [{"case_id": i, "requested_rewrite": {"subject": f"S{i}", "prompt": "{} lives in", "target_true": {"str": "Rome"}, "target_new": {"str": "Paris"}}} for i in range(8)]
    manifest = tmp_path / "facts.json"
    write_case_manifest(manifest, generate_random_case_manifest(count=8, seed=42, dataset_name="fake", split="train", dataset=facts))
    models = tmp_path / "models.json"
    models.write_text(json.dumps({"models": [{"model_id": "org/one"}, {"model_id": "org/two"}]}))
    return fleet.parse_args(["--base-model", "gpt2-xl", "--models-manifest", str(models),
        "--model-count", "2", "--run-root", str(tmp_path / "run"), "--case-index-file", str(manifest),
        "--case-start", "2", "--case-stop", "4", "--no-graphs", "--finetuned-covariance"])


class FakeApi:
    def __init__(self):
        self.calls = []

    def model_info(self, model, revision=None):
        self.calls.append((model, revision))
        return SimpleNamespace(sha="a" * 40, siblings=[SimpleNamespace(rfilename=n) for n in
            ("config.json", "model.safetensors", "pytorch_model.bin", "tokenizer.json", "optimizer.pt", "adapter_model.safetensors")])


def test_default_reuses_original_covariance_without_checkpoint_computation(tmp_path, monkeypatch):
    from jobs.paper_fleet import model_second_moment_files
    prepared = fixture_args(tmp_path)
    args = fleet.parse_args(["--base-model", "gpt2-xl", "--models-manifest", prepared.models_manifest,
        "--model-count", "2", "--run-root", prepared.run_root, "--case-index-file", prepared.case_index_file,
        "--case-start", "2", "--case-stop", "4", "--no-graphs"])
    assert not args.finetuned_covariance
    base = load_model_config(args.base_model)
    base.second_moment_dir = str(tmp_path / "classic-stats")
    covariance = Path(base.second_moment_dir) / f"{base.name.replace('/', '_')}_{base.layer}_SM_Method.WIKIPEDIA_100000.pt"
    covariance.parent.mkdir()
    covariance.write_bytes(b"original statistics")
    base.second_moment_path = str(covariance)
    real_load = fleet.load_model_config
    monkeypatch.setattr(fleet, "load_model_config", lambda name, **kwargs:
        base if name == args.base_model else real_load(name, **kwargs))
    seen = []

    def gram(params):
        cfg = real_load(params.models[0])
        assert params.skip_second_moment
        assert cfg.name in ("org/one", "org/two")
        assert cfg.second_moment_model_name == base.name
        assert cfg.second_moment_path == str(covariance)
        assert model_second_moment_files(params.models[0], int(base.layer), 100000) == [covariance]
        assert model_second_moment_files(params.models[0], int(base.layer), 50000) == []
        seen.append(cfg.name)
        return 0

    monkeypatch.setattr(fleet.gram_fleet, "run", gram)
    assert fleet.run(args, FakeApi(), lambda **kwargs: None) == 0
    assert seen == ["org/one", "org/two"]
    assert covariance.read_bytes() == b"original statistics"
    assert not list(Path(args.run_root).glob("models/*/covariance"))
    args.finetuned_covariance = True
    with pytest.raises(ValueError, match="selection/configuration changed"):
        fleet.run(args, FakeApi(), lambda **kwargs: None)


def test_missing_original_covariance_fails_before_downloads(tmp_path, monkeypatch):
    args = fixture_args(tmp_path)
    args.finetuned_covariance = False
    base = load_model_config(args.base_model)
    base.second_moment_dir = str(tmp_path / "missing-stats")
    base.second_moment_path = None
    monkeypatch.setattr(fleet, "load_model_config", lambda *a, **kw: base)
    calls = []
    with pytest.raises(FileNotFoundError, match="Missing original-model covariance"):
        fleet.run(args, FakeApi(), lambda **kwargs: calls.append(kwargs))
    assert calls == []
    args.prepare_only = True
    assert fleet.run(args, FakeApi(), lambda **kwargs: calls.append(kwargs)) == 0
    assert calls == []


def test_checkpoint_stop_alias_preserves_frozen_manifest_range(tmp_path):
    prepared = fixture_args(tmp_path)
    common = ["--base-model", "gpt2-xl", "--models-manifest", prepared.models_manifest,
        "--model-count", "100", "--run-root", prepared.run_root,
        "--case-index-file", prepared.case_index_file]
    first = fleet.parse_args(common + ["--checkpoint-start", "0", "--checkpoint-stop", "20"])
    second = fleet.parse_args(common + ["--checkpoint-start", "20", "--checkpoint-stop", "40"])
    legacy = fleet.parse_args(common + ["--checkpoint-start", "20", "--checkpoint-limit", "40"])
    assert (first.checkpoint_start, first.checkpoint_limit) == (0, 20)
    assert (second.checkpoint_start, second.checkpoint_limit) == (20, 40)
    assert (legacy.checkpoint_start, legacy.checkpoint_limit) == (20, 40)
    assert first.model_count == second.model_count == 100


def test_configs_keep_classic_layer_and_use_checkpoint_covariance(tmp_path, monkeypatch):
    args = fixture_args(tmp_path)
    base = load_model_config("gpt2-xl")
    api = FakeApi()
    selection = fleet.freeze_selection(args, base, api)
    directory = fleet.save_configs(Path(args.run_root), selection)
    monkeypatch.setenv("LATIUM_MODEL_CONFIG_DIR", str(directory))
    cfg = load_model_config("fleet_org_one")
    assert canonical_model_name("org/one") == "fleet_org_one"
    assert cfg.layer == base.layer
    assert cfg.layer_name_template == base.layer_name_template
    assert cfg.lr == base.lr and cfg.prefix_mode == base.prefix_mode
    assert cfg.second_moment_path is None
    assert cfg.second_moment_dir == str(Path(args.run_root) / "models/fleet_org_one/covariance")
    assert cfg.checkpoint_revision == "a" * 40
    with hydra.initialize_config_dir(config_dir=str(fleet.ROOT / "src/config"), version_base=None):
        composed = hydra.compose(config_name="latium", overrides=["command=second_moment",
            f"hydra.searchpath=['file://{directory.parent}']", "model=fleet_org_one"])
    assert composed.model.name == "org/one" and composed.model.layer == base.layer
    assert fleet.freeze_selection(args, base, api) == selection
    assert len(api.calls) == 2  # retries do not resolve a moving HF main branch
    args.model_count = 1
    with pytest.raises(ValueError, match="selection/configuration changed"):
        fleet.freeze_selection(args, base, api)


def test_download_filters_and_cleanup_containment(tmp_path):
    files = fleet.checkpoint_files(["config.json", "model.safetensors", "tokenizer.json", "pytorch_model.bin",
                                   "adapter_model.safetensors", "optimizer.pt", "onnx/model.onnx"])
    assert files == ["config.json", "model.safetensors", "tokenizer.json"]
    assert fleet.checkpoint_files(["adapter_config.json", "adapter_model.safetensors", "optimizer.pt"]) == ["adapter_config.json", "adapter_model.safetensors"]
    with pytest.raises(ValueError, match="neither full"):
        fleet.checkpoint_files(["model.gguf"])
    owned = tmp_path / "owned"; owned.mkdir()
    victim = tmp_path / "other"; victim.mkdir()
    (victim / "keep").write_text("keep")
    link = owned / "checkpoint"; link.symlink_to(victim, target_is_directory=True)
    with pytest.raises(ValueError, match="escaped"):
        fleet.remove_download(link, owned)
    assert (victim / "keep").exists()
    with pytest.raises(ValueError, match="unowned"):
        fleet.own_download_root(victim, tmp_path / "run")


@pytest.mark.parametrize("download_failure,gram_failure", [(False, False), (True, False), (False, True)])
def test_sequential_order_failure_cleanup_append_and_retry(tmp_path, monkeypatch, download_failure, gram_failure):
    args = fixture_args(tmp_path)
    base_layer = int(load_model_config(args.base_model).layer)
    api = FakeApi()
    events = []

    def download(**kwargs):
        directory = Path(kwargs["local_dir"])
        # Each previous checkpoint's weights have already been removed.
        assert not list((Path(args.run_root) / ".downloads").glob("*/models/*/*/weights"))
        directory.mkdir(parents=True)
        (directory / "weights").write_text("fake")
        assert kwargs["revision"] == "a" * 40
        assert "pytorch_model.bin" not in kwargs["allow_patterns"]
        events.append(("download", kwargs["repo_id"]))
        if download_failure and kwargs["repo_id"] == "org/one":
            raise RuntimeError("fake partial download failure")

    def gram(args):
        assert args.case_start == 2 and args.case_stop == 4 and args.n_tests == 2
        assert args.workflow == "gram"
        assert load_model_config(args.models[0]).layer == base_layer
        assert (Path(load_model_config(args.models[0]).models_dir) / load_model_config(args.models[0]).name / "weights").exists()
        events.append(("baseline-rome-edited-gram", args.models[0]))
        return int(gram_failure and args.models[0].endswith("one"))

    monkeypatch.setattr(fleet.gram_fleet, "run", gram)
    monkeypatch.setattr(fleet.gram_fleet, "verify_batch", lambda *a: None)
    assert fleet.run(args, api, download) == int(download_failure or gram_failure)
    expected = ["download"] if download_failure else ["download", "baseline-rome-edited-gram"]
    assert [v[0] for v in events] == expected + ["download", "baseline-rome-edited-gram"]
    assert not list((Path(args.run_root) / ".downloads").glob("*/models"))
    catalog = json.loads((Path(args.run_root) / "experiment.json").read_text())
    assert len(catalog["models"]) == 2
    states = list((Path(args.run_root) / "models").glob("*/fleet-batches/*/state.json"))
    assert len(states) == 2
    assert all(json.loads(p.read_text())["revision"] == "a" * 40 for p in states)
    if not download_failure and not gram_failure:
        leftover = Path(args.run_root) / ".downloads/fleet_org_two/models/org/two"
        leftover.mkdir(parents=True)
        (leftover / "weights").write_text("interrupted download")
        events.clear()
        assert fleet.run(args, api, download) == 0
        assert not leftover.exists()
        assert events == []  # complete retries download/load no models
        assert len(api.calls) == 2
        args.case_start = 4; args.case_stop = 6
        monkeypatch.setattr(fleet.gram_fleet, "run", lambda args: 0)
        assert fleet.run(args, api, download) == 0
        catalog = json.loads((Path(args.run_root) / "experiment.json").read_text())
        assert all(len(m["batches"]) == 2 for m in catalog["models"].values())


def test_cli_loads_generated_config_for_covariance(tmp_path, monkeypatch):
    from src.main import run_hydra
    import src.commands
    args = fixture_args(tmp_path)
    base_layer = int(load_model_config(args.base_model).layer)
    directory = fleet.save_configs(Path(args.run_root), fleet.freeze_selection(args, load_model_config("gpt2-xl"), FakeApi()))
    monkeypatch.setenv("LATIUM_MODEL_CONFIG_DIR", str(directory))
    configs = []
    monkeypatch.setattr(src.commands, "run_command", lambda cfg: configs.append(cfg) or 0)
    assert run_hydra(["command=second_moment", "model=fleet_org_one"]) == 0
    assert all(cfg.model.name == "org/one" and cfg.model.layer == base_layer for cfg in configs)
    assert all(cfg.model.second_moment_path is None for cfg in configs)


def test_adapter_downloads_pinned_base_and_preserves_unusable_top_rank(tmp_path, monkeypatch):
    args = fixture_args(tmp_path)
    args.keep_downloads = True
    manifest = Path(args.models_manifest)
    manifest.write_text(json.dumps({"models": [
        {"model_id": "org/broken", "downloads": 100, "selection_error": "Missing weights"},
        {"model_id": "org/adapter", "downloads": 90, "revision": "a" * 40,
         "files": ["adapter_config.json", "adapter_model.safetensors"],
         "adapter_base_model": "org/base", "adapter_base_revision": "b" * 40,
         "adapter_base_files": ["config.json", "model.safetensors"]}]}))
    calls = []

    def download(**kwargs):
        calls.append((kwargs["repo_id"], kwargs["revision"]))
        directory = Path(kwargs["local_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        for name in kwargs["allow_patterns"]:
            (directory / name).write_text("fake")

    def gram(params):
        cfg = load_model_config(params.models[0])
        assert cfg.name == "org/adapter" and cfg.adapter_base_revision == "b" * 40
        assert Path(cfg.adapter_base_path, "config.json").exists()
        assert cfg.layer == load_model_config("gpt2-xl", config_dir=fleet.MODEL_CONFIG_DIR).layer
        return 0

    monkeypatch.setattr(fleet.gram_fleet, "run", gram)
    assert fleet.run(args, FakeApi(), download) == 1
    assert calls == [("org/adapter", "a" * 40), ("org/base", "b" * 40)]
    frozen = json.loads((Path(args.run_root) / "checkpoints.json").read_text())
    assert [e["model_id"] for e in frozen["models"]] == ["org/broken", "org/adapter"]
    assert [e["rank"] for e in frozen["models"]] == [1, 2]


def test_retained_downloads_retry_rome_failure_and_skip_complete_rerun(tmp_path, monkeypatch):
    args = fixture_args(tmp_path)
    args.case_start = 0; args.case_stop = 1
    args.keep_downloads = True; args.retry_failed_facts = True; args.checkpoint_limit = 1
    calls = []

    def download(**kwargs):
        p = Path(kwargs["local_dir"])
        p.mkdir(parents=True, exist_ok=True)
        (p / "weights").write_text("pinned weights")
        calls.append("download")

    def gram(params):
        batch, catalog = fleet.gram_fleet.prepare(params)
        root = Path(params.run_root) / catalog["models"][params.models[0]]["run_root"]
        root.mkdir(parents=True, exist_ok=True)
        position = params.case_start
        artifact_id = f"exec-{position}"
        path = f"execution-{position}.json"
        manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {"artifacts": {}}
        manifest["artifacts"][artifact_id] = {"kind": "execution", "edit_method": "rome", "plan_id": catalog["models"][params.models[0]]["batches"][batch]["plan_id"], "path": path}
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / path).write_text(json.dumps({"cases": [{"case_id": position, "status": "complete", "edit": {"success": position > 0}, "detected_layer": 999}]}))
        calls.append(position)
        return 0

    monkeypatch.setattr(fleet.gram_fleet, "run", gram)
    monkeypatch.setattr(fleet.gram_fleet, "verify_batch", lambda *a: None)
    assert fleet.run(args, FakeApi(), download) == 0
    assert calls == ["download", 0, 1]  # Wrong GRAM layer does not trigger another attempt.
    state = json.loads(next((Path(args.run_root) / "models").glob("*/fleet-batches/*/state.json")).read_text())
    assert state["accepted_position"] == 1
    assert [a["status"] for a in state["attempts"]] == ["rome_failed", "complete"]
    weights = list((Path(args.run_root) / ".downloads").glob("*/models/*/*/weights"))
    assert len(weights) == 1
    calls.clear()
    assert fleet.run(args, FakeApi(), download) == 0
    assert calls == [] and weights[0].is_file()
