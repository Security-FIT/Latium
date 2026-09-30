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
        "--case-start", "2", "--case-stop", "4", "--no-graphs"])


class FakeApi:
    def __init__(self):
        self.calls = []

    def model_info(self, model, revision=None):
        self.calls.append((model, revision))
        return SimpleNamespace(sha="a" * 40, siblings=[SimpleNamespace(rfilename=n) for n in
            ("config.json", "model.safetensors", "pytorch_model.bin", "tokenizer.json", "optimizer.pt", "adapter_model.safetensors")])


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
    with pytest.raises(ValueError, match="full Transformers"):
        fleet.checkpoint_files(["adapter_config.json", "adapter_model.safetensors"])
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
        assert load_model_config(args.models[0]).layer == 18
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
    directory = fleet.save_configs(Path(args.run_root), fleet.freeze_selection(args, load_model_config("gpt2-xl"), FakeApi()))
    monkeypatch.setenv("LATIUM_MODEL_CONFIG_DIR", str(directory))
    configs = []
    monkeypatch.setattr(src.commands, "run_command", lambda cfg: configs.append(cfg) or 0)
    assert run_hydra(["command=second_moment", "model=fleet_org_one"]) == 0
    assert all(cfg.model.name == "org/one" and cfg.model.layer == 18 for cfg in configs)
    assert all(cfg.model.second_moment_path is None for cfg in configs)
