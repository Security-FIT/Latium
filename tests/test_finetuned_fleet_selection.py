import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from jobs import prepare_finetuned_gram_fleets as preparation
from scripts import fetch_finetuned_qwen3_8b as fetcher


def test_fetcher_uses_api_tag_filter_and_stable_download_sort(monkeypatch):
    tag = "base_model:finetune:org/base"

    def request(url, token=None):
        query = parse_qs(urlparse(url).query)
        assert query["filter"] in ([tag], ["base_model:adapter:org/base"]) and "other" not in query
        assert query["direction"] == ["-1"]
        if query["filter"] == ["base_model:adapter:org/base"]:
            return [{"id": "org/adapter", "downloads": 20, "tags": query["filter"]}]
        return [{"id": "org/z", "downloads": 10, "tags": [tag]},
                {"id": "org/a", "downloads": 10, "tags": [tag]},
                {"id": "org/unrelated", "downloads": 999, "tags": []}]

    monkeypatch.setattr(fetcher, "_request_json", request)
    assert [e["model_id"] for e in fetcher.fetch_models(base_model="org/base", limit=3)] == ["org/adapter", "org/a", "org/z"]


def details(model, files, revision="a" * 40):
    return SimpleNamespace(id=model, sha=revision, card_data=None,
                           siblings=[SimpleNamespace(rfilename=name, size=20) for name in files])


CONFIG = {"model_type": "toy", "hidden_size": 4, "num_hidden_layers": 2,
          "architectures": ["ToyForCausalLM"]}


def metadata_file(directory, model, filename, value):
    path = directory / (model.replace("/", "_") + "_" + filename)
    path.write_text(json.dumps(value))
    return str(path)


def test_preparation_freezes_top_ranks_including_adapters_and_errors(tmp_path, monkeypatch):
    calls = []

    class Api:
        def model_info(self, model, revision=None, files_metadata=False):
            calls.append((model, revision))
            if model == "org/unsupported":
                return details(model, ["model.gguf"])
            if model == "org/adapter":
                return details(model, ["adapter_config.json", "adapter_model.safetensors"])
            return details(model, ["config.json", "model.safetensors"], "b" * 40 if revision else "a" * 40)

        def list_models(self, *, filter, sort, limit, full):
            assert sort == "downloads"
            if filter == "base_model:finetune:org/base":
                return [SimpleNamespace(id=name, downloads=downloads, tags=[filter], pipeline_tag="feature-extraction")
                        for name, downloads in [("org/unsupported", 30), ("org/later-full", 10)]]
            assert filter == "base_model:adapter:org/base"
            return [SimpleNamespace(id="org/adapter", downloads=20, tags=[filter], pipeline_tag="text-classification")]

    def metadata(model, filename, **kwargs):
        cfg = {"base_model_name_or_path": "org/base", "revision": "v1"} if filename == "adapter_config.json" else CONFIG
        return metadata_file(tmp_path, model, filename, cfg)

    monkeypatch.setattr(preparation, "load_model_config", lambda model: SimpleNamespace(name="org/base", layer=1))
    monkeypatch.setattr(preparation, "hf_hub_download", metadata)
    result = preparation.prepare_family("toy", tmp_path / "out", 2, 300, Api(), None)
    assert result["selected"] == 2 and result["adapter"] == 1 and result["errors"] == 1
    manifest = json.loads(Path(result["manifest"]).read_text())
    records = manifest["models"]
    assert [record["model_id"] for record in records] == ["org/unsupported", "org/adapter"]
    assert [record["rank"] for record in records] == [1, 2]
    assert records[0]["selection_error"]
    assert records[1]["adapter_base_revision"] == "b" * 40
    assert records[1]["adapter_base_files"] == ["config.json", "model.safetensors"]
    assert ("org/base", "v1") in calls
    assert not any(model == "org/later-full" for model, _ in calls)
    # A repeated metadata preparation keeps types and resolved revisions from its audit.
    calls.clear()
    repeated = preparation.prepare_family("toy", tmp_path / "out", 2, 300, Api(), None)
    assert repeated["adapter"] == 1 and calls == [("org/base", None)]


def test_preparation_keeps_full_models_and_records_runtime_diagnostics(tmp_path, monkeypatch):
    directory = tmp_path / "out/toy"
    directory.mkdir(parents=True)
    (directory / "selection-audit.json").write_text(json.dumps({"selection_policy": "top-downloads-including-adapters-v1", "models": [
        {"model_id": "unsloth/base", "selection_error": "Base/classic redistribution"},
        {"model_id": "org/wrong", "selection_error": "Architecture differs from the classic config"}]}))

    class Api:
        def model_info(self, model, files_metadata=False):
            return details(model, ["config.json", "model.safetensors"])

        def list_models(self, *, filter, sort, limit, full):
            if filter.startswith("base_model:adapter:"):
                return []
            return [SimpleNamespace(id=name, downloads=10, tags=[filter], pipeline_tag="feature-extraction")
                    for name in ["unsloth/base", "org/wrong", "org/quantized", "org/other-size"]]

    monkeypatch.setattr(preparation, "load_model_config", lambda model: SimpleNamespace(name="org/base", layer=1))
    configs = {"org/wrong": {**CONFIG, "architectures": ["ToyModel"]},
               "org/quantized": {**CONFIG, "quantization_config": {"bits": 4}},
               "org/other-size": {**CONFIG, "hidden_size": 8, "architectures": None}}
    monkeypatch.setattr(preparation, "hf_hub_download", lambda model, filename, **kwargs:
                        metadata_file(tmp_path, model, filename, configs.get(model, CONFIG)))
    result = preparation.prepare_family("toy", tmp_path / "out", 100, 300, Api(), None)
    assert result["requested"] == 100 and result["selected"] == 4 and result["full"] == 4 and result["errors"] == 2
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert [entry["model_id"] for entry in manifest["models"]] == ["org/other-size", "org/quantized", "org/wrong", "unsloth/base"]
    assert manifest["models"][0]["matches_classic_dimensions"] is False
    assert "selection_error" not in manifest["models"][0]
    assert "Quantized" in manifest["models"][1]["selection_error"]
    assert "initialize missing weights" in manifest["models"][2]["selection_error"]
    assert "selection_error" not in manifest["models"][3]
    # Migrate the prior dimension rejection using frozen metadata without HF refetches.
    audit_path = directory / "selection-audit.json"
    audit = json.loads(audit_path.read_text())
    audit["models"][0]["selection_error"] = "ValueError: Checkpoint dimensions differ from the classic model config"
    audit["models"][0].pop("matches_classic_dimensions")
    audit_path.write_text(json.dumps(audit))
    monkeypatch.setattr(Api, "model_info", lambda self, model, **kwargs:
                        details(model, ["config.json", "model.safetensors"]) if model == "org/base"
                        else (_ for _ in ()).throw(AssertionError("Cached checkpoint metadata refetched")))
    repeated = preparation.prepare_family("toy", tmp_path / "out", 100, 300, Api(), None)
    assert repeated["errors"] == 2
    assert json.loads(Path(repeated["manifest"]).read_text())["models"][0]["matches_classic_dimensions"] is False


def test_preparation_resolves_shared_adapter_base_once(tmp_path, monkeypatch):
    calls = []

    class Api:
        def model_info(self, model, revision=None, files_metadata=False):
            calls.append((model, revision, files_metadata))
            return details(model, ["config.json", "model.safetensors"] if model == "org/base"
                           else ["adapter_config.json", "adapter_model.bin"])

        def list_models(self, *, filter, sort, limit, full):
            if filter.startswith("base_model:finetune:"):
                return []
            return [SimpleNamespace(id=name, downloads=10, tags=[filter]) for name in ["org/b", "org/a"]]

    monkeypatch.setattr(preparation, "load_model_config", lambda model: SimpleNamespace(name="org/base", layer=1))
    monkeypatch.setattr(preparation, "hf_hub_download", lambda model, filename, **kwargs:
                        metadata_file(tmp_path, model, filename, {"base_model_name_or_path": "org/base", "revision": "v2"}
                                      if filename == "adapter_config.json" else CONFIG))
    result = preparation.prepare_family("toy", tmp_path / "out", 100, 300, Api(), None)
    assert result["adapter"] == 2
    assert calls.count(("org/base", "v2", True)) == 1


def test_preparation_does_not_classify_rate_limits_as_unsupported(tmp_path, monkeypatch):
    import pytest

    class RateLimit(Exception):
        response = SimpleNamespace(status_code=429)

    class Api:
        def model_info(self, model, files_metadata=False):
            if model != "org/base":
                raise RateLimit("429 Too many requests")
            return details(model, ["config.json", "model.safetensors"])

        def list_models(self, *, filter, sort, limit, full):
            return [SimpleNamespace(id="org/fine", downloads=10, tags=[filter])]

    monkeypatch.setattr(preparation, "load_model_config", lambda model: SimpleNamespace(name="org/base", layer=1))
    monkeypatch.setattr(preparation, "hf_hub_download", lambda model, filename, **kwargs:
                        metadata_file(tmp_path, model, filename, CONFIG))
    with pytest.raises(RateLimit):
        preparation.prepare_family("toy", tmp_path / "out", 100, 300, Api(), None)
