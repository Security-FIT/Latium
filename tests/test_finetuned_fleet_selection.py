import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from jobs import prepare_finetuned_gram_fleets as preparation
from scripts import fetch_finetuned_qwen3_8b as fetcher


def test_fetcher_uses_api_tag_filter_and_stable_download_sort(monkeypatch):
    tag = "base_model:finetune:org/base"
    calls = []

    def request(url, token=None):
        query = parse_qs(urlparse(url).query)
        calls.append(query)
        assert query["filter"] == [tag] and "other" not in query
        assert query["pipeline_tag"] == ["text-generation"]
        assert query["direction"] == ["-1"]
        return [{"id": "org/z", "downloads": 10, "tags": [tag], "pipeline_tag": "text-generation"},
                {"id": "org/a", "downloads": 10, "tags": [tag], "pipeline_tag": "text-generation"},
                {"id": "org/unrelated", "downloads": 999, "tags": [], "pipeline_tag": "text-generation"},
                {"id": "org/wrong-task", "downloads": 999, "tags": [tag], "pipeline_tag": "feature-extraction"}]

    monkeypatch.setattr(fetcher, "_request_json", request)
    assert [e["model_id"] for e in fetcher.fetch_models(base_model="org/base", limit=3)] == ["org/a", "org/z"]
    assert len(calls) == 1


def details(model, files, revision="a" * 40):
    return SimpleNamespace(id=model, sha=revision, card_data=None,
                           siblings=[SimpleNamespace(rfilename=name, size=20) for name in files])


def test_fetcher_reads_all_pages_before_selecting_top_models(monkeypatch):
    import io

    tag = "base_model:finetune:org/base"
    urls = []

    def response(request, timeout):
        urls.append(request.full_url)
        last = "cursor=next" in request.full_url
        stream = io.BytesIO(json.dumps([{"id": "org/a" if last else "org/z", "downloads": 10,
                                         "tags": [tag], "pipeline_tag": "text-generation"}]).encode())
        stream.headers = {} if last else {"Link": '<https://huggingface.co/api/models?cursor=next>; rel="next"'}
        return stream

    monkeypatch.setattr(fetcher.urllib.request, "urlopen", response)
    assert [e["model_id"] for e in fetcher.fetch_models(base_model="org/base", limit=1)] == ["org/a"]
    assert len(urls) == 2


CONFIG = {"model_type": "toy", "hidden_size": 4, "num_hidden_layers": 2,
          "architectures": ["ToyForCausalLM"]}


def metadata_file(directory, model, filename, value):
    path = directory / (model.replace("/", "_") + "_" + filename)
    path.write_text(json.dumps(value))
    return str(path)


def test_preparation_freezes_top_ranks_including_adapters_and_errors(tmp_path, monkeypatch):
    calls = []
    discovery_calls = []

    class Api:
        def model_info(self, model, revision=None, files_metadata=False):
            calls.append((model, revision))
            if model == "org/base":
                result = details("org/canonical", ["config.json", "model.safetensors"], "b" * 40 if revision else "a" * 40)
                result.card_data = SimpleNamespace(base_model="org/parent")
                return result
            if model == "org/unsupported":
                return details(model, ["model.gguf"])
            if model == "org/adapter":
                return details(model, ["adapter_config.json", "adapter_model.safetensors"])
            return details(model, ["config.json", "model.safetensors"], "b" * 40 if revision else "a" * 40)

        def list_models(self, *, filter, pipeline_tag, sort, full):
            discovery_calls.append(filter)
            assert filter == "base_model:finetune:org/base"
            assert pipeline_tag == "text-generation" and sort == "downloads" and full is True
            return [SimpleNamespace(id=name, downloads=downloads, tags=[filter], pipeline_tag=pipeline_tag)
                    for name, downloads in [("org/unsupported", 30), ("org/adapter", 20), ("org/later-full", 10)]] + [
                SimpleNamespace(id="org/wrong-task", downloads=999, tags=[filter], pipeline_tag="text-classification"),
                SimpleNamespace(id="org/wrong-base", downloads=999, tags=["base_model:finetune:org/base-Base"], pipeline_tag=pipeline_tag),
                SimpleNamespace(id="org/adapter-only", downloads=999, tags=["base_model:adapter:org/base"], pipeline_tag=pipeline_tag)]

    def metadata(model, filename, **kwargs):
        cfg = {"base_model_name_or_path": "org/base", "revision": "v1"} if filename == "adapter_config.json" else CONFIG
        return metadata_file(tmp_path, model, filename, cfg)

    monkeypatch.setattr(preparation, "load_model_config", lambda model: SimpleNamespace(name="org/base", layer=1))
    monkeypatch.setattr(preparation, "hf_hub_download", metadata)
    result = preparation.prepare_family("qwen3-8b", tmp_path / "out", 2, Api(), None)
    assert result["selected"] == 2 and result["adapter"] == 1 and result["errors"] == 1
    manifest = json.loads(Path(result["manifest"]).read_text())
    records = manifest["models"]
    assert [record["model_id"] for record in records] == ["org/unsupported", "org/adapter"]
    assert [record["rank"] for record in records] == [1, 2]
    assert all(record["pipeline_tag"] == "text-generation" for record in records)
    assert manifest["discovery_filter"] == ["base_model:finetune:org/base"]
    assert manifest["relations"] == ["finetune"]
    assert discovery_calls == ["base_model:finetune:org/base"]
    discovery = json.loads((Path(result["manifest"]).parent / "discovery.json").read_text())
    assert [record["model_id"] for record in discovery["models"]] == ["org/unsupported", "org/adapter", "org/later-full"]
    assert records[0]["selection_error"]
    assert records[1]["adapter_base_revision"] == "b" * 40
    assert records[1]["adapter_base_files"] == ["config.json", "model.safetensors"]
    assert ("org/base", "v1") in calls
    assert not any(model == "org/later-full" for model, _ in calls)
    # A repeated metadata preparation keeps types and resolved revisions from its audit.
    calls.clear()
    repeated = preparation.prepare_family("qwen3-8b", tmp_path / "out", 2, Api(), None)
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

        def list_models(self, *, filter, pipeline_tag, sort, full):
            return [SimpleNamespace(id=name, downloads=10, tags=[filter], pipeline_tag=pipeline_tag)
                    for name in ["unsloth/base", "org/wrong", "org/quantized", "org/other-size"]]

    monkeypatch.setattr(preparation, "load_model_config", lambda model: SimpleNamespace(name="org/base", layer=1))
    configs = {"org/wrong": {**CONFIG, "architectures": ["ToyModel"]},
               "org/quantized": {**CONFIG, "quantization_config": {"bits": 4}},
               "org/other-size": {**CONFIG, "hidden_size": 8, "architectures": None}}
    monkeypatch.setattr(preparation, "hf_hub_download", lambda model, filename, **kwargs:
                        metadata_file(tmp_path, model, filename, configs.get(model, CONFIG)))
    result = preparation.prepare_family("toy", tmp_path / "out", 100, Api(), None)
    assert result["requested"] == 100 and result["selected"] == 4 and result["full"] == 4 and result["errors"] == 2
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert [entry["model_id"] for entry in manifest["models"]] == ["org/other-size", "org/quantized", "org/wrong", "unsloth/base"]
    assert manifest["models"][0]["matches_classic_dimensions"] is False
    assert "selection_error" not in manifest["models"][0]
    assert "Quantized" in manifest["models"][1]["selection_error"]
    assert "initialize missing weights" in manifest["models"][2]["selection_error"]
    assert "selection_error" not in manifest["models"][3]


def test_preparation_resolves_shared_adapter_base_once(tmp_path, monkeypatch):
    calls = []

    class Api:
        def model_info(self, model, revision=None, files_metadata=False):
            calls.append((model, revision, files_metadata))
            return details(model, ["config.json", "model.safetensors"] if model == "org/base"
                           else ["adapter_config.json", "adapter_model.bin"])

        def list_models(self, *, filter, pipeline_tag, sort, full):
            return [SimpleNamespace(id=name, downloads=10, tags=[filter], pipeline_tag=pipeline_tag) for name in ["org/b", "org/a"]]

    monkeypatch.setattr(preparation, "load_model_config", lambda model: SimpleNamespace(name="org/base", layer=1))
    monkeypatch.setattr(preparation, "hf_hub_download", lambda model, filename, **kwargs:
                        metadata_file(tmp_path, model, filename, {"base_model_name_or_path": "org/base", "revision": "v2"}
                                      if filename == "adapter_config.json" else CONFIG))
    result = preparation.prepare_family("toy", tmp_path / "out", 100, Api(), None)
    assert result["adapter"] == 2
    assert calls.count(("org/base", "v2", True)) == 1


def test_preparation_selects_exact_top_100_before_runtime_checks(tmp_path, monkeypatch):
    inspected = []

    class Api:
        def model_info(self, model, files_metadata=False):
            inspected.append(model)
            return details(model, ["model.gguf"] if model == "org/fine104"
                           else ["config.json", "model.safetensors"])

        def list_models(self, *, filter, pipeline_tag, sort, full):
            return [SimpleNamespace(id=f"org/fine{index:03d}", downloads=index,
                                    tags=[filter], pipeline_tag=pipeline_tag) for index in range(105)]

    monkeypatch.setattr(preparation, "load_model_config", lambda model: SimpleNamespace(name="org/base", layer=1))
    monkeypatch.setattr(preparation, "hf_hub_download", lambda model, filename, **kwargs:
                        metadata_file(tmp_path, model, filename, CONFIG))
    result = preparation.prepare_family("toy", tmp_path / "out", 100, Api(), None)
    records = json.loads(Path(result["manifest"]).read_text())["models"]
    assert result["selected"] == 100 and result["errors"] == 1
    assert [entry["model_id"] for entry in records] == [f"org/fine{index:03d}" for index in range(104, 4, -1)]
    assert [entry["rank"] for entry in records] == list(range(1, 101))
    assert "org/fine004" not in inspected


def test_preparation_does_not_classify_rate_limits_as_unsupported(tmp_path, monkeypatch):
    import pytest

    class RateLimit(Exception):
        response = SimpleNamespace(status_code=429)

    class Api:
        def model_info(self, model, files_metadata=False):
            if model != "org/base":
                raise RateLimit("429 Too many requests")
            return details(model, ["config.json", "model.safetensors"])

        def list_models(self, *, filter, pipeline_tag, sort, full):
            return [SimpleNamespace(id="org/fine", downloads=10, tags=[filter], pipeline_tag=pipeline_tag)]

    monkeypatch.setattr(preparation, "load_model_config", lambda model: SimpleNamespace(name="org/base", layer=1))
    monkeypatch.setattr(preparation, "hf_hub_download", lambda model, filename, **kwargs:
                        metadata_file(tmp_path, model, filename, CONFIG))
    with pytest.raises(RateLimit):
        preparation.prepare_family("toy", tmp_path / "out", 100, Api(), None)
