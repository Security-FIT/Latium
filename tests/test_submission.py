"""CSF gates: frozen assignments, covariance ownership and report semantics."""
import json
from pathlib import Path

import pytest
import torch

from src.graphs.gram import case_rows
from src.submission.inputs import INPUTS, load_campaign, verify_packaged_inputs
from src.submission.run import _pin, compute_covariance, download_checkpoint, relocate_model


def test_packaged_campaigns_keep_the_original_assignments_and_repairs():
    counts = verify_packaged_inputs()
    assert counts["classic_facts"] == 1000
    assert counts["basecov_workflows"] == 790
    classic = json.loads((INPUTS / "classic-facts.json").read_text())
    for entry in load_campaign("classic")["entries"]:
        assigned = [case for batch in entry["batches"] for case in batch["selected_case_ids"]]
        assert len(assigned) == len(set(assigned)) == 1000
        assert set(assigned) == set(map(str, classic["case_ids"]))
    repaired = json.loads((INPUTS / "classic-repair24-facts.json").read_text())
    qwen = next(e for e in load_campaign("classic")["entries"] if e["family"] == "qwen3.5-4b")
    batch = qwen["batches"][-1]
    assert batch["count"] == repaired["count"] == 24
    assert batch["selected_case_ids"] == list(map(str, repaired["case_ids"]))
    base = {e["model_id"] for e in load_campaign("finetune-base")["entries"]}
    own = load_campaign("finetune-own")["entries"]
    assert {e["model_id"] for e in own} <= base
    assert sum(e["evaluated"] for e in own) == 744
    prior = {e["model_id"]: e for e in json.loads((INPUTS / "basecov-selected-checkpoints.json").read_text())}
    assert {e["model_id"] for e in own} == {name for name, e in prior.items() if e["rome_efficacy_success"]}


def test_missing_revision_never_resolves_a_current_hub_tip():
    for revision in (None, "main", "latest", "123"):
        with pytest.raises(ValueError, match="immutable revision"):
            _pin(revision, "original/base")


def test_download_resolves_historical_alias_at_the_exact_pin(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import huggingface_hub
    calls = []
    pin = "a" * 40
    class Api:
        def __init__(self, token):
            pass
        def model_info(self, *, repo_id, revision):
            calls.append((repo_id, revision))
            return SimpleNamespace(id="openai-community/gpt2-large", sha=pin)
    def download(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs["local_dir"])
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    path = download_checkpoint("gpt2-large", pin, ["config.json"], {"config.json": 2}, tmp_path)
    assert calls[0] == ("gpt2-large", pin)
    assert (calls[1]["repo_id"], calls[1]["revision"]) == ("openai-community/gpt2-large", pin)
    assert path.name == "gpt2-large"
    assert json.loads((path / ".latium-snapshot.json").read_text())["resolved_repo"] == "openai-community/gpt2-large"
    with pytest.raises(ValueError, match="different revision"):
        download_checkpoint("gpt2-large", "b" * 40, ["config.json"], None, tmp_path)
    assert len(calls) == 3


@pytest.mark.parametrize("repo", ["gpt2-medium", "author/finetune"])
def test_relocated_checkpoint_cache_loads_the_exact_snapshot(repo, tmp_path, monkeypatch):
    import src.submission.run as runner
    def download(name, revision, files, sizes, cache):
        return Path(cache) / "pinned" / "models" / name
    monkeypatch.setattr(runner, "download_checkpoint", download)
    entry = {"model_id": repo, "model_key": "checkpoint", "revision": "a" * 40, "checkpoint_type": "full"}
    config = relocate_model(entry, {"name": repo, "layer": 8, "epochs": 20, "prefix_mode": "self"}, tmp_path, {})
    assert Path(config["models_dir"]) / repo == tmp_path / "weights/pinned/models" / repo
    assert config["checkpoint_revision"] == "a" * 40
    assert (config["layer"], config["epochs"]) == (8, 20)


def test_fresh_covariance_is_recomputed_on_the_finetuned_checkpoint(tmp_path, monkeypatch):
    import src.handlers.rome as handlers
    import src.rome.common as rome
    calls = []
    class Handler:
        def __init__(self, cfg):
            self.cfg = cfg
        def remove_hooks(self):
            pass
    def compute(handler, N_rounds, N_k):
        calls.append((handler.cfg.model.name, N_rounds * N_k))
        return torch.eye(3), N_rounds * N_k, "SM_Method.WIKIPEDIA"
    monkeypatch.setattr(handlers, "ModelHandler", Handler)
    monkeypatch.setattr(rome, "compute_second_moment", compute)
    config = {"name": "author/finetune", "checkpoint_revision": "a" * 40, "layer": 8}
    compute_covariance(config, tmp_path, 0, 100000, fresh=True)
    compute_covariance(config, tmp_path, 0, 100000, fresh=True)
    assert calls == [("author/finetune", 100000), ("author/finetune", 100000)]
    compute_covariance(config, tmp_path, 0, 100000, fresh=False)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="Incompatible covariance"):
        compute_covariance({**config, "checkpoint_revision": "b" * 40}, tmp_path, 0, 100000, fresh=False)


def test_baseline_fp_uses_future_configured_edit_layer_and_efficacy_score():
    execution = {"run": {"model": "checkpoint", "plan_id": "one", "edit_method": "rome"},
                 "config": {"model": {"target_layer": 7}}, "summary": {"target_layer": 7},
                 "cases": [{"case_id": "fact", "status": "complete", "edit": {"success": False, "metrics": {"efficacy_score": 1.0}}}]}
    baseline = {"producer": "gram-localization", "run": {"model": "checkpoint", "plan_id": "one", "edit_method": None},
                "cases": [{"case_id": "baseline", "status": "complete", "data": {"anomalous_layer": 7}}]}
    edited = {"producer": "gram-localization", "run": execution["run"],
              "cases": [{"case_id": "fact", "status": "complete", "data": {"anomalous_layer": 8}}]}
    rows, _ = case_rows([execution], [baseline, edited])
    assert rows[0]["baseline_fp"] is True
    assert rows[0]["rome_success"] is True
    assert rows[0]["exact"] is False
    execution["summary"]["target_layer"] = 8
    with pytest.raises(ValueError, match="configured layer"):
        case_rows([execution], [baseline, edited])


def test_new_renderer_selects_repairs_once_and_preserves_source_bytes(tmp_path):
    from test_gram_workflow import cohort_file, write_batch
    from src.submission.render import render_new_campaign
    _, cohort, _ = cohort_file(tmp_path, n=3)
    root = tmp_path / "reproduced"
    entry = {"family": "gpt2-medium", "model_id": "original/base", "model_key": "model", "revision": "a" * 40}
    for name, start, stop, selected in (("first", 0, 3, cohort["case_ids"][:2]), ("repair", 2, 3, cohort["case_ids"][2:])):
        run = root / "runs/model" / name / "run"
        write_batch(run, "model", cohort, start, stop)
        receipt = {"campaign": "classic", "validation_subset": True, "entry": entry,
                   "batch": {"selected_case_ids": list(map(str, selected))}, "run_root": str(run)}
        (run.parent / "reproduction.json").write_text(json.dumps(receipt))
    original = {p: p.read_bytes() for p in root.rglob("*.json")}
    result = render_new_campaign("classic", root, tmp_path / "graphs")
    assert result["cases"] == 3 and result["validation_subset"] is True
    rows = json.loads((tmp_path / "graphs/native/cases.json").read_text())
    assert len({r["case_id"] for r in rows}) == 3
    repaired = next(r for r in rows if r["case_id"] == str(cohort["case_ids"][2]))
    assert repaired["cohort_position"] == 2
    assert all(p.read_bytes() == content for p, content in original.items())
    assert (tmp_path / "graphs/profiles/gpt2-medium-linear.pdf").is_file()


def test_saved_artifact_tampering_fails_before_reporting(tmp_path):
    from test_gram_workflow import cohort_file, write_batch
    from src.submission.artifacts import read_run
    _, cohort, _ = cohort_file(tmp_path, n=1)
    run = tmp_path / "run"
    write_batch(run, "model", cohort, 0, 1)
    reader, _ = read_run(run)
    record = next(reader.records(kind="execution"))
    path = run / record["path"]
    payload = json.loads(path.read_text())
    payload["cases"][0]["edit"]["metrics"]["efficacy_score"] = 0.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Content hash mismatch"):
        read_run(run)
