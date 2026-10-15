"""Case selection and GRAM report tests."""
import json
from pathlib import Path
import hydra
import pytest
from src.counterfact_selection import generate_random_case_manifest, load_cases_from_manifest, write_case_manifest
from src.graphs.gram import summarize
from src.graphs.runtime import render_run
from src.main import main
from src.results import ArtifactWriter, RunLayout, build_artifact, config_hash
from src.results.ids import execution_id, capture_id, analysis_id
from src.structural.planning import build_plan_summary
from src.structural.hydra_config import structural_config_from_hydra

ROOT = Path(__file__).resolve().parents[1]


def dataset(n=2000):
    return [{"case_id": i+10000, "requested_rewrite": {"subject": f"S{i}", "prompt": "{} lives in", "target_new": {"str": "Paris"}, "target_true": {"str": "Rome"}}} for i in range(n)]


def cohort_file(tmp_path, n=1000):
    ds = dataset()
    cohort = generate_random_case_manifest(count=n, seed=42, dataset_name="fake", split="train", dataset=ds, revision="frozen")
    path = tmp_path / "facts.json"
    write_case_manifest(path, cohort)
    return path, cohort, ds


def test_random_cohort_ranges_and_identity(tmp_path):
    path, cohort, ds = cohort_file(tmp_path)
    assert len(set(cohort["indices"])) == 1000
    assert generate_random_case_manifest(count=1000, seed=42, dataset_name="fake", split="train", dataset=ds, revision="frozen") == cohort
    first = load_cases_from_manifest(path, start_idx=0, n_tests=100, dataset=ds)[1]
    second = load_cases_from_manifest(path, start_idx=100, n_tests=100, dataset=ds)[1]
    assert {c["case_id"] for c in first}.isdisjoint(c["case_id"] for c in second)
    assert [c["case_id"] for c in second] == cohort["case_ids"][100:200]
    assert [c["cohort_position"] for c in second] == list(range(100,200))
    for start, count in ((-1,100),(950,100),(100,-1)):
        with pytest.raises(ValueError):
            load_cases_from_manifest(path, start_idx=start, n_tests=count, dataset=ds)
    changed = dataset(); changed[cohort["indices"][100]]["case_id"] = -1
    with pytest.raises(ValueError, match="case ID mismatch"):
        load_cases_from_manifest(path, start_idx=100, n_tests=1, dataset=changed)
    changed = dataset(); changed[cohort["indices"][100]]["requested_rewrite"]["subject"] = "changed"
    with pytest.raises(ValueError, match="content mismatch"):
        load_cases_from_manifest(path, start_idx=100, n_tests=1, dataset=changed)


def test_case_loader_two_models_use_same_range(tmp_path, monkeypatch):
    from src import counterfact_selection as cf
    from src.structural.execution.case_selection import load_test_cases
    path, cohort, ds = cohort_file(tmp_path)
    monkeypatch.setattr(cf, "load_counterfact_split", lambda *a, **kw: ds)
    cases, metadata = load_test_cases(100, 100, dataset_name="fake", split="train", case_index_file=str(path))
    other, other_metadata = load_test_cases(100, 100, dataset_name="fake", split="train", case_index_file=str(path))
    assert cases == other
    assert metadata == other_metadata
    assert metadata["selected_case_ids"] == cohort["case_ids"][100:200]
    assert metadata["start_idx"] == 100 and metadata["stop_idx"] == 200


def test_gram_preset_resolves_only_gram_even_with_graphs():
    with hydra.initialize_config_dir(config_dir=str(ROOT / "src/config"), version_base=None):
        cfg = hydra.compose(config_name="latium", overrides=["command=structural/plan", "structural=gram", "structural.render.enabled=true"])
    config = structural_config_from_hydra(cfg, run_analysis=True)
    plan = build_plan_summary(config)
    assert plan["resolved_captures"] == ["gram-localization"]
    assert plan["resolved_analyses"] == ["gram-localization"]
    assert plan["resolved_renderers"] == ["gram-report"]
    assert plan["edit_methods"] == ["rome"]


def write_batch(root, model, cohort, start, stop):
    plan = f"cf_{cohort['manifest_hash'][:12]}_m{start:04d}-{stop:04d}_r01"
    writer = ArtifactWriter(root, run_id="run")
    layout = RunLayout(root)
    selected = cohort["case_ids"][start:stop]
    config = {"case_selection": {"manifest_hash":cohort["manifest_hash"], "start_idx":start, "selected_case_ids":selected}}
    execution = build_artifact(artifact_id=execution_id(model,plan,"rome"), kind="execution", producer="rome", run_id="run", model=model, plan_id=plan, edit_method="rome", status="complete", config=config, config_hash=config_hash(config), inputs=[], created_at="now", cases=[{"case_id":str(i),"status":"complete", "edit":{"success":True,"metrics":{"efficacy_score":1.,"paraphrase_score":.5 if start==0 else 1.,"neighborhood_score":1.}}, "error":None} for i in selected], summary={"target_layer":7})
    writer.write(layout.execution_path(model,plan,edit_method="rome"), execution)
    for method, ids in ((None,["baseline"]),("rome",selected)):
        if method is None:
            baseline = build_artifact(artifact_id=execution_id(model,plan,None), kind="execution",producer="baseline",run_id="run",model=model,plan_id=plan,edit_method=None,status="complete",config={},config_hash=config_hash({}),inputs=[],created_at="now",cases=[{"case_id":"baseline","status":"complete"}],summary={})
            writer.write(layout.execution_path(model,plan,edit_method=None),baseline)
        capture = build_artifact(artifact_id=capture_id(model,plan,"gram-localization",method),kind="capture",producer="gram-localization",run_id="run",model=model,plan_id=plan,edit_method=method,status="complete",config={},config_hash=config_hash({}),inputs=[],created_at="now",cases=[{"case_id":str(i),"status":"complete","data":{}} for i in ids],summary={})
        writer.write(layout.capture_path(model,plan,"gram-localization",edit_method=method),capture)
        analysis = build_artifact(artifact_id=analysis_id(model,plan,method,"detection","gram-localization",config_hash({})),kind="analysis",producer="gram-localization",run_id="run",model=model,plan_id=plan,edit_method=method,status="complete",config={},config_hash=config_hash({}),inputs=[],created_at="now",cases=[{"case_id":str(i),"status":"complete","data":{"anomalous_layer":7,"localization":{"selected_layer":7,"layer_scores":{"6":1.,"7":2.,"8":1.}}}} for i in ids],summary={})
        writer.write(layout.analysis_path(model,plan,method,"detection","gram-localization",config_hash({})),analysis)
    return plan


def test_render_saved_artifacts_and_reuse(tmp_path):
    _, cohort, _ = cohort_file(tmp_path)
    root = tmp_path / "run"
    write_batch(root, "a", cohort, 0, 100)
    originals = {str(p): p.read_bytes() for p in (root / "plans").rglob("*.json")}
    assert main(["graphs", "run", str(root)]) == 0
    assert (root / "graphs/gram-report/profiles.pdf").is_file()
    report = json.loads((root / "graphs/gram-report/summary.json").read_text())
    assert report["models"]["a"]["selected"] == 100
    assert report["models"]["a"]["gram_exact"] == 1.
    assert render_run(root, preset="gram-report",
                      renderer_options={"gram-report": {"formats": ["png", "pdf", "json"]}})["skipped"]
    for path, original in originals.items():
        assert Path(path).read_bytes() == original
    (root / "graphs/gram-report/accuracy.png").unlink()
    assert main(["graphs", "run", str(root), "graphs.force=true"]) == 0
    assert (root / "graphs/gram-report/accuracy.png").is_file()


def test_errors_pending_and_baseline_denominators():
    rows = [{"status":"complete","predicted_layer":7,"exact":True,"rome_success":False,"metrics":{}},
            {"status":"error","predicted_layer":None,"exact":False,"rome_success":False,"metrics":{}},
            {"status":"pending","predicted_layer":None,"exact":False,"rome_success":False,"metrics":{}}]
    summary = summarize(rows)
    assert summary["selected"] == 3 and summary["exact_count"] == 1
    assert summary["gram_exact"] == 1/3 and summary["gram_exact_evaluated"] == 1
    assert summary["coverage"] == 1/3 and summary["rome_success"] == 0








def test_counterfact_disk_cache_avoids_hub_and_preserves_manifest(tmp_path, monkeypatch):
    import datasets
    from src.counterfact_selection import load_counterfact_split
    ds = datasets.Dataset.from_list(dataset(10))
    path = tmp_path / "cached-counterfact"
    datasets.DatasetDict({"train": ds}).save_to_disk(path)
    manifest = generate_random_case_manifest(count=5, seed=42, dataset_name="fake", split="train", dataset=ds, revision="pinned")
    manifest_path = tmp_path / "cases.json"
    write_case_manifest(manifest_path, manifest)
    monkeypatch.setenv("LATIUM_COUNTERFACT_CACHE", str(path))
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: pytest.fail("Unexpected Hub access"))
    loaded = load_counterfact_split("fake", "train", revision="pinned")
    assert len(loaded) == len(ds)
    metadata, cases = load_cases_from_manifest(manifest_path, start_idx=0, n_tests=5)
    assert [c["case_id"] for c in cases] == manifest["case_ids"]
    assert metadata["manifest_hash"] == manifest["manifest_hash"]

    # A different representation is allowed; different facts are still rejected.
    changed = loaded.to_list()
    changed[manifest["indices"][0]]["requested_rewrite"]["subject"] = "changed"
    monkeypatch.setattr(datasets, "load_from_disk", lambda _: datasets.DatasetDict({"train": datasets.Dataset.from_list(changed)}))
    with pytest.raises(ValueError, match="content mismatch"):
        load_cases_from_manifest(manifest_path, n_tests=5)
