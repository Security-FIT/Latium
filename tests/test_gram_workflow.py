"""Regression gates for shared cohorts, batch append and cumulative Gram reports."""
import json
import subprocess
from pathlib import Path
import hydra
import pytest
from src.counterfact_selection import generate_random_case_manifest, load_cases_from_manifest, write_case_manifest
from src.gram_experiment import register_batch
from src.graphs.gram import report_experiment, summarize
from src.graphs.runtime import render_run
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


def test_batch_identity_retries_overlaps_and_config_changes(tmp_path):
    _, cohort, _ = cohort_file(tmp_path)
    root = tmp_path / "experiment"
    models = {"a": {"config_hash": "same"}, "b": {"config_hash": "same"}}
    register_batch(root, cohort, models, 0,100,{"workflow":"gram"})
    register_batch(root, cohort, models, 0,100,{"workflow":"gram"})
    register_batch(root, cohort, models, 100,200,{"workflow":"gram"})
    assert len(json.loads((root/"experiment.json").read_text())["models"]["a"]["batches"]) == 2
    with pytest.raises(ValueError, match="Overlapping"):
        register_batch(root, cohort, models,50,150,{"workflow":"gram"})
    with pytest.raises(ValueError, match="setup changed"):
        register_batch(root, cohort, models,200,300,{"workflow":"other"})
    with pytest.raises(ValueError, match="configuration changed"):
        register_batch(root, cohort,{"a":{"config_hash":"changed"}},200,300,{"workflow":"gram"})


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


def test_append_100_to_200_preserves_first_batch_and_invalidates_graphs(tmp_path):
    _, cohort, _ = cohort_file(tmp_path)
    root = tmp_path / "experiment"
    models = {"a":{"config_hash":"a"},"b":{"config_hash":"b"}}
    register_batch(root,cohort,models,0,100,{"workflow":"gram"})
    for model in models:
        write_batch(root/"models"/model/"run",model,cohort,0,100)
    report_experiment(root,graphs=False)
    first = json.loads((root/"report/summary.json").read_text())
    assert first["models"]["a"]["selected"] == 100
    rendered = render_run(root/"models/a/run",preset="gram-report")
    assert rendered["written"]
    assert render_run(root/"models/a/run",preset="gram-report")["skipped"]
    frozen = {str(p):p.read_bytes() for p in (root/"models/a/run/plans").rglob("*.json")}
    register_batch(root,cohort,models,100,200,{"workflow":"gram"})
    for model in models:
        write_batch(root/"models"/model/"run",model,cohort,100,200)
    for path, original in frozen.items():
        assert Path(path).read_bytes() == original
    report_experiment(root,graphs=False)
    report = json.loads((root/"report/summary.json").read_text())
    rows = json.loads((root/"report/cases.json").read_text())
    assert report["models"]["a"]["selected"] == 200
    assert report["models"]["a"]["gram_exact"] == 1.
    assert len({(r["model"],r["case_id"]) for r in rows}) == 400
    assert report["paired_coverage"]["common_evaluated_count"] == 200
    assert report["input_hash"] != first["input_hash"]
    assert report["models"]["a"]["mean_overall_score"] == pytest.approx(3/(1+1/.75+1))
    assert render_run(root/"models/a/run",preset="gram-report")["written"]
    assert render_run(root/"models/a/run",preset="gram-report")["skipped"]
    (root/"models/a/run/graphs/gram-report/accuracy.png").unlink()
    assert render_run(root/"models/a/run",preset="gram-report")["written"]
    report_experiment(root,graphs=False)
    assert json.loads((root/"report/summary.json").read_text())["models"]["a"]["selected"] == 200


def test_errors_pending_and_baseline_denominators():
    rows = [{"status":"complete","predicted_layer":7,"exact":True,"rome_success":False,"metrics":{}},
            {"status":"error","predicted_layer":None,"exact":False,"rome_success":False,"metrics":{}},
            {"status":"pending","predicted_layer":None,"exact":False,"rome_success":False,"metrics":{}}]
    summary = summarize(rows)
    assert summary["selected"] == 3 and summary["exact_count"] == 1
    assert summary["gram_exact"] == 1/3 and summary["gram_exact_evaluated"] == 1
    assert summary["coverage"] == 1/3 and summary["rome_success"] == 0


def test_fleet_command_and_pbs_arguments(tmp_path):
    from jobs.paper_fleet import parse_args
    from jobs.gram_fleet import structural_command
    path, _, _ = cohort_file(tmp_path)
    args = parse_args(["--workflow","gram","--case-index-file",str(path),"--case-start","100","--case-stop","200","--models","gpt2-xl","--run-root",str(tmp_path/"exp")])
    assert args.n_tests == 100
    command = structural_command(args,"gpt2-xl",tmp_path/"exp/models/gpt2-xl/run","m0100-0200")
    assert "structural=gram" in command and "structural.run.start_idx=100" in command
    assert not any("ccs" in v or "experiments" in v for v in command)
    result = subprocess.run(["bash",str(ROOT/"jobs/submit_paper_fleet.sh"),"--dry-run","--workflow","gram","--case-index-file",str(path),"--case-start","100","--case-stop","200","--models","gpt2-xl","--run-root",str(tmp_path/"exp")],capture_output=True,text=True,check=True)
    assert "[dry-run]" in result.stdout


def test_fleet_two_batches_resume_without_duplicate_edits(tmp_path, monkeypatch):
    from jobs import gram_fleet, paper_fleet
    path, cohort, _ = cohort_file(tmp_path)
    root = tmp_path / "fleet"
    commands = []
    monkeypatch.setattr(paper_fleet, "configure_logging", lambda *a: None)
    monkeypatch.setattr(paper_fleet, "model_second_moment_files", lambda *a: [Path("covariance.pt")])
    def fake_run(command, **kwargs):
        commands.append(command)
        start = int(next(c.split("=",1)[1] for c in command if c.startswith("structural.run.start_idx=")))
        count = int(next(c.split("=",1)[1] for c in command if c.startswith("structural.run.n_tests=")))
        write_batch(root/"models/gpt2-xl/run", "gpt2-xl", cohort, start, start+count)
    monkeypatch.setattr(paper_fleet, "run_logged", fake_run)
    def args(start, stop):
        return paper_fleet.parse_args(["--workflow","gram","--case-index-file",str(path),"--case-start",str(start),"--case-stop",str(stop),"--models","gpt2-xl","--run-root",str(root),"--reuse-covariance","--no-graphs"])
    assert gram_fleet.run(args(0,3)) == 0
    assert gram_fleet.run(args(3,6)) == 0
    assert len(commands) == 2
    assert gram_fleet.run(args(0,3)) == 0
    assert gram_fleet.run(args(3,6)) == 0
    assert len(commands) == 2
    assert json.loads((root/"report/summary.json").read_text())["models"]["gpt2-xl"]["selected"] == 6
    assert json.loads((root/"models/gpt2-xl/batches/m0003-0006/state.json").read_text())["status"] == "complete"


def test_append_ignores_reporting_changes_but_rejects_computation_changes(tmp_path, monkeypatch):
    from jobs import gram_fleet, paper_fleet
    path, _, _ = cohort_file(tmp_path)
    source = tmp_path / "source"
    algorithm = source / "src/rome/algorithm.py"
    algorithm.parent.mkdir(parents=True)
    algorithm.write_text("version = 1\n")
    monkeypatch.setattr(paper_fleet, "ROOT", source)
    args = paper_fleet.parse_args(["--workflow", "gram", "--case-index-file", str(path),
                                  "--models", "gpt2-xl", "--run-root", str(tmp_path / "run"),
                                  "--case-start", "0", "--case-stop", "3"])
    gram_fleet.prepare(args)
    graph = source / "src/graphs/plot.py"
    graph.parent.mkdir(parents=True)
    graph.write_text("new_plot = True\n")
    unrelated = source / "src/config/model/another.yaml"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("name: another-model\n")
    args.case_start, args.case_stop = 3, 6
    gram_fleet.prepare(args)
    algorithm.write_text("version = 2\n")
    args.case_start, args.case_stop = 6, 9
    with pytest.raises(ValueError, match="setup changed"):
        gram_fleet.prepare(args)
