"""Gram-only adapter for the existing MetaCentrum fleet and artifact runtime."""
import json
from pathlib import Path
from src.counterfact_selection import load_case_manifest
from src.gram_experiment import digest, locked, register_batch, write_json
from src.common.model_config import load_model_config
from src.common.config import plain


def prepare(args):
    from jobs.paper_fleet import ROOT
    cohort = load_case_manifest(args.case_index_file)
    # Freeze computation code; plots, PBS wrappers and unrelated model configs
    # can change without invalidating the cohort. Selected configs are checked below.
    sources = {}
    for directory in ("src/common", "src/handlers", "src/editing", "src/rome", "src/structural"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix == ".py":
                sources[str(path.relative_to(ROOT))] = digest(path.read_text())
    setup = {"workflow": "gram", "seed": 0, "covariance_samples": args.covariance_samples,
             "source_hash": digest(sources)}
    models = {model: {"config_hash": digest(plain(load_model_config(model)))} for model in args.models}
    batch, catalog = register_batch(args.run_root, cohort, models, args.case_start, args.case_stop, setup)
    return batch, catalog


def structural_command(args, model, root, batch):
    return [args.python, "-m", "src", "structural", "run", "structural=gram",
        f"structural.run.models=[{model}]", f"structural.run.n_tests={args.n_tests}",
        f"structural.run.start_idx={args.case_start}",
        f"structural.run.case_index_file='{Path(args.run_root).resolve() / 'cases.json'}'",
        f"structural.run.output_dir='{root.parent}'", "structural.run.run_id=run",
        f"structural.run.progress_file='{root.parent / 'batches' / batch / 'progress.json'}'",
        "structural.render.enabled=false",
        f"structural.tracking.provider={args.tracking}",
        f"structural.tracking.project={args.wandb_project}",
        f"structural.tracking.group='{args.wandb_group or Path(args.run_root).name}'",
        f"structural.tracking.run_name={model}-{batch}"]


def verify_batch(root, model, plan_id, cohort, start, stop):
    from jobs.paper_fleet import manifest_records, record_payload
    _, records = manifest_records(root)
    records = [r for r in records if r.get("plan_id") == plan_id]
    expected_ids = {str(v) for v in cohort["case_ids"][start:stop]}
    for kind, producer in (("execution", None), ("capture", "gram-localization"), ("analysis", "gram-localization")):
        matches = [r for r in records if r.get("kind") == kind and (producer is None or r.get("producer") == producer)]
        for baseline in (True, False):
            found = [r for r in matches if (r.get("edit_method") in (None, "baseline")) == baseline]
            if len(found) != 1 or found[0].get("status") != "complete":
                raise RuntimeError(f"Missing/failed {kind} for {model}/{plan_id}, baseline={baseline}")
            payload = record_payload(root, found[0])
            cases = payload.get("cases", [])
            expected = {"baseline"} if baseline else expected_ids
            if len(cases) != len(expected) or {str(c["case_id"]) for c in cases} != expected:
                raise RuntimeError(f"Case coverage mismatch in {kind} for {model}/{plan_id}")
    extras = [r for r in records if r.get("kind") in ("capture", "analysis") and r.get("producer") != "gram-localization"]
    if extras:
        raise RuntimeError("Gram batch unexpectedly contains additional captures/analyses")


def run(args):
    from jobs.paper_fleet import configure_logging, run_logged, model_second_moment_files, utc_now
    batch, catalog = prepare(args)
    if args.prepare_only:
        print(f"Prepared {args.run_root}: {batch}, {len(args.models)} models")
        return 0
    root = Path(args.run_root).resolve()
    configure_logging(root)
    cohort = load_case_manifest(root / "cases.json")
    failures = []
    for model in args.models:
        model_dir = root / "models" / model
        state_path = model_dir / "batches" / batch / "state.json"
        state = {"model": model, "batch": batch, "status": "running", "started_at": utc_now()}
        run_root = root / catalog["models"][model]["run_root"]
        plan_id = catalog["models"][model]["batches"][batch]["plan_id"]
        # Independent model/batch lock prevents duplicate PBS retries racing on the same artifacts.
        with locked(state_path.parent):
            try:
                if args.resume and state_path.exists() and json.loads(state_path.read_text()).get("status") == "complete":
                    try:
                        verify_batch(run_root, model, plan_id, cohort, args.case_start, args.case_stop)
                    except (FileNotFoundError, RuntimeError):
                        pass  # Recompute missing artifacts in this resume attempt.
                    else:
                        continue
                write_json(state_path, state)
                config = load_model_config(model)
                files = model_second_moment_files(model, int(config.layer), args.covariance_samples)
                if not files:
                    if args.skip_second_moment:
                        raise FileNotFoundError(f"No configured covariance for {model}; reuse forbids recomputation")
                    run_logged([args.python, "-m", "src", "second-moment", f"model={model}",
                        "model.second_moment_path=null", f"model.second_moment_target_samples={args.covariance_samples}"], stage="covariance", model=model)
                    files = model_second_moment_files(model, int(config.layer), args.covariance_samples)
                if not files:
                    raise FileNotFoundError(f"Missing configured covariance for {model}")
                if not args.covariance_only:
                    run_logged(structural_command(args, model, run_root, batch), stage="gram", model=model)
                    verify_batch(run_root, model, plan_id, cohort, args.case_start, args.case_stop)
                    if not args.no_graphs:
                        run_logged([args.python, "-m", "src", "graphs", "run", str(run_root),
                                    "graphs.renderer_preset=gram-report"], stage="gram-report", model=model)
                state.update(status="covariance_complete" if args.covariance_only else "complete", completed_at=utc_now())
            except Exception as exc:
                state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                failures.append(model)
            write_json(state_path, state)
    if not args.covariance_only:
        from src.graphs.gram import report_experiment
        report_experiment(root, graphs=not args.no_graphs)
    return int(bool(failures))
