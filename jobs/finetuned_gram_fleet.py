#!/usr/bin/env python3
"""Sequential HF checkpoints: baseline Gram, ROME, edited Gram."""
import argparse
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from src.common.model_config import MODEL_CONFIG_DIR, fleet_model_key, load_model_config
from src.counterfact_selection import load_case_manifest
from src.gram_experiment import digest, locked, write_json
from jobs import gram_fleet
from jobs.paper_fleet import parse_args as gram_args, utc_now


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Classic model config key, e.g. qwen3-8b")
    parser.add_argument("--hf-base-model", help="HF finetune tag to discover; defaults to config.name")
    parser.add_argument("--models-manifest", help="Existing fleet JSON instead of HF discovery")
    parser.add_argument("--model-count", type=int, default=100)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--download-root", help="Dedicated download parent; keep the same path on retries")
    parser.add_argument("--case-index-file", default=str(ROOT / "manifests/counterfact_seed42_n1000.json"))
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--case-stop", type=int)
    parser.add_argument("--n-tests", type=int, default=1, help="ROME facts per checkpoint (independent of model-count)")
    parser.add_argument("--covariance-samples", type=int, default=100000)
    parser.add_argument("--finetuned-covariance", action="store_true",
                        help="Compute covariance per checkpoint instead of reusing the original model's statistics")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tracking", choices=("none", "wandb"), default="none")
    parser.add_argument("--wandb-project", default="latium")
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--keep-downloads", action=argparse.BooleanOptionalAction, default=True,
                        help="Retain pinned checkpoint files for reruns (default); --no-keep-downloads uses disposable run-local files")
    parser.add_argument("--retry-failed-facts", action="store_true", help="Try reserve facts until one ROME edit succeeds")
    parser.add_argument("--random-facts", action="store_true",
                        help="Pick a random starting fact per checkpoint, save it for resume, and wrap for reserve facts")
    parser.add_argument("--max-fact-attempts", type=int, default=1000,
                        help="Maximum candidate facts per checkpoint (default: 1000), including tracing rejections and ROME failures")
    parser.add_argument("--causal-kuba-fix", action="store_true", help="Trace the same fact before ROME; save trace artifacts")
    parser.add_argument("--checkpoint-stop", "--checkpoint-limit", dest="checkpoint_limit", type=int,
                        help="Exclusive frozen checkpoint stop; cohort stays fixed (default: all)")
    parser.add_argument("--checkpoint-start", type=int, default=0, help="Start at this zero-based frozen checkpoint position")
    parser.add_argument("--prefix-cache-file", help="Existing classic external prefix pool")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze IDs/revisions/configs and facts without downloading weights")
    args = parser.parse_args(argv)
    args.run_root = str(Path(args.run_root).resolve())
    args.case_index_file = str(Path(args.case_index_file).resolve())
    args.case_stop = args.case_stop if args.case_stop is not None else args.case_start + args.n_tests
    if (args.model_count <= 0 or args.covariance_samples <= 0 or args.max_fact_attempts <= 0 or args.case_start < 0
            or args.case_stop <= args.case_start or args.case_stop > load_case_manifest(args.case_index_file)["count"]):
        parser.error("Invalid model count, fact range, or covariance sample count")
    if args.checkpoint_limit is not None and not 0 < args.checkpoint_limit <= args.model_count:
        parser.error("checkpoint-limit must be between 1 and model-count")
    if not 0 <= args.checkpoint_start < (args.checkpoint_limit or args.model_count):
        parser.error("checkpoint-start must precede checkpoint-limit/model-count")
    if (args.retry_failed_facts or args.causal_kuba_fix) and args.case_stop != args.case_start + 1:
        parser.error("Fact retries and tracing require one initial fact per checkpoint")
    if args.random_facts and (args.case_start != 0 or args.case_stop != 1):
        parser.error("Random facts require one edit per checkpoint with case-start=0 and case-stop=1")
    return args


def rome_case(root, plan_id):
    from jobs.paper_fleet import manifest_records, record_payload
    _, records = manifest_records(root)
    execution = [r for r in records if r.get("kind") == "execution" and r.get("edit_method") == "rome"
                 and r.get("plan_id") == plan_id]
    if len(execution) != 1:
        raise RuntimeError("Missing ROME execution artifact")
    cases = record_payload(root, execution[0])["cases"]
    if len(cases) != 1 or cases[0]["status"] != "complete":
        raise RuntimeError(f"ROME execution failed: {cases}")
    return cases[0]


def checkpoint_files(names):
    """Keep one weight format (full or adapter) and inference metadata."""
    names = [name for name in names if "/" not in name]
    weights = [name for name in names if name.endswith(".safetensors") and not name.startswith("adapter_")
               and (name == "model.safetensors" or "model.safetensors.index.json" in names)]
    if not weights:
        weights = [name for name in names if name.endswith(".bin")
                   and (name == "pytorch_model.bin" or
                        (name.startswith("pytorch_model") and "pytorch_model.bin.index.json" in names))]
    metadata = [name for name in names if name.endswith((".json", ".txt", ".model", ".tiktoken", ".vocab", ".merges", ".jinja"))]
    if "config.json" in names and weights:
        metadata = [name for name in metadata if not name.startswith("adapter_")]
    elif "adapter_config.json" in names:
        weights = [name for name in names if name == "adapter_model.safetensors"] or [name for name in names if name == "adapter_model.bin"]
        if not weights:
            raise ValueError("Adapter checkpoint has no adapter weights")
    else:
        raise ValueError("Repository has neither full Transformers weights nor PEFT adapter weights")
    return sorted(set(weights + metadata))


def freeze_selection(args, base, api):
    root = Path(args.run_root)
    path = root / "checkpoints.json"
    existing = json.loads(path.read_text()) if path.exists() else None
    shared = (existing["identity"].get("download_layout") == "shared-revision-v1" if existing else
              bool(args.keep_downloads and args.download_root))
    downloads = (Path(args.download_root).resolve() if shared else
                 Path(args.download_root).resolve() / digest(str(root))[:12] if args.download_root else root / ".downloads")
    base = OmegaConf.to_container(base, resolve=True)
    # Reuse the classic external prefix pool without downloading a helper model.
    prefix_hash = None
    if base.get("prefix_mode") == "external":
        source = Path(str(base.get("prefix_source") or ""))
        cache = source if source.is_file() else Path(str(base.get("prefix_cache_path") or ""))
        cache = cache if cache.is_absolute() else ROOT / cache
        if not cache.is_file():
            raise FileNotFoundError(f"Classic external prefix cache is required: {cache}")
        pool = json.loads(cache.read_text())
        pool = pool.get("templates", []) if isinstance(pool, dict) else pool
        if not isinstance(pool, list) or not any(str(value).strip() for value in pool):
            raise ValueError(f"Classic external prefix cache has no templates: {cache}")
        prefix_hash = digest(cache.read_text())
        base["prefix_source"] = str(cache.resolve())
        base["prefix_cache_path"] = str(cache.resolve())
    supplied = json.loads(Path(args.models_manifest).read_text()) if args.models_manifest else None
    identity = {"base_model": args.base_model, "hf_base_model": args.hf_base_model or base["name"],
                "base_config": base, "prefix_hash": prefix_hash, "model_count": args.model_count,
                "supplied_manifest_hash": digest(supplied) if supplied else None,
                "download_root": str(downloads)}
    identity.update(covariance_source="finetuned" if args.finetuned_covariance else "base",
                     covariance_samples=args.covariance_samples)
    if args.random_facts:
        identity["fact_selection"] = "random-start-with-wrap-v1"
    if shared:
        identity["download_layout"] = "shared-revision-v1"
        if not args.keep_downloads:
            raise ValueError("Shared revision caches must be retained")
    if existing:
        selection = existing
        if selection["identity"] != identity:
            raise ValueError("Checkpoint selection/configuration changed; use a new run root")
        return selection
    if supplied is not None:
        records = supplied["models"][:args.model_count]
    else:
        from scripts.fetch_finetuned_qwen3_8b import fetch_models
        records = fetch_models(base_model=identity["hf_base_model"], limit=args.model_count,
                               token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    if len(records) != args.model_count:
        raise ValueError(f"Requested {args.model_count} checkpoints but found {len(records)}")
    models, seen = [], set()
    for record in records:
        model_id = str(record["model_id"])
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", model_id) or any(p in (".", "..") for p in model_id.split("/")):
            raise ValueError(f"Invalid HF repository ID: {model_id}")
        key = fleet_model_key(model_id)
        if key in seen:
            raise ValueError(f"Duplicate checkpoint or config key: {model_id}")
        seen.add(key)
        entry = {"model_id": model_id, "key": key, "rank": len(models) + 1,
                 "discovery_rank": record.get("discovery_rank", record.get("rank", len(models) + 1)),
                 "downloads": record.get("downloads"), "revision": record.get("revision"),
                  "checkpoint_type": record.get("checkpoint_type")}
        if record.get("file_sizes"):
            entry["file_sizes"] = record["file_sizes"]
        try:
            if record.get("selection_error"):
                raise ValueError(record["selection_error"])
            if re.fullmatch(r"[0-9a-f]{40}", str(record.get("revision", ""))) and isinstance(record.get("files"), list):
                entry.update(revision=record["revision"], files=checkpoint_files(record["files"]))
            else:
                info = api.model_info(model_id, revision=record.get("revision"))
                if not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
                    raise ValueError("HF did not return a full checkpoint revision")
                entry.update(revision=info.sha, files=checkpoint_files([s.rfilename for s in info.siblings]))
            entry["checkpoint_type"] = "adapter" if "adapter_config.json" in entry["files"] else "full"
            if entry["checkpoint_type"] == "adapter":
                if record.get("adapter_base_revision") and record.get("adapter_base_files"):
                    for key_name in ("adapter_base_model", "adapter_base_revision", "adapter_base_files"):
                        entry[key_name] = record[key_name]
                    for key_name in ("adapter_base_file_sizes", "adapter_config"):
                        if key_name in record:
                            entry[key_name] = record[key_name]
                else:
                    from huggingface_hub import hf_hub_download
                    adapter = json.loads(Path(hf_hub_download(model_id, "adapter_config.json", revision=entry["revision"], token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))).read_text())
                    base_id = adapter.get("base_model_name_or_path")
                    if not base_id or Path(base_id).is_absolute():
                        raise ValueError(f"Adapter has no usable HF base model ID: {base_id}")
                    base_info = api.model_info(base_id, revision=adapter.get("revision"))
                    entry.update(adapter_base_model=base_id, adapter_base_revision=base_info.sha,
                                 adapter_base_files=checkpoint_files([s.rfilename for s in base_info.siblings]))
                if not re.fullmatch(r"[0-9a-f]{40}", str(entry["adapter_base_revision"])) or "adapter_config.json" in entry["adapter_base_files"]:
                    raise ValueError("Adapter base must be a pinned full Transformers checkpoint")
        except Exception as exc:
            entry["selection_error"] = f"{type(exc).__name__}: {exc}"
        models.append(entry)
    selection = {"identity": identity, "created_at": utc_now(), "models": models}
    write_json(path, selection)
    return selection


def save_configs(root, selection):
    config_dir = root / "checkpoint-configs" / "model"
    config_dir.mkdir(parents=True, exist_ok=True)
    base = selection["identity"]["base_config"]
    shared = selection["identity"]["covariance_source"] == "base"
    if shared:
        from jobs.paper_fleet import model_second_moment_files
        from src.common.paths import resolve_project_path
        files = model_second_moment_files(selection["identity"]["base_model"], int(base["layer"]),
            selection["identity"]["covariance_samples"], config=OmegaConf.create(base))
        directory = resolve_project_path(base["second_moment_dir"])
        # Preparation can run before the classic covariance has been generated.
        stem = f"{base['name'].replace('/', '_')}_{base['layer']}_SM_Method.WIKIPEDIA"
        expected = directory / f"{stem}_{selection['identity']['covariance_samples']}.pt"
        path = files[0] if files else resolve_project_path(base.get("second_moment_path") or
                                                          expected)
    for entry in selection["models"]:
        cfg = dict(base)
        cfg.update(name=entry["model_id"], save_to_local=False,
                   models_dir=str(checkpoint_download_path(selection, entry) / "models"),
                   second_moment_dir=str(root / "models" / entry["key"] / "covariance"),
                   second_moment_path=None)
        if shared:
            cfg.update(second_moment_dir=str(directory), second_moment_path=str(path),
                       second_moment_model_name=base["name"])
        # Provenance is also part of the Gram config hash, even after weights are deleted.
        cfg["checkpoint_revision"] = entry.get("revision")
        if entry.get("checkpoint_type") == "adapter" and not entry.get("selection_error"):
            cfg["adapter_base_model"] = entry["adapter_base_model"]
            cfg["adapter_base_revision"] = entry["adapter_base_revision"]
            cfg["adapter_base_path"] = str(adapter_base_path(selection, entry))
        OmegaConf.save(OmegaConf.create(cfg), config_dir / f"{entry['key']}.yaml")
    return config_dir


def adapter_base_path(selection, entry):
    if selection["identity"].get("download_layout") == "shared-revision-v1":
        base = {"model_id": entry["adapter_base_model"], "key": fleet_model_key(entry["adapter_base_model"]),
                "revision": entry["adapter_base_revision"]}
        return checkpoint_download_path(selection, base) / "models" / entry["adapter_base_model"]
    identity = {"model": entry["adapter_base_model"], "revision": entry["adapter_base_revision"]}
    return Path(selection["identity"]["download_root"]) / "bases" / digest(identity)[:16]


def checkpoint_download_path(selection, entry):
    root = Path(selection["identity"]["download_root"])
    if selection["identity"].get("download_layout") == "shared-revision-v1":
        return root / "checkpoints" / entry["key"] / str(entry.get("revision") or "unresolved")
    return root / entry["key"]


def ensure_download(downloader, *, repo_id, revision, local_dir, files, token, file_sizes=None):
    """Reuse complete pinned snapshots; serialize downloads shared by fleet jobs."""
    directory = Path(local_dir)
    if directory.is_symlink():
        raise ValueError(f"Checkpoint cache must not be a symlink: {directory}")
    identity = {"repo_id": repo_id, "revision": revision, "files": sorted(files)}
    marker = directory / ".latium-snapshot.json"
    with locked(directory.parent / ("." + directory.name + "-download.lock")):
        try:
            saved = json.loads(marker.read_text()) if marker.exists() else None
        except (OSError, json.JSONDecodeError):
            saved = None
        if (saved and saved.get("identity") == identity and
                all((directory / name).is_file() and (directory / name).stat().st_size == size
                    for name, size in saved["file_sizes"].items()) and
                set(saved["file_sizes"]) == set(files)):
            return "cache"
        downloader(repo_id=repo_id, revision=revision, local_dir=str(directory), allow_patterns=files, token=token,
                   force_download=bool(saved))
        missing = [name for name in files if not (directory / name).is_file()]
        if missing:
            raise RuntimeError(f"Incomplete pinned download for {repo_id}: {missing[:6]}")
        actual = {name: (directory / name).stat().st_size for name in files}
        if file_sizes and any(actual[name] != size for name, size in file_sizes.items() if size is not None and name in actual):
            raise RuntimeError(f"Downloaded file sizes do not match pinned metadata for {repo_id}")
        write_json(marker, {"identity": identity, "file_sizes": actual})
        return "download"


def remove_download(download, parent):
    """Delete only this runner's checkpoint directory, never the global HF cache."""
    if download.is_symlink() or download.resolve().parent != parent.resolve():
        raise ValueError(f"Download cleanup escaped its owned parent: {download}")
    if download.exists():
        shutil.rmtree(download)


def own_download_root(downloads, root):
    marker = downloads / "owner.json"
    if downloads.is_symlink():
        raise ValueError("Download root must not be a symlink")
    if downloads.exists() and any(downloads.iterdir()) and not marker.is_file():
        raise ValueError(f"Refusing to use an existing unowned download directory: {downloads}")
    owner = {"run_root": str(root)}
    if marker.exists() and json.loads(marker.read_text()) != owner:
        raise ValueError("Download directory belongs to another fleet")
    write_json(marker, owner)


def run(args, api=None, downloader=None):
    from huggingface_hub import HfApi, snapshot_download
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = api if api is not None else HfApi(token=token)
    downloader = downloader if downloader is not None else snapshot_download
    root = Path(args.run_root)
    # A single fleet lock prevents two jobs downloading/deleting the same checkpoint.
    with locked(root / "fleet-lock"):
        base = load_model_config(args.base_model, config_dir=MODEL_CONFIG_DIR)
        if args.prefix_cache_file:
            base.prefix_source = str(Path(args.prefix_cache_file).resolve())
            base.prefix_cache_path = base.prefix_source
        selection = freeze_selection(args, base, api)
        config_dir = save_configs(root, selection)
        previous = os.environ.get("LATIUM_MODEL_CONFIG_DIR")
        os.environ["LATIUM_MODEL_CONFIG_DIR"] = str(config_dir)
        try:
            forwarded = ["--workflow", "gram", "--run-root", str(root), "--models",
                         *[e["key"] for e in selection["models"]],
                         "--case-index-file", args.case_index_file, "--case-start", str(args.case_start),
                         "--case-stop", str(args.case_stop), "--covariance-samples", str(args.covariance_samples),
                         "--python", args.python, "--tracking", args.tracking, "--wandb-project", args.wandb_project]
            if args.no_graphs:
                forwarded.append("--no-graphs")
            gram = gram_args(forwarded)
            gram.skip_second_moment = not args.finetuned_covariance
            if args.retry_failed_facts:
                gram.models = []  # Freeze the cohort; register only facts actually attempted by ROME.
            batch, catalog = gram_fleet.prepare(gram)
            if args.prepare_only:
                print(f"Prepared {len(selection['models'])} pinned checkpoints at {root}")
                return 0
            cohort = load_case_manifest(root / "cases.json")
            from jobs.paper_fleet import configure_logging
            configure_logging(root)
            if not args.finetuned_covariance:
                from jobs.paper_fleet import model_second_moment_files
                cfg = load_model_config(selection["models"][0]["key"])
                if not model_second_moment_files(selection["models"][0]["key"], int(cfg.layer), args.covariance_samples):
                    raise FileNotFoundError(f"Missing original-model covariance for {args.base_model}, "
                        f"layer={cfg.layer}, samples={args.covariance_samples}. Prepare it with "
                        f"python jobs/paper_fleet.py --workflow gram --models {args.base_model} "
                        f"--covariance-only --covariance-samples {args.covariance_samples} "
                        "--run-root analysis_out/base-covariance, or select --finetuned-covariance.")
            failures = []
            downloads = Path(selection["identity"]["download_root"])
            shared = selection["identity"].get("download_layout") == "shared-revision-v1"
            if shared:
                if downloads.is_symlink():
                    raise ValueError("Shared cache root must not be a symlink")
                downloads.mkdir(parents=True, exist_ok=True)
            else:
                own_download_root(downloads, root)
            # A killed job can leave a later checkpoint behind. Clear all owned
            # leftovers before starting so retries still keep only one model on disk.
            if not args.keep_downloads:
                for entry in selection["models"]:
                    remove_download(downloads / entry["key"], downloads)
                remove_download(downloads / "bases", downloads)
            trace_implementation = None
            if args.causal_kuba_fix:
                from jobs.finetuned_trace import trace_implementation_hash
                trace_implementation = trace_implementation_hash()
            for entry in selection["models"][args.checkpoint_start:args.checkpoint_limit]:
                model = entry["key"]
                state_path = root / "models" / model / "fleet-batches" / batch / "state.json"
                download = checkpoint_download_path(selection, entry)
                previous_state = json.loads(state_path.read_text()) if state_path.exists() else {}
                if previous_state and (args.causal_kuba_fix or previous_state.get("causal_kuba_fix")):
                    if (previous_state.get("causal_kuba_fix") != args.causal_kuba_fix
                            or previous_state.get("trace_implementation") != trace_implementation):
                        raise ValueError("Causal tracing implementation changed; use a new run root")
                state = {**entry, "status": "running", "started_at": utc_now()}
                try:
                    if previous_state.get("status") == "complete" and previous_state.get("causal_kuba_fix", False) == args.causal_kuba_fix:
                        try:
                            accepted = previous_state.get("accepted_position", args.case_start)
                            accepted_batch = f"m{accepted:04d}-{accepted + (args.case_stop - args.case_start):04d}"
                            gram_fleet.verify_batch(root / catalog["models"][model]["run_root"], model,
                                catalog["models"][model]["batches"][accepted_batch]["plan_id"], cohort,
                                accepted, accepted + (args.case_stop - args.case_start))
                            if args.causal_kuba_fix:
                                trace = root / "models" / model / "run" / "causal-kuba-fix" / f"m{accepted:04d}" / "artifact.json"
                                payload = json.loads(trace.read_text())
                                if payload["status"] != "complete" or not all((root / "models" / model / "run" / p).is_file() for p in payload["summary"]["outputs"]):
                                    raise RuntimeError("Missing trace outputs")
                            continue
                        except (FileNotFoundError, RuntimeError):
                            pass  # Repair missing artifacts after redownloading the pinned checkpoint.
                    state["attempts"] = previous_state.get("attempts", [])
                    if args.random_facts:
                        state["fact_candidate_start"] = previous_state.get("fact_candidate_start")
                        if state["fact_candidate_start"] is None:
                            state["fact_candidate_start"] = random.randrange(cohort["count"])
                    state["causal_kuba_fix"] = args.causal_kuba_fix
                    if args.causal_kuba_fix:
                        state["trace_implementation"] = trace_implementation
                    write_json(state_path, state)
                    if entry.get("selection_error"):
                        raise ValueError(entry["selection_error"])
                    if not args.keep_downloads:
                        remove_download(download, downloads)
                        if entry.get("checkpoint_type") == "adapter" and not entry.get("selection_error"):
                            base_download = adapter_base_path(selection, entry)
                            remove_download(base_download, base_download.parent)
                    local_dir = download / "models" / entry["model_id"]
                    print(f"[{model}] downloading {entry['revision']}", flush=True)
                    state["download_source"] = ensure_download(downloader, repo_id=entry["model_id"], revision=entry["revision"],
                               local_dir=str(local_dir), files=entry["files"], token=token, file_sizes=entry.get("file_sizes"))
                    if entry.get("checkpoint_type") == "adapter":
                        state["adapter_base_download_source"] = ensure_download(downloader,
                                   repo_id=entry["adapter_base_model"], revision=entry["adapter_base_revision"],
                                   local_dir=str(adapter_base_path(selection, entry)), files=entry["adapter_base_files"],
                                   token=token, file_sizes=entry.get("adapter_base_file_sizes"))
                    state["download_completed_at"] = utc_now()
                    position = state["fact_candidate_start"] if args.random_facts else args.case_start
                    attempt_limit = min(args.max_fact_attempts if args.retry_failed_facts else 1,
                                        cohort["count"] - (0 if args.random_facts else args.case_start))
                    stop = min(cohort["count"], position + attempt_limit)
                    wrap_stop = attempt_limit - (stop - position) if args.random_facts else 0
                    accepted = False
                    while position < stop or wrap_stop:
                        if position >= stop:
                            position, stop, wrap_stop = 0, wrap_stop, 0
                        rejected = {a["position"] for a in state["attempts"] if a["status"] in ("trace_rejected", "rome_failed")}
                        if position in rejected:
                            position += 1
                            continue
                        attempt = {"position": position, "case_id": cohort["case_ids"][position], "status": "running", "started_at": utc_now()}
                        state["attempts"] = [a for a in state["attempts"] if a["position"] != position] + [attempt]
                        write_json(state_path, state)
                        if args.causal_kuba_fix:
                            from jobs.finetuned_trace import run_trace
                            # Do not rescan previously rejected facts on a resumed run.
                            trace_stop = min((p for p in rejected if position < p < stop), default=stop)
                            result = run_trace(args, model, root / "models" / model / "run", cohort, position, trace_stop)
                            for row in result["rejections"]:
                                record = {**row, "status": "trace_rejected", "started_at": attempt["started_at"], "completed_at": utc_now()}
                                state["attempts"] = [a for a in state["attempts"] if a["position"] != row["position"]] + [record]
                            write_json(state_path, state)
                            if result["accepted_position"] is None:
                                if not args.retry_failed_facts:
                                    raise RuntimeError(f"Fact rejected by causal tracing: {result['error']}")
                                position = trace_stop
                                continue
                            position = result["accepted_position"]
                            if attempt["position"] != position:
                                attempt = {"position": position, "case_id": cohort["case_ids"][position], "status": "running", "started_at": utc_now()}
                                state["attempts"] = [a for a in state["attempts"] if a["position"] != position] + [attempt]
                                write_json(state_path, state)
                        gram.models = [model]
                        gram.case_start = position
                        gram.case_stop = position + (args.case_stop - args.case_start)
                        gram.n_tests = gram.case_stop - gram.case_start
                        if gram_fleet.run(gram):
                            raise RuntimeError("Gram batch failed; see batches/*/state.json and fleet.log")
                        if args.retry_failed_facts:
                            current_batch, catalog = gram_fleet.prepare(gram)
                            case = rome_case(root / catalog["models"][model]["run_root"], catalog["models"][model]["batches"][current_batch]["plan_id"])
                            if not case["edit"]["success"]:
                                attempt.update(status="rome_failed", error="Efficacy evaluation did not pass", execution=case)
                                write_json(state_path, state)
                                position += 1
                                continue
                        attempt.update(status="complete", completed_at=utc_now())
                        state["accepted_position"] = position
                        accepted = True
                        break
                    if not accepted:
                        raise RuntimeError(f"No successful edit within {attempt_limit} candidate facts "
                                           f"(max-fact-attempts={args.max_fact_attempts})")
                    state.update(status="complete", completed_at=utc_now())
                except Exception as exc:
                    state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    if state.get("attempts") and state["attempts"][-1].get("status") == "running":
                        state["attempts"][-1].update(status="runtime_error", error=state["error"], completed_at=utc_now())
                    failures.append(model)
                    print(f"[{model}] {state['error']}", flush=True)
                finally:
                    if not args.keep_downloads:
                        remove_download(download, downloads)
                    if state.get("status") != "running":
                        write_json(state_path, state)
            return int(bool(failures))
        finally:
            if previous is None:
                os.environ.pop("LATIUM_MODEL_CONFIG_DIR", None)
            else:
                os.environ["LATIUM_MODEL_CONFIG_DIR"] = previous


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
