"""Portable execution of frozen campaign entries on a Linux GPU host."""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess

from src.gram_experiment import digest, write_json
from src.submission.archive import sha256
from src.submission.inputs import INPUTS, ROOT, load_campaign, select_entries


@contextmanager
def model_config_directory(path):
    old = os.environ.get("LATIUM_MODEL_CONFIG_DIR")
    os.environ["LATIUM_MODEL_CONFIG_DIR"] = str(path)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("LATIUM_MODEL_CONFIG_DIR", None)
        else:
            os.environ["LATIUM_MODEL_CONFIG_DIR"] = old


def source_identity():
    code = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*")):
        if path.suffix in {".py", ".yaml"}:
            code.update(str(path.relative_to(ROOT)).encode())
            code.update(path.read_bytes())
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    versions = {}
    for package in ("torch", "transformers", "datasets", "peft", "numpy", "hydra-core"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"git_revision": revision, "source_sha256": code.hexdigest(), "packages": versions}


def _pin(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"No recorded immutable revision for {name}. Supply the original revision with --base-revisions; current Hub main is never substituted.")
    return value


def download_checkpoint(repo, revision, files, sizes, cache):
    from huggingface_hub import snapshot_download
    revision = _pin(revision, repo)
    location = Path(cache) / digest([repo, revision])[:20] / "models" / repo
    snapshot_download(repo_id=repo, revision=revision, local_dir=location, allow_patterns=files or None,
                      token=os.environ.get("HF_TOKEN") or None)
    for name in files or ():
        path = location / name
        if not path.is_file() or (sizes and name in sizes and path.stat().st_size != sizes[name]):
            raise ValueError(f"Incomplete pinned snapshot: {repo}@{revision}/{name}")
    return location


def relocate_model(entry, config, output, revisions):
    """Change storage paths only; retain every archived scientific setting."""
    config = dict(config)
    checkpoint = entry.get("checkpoint", {})
    revision = entry.get("revision") or revisions.get(config["name"])
    location = download_checkpoint(config["name"], revision, checkpoint.get("files"), checkpoint.get("file_sizes"), output / "weights")
    config.update(models_dir=str(location.parents[len(Path(config["name"]).parts) - 1]), checkpoint_revision=revision, save_to_local=False)
    if entry.get("checkpoint_type") == "adapter":
        base = download_checkpoint(checkpoint["adapter_base_model"], checkpoint["adapter_base_revision"],
                                   checkpoint["adapter_base_files"], checkpoint.get("adapter_base_file_sizes"), output / "weights")
        config.update(adapter_base_path=str(base), adapter_base_model=checkpoint["adapter_base_model"],
                      adapter_base_revision=checkpoint["adapter_base_revision"])
    if config.get("prefix_mode") == "external":
        source = INPUTS / "prefix_cache" / Path(config.get("prefix_cache_path") or config["prefix_source"]).name
        expected = entry.get("selection_identity", {}).get("prefix_hash")
        if not source.is_file() or (expected and digest(source.read_text()) != expected):
            raise ValueError(f"Original external prefix cache unavailable or changed: {source}")
        config.update(prefix_source=str(source), prefix_cache_path=str(source))
    elif config.get("prefix_cache_path"):
        # Self-generated caches belong to the selected checkpoint and batch.
        config["prefix_cache_path"] = str(output / "prefix_cache" / (entry["model_key"] + ".json"))
    if config.get("tokenizer_path"):
        # A historical tokenizer override must come from its pinned original
        # source; never recover a different vocabulary from a current Hub tip.
        repo = checkpoint.get("tokenizer_base_model") or config.get("tokenizer_base_model")
        pin = checkpoint.get("tokenizer_base_revision") or config.get("tokenizer_base_revision")
        if not repo or not pin:
            raise ValueError(f"Unresolved historical tokenizer override for {entry['model_id']}")
        config["tokenizer_path"] = str(download_checkpoint(repo, pin, ["*.json", "*.model", "*.txt"], None, output / "weights"))
    return config


def compute_covariance(config, directory, seed, samples, *, fresh):
    import gc
    import torch
    from omegaconf import OmegaConf
    from src.handlers.rome import ModelHandler
    from src.rome.common import compute_second_moment
    from src.runtime import set_global_seed
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{config['name'].replace('/', '_')}_{config['layer']}_SM_Method.WIKIPEDIA_{samples}.pt"
    path = directory / name
    receipt = path.with_suffix(".receipt.json")
    identity = {"model": config["name"], "revision": config.get("checkpoint_revision"),
                "adapter_base_revision": config.get("adapter_base_revision"), "layer": config["layer"],
                "samples": samples, "seed": seed,
                "settings": {k: v for k, v in config.items() if k.startswith("second_moment_") and k not in {"second_moment_dir", "second_moment_path"}},
                "dataset": "Salesforce/wikitext", "config": "wikitext-103-raw-v1", "splits": ["train", "test", "validation"]}
    if not fresh and path.is_file() and receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved["identity"] == identity and saved["sha256"] == sha256(path):
            return path, saved
        raise ValueError(f"Incompatible covariance receipt: {path}")
    set_global_seed(seed)
    cfg = OmegaConf.create({"seed": seed, "runtime": {"hf_token": os.environ.get("HF_TOKEN")},
                           "model": {**config, "second_moment_dir": str(directory), "second_moment_path": None},
                           "generation": OmegaConf.load(ROOT / "src/config/generation/generation.yaml"),
                           "dataset_sm": OmegaConf.load(ROOT / "src/config/dataset_sm/wikitext.yaml")})
    handler = ModelHandler(cfg)
    try:
        matrix, count, method = compute_second_moment(handler, N_rounds=1, N_k=samples)
        if count != samples:
            raise ValueError(f"Incomplete covariance: {count} != {samples}")
        torch.save(matrix, path)
        saved = {"identity": identity, "fresh_computation": True, "count": count,
                 "method": str(method), "sha256": sha256(path), "source": source_identity()}
        write_json(receipt, saved)
        return path, saved
    finally:
        handler.remove_hooks()
        del handler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def execute(args, *, smoke=False):
    from omegaconf import OmegaConf
    from src.structural.config import StructuralBenchmarkConfig
    from src.structural.runner import run_structural_benchmark
    from src.submission.artifacts import read_run
    from src.submission.inputs import verify_packaged_inputs
    verify_packaged_inputs()
    payload = load_campaign(args.campaign)
    entries = select_entries(payload, family=args.family, checkpoint=args.checkpoint, limit=args.limit_checkpoints)
    output = Path(args.output).resolve()
    if output.is_relative_to(INPUTS.resolve()):
        raise ValueError("Experiment outputs must be outside the frozen input directory")
    output.mkdir(parents=True, exist_ok=True)
    revisions = json.loads(Path(args.base_revisions).read_text()) if args.base_revisions else {}
    if args.campaign == "finetune-base" and not smoke and not args.base_covariance_dir:
        for entry in entries:
            repo = entry["selection_identity"]["base_config"]["name"]
            _pin(revisions.get(repo), repo)
    outputs, failures = [], []
    for entry in entries:
        if not entry.get("evaluated", True):
            failure = {"model_id": entry["model_id"], "status": "recorded-download-failure", "recorded_status": entry.get("recorded_status")}
            failures.append(failure)
            continue
        batches = entry.get("batches") or [{"name": "assigned", "model_config": entry["archived_model_config"],
                                            "seed": entry["execution_seed"], "fact_manifest": payload["fact_manifest"],
                                            "start": entry["fact_position"], "count": 1}]
        for batch in batches:
            run_dir = output / "runs" / entry["model_key"] / batch["name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                config = relocate_model(entry, batch["model_config"], run_dir, revisions)
            except Exception as exc:
                write_json(run_dir / "failure.json", {"campaign": args.campaign, "entry": entry,
                           "batch": batch, "stage": "pinned-model-loading", "status": "error", "error": str(exc)})
                raise
            seed = int(batch["seed"])
            samples = args.covariance_samples if smoke else int(entry.get("selection_identity", {}).get("covariance_samples", 100000))
            if args.campaign == "finetune-own":
                cov_path, covariance = compute_covariance(config, run_dir / "covariance", seed, samples, fresh=True)
            else:
                base_config = dict(entry.get("selection_identity", {}).get("base_config", config))
                base_config.pop("adapter_base_path", None)
                base_config.pop("adapter_base_model", None)
                base_config.pop("adapter_base_revision", None)
                base_entry = {"model_id": base_config["name"], "model_key": entry["family"],
                              "revision": revisions.get(base_config["name"]) or (entry.get("revision") if args.campaign == "classic" else None)}
                if smoke and base_entry["revision"] is None and entry.get("checkpoint_type") == "adapter":
                    base_entry["revision"] = entry["checkpoint"]["adapter_base_revision"]
                    base_config["name"] = entry["checkpoint"]["adapter_base_model"]
                    base_entry["checkpoint"] = {"files": entry["checkpoint"]["adapter_base_files"],
                                                "file_sizes": entry["checkpoint"].get("adapter_base_file_sizes")}
                if args.base_covariance_dir and not smoke:
                    cov_path = Path(args.base_covariance_dir).resolve() / f"{base_config['name'].replace('/', '_')}_{base_config['layer']}_SM_Method.WIKIPEDIA_{samples}.pt"
                    if not cov_path.is_file():
                        raise FileNotFoundError(cov_path)
                    covariance = {"source": "supplied-original-base-covariance", "samples": samples, "sha256": sha256(cov_path)}
                else:
                    base_config = relocate_model(base_entry, base_config, output / "base-models" / entry["family"], revisions)
                    cov_path, covariance = compute_covariance(base_config, output / "base-covariance" / entry["family"], seed, samples, fresh=False)
            config.update(second_moment_dir=str(cov_path.parent), second_moment_path=str(cov_path))
            config_dir = run_dir / "config/model"
            config_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(OmegaConf.create(config), config_dir / (entry["model_key"] + ".yaml"))
            count = min(batch["count"], args.limit_facts) if args.limit_facts else batch["count"]
            if smoke:
                count = min(count, 1)
            with model_config_directory(config_dir):
                cfg = StructuralBenchmarkConfig(models=(entry["model_key"],), n_tests=count, start_idx=batch["start"],
                    case_index_file=str(INPUTS / batch["fact_manifest"]), output_dir=run_dir, run_id="run", seed=seed,
                    analysis_preset="gram-localization", render_graphs=False, fail_on_missing_second_moment=True,
                    analysis_continue_on_error=True,
                    hf_token=os.environ.get("HF_TOKEN"), force=True)
                result = run_structural_benchmark(cfg)
            run_root = run_dir / "run"
            reader, artifacts = read_run(run_root)
            selected = set(batch.get("selected_case_ids", [str(entry.get("fact_case_id"))]))
            errors = [p["artifact_id"] for p in artifacts.values()
                      if any(c.get("status") != "complete" for c in p.get("cases", [])
                             if c.get("case_id") == "baseline" or str(c.get("case_id")) in selected)]
            receipt = {"campaign": args.campaign, "validation_subset": smoke or bool(args.limit_facts or args.limit_checkpoints or args.checkpoint or args.family),
                       "entry": entry, "batch": batch, "resolved_model_config": config, "seed": seed,
                       "covariance": covariance, "source": source_identity(), "run_root": str(run_root),
                       "status": "error" if errors else "complete", "errors": errors}
            write_json(run_dir / "reproduction.json", receipt)
            outputs.append(receipt)
            if errors:
                raise RuntimeError(f"Saved workflow failed: {run_root}: {errors}")
            if smoke:
                from src.submission.render import render_saved_run
                render_saved_run(run_root, run_dir / "rendered")
    write_json(output / "recorded-download-failures.json", failures)
    return {"completed_runs": len(outputs), "recorded_download_failures": len(failures), "output": str(output)}
