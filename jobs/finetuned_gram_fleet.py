#!/usr/bin/env python3
"""Sequential HF checkpoints: baseline Gram, ROME, edited Gram."""
import argparse
import json
import os
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
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tracking", choices=("none", "wandb"), default="none")
    parser.add_argument("--wandb-project", default="latium")
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze IDs/revisions/configs and facts without downloading weights")
    args = parser.parse_args(argv)
    args.run_root = str(Path(args.run_root).resolve())
    args.case_index_file = str(Path(args.case_index_file).resolve())
    args.case_stop = args.case_stop if args.case_stop is not None else args.case_start + args.n_tests
    if (args.model_count <= 0 or args.covariance_samples <= 0 or args.case_start < 0
            or args.case_stop <= args.case_start or args.case_stop > load_case_manifest(args.case_index_file)["count"]):
        parser.error("Invalid model count, fact range, or covariance sample count")
    return args


def checkpoint_files(names):
    """Keep one full PyTorch weight format and tokenizer/config files, not training state."""
    names = [name for name in names if "/" not in name]
    weights = [name for name in names if name.endswith(".safetensors") and not name.startswith("adapter_")]
    if not weights:
        weights = [name for name in names if name.startswith("pytorch_model") and name.endswith(".bin")]
    if "config.json" not in names or not weights:
        raise ValueError("Repository is not a full Transformers checkpoint (adapters/GGUF are unsupported)")
    metadata = [name for name in names if name.endswith((".json", ".txt", ".model", ".tiktoken", ".vocab", ".merges", ".jinja"))]
    return sorted(set(weights + metadata))


def freeze_selection(args, base, api):
    root = Path(args.run_root)
    downloads = (Path(args.download_root).resolve() / digest(str(root))[:12]
                 if args.download_root else root / ".downloads")
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
    path = root / "checkpoints.json"
    if path.exists():
        selection = json.loads(path.read_text())
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
        entry = {"model_id": model_id, "key": key}
        try:
            info = api.model_info(model_id, revision=record.get("revision"))
            if not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
                raise ValueError("HF did not return a full checkpoint revision")
            entry.update(revision=info.sha, files=checkpoint_files([s.rfilename for s in info.siblings]))
        except Exception as exc:
            entry["selection_error"] = f"{type(exc).__name__}: {exc}"
        models.append(entry)
    selection = {"identity": identity, "created_at": utc_now(), "models": models}
    write_json(path, selection)
    return selection


def save_configs(root, selection):
    config_dir = root / "checkpoint-configs" / "model"
    config_dir.mkdir(parents=True, exist_ok=True)
    for entry in selection["models"]:
        cfg = dict(selection["identity"]["base_config"])
        cfg.update(name=entry["model_id"], save_to_local=False,
                   models_dir=str(Path(selection["identity"]["download_root"]) / entry["key"] / "models"),
                   second_moment_dir=str(root / "models" / entry["key"] / "covariance"),
                   second_moment_path=None)
        # Provenance is also part of the Gram config hash, even after weights are deleted.
        cfg["checkpoint_revision"] = entry.get("revision")
        OmegaConf.save(OmegaConf.create(cfg), config_dir / f"{entry['key']}.yaml")
    return config_dir


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
            batch, catalog = gram_fleet.prepare(gram)
            if args.prepare_only:
                print(f"Prepared {len(selection['models'])} pinned checkpoints at {root}")
                return 0
            cohort = load_case_manifest(root / "cases.json")
            failures = []
            downloads = Path(selection["identity"]["download_root"])
            own_download_root(downloads, root)
            # A killed job can leave a later checkpoint behind. Clear all owned
            # leftovers before starting so retries still keep only one model on disk.
            for entry in selection["models"]:
                remove_download(downloads / entry["key"], downloads)
            for entry in selection["models"]:
                model = entry["key"]
                state_path = root / "models" / model / "fleet-batches" / batch / "state.json"
                download = downloads / model
                state = {**entry, "status": "running", "started_at": utc_now()}
                try:
                    if state_path.exists() and json.loads(state_path.read_text())["status"] == "complete":
                        try:
                            gram_fleet.verify_batch(root / catalog["models"][model]["run_root"], model,
                                catalog["models"][model]["batches"][batch]["plan_id"], cohort, args.case_start, args.case_stop)
                            continue
                        except (FileNotFoundError, RuntimeError):
                            pass  # Repair missing artifacts after redownloading the pinned checkpoint.
                    write_json(state_path, state)
                    if entry.get("selection_error"):
                        raise ValueError(entry["selection_error"])
                    remove_download(download, downloads)
                    local_dir = download / "models" / entry["model_id"]
                    print(f"[{model}] downloading {entry['revision']}", flush=True)
                    downloader(repo_id=entry["model_id"], revision=entry["revision"],
                               local_dir=str(local_dir), allow_patterns=entry["files"], token=token)
                    state["download_completed_at"] = utc_now()
                    gram.models = [model]
                    if gram_fleet.run(gram):
                        raise RuntimeError("Gram batch failed; see batches/*/state.json and fleet.log")
                    state.update(status="complete", completed_at=utc_now())
                except Exception as exc:
                    state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    failures.append(model)
                    print(f"[{model}] {state['error']}", flush=True)
                finally:
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
