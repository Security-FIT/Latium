#!/usr/bin/env python3
"""Freeze top HF text-generation fine-tunes tagged with the configured base model."""
import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi, hf_hub_download
from jobs.finetuned_gram_fleet import checkpoint_files
from jobs.paper_fleet import utc_now
from src.common.model_config import load_model_config
from src.gram_experiment import write_json

MODELS = ["deepseek-7b-base", "falcon-7b", "gemma-4-12b", "gpt-j-6b", "gpt2-large",
          "gpt2-medium", "gpt2-xl", "granite-4.1-8b", "granite4-micro", "llama2-7b",
          "ministral-3-8b", "mistral-7b-v0.1", "mistral-7b-v0.3", "olmo-3-1025-7b",
          "opt-6.7b", "qwen3-4b", "qwen3-8b", "qwen3.5-4b"]


def signature(cfg):
    while isinstance(cfg.get("text_config"), dict):
        cfg = cfg["text_config"]
    fields = ("model_type", "hidden_size", "n_embd", "num_hidden_layers", "n_layer",
              "intermediate_size", "ffn_dim", "num_attention_heads", "n_head",
              "num_key_value_heads", "num_local_experts")
    return {key: cfg[key] for key in fields if key in cfg}


def prepare_family(model, root, count, api, token):
    base = load_model_config(model)
    directory = root / model
    directory.mkdir(parents=True, exist_ok=True)
    cache = str(root / "metadata-cache")
    info = api.model_info(str(base.name))
    def config(model_id, revision, filename="config.json"):
        return json.loads(Path(hf_hub_download(model_id, filename, revision=revision,
                                               cache_dir=cache, token=token)).read_text())
    original = config(info.id, info.sha)
    tag = f"base_model:finetune:{base.name}"
    discovered = {}
    for entry in api.list_models(filter=tag, pipeline_tag="text-generation", sort="downloads", full=True):
        if tag in (entry.tags or []) and entry.pipeline_tag == "text-generation":
            discovered[entry.id] = entry
    ordered = sorted(discovered.values(), key=lambda e: (-int(e.downloads or 0), e.id.lower()))
    previous = directory / "selection-audit.json"
    policy = "top-downloads-finetune-text-generation-exact-base-v1"
    audit = json.loads(previous.read_text()) if previous.exists() else {}
    cached = {e["model_id"]: e for e in audit.get("models", [])} if audit.get("selection_policy") == policy else {}
    write_json(directory / "discovery.json", {"base_model": str(base.name), "discovery_filter": [tag],
               "pipeline_tag": "text-generation", "fetched_at": utc_now(),
               "sort": "downloads descending, model ID ascending for ties",
               "models": [{"model_id": e.id, "downloads": e.downloads, "tags": e.tags,
                           "pipeline_tag": e.pipeline_tag} for e in ordered]})
    base_metadata, base_lock = {}, threading.Lock()

    def validate(cfg, record):
        architectures = cfg.get("architectures") or []
        if architectures and original.get("architectures") and not set(architectures) & set(original["architectures"]):
            raise ValueError("Checkpoint architecture is not the classic causal LM; loading could initialize missing weights")
        record["matches_classic_dimensions"] = signature(cfg) == signature(original)
        if cfg.get("quantization_config"):
            raise ValueError("Quantized weights are unsupported by the float ROME/GRAM runtime")

    def inspect(entry):
        record = {"model_id": entry.id, "downloads": int(entry.downloads or 0), "pipeline_tag": entry.pipeline_tag,
                  "tags": entry.tags,
                  "checkpoint_type": "unsupported", "runtime_validation": 1}
        earlier = cached.get(entry.id)
        if earlier and earlier.get("runtime_validation") == 1 and not any(code in earlier.get("selection_error", "") for code in
                               ("429", "500", "502", "503", "504", "no resolvable HF base repository")):
            return {**earlier, "model_id": entry.id, "downloads": record["downloads"],
                    "pipeline_tag": record["pipeline_tag"], "tags": record["tags"]}
        try:
            details = api.model_info(entry.id, files_metadata=True)
            if not re.fullmatch(r"[0-9a-f]{40}", details.sha or ""):
                raise ValueError("HF did not return a full checkpoint revision")
            record["revision"] = details.sha
            files = checkpoint_files([f.rfilename for f in details.siblings])
            kind = "adapter" if "adapter_config.json" in files else "full"
            record.update(checkpoint_type=kind, files=files)
            sizes = {f.rfilename: f.size for f in details.siblings}
            if any(sizes.get(name) is None for name in files):
                raise ValueError("Missing file sizes")
            record["file_bytes"] = sum(sizes[name] for name in files)
            if kind == "adapter":
                cfg = config(entry.id, details.sha, "adapter_config.json")
                base_id = str(cfg.get("base_model_name_or_path") or "")
                if not re.fullmatch(r"(?:[\w.-]+/)?[\w.-]+", base_id) or any(p in (".", "..") for p in base_id.split("/")):
                    raise ValueError(f"Adapter declares no resolvable HF base repository: {base_id!r}")
                requested_revision = cfg.get("revision") or None
                record.update(adapter_base_model=base_id, adapter_config_revision=requested_revision)
                with base_lock:
                    key = (base_id, requested_revision)
                    if key not in base_metadata:
                        base_info = api.model_info(base_id, revision=requested_revision, files_metadata=True)
                        if not re.fullmatch(r"[0-9a-f]{40}", base_info.sha or ""):
                            raise ValueError("HF did not return a full adapter base revision")
                        base_files = checkpoint_files([f.rfilename for f in base_info.siblings])
                        if "adapter_config.json" in base_files:
                            raise ValueError("Adapter base is itself an adapter")
                        base_metadata[key] = ({"adapter_base_revision": base_info.sha, "adapter_base_files": base_files},
                                              config(base_id, base_info.sha))
                    record.update(base_metadata[key][0])
                    cfg = base_metadata[key][1]
            else:
                cfg = config(entry.id, details.sha)
            # Diagnose unsafe execution after freezing membership; never backfill ranks.
            validate(cfg, record)
        except Exception as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) in (429, 500, 502, 503, 504):
                raise
            record["selection_error"] = f"{type(exc).__name__}: {exc}"
        return record

    # Freeze membership before checking files: an unusable repository keeps its rank.
    chosen = ordered[:count]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        selected = [{**record, "rank": rank} for rank, record in enumerate(pool.map(inspect, chosen), start=1)]
    write_json(directory / "selection-audit.json", {"selection_policy": policy, "models": selected})
    manifest = {"base_model": str(base.name), "base_config": model, "created_at": utc_now(),
                "selection_policy": policy, "discovery_filter": [tag], "relations": ["finetune"],
                "pipeline_tag": "text-generation",
                "sort": "downloads descending, model ID ascending for ties; membership fixed before metadata checks",
                "requested_count": count, "count": len(selected), "models": selected}
    write_json(directory / "checkpoints.json", manifest)
    return {"model": model, "requested": count, "selected": len(selected), "discovered": len(ordered),
            "full": sum(e["checkpoint_type"] == "full" for e in selected),
            "adapter": sum(e["checkpoint_type"] == "adapter" for e in selected),
            "unsupported": sum(e["checkpoint_type"] == "unsupported" for e in selected),
            "errors": sum(bool(e.get("selection_error")) for e in selected),
            "weight_and_metadata_bytes": sum(e.get("file_bytes", 0) for e in selected),
            "manifest": str(directory / "checkpoints.json"), "rome_layer": int(base.layer)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--model-count", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)
    summary = []
    for model in args.models:
        for retry in range(3):
            try:
                record = prepare_family(model, root, args.model_count, api, token)
                break
            except Exception as exc:
                response = getattr(exc, "response", None)
                if getattr(response, "status_code", None) == 429 and retry < 2:
                    delay = float(response.headers.get("Retry-After", 180)) + 1
                    print(f"{model}: HF rate limit; retry after {delay:.0f}s", flush=True)
                    while delay > 0:
                        pause = min(30, delay)
                        time.sleep(pause)
                        delay -= pause
                    continue
                record = {"model": model, "selected": 0, "error": f"{type(exc).__name__}: {exc}"}
                break
        summary.append(record)
        write_json(root / "summary.json", {"created_at": utc_now(), "families": summary})
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
