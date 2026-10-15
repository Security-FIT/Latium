"""Run the paper's saved checkpoint/fact selections through the normal CLI."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from huggingface_hub import HfApi, snapshot_download
from omegaconf import OmegaConf

from src.common.io import write_json
from src.common.model_config import load_model_config
from src.counterfact_selection import load_case_manifest
from src.main import main as latium


INPUTS = Path(__file__).resolve().parents[1] / "manifests" / "paper"
SAMPLES = 100_000


def download(api, model_id, revision, directory, files=None):
    # Resolve aliases before requesting pinned files (e.g. gpt2-medium).
    info = api.model_info(model_id, revision=revision)
    if revision and info.sha != revision:
        raise ValueError(f"Requested revision differs from resolved revision: {model_id}")
    destination = directory / info.id
    if files is None:
        safetensors = any(item.rfilename.startswith("model") and item.rfilename.endswith(".safetensors")
                          for item in info.siblings)
        files = ["*.json", "*.txt", "*.model", "*.tiktoken", "*.vocab", "*.merges", "*.jinja",
                 "model*.safetensors" if safetensors else "pytorch_model*.bin"]
    snapshot_download(
        repo_id=info.id, revision=info.sha, local_dir=str(destination),
        allow_patterns=files,
    )
    return info.id, info.sha


def covariance_config(config, directory):
    config["second_moment_dir"] = str(directory)
    config["second_moment_target_samples"] = SAMPLES
    filename = f"{config['name'].replace('/', '_')}_{config['layer']}_SM_Method.WIKIPEDIA_{SAMPLES}.pt"
    config["second_moment_path"] = str(directory / filename)
    return config


def save_config(directory, key, config):
    OmegaConf.save(OmegaConf.create(config), directory / f"{key}.yaml")


def compute_covariance(directory, key, config):
    save_config(directory, key, config)
    if not Path(config["second_moment_path"]).is_file():
        if latium(["second-moment", f"model={key}"]):
            raise RuntimeError(f"Covariance computation failed: {key}")


def run(args):
    selection = json.loads((INPUTS / f"finetunes-{args.covariance}" / f"{args.family}.json").read_text())
    facts_path = INPUTS / "rome-facts-100.json"
    facts = load_case_manifest(facts_path)
    if selection["fact_manifest_hash"] != facts["manifest_hash"]:
        raise ValueError("Checkpoint selection and fact manifest do not match")
    entries = selection["models"][:args.limit]
    for entry in entries:
        if facts["case_ids"][entry["fact_position"]] != entry["fact_case_id"]:
            raise ValueError(f"Fact assignment differs: {entry['model_id']}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output or f"analysis_out/finetunes-{args.covariance}-{args.family}-{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    configs = output / "config" / "model"
    configs.mkdir(parents=True)
    base = OmegaConf.to_container(load_model_config(args.family), resolve=True)
    base.update(layer=selection["edit_layer"], save_to_local=False)
    api = HfApi()
    previous = os.environ.get("LATIUM_MODEL_CONFIG_DIR")
    os.environ["LATIUM_MODEL_CONFIG_DIR"] = str(configs)
    failures = []
    try:
        base_covariance = None
        if args.covariance == "base":
            with tempfile.TemporaryDirectory(prefix="latium-base-", dir=output) as temporary:
                config = deepcopy(base)
                directory = Path(temporary)
                config["name"], config["checkpoint_revision"] = download(api, base["name"], None, directory)
                config["models_dir"] = str(directory)
                covariance_config(config, output / "covariance" / "base")
                compute_covariance(configs, "covariance", config)
                base_covariance = config["second_moment_path"]

        for entry in entries:
            key = entry["key"]
            print(f"[{key}] fact={entry['fact_case_id']} revision={entry['revision']}", flush=True)
            try:
                with tempfile.TemporaryDirectory(prefix="latium-checkpoint-", dir=output) as temporary:
                    directory = Path(temporary) / "models"
                    config = deepcopy(base)
                    config["name"], config["checkpoint_revision"] = download(
                        api, entry["model_id"], entry["revision"], directory, entry["files"],
                    )
                    config["models_dir"] = str(directory)
                    if entry["checkpoint_type"] == "adapter":
                        base_id, base_revision = download(
                            api, entry["adapter_base_model"], entry["adapter_base_revision"],
                            directory, entry["adapter_base_files"],
                        )
                        config.update(adapter_base_model=base_id, adapter_base_revision=base_revision,
                                      adapter_base_path=str(directory / base_id))
                    covariance_config(config, output / "covariance" / key)
                    if base_covariance:
                        config["second_moment_path"] = base_covariance
                        save_config(configs, key, config)
                    else:
                        compute_covariance(configs, key, config)
                    if latium([
                        "structural", "run", "structural=gram", f"structural.run.models=[{key}]",
                        f"structural.run.case_index_file={json.dumps(str(facts_path))}",
                        f"structural.run.start_idx={entry['fact_position']}", "structural.run.n_tests=1",
                        f"structural.run.output_dir={json.dumps(str(output / 'runs'))}",
                        f"structural.run.run_id={key}", "structural.render.enabled=true", "seed=0",
                    ]):
                        raise RuntimeError("ROME/GRAM run failed")
            except Exception as error:
                failures.append({"model_id": entry["model_id"], "revision": entry["revision"], "error": str(error)})
                write_json(output / "failures.json", failures)
                print(f"[{key}] failed: {error}", flush=True)
    finally:
        if previous is None:
            os.environ.pop("LATIUM_MODEL_CONFIG_DIR", None)
        else:
            os.environ["LATIUM_MODEL_CONFIG_DIR"] = previous
    print(f"Results: {output}", flush=True)
    return int(bool(failures))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, help="Model config key, e.g. gpt2-medium")
    parser.add_argument("--covariance", choices=("base", "own"), required=True)
    parser.add_argument("--limit", type=int, help="Run only the first N saved checkpoints")
    parser.add_argument("--output", help="New output directory; defaults to a timestamped directory")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not (INPUTS / f"finetunes-{args.covariance}" / f"{args.family}.json").is_file():
        parser.error(f"No saved {args.covariance}-covariance selection for {args.family}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
