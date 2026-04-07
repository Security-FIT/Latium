#!/usr/bin/env python3
"""
ROME-only benchmark runner.

Runs a fixed number of independent ROME edits and reports success rate.
No structural detectors are executed.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas
import torch
from omegaconf import OmegaConf

from src.handlers.rome import ModelHandler
from src.rome.common import gather_k, optimize_v, insert_kv
from src.utils import load_dataset


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG_DIR = Path(__file__).parent / "src" / "config"
MODEL_CONFIG_DIR = CONFIG_DIR / "model"


def load_model_config(model_name: str):
    yaml_path = MODEL_CONFIG_DIR / f"{model_name}.yaml"
    if yaml_path.exists():
        return OmegaConf.load(yaml_path)

    for path in sorted(MODEL_CONFIG_DIR.glob("*.yaml")):
        if path.name == "boilerplate.yaml":
            continue
        cfg = OmegaConf.load(path)
        if getattr(cfg, "name", "") == model_name:
            return cfg

    available = ", ".join(
        p.stem for p in sorted(MODEL_CONFIG_DIR.glob("*.yaml")) if p.name != "boilerplate.yaml"
    )
    raise FileNotFoundError(f"No model config for '{model_name}'. Available: {available}")


def build_cfg(model_name: str):
    model_cfg = load_model_config(model_name)
    cfg = OmegaConf.create(
        {
            "model": model_cfg,
            "generation": OmegaConf.load(CONFIG_DIR / "generation" / "generation.yaml"),
            "dataset_facts": OmegaConf.load(CONFIG_DIR / "dataset_facts" / "counterfact.yaml"),
            "dataset_sm": OmegaConf.load(CONFIG_DIR / "dataset_sm" / "wikitext.yaml"),
        }
    )
    return cfg


def apply_overrides(cfg, overrides: list[str]):
    if not overrides:
        return cfg
    override_cfg = OmegaConf.from_dotlist(overrides)
    return OmegaConf.merge(cfg, override_cfg)


def run_single_model(model_name: str, n_tests: int, start_idx: int = 0, overrides: list[str] | None = None) -> dict:
    cfg = build_cfg(model_name)
    cfg = apply_overrides(cfg, overrides or [])
    cfg.model.device = "cuda"
    cfg.model.cuda_mode = "strict"

    handler = ModelHandler(cfg)
    dataset = load_dataset(cfg)
    df = pandas.DataFrame(dataset)

    layer_name = handler._layer_name_template.format(handler._layer)
    tested = 0
    successes = 0
    skipped = 0
    results = []

    LOGGER.info("Model=%s layer=%s n_tests=%d", cfg.model.name, handler._layer, n_tests)

    for i, prompt_dict in enumerate(df.itertuples()):
        if i < start_idx:
            continue
        if tested >= n_tests:
            break

        tested += 1
        module = handler._get_module(layer_name)
        original_weight = module.weight.detach().clone()

        case_id = int(getattr(prompt_dict, "case_id", prompt_dict.Index))
        rewrite = prompt_dict.requested_rewrite
        fact_tuple = (
            rewrite["prompt"],
            rewrite["subject"],
            " " + rewrite["target_new"]["str"],
            " " + rewrite["target_true"]["str"],
        )

        success = False
        predicted = None
        expected = None
        error = None

        try:
            k = gather_k(handler, fact_tuple=fact_tuple, N=getattr(cfg.generation, "k_N", 40))
            delta = optimize_v(
                handler,
                fact_tuple=fact_tuple,
                N_prompts=getattr(cfg.generation, "v_N", 20),
                N_optim_steps=handler.epochs,
                verbose=False,
            )
            if delta is None:
                raise RuntimeError("optimize_v returned None")

            insert_kv(handler, k, delta)

            prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
            target_ids = handler.tokenize_prompt(f"{fact_tuple[2]}").input_ids[0]
            bos_id = getattr(handler.tokenizer, "bos_token_id", None)
            if bos_id is not None and target_ids.numel() > 1 and int(target_ids[0].item()) == int(bos_id):
                target_ids = target_ids[1:]

            target_len = int(target_ids.shape[0])
            outputs = handler.model.generate(
                **prompt,
                max_length=prompt.input_ids.shape[1] + target_len,
            )
            generated_ids = outputs[0, prompt.input_ids.shape[1] : prompt.input_ids.shape[1] + target_len]
            predicted = handler.tokenizer.decode(generated_ids, skip_special_tokens=True)
            expected = handler.tokenizer.decode(target_ids, skip_special_tokens=True)
            success = predicted.strip() == expected.strip()
            if success:
                successes += 1

        except Exception as exc:
            error = str(exc)
            skipped += 1
        finally:
            handler._get_module(layer_name).weight = torch.nn.Parameter(original_weight)
            handler.remove_hooks()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append(
            {
                "case_id": case_id,
                "relation_id": rewrite.get("relation_id", ""),
                "subject": rewrite["subject"],
                "target_new": rewrite["target_new"]["str"],
                "target_eval": expected if expected is not None else rewrite["target_new"]["str"],
                "predicted": predicted,
                "success": bool(success),
                "error": error,
            }
        )

        LOGGER.info("%s case=%s success=%s", cfg.model.name, case_id, success)

    success_rate = (successes / tested) if tested else 0.0
    summary = {
        "model_key": model_name,
        "model_name": cfg.model.name,
        "layer": int(handler._layer),
        "tested": tested,
        "successes": successes,
        "skipped": skipped,
        "success_rate": success_rate,
    }

    return {"summary": summary, "cases": results}


def main():
    parser = argparse.ArgumentParser(description="ROME-only benchmark")
    parser.add_argument("--models", nargs="+", required=True, help="Model config stems (e.g., mistral-7b-v0.3)")
    parser.add_argument("--n-tests", type=int, default=10)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--output-dir", default="./analysis_out")
    parser.add_argument("--override", nargs="*", default=[], help="OmegaConf dotlist overrides (e.g. model.layer=8 model.lr=1.0)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    aggregate = {"timestamp": ts, "n_tests": args.n_tests, "results": {}}

    for model in args.models:
        result = run_single_model(model, args.n_tests, args.start_idx, args.override)
        aggregate["results"][model] = result

        out_file = out_dir / f"rome_only_{model}_{ts}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        LOGGER.info("Saved %s", out_file)
        LOGGER.info("Summary %s", result["summary"])

    agg_file = out_dir / f"rome_only_all_{ts}.json"
    with open(agg_file, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    LOGGER.info("Saved %s", agg_file)


if __name__ == "__main__":
    main()
