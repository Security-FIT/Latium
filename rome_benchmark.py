#!/usr/bin/env python3
"""
ROME-only benchmark runner.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>

Runs a fixed number of independent ROME edits and reports success rate.
No structural detectors are executed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas
import torch
from omegaconf import OmegaConf

from src.handlers.rome import ModelHandler
from src.evaluation.rome import compute_rome_metrics
from src.common.loading import load_dataset
from src.common.model_config import load_model_config
from src.rome.optimization import gather_k, insert_kv, optimize_v
from src.rome.prefixes import resolve_rome_sample_count
from src.runtime import set_global_seed


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG_DIR = Path(__file__).parent / "src" / "config"


def build_cfg(model_name: str, *, runtime: dict | None, seed: int):
    model_cfg = load_model_config(model_name)
    cfg = OmegaConf.create(
        {
            "seed": int(seed),
            "runtime": dict(runtime or {}),
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


def run_single_model(
    model_name: str,
    n_tests: int,
    start_idx: int,
    overrides: list[str],
    *,
    runtime: dict | None,
    seed: int,
) -> dict:
    set_global_seed(seed)
    cfg = build_cfg(model_name, runtime=runtime, seed=seed)
    cfg = apply_overrides(cfg, overrides)
    if not hasattr(cfg.model, "device"):
        cfg.model.device = "cuda"

    handler = ModelHandler(cfg)
    dataset = load_dataset(cfg)
    df = pandas.DataFrame(dataset)

    layer_name = handler._layer_name_template.format(handler._layer)
    tested = 0
    skipped = 0
    results = []
    # Accumulators for aggregate ROME paper metrics
    es_scores = []
    em_scores = []
    ps_scores = []
    ns_scores = []
    s_scores = []

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
        target_new_str = rewrite["target_new"]["str"]
        target_true_str = rewrite["target_true"]["str"]
        prompt_text = rewrite["prompt"].format(rewrite["subject"])

        paraphrase_prompts = getattr(prompt_dict, "paraphrase_prompts", None)
        neighborhood_prompts = getattr(prompt_dict, "neighborhood_prompts", None)
        if isinstance(paraphrase_prompts, float):
            paraphrase_prompts = None
        if isinstance(neighborhood_prompts, float):
            neighborhood_prompts = None

        metrics = None
        error = None

        try:
            k = gather_k(handler, fact_tuple=fact_tuple, N=resolve_rome_sample_count(cfg, "k_N"))
            delta = optimize_v(
                handler,
                fact_tuple=fact_tuple,
                N_prompts=resolve_rome_sample_count(cfg, "v_N"),
                N_optim_steps=handler.epochs,
                verbose=False,
            )
            if delta is None:
                raise RuntimeError("optimize_v returned None")

            insert_kv(handler, k, delta)

            # Compute ROME paper metrics (probability-based)
            metrics = compute_rome_metrics(
                handler,
                prompt_text,
                target_new_str,
                target_true_str,
                paraphrase_prompts=paraphrase_prompts or [],
                neighborhood_prompts=neighborhood_prompts or [],
            )

            es_scores.append(metrics["efficacy_score"])
            em_scores.append(metrics["efficacy_magnitude"])
            if metrics["paraphrase_score"] is not None:
                ps_scores.append(metrics["paraphrase_score"])
            if metrics["neighborhood_score"] is not None:
                ns_scores.append(metrics["neighborhood_score"])
            s_scores.append(metrics["overall_score"])

        except Exception as exc:
            error = str(exc)
            skipped += 1
        finally:
            handler._get_module(layer_name).weight = torch.nn.Parameter(original_weight)
            handler.remove_hooks()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        case_result = {
            "case_id": case_id,
            "relation_id": rewrite.get("relation_id", ""),
            "subject": rewrite["subject"],
            "target_new": target_new_str,
            "target_true": target_true_str,
            "error": error,
        }
        if metrics is not None:
            case_result.update(metrics)

        results.append(case_result)

        if metrics:
            LOGGER.info(
                "%s case=%s ES=%.1f EM=%.4f PS=%s NS=%s S=%.4f",
                cfg.model.name,
                case_id,
                metrics["efficacy_score"],
                metrics["efficacy_magnitude"],
                f'{metrics["paraphrase_score"]:.4f}' if metrics["paraphrase_score"] is not None else "N/A",
                f'{metrics["neighborhood_score"]:.4f}' if metrics["neighborhood_score"] is not None else "N/A",
                metrics["overall_score"],
            )
        else:
            LOGGER.info("%s case=%s SKIPPED: %s", cfg.model.name, case_id, error)

    n_evaluated = tested - skipped
    summary = {
        "model_key": model_name,
        "model_name": cfg.model.name,
        "layer": int(handler._layer),
        "tested": tested,
        "skipped": skipped,
        "n_evaluated": n_evaluated,
        "mean_efficacy_score": float(np.mean(es_scores)) if es_scores else 0.0,
        "mean_efficacy_magnitude": float(np.mean(em_scores)) if em_scores else 0.0,
        "mean_paraphrase_score": float(np.mean(ps_scores)) if ps_scores else 0.0,
        "mean_neighborhood_score": float(np.mean(ns_scores)) if ns_scores else 0.0,
        "mean_overall_score": float(np.mean(s_scores)) if s_scores else 0.0,
    }

    return {"summary": summary, "cases": results}


def main(argv: list[str] | None = None) -> int:
    from src.main import run_hydra

    raw = list(sys.argv[1:] if argv is None else argv)
    if any(arg.startswith("-") for arg in raw):
        print(
            "rome_benchmark.py now uses Hydra overrides. Example: "
            "python rome_benchmark.py rome_benchmark.models=[gpt2-large]",
            file=sys.stderr,
        )
        return 2
    return run_hydra(["command=rome_benchmark", *raw])


if __name__ == "__main__":
    raise SystemExit(main())
