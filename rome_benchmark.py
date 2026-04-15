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
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
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


def normalize_target_text(text: str) -> str:
    """Normalize short targets for tolerant textual comparison."""
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = re.sub(r"[\s\.,;:!?\"'`]+$", "", s)
    s = re.sub(r"^[\s\"'`]+", "", s)
    return s


# ---------------------------------------------------------------------------
# ROME paper evaluation metrics (probability-based)
# ---------------------------------------------------------------------------

def _get_target_token_ids(tok, text: str) -> list[int]:
    """Tokenize target text without BOS/special tokens."""
    ids = tok(f" {text}", add_special_tokens=False)["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if not ids:
        ids = tok(f" {text}")["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        bos_id = getattr(tok, "bos_token_id", None)
        if bos_id is not None and len(ids) > 1 and ids[0] == bos_id:
            ids = ids[1:]
    return ids


def _test_batch_prediction(
    model, tok, prefixes: list[str],
    target_new: str, target_true: str, device,
    batch_size: int = 8,
) -> list[dict]:
    """
    Compute mean NLL for target_new and target_true given each prefix.
    Based on the original ROME paper evaluation code in test_batch_prediction.
    Processes in chunks to control GPU memory.
    """
    a_tok = _get_target_token_ids(tok, target_new)
    b_tok = _get_target_token_ids(tok, target_true)
    choice_a_len, choice_b_len = len(a_tok), len(b_tok)

    all_results = []

    for chunk_start in range(0, len(prefixes), batch_size):
        chunk_prefixes = prefixes[chunk_start:chunk_start + batch_size]
        prefix_lens = [len(n) for n in tok(chunk_prefixes)["input_ids"]]

        prompt_tok = tok(
            [f"{prefix} {suffix}" for prefix in chunk_prefixes for suffix in [target_new, target_true]],
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Account for left-padding: count leading pad tokens per row
        pad_offsets = (prompt_tok["attention_mask"] == 0).sum(dim=1).tolist()

        with torch.no_grad():
            logits = model(**prompt_tok, use_cache=False).logits

        chunk_results = np.zeros((logits.size(0),), dtype=np.float32)
        for i in range(logits.size(0)):
            cur_len = choice_a_len if i % 2 == 0 else choice_b_len
            cur_tok_ids = a_tok if i % 2 == 0 else b_tok
            offset = pad_offsets[i]
            for j in range(cur_len):
                cur_tok = cur_tok_ids[j]
                chunk_results[i] += -torch.nn.functional.log_softmax(
                    logits[i, offset + prefix_lens[i // 2] + j - 1, :], dim=0
                )[cur_tok].item()
            chunk_results[i] /= cur_len

        for i in range(0, len(chunk_results), 2):
            all_results.append({
                "target_new": chunk_results[i].item(),
                "target_true": chunk_results[i + 1].item(),
            })

        del prompt_tok, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


def compute_rome_metrics(
    handler, prompt_text: str,
    target_new_str: str, target_true_str: str,
    paraphrase_prompts: list[str] | None = None,
    neighborhood_prompts: list[str] | None = None,
) -> dict:
    """
    Compute ROME paper evaluation metrics using probability comparison.

    ES  – Efficacy Score:     1{P[o*] > P[o_c]} on rewrite prompt
    EM  – Efficacy Magnitude: P[o*] - P[o_c]
    PS  – Paraphrase Score:   mean ES across paraphrase prompts
    NS  – Neighborhood Score: mean 1{P[o_c] > P[o*]} across neighborhood prompts
    S   – Overall Score:      harmonic mean of ES, PS, NS
    """
    model = handler.model
    tok = handler.tokenizer
    device = handler.device

    all_prompts = [prompt_text]
    n_para = len(paraphrase_prompts) if paraphrase_prompts else 0
    n_neigh = len(neighborhood_prompts) if neighborhood_prompts else 0
    if paraphrase_prompts:
        all_prompts.extend(paraphrase_prompts)
    if neighborhood_prompts:
        all_prompts.extend(neighborhood_prompts)

    probs = _test_batch_prediction(model, tok, all_prompts, target_new_str, target_true_str, device)

    rewrite_prob = probs[0]
    para_probs = probs[1:1 + n_para] if n_para else []
    neigh_probs = probs[1 + n_para:] if n_neigh else []

    # ES: lower NLL = higher probability
    es = 1.0 if rewrite_prob["target_new"] < rewrite_prob["target_true"] else 0.0
    # EM: probability difference (geometric-mean per-token probability)
    em = math.exp(-rewrite_prob["target_new"]) - math.exp(-rewrite_prob["target_true"])

    # PS: mean ES across paraphrase prompts
    ps = (
        sum(1.0 for p in para_probs if p["target_new"] < p["target_true"]) / len(para_probs)
        if para_probs else None
    )

    # NS: neighborhood should still predict target_true
    ns = (
        sum(1.0 for p in neigh_probs if p["target_true"] < p["target_new"]) / len(neigh_probs)
        if neigh_probs else None
    )

    # S: harmonic mean of available scores
    components = [s for s in [es, ps, ns] if s is not None]
    if components and all(s > 0 for s in components):
        overall = len(components) / sum(1.0 / s for s in components)
    else:
        overall = 0.0

    return {
        "efficacy_score": es,
        "efficacy_magnitude": em,
        "paraphrase_score": ps,
        "neighborhood_score": ns,
        "overall_score": overall,
        "rewrite_nll": rewrite_prob,
        "paraphrase_nll": para_probs,
        "neighborhood_nll": neigh_probs,
    }


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

            # Compute ROME paper metrics (probability-based)
            metrics = compute_rome_metrics(
                handler, prompt_text,
                target_new_str, target_true_str,
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
                cfg.model.name, case_id,
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
