#!/usr/bin/env python3
"""
template_prefix_spectral_experiment.py

Run a single-fact ROME edit on one model multiple times while varying
prefix/template construction, then measure how much spectral/edit signals shift.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from structural_benchmark import (
    build_cfg,
    extract_weights,
    find_second_moment_files,
    get_fc_template,
    load_test_cases,
    resolve_trim,
    spectral_signal_delta,
    to_serializable,
)
from src.handlers.rome import ModelHandler
from src.rome.common import PrefixMode, PrefixGenerationHandler, gather_k, insert_kv, optimize_v
from src.structural.edit_presence_detector import RomeEditPresenceDetector
from src.structural.spectral_detector import SpectralDetector

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class RunConfig:
    name: str
    prefix_mode: PrefixMode
    prefix_source: Optional[str]
    n_prompts: int
    prefix_range: Tuple[int, int]
    subject_template: str
    k_additional_prompts: List[str]
    seed: int


def _token_prediction_metrics(handler: ModelHandler, fact_tuple: Tuple[str, str, str, str]) -> Dict[str, object]:
    prompt_text = fact_tuple[0].format(fact_tuple[1])
    tokens = handler.tokenize_prompt(prompt_text)
    with torch.no_grad():
        out = handler.model(**tokens)
        last_logits = out.logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)

    predicted_token_id = int(torch.argmax(last_logits).item())
    predicted_token = handler.tokenizer.decode([predicted_token_id])

    target_ids = handler.tokenize_prompt(fact_tuple[2]).input_ids[0].tolist()
    target_id = int(target_ids[0]) if target_ids else predicted_token_id
    target_token = handler.tokenizer.decode([target_id])
    target_prob = float(probs[target_id].item())
    top_prob = float(probs[predicted_token_id].item())

    return {
        "prompt": prompt_text,
        "predicted_token": predicted_token,
        "predicted_token_id": predicted_token_id,
        "target_token": target_token,
        "target_token_id": target_id,
        "target_prob": target_prob,
        "top_prob": top_prob,
        "is_target_top1": predicted_token.strip().lower() == fact_tuple[2].strip().lower(),
    }


def _write_prefix_cache(cache_path: Path, templates: List[str]) -> str:
    payload = {"templates": templates}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(cache_path)


def _build_fact_related_template_sets(fact_tuple: Tuple[str, str, str, str]) -> Dict[str, List[str]]:
    prompt_tmpl, subject, target_new, target_true = fact_tuple
    prompt_core = " ".join(prompt_tmpl.replace("{}", "").split()).strip(" ,.;:")
    subject_clean = subject.strip()
    new_clean = target_new.strip()
    true_clean = target_true.strip()

    relation_templates = [
        "{}",
        f"In relation to {prompt_core}, {{}}",
        f"Focused on {prompt_core}, {{}}",
        f"When discussing {prompt_core}, {{}}",
        f"According to the claim about {prompt_core}, {{}}",
        f"In that same {prompt_core} context, {{}}",
        f"For the {prompt_core} record, {{}}",
        f"Within this {prompt_core} statement, {{}}",
        f"In the hidden-fact framing of {prompt_core}, {{}}",
        f"Given the {prompt_core} prompt, {{}}",
    ]

    target_templates = [
        "{}",
        f"A statement linked to '{new_clean}': {{}}",
        f"A rewrite aiming at '{new_clean}': {{}}",
        f"Claim involving {new_clean}: {{}}",
        f"Assertion near '{new_clean}': {{}}",
        f"Context that would support {new_clean}: {{}}",
        f"Line of reasoning toward {new_clean}: {{}}",
        f"Edited-fact version around {new_clean}: {{}}",
        f"Target-oriented framing ({new_clean}): {{}}",
        f"In a new-target narrative about {new_clean}, {{}}",
    ]

    contrast_templates = [
        "{}",
        f"Compare old '{true_clean}' vs new '{new_clean}': {{}}",
        f"Before/after fact switch ({true_clean} -> {new_clean}): {{}}",
        f"Contrasting true and edited answer, {{}}",
        f"Difference between '{true_clean}' and '{new_clean}': {{}}",
        f"Under the edited-vs-original lens, {{}}",
        f"Across conflicting answers ({true_clean} / {new_clean}), {{}}",
        f"While replacing {true_clean} with {new_clean}, {{}}",
        f"In a contradiction-focused framing, {{}}",
        f"During fact substitution analysis, {{}}",
    ]

    subject_templates = [
        "{}",
        f"Profile note on {subject_clean}: {{}}",
        f"Bio context for {subject_clean}: {{}}",
        f"Entity-focused framing ({subject_clean}), {{}}",
        f"Subject-specific statement for {subject_clean}: {{}}",
        f"Investigating facts about {subject_clean}, {{}}",
        f"Record about {subject_clean}: {{}}",
        f"Data point tied to {subject_clean}: {{}}",
        f"Fact probe around {subject_clean}: {{}}",
        f"Hidden-edit check for {subject_clean}: {{}}",
    ]

    return {
        "fact_relation": relation_templates,
        "fact_target": target_templates,
        "fact_contrast": contrast_templates,
        "fact_subject": subject_templates,
    }


def _build_extended_fact_template_sets(fact_tuple: Tuple[str, str, str, str]) -> Dict[str, List[str]]:
    prompt_tmpl, subject, target_new, target_true = fact_tuple
    prompt_core = " ".join(prompt_tmpl.replace("{}", "").split()).strip(" ,.;:")
    subject_clean = subject.strip()
    new_clean = target_new.strip()
    true_clean = target_true.strip()

    relation_long = [
        "{}",
        f"In an evidence-grounded discussion of '{prompt_core}', {{}}",
        f"Within a fact-verification pass focused on {prompt_core}, {{}}",
        f"In a context explicitly anchored to the edited relation ({prompt_core}), {{}}",
        f"From a retrieval-oriented perspective about {prompt_core}, {{}}",
        f"Cross-checking the rewritten claim around {prompt_core}, {{}}",
        f"Under a detailed relational framing for {prompt_core}, {{}}",
        f"Using extended context tied to {prompt_core}, {{}}",
        f"While preserving topical continuity for {prompt_core}, {{}}",
        f"In a long-form relation-aware prefix for {prompt_core}, {{}}",
    ]

    target_long = [
        "{}",
        f"In a continuation that repeatedly reinforces '{new_clean}', {{}}",
        f"Given an edited-memory framing centered on {new_clean}, {{}}",
        f"Using an expanded prefix emphasizing {new_clean} as the desired fact, {{}}",
        f"In a targeted recall setup where {new_clean} is foregrounded, {{}}",
        f"Inside a semantically aligned narrative that points to {new_clean}, {{}}",
        f"With explicit lexical overlap around {new_clean}, {{}}",
        f"In a stronger target-conditioned preamble for {new_clean}, {{}}",
        f"During revised-fact reinforcement near {new_clean}, {{}}",
        f"In a high-relatedness prompt extension keyed to {new_clean}, {{}}",
    ]

    subject_long = [
        "{}",
        f"Entity dossier for {subject_clean}, keeping factual focus explicit: {{}}",
        f"Extended biographical framing about {subject_clean} and relevant claims: {{}}",
        f"Subject-centric context block around {subject_clean} for edit probing: {{}}",
        f"In a profile narrative where {subject_clean} remains the central entity, {{}}",
        f"When compiling a long prefix about {subject_clean}, {{}}",
        f"From records discussing {subject_clean} in detail, {{}}",
        f"In an entity-memory consistency check tied to {subject_clean}, {{}}",
        f"Subject-conditioned continuation about {subject_clean}, {{}}",
        f"High-context prompt focused on {subject_clean} before answering, {{}}",
    ]

    contrast_long = [
        "{}",
        f"Explicit old-vs-new analysis ({true_clean} -> {new_clean}) before completion: {{}}",
        f"In a contradiction-aware scaffold comparing '{true_clean}' and '{new_clean}', {{}}",
        f"With an edit-audit framing that contrasts old and edited targets, {{}}",
        f"While stress-testing substitution from {true_clean} to {new_clean}, {{}}",
        f"Through a longer conflict-sensitive preamble about fact replacement, {{}}",
        f"During a revised-knowledge consistency check ({true_clean} versus {new_clean}), {{}}",
        f"Under high-contrast framing for true/edited answers, {{}}",
        f"Before answering, compare historical target {true_clean} with edited target {new_clean}: {{}}",
        f"In a long comparative template centered on this fact change, {{}}",
    ]

    return {
        "fact_relation_long": relation_long,
        "fact_target_long": target_long,
        "fact_subject_long": subject_long,
        "fact_contrast_long": contrast_long,
    }


def build_default_run_configs(
    base_seed: int,
    cache_dir: Path,
    fact_tuple: Tuple[str, str, str, str],
) -> List[RunConfig]:
    cache_news = _write_prefix_cache(
        cache_dir / "external_news.json",
        [
            "{}",
            "Breaking report: {}",
            "News brief: {}",
            "According to witnesses, {}",
            "Analysts noted that {}",
            "In an official statement, {}",
            "International desk: {}",
            "Developing story: {}",
            "From archived reports, {}",
            "Public records indicate {}",
        ],
    )
    cache_qa = _write_prefix_cache(
        cache_dir / "external_qa.json",
        [
            "{}",
            "Q: What is true here? A: {}",
            "FAQ entry: {}",
            "Short answer: {}",
            "Question context: {}",
            "Exam note: {}",
            "Tutor hint: {}",
            "Study card: {}",
            "Review answer: {}",
            "Verification step: {}",
        ],
    )
    cache_temporal = _write_prefix_cache(
        cache_dir / "external_temporal.json",
        [
            "{}",
            "Historically, {}",
            "By 1950, {}",
            "In modern references, {}",
            "Over the decades, {}",
            "In recent archives, {}",
            "Earlier records show {}",
            "Contemporary sources state {}",
            "Chronologically, {}",
            "As of today, {}",
        ],
    )
    cache_noisy = _write_prefix_cache(
        cache_dir / "external_noisy.json",
        [
            "{}",
            "FYI -- {}",
            "Note (quick): {}",
            "Heads-up: {}",
            "Random context: {}",
            "Side remark: {}",
            "Unfiltered take: {}",
            "Offhand summary: {}",
            "Loose wording: {}",
            "Messy note: {}",
        ],
    )
    fact_sets = _build_fact_related_template_sets(fact_tuple)
    extended_fact_sets = _build_extended_fact_template_sets(fact_tuple)
    cache_fact_relation = _write_prefix_cache(cache_dir / "external_fact_relation.json", fact_sets["fact_relation"])
    cache_fact_target = _write_prefix_cache(cache_dir / "external_fact_target.json", fact_sets["fact_target"])
    cache_fact_contrast = _write_prefix_cache(cache_dir / "external_fact_contrast.json", fact_sets["fact_contrast"])
    cache_fact_subject = _write_prefix_cache(cache_dir / "external_fact_subject.json", fact_sets["fact_subject"])
    cache_fact_relation_long = _write_prefix_cache(
        cache_dir / "external_fact_relation_long.json",
        extended_fact_sets["fact_relation_long"],
    )
    cache_fact_target_long = _write_prefix_cache(
        cache_dir / "external_fact_target_long.json",
        extended_fact_sets["fact_target_long"],
    )
    cache_fact_subject_long = _write_prefix_cache(
        cache_dir / "external_fact_subject_long.json",
        extended_fact_sets["fact_subject_long"],
    )
    cache_fact_contrast_long = _write_prefix_cache(
        cache_dir / "external_fact_contrast_long.json",
        extended_fact_sets["fact_contrast_long"],
    )

    runs = [
        RunConfig("self_short", PrefixMode.SELF, None, 20, (2, 8), "{} is a", [], base_seed + 1),
        RunConfig("self_long", PrefixMode.SELF, None, 20, (8, 20), "{} is a", [], base_seed + 2),
        RunConfig("self_with_k_hints", PrefixMode.SELF, None, 20, (2, 14), "{} is a", ["For context, {}"], base_seed + 3),
        RunConfig(
            "self_extended_related",
            PrefixMode.SELF,
            None,
            30,
            (10, 24),
            "{} is a",
            ["In the same fact setting, {}", "Keep the subject context aligned: {}"],
            base_seed + 4,
        ),
        RunConfig("template_short", PrefixMode.TEMPLATE, None, 20, (2, 8), "{} is a", [], base_seed + 5),
        RunConfig("template_long", PrefixMode.TEMPLATE, None, 20, (8, 20), "{} is a", [], base_seed + 6),
        RunConfig("template_alt_subject", PrefixMode.TEMPLATE, None, 20, (4, 14), "{} can be described as", [], base_seed + 7),
        RunConfig(
            "template_extended_related",
            PrefixMode.TEMPLATE,
            None,
            30,
            (10, 24),
            "{} can be characterized as",
            ["Fact-aware preface: {}", "Subject continuity note: {}"],
            base_seed + 8,
        ),
        RunConfig("external_qa", PrefixMode.EXTERNAL, cache_qa, 20, (2, 14), "{} is a", [], base_seed + 10),
        RunConfig("external_temporal", PrefixMode.EXTERNAL, cache_temporal, 20, (2, 14), "{} is a", [], base_seed + 11),
        RunConfig("external_fact_relation", PrefixMode.EXTERNAL, cache_fact_relation, 20, (2, 14), "{} is a", [], base_seed + 13),
        RunConfig("external_fact_target", PrefixMode.EXTERNAL, cache_fact_target, 20, (2, 14), "{} is a", [], base_seed + 14),
        RunConfig("external_fact_contrast", PrefixMode.EXTERNAL, cache_fact_contrast, 20, (2, 14), "{} is a", [], base_seed + 15),
        RunConfig("external_fact_subject", PrefixMode.EXTERNAL, cache_fact_subject, 20, (2, 14), "{} is a", [], base_seed + 16),
        RunConfig(
            "external_fact_relation_long",
            PrefixMode.EXTERNAL,
            cache_fact_relation_long,
            30,
            (10, 24),
            "{} is a",
            [],
            base_seed + 17,
        ),
        RunConfig(
            "external_fact_target_long",
            PrefixMode.EXTERNAL,
            cache_fact_target_long,
            30,
            (10, 24),
            "{} is a",
            [],
            base_seed + 18,
        ),
        RunConfig(
            "external_fact_subject_long",
            PrefixMode.EXTERNAL,
            cache_fact_subject_long,
            30,
            (10, 24),
            "{} is a",
            [],
            base_seed + 19,
        ),
        RunConfig(
            "external_fact_contrast_long",
            PrefixMode.EXTERNAL,
            cache_fact_contrast_long,
            30,
            (10, 24),
            "{} is a",
            [],
            base_seed + 20,
        ),
        RunConfig(
            "external_fact_target_long_hints",
            PrefixMode.EXTERNAL,
            cache_fact_target_long,
            30,
            (12, 26),
            "{} can be described as",
            ["Edited claim focus: {}", "Aligned relation context: {}"],
            base_seed + 21,
        ),
        RunConfig(
            "external_fact_subject_long_hints",
            PrefixMode.EXTERNAL,
            cache_fact_subject_long,
            30,
            (12, 26),
            "{} can be characterized as",
            ["Entity-specific context: {}", "Keep biography aligned: {}"],
            base_seed + 22,
        ),
    ]
    if len(runs) != 20:
        raise ValueError(f"Expected 20 run configurations, got {len(runs)}")
    return runs


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(
    model_name: str,
    case_idx: int,
    output_dir: str,
    spectral_top_k: int,
    trim_first: Optional[int],
    trim_last: Optional[int],
    spectral_neighbor_layers: int,
) -> Dict[str, object]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cache_dir = out_dir / f"prefix_caches_{model_name.replace('/', '_')}_{ts}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(model_name)
    model_cfg = cfg.model
    sm_files, sm_dir = find_second_moment_files(model_cfg)
    if not sm_files:
        raise FileNotFoundError(
            f"Missing second moment stats for model={model_cfg.name} layer={model_cfg.layer} in {sm_dir}"
        )

    handler = ModelHandler(cfg)
    proj_template = handler._layer_name_template
    fc_template = get_fc_template(proj_template)
    original_proj = extract_weights(handler, proj_template)
    original_fc = extract_weights(handler, fc_template)

    eff_trim_first, eff_trim_last = resolve_trim(
        num_layers=handler.num_of_layers,
        trim_first=trim_first,
        trim_last=trim_last,
    )
    spectral_detector = SpectralDetector(
        top_k=spectral_top_k,
        boundary=2,
        trim_first_layers=eff_trim_first,
        trim_last_layers=eff_trim_last,
        neighbor_layers=spectral_neighbor_layers,
    )
    presence_detector = RomeEditPresenceDetector()
    baseline_spectral = spectral_detector.detect(original_proj, fc_weights=original_fc)

    cases = load_test_cases(n_tests=case_idx + 1, start_idx=0)
    case = cases[case_idx]
    fact = case["fact_tuple"]

    base_seed = 1337 + case_idx * 100
    run_cfgs = build_default_run_configs(base_seed=base_seed, cache_dir=cache_dir, fact_tuple=fact)
    runs: List[Dict[str, object]] = []

    layer_name = proj_template.format(handler._layer)

    for i, run_cfg in enumerate(run_cfgs):
        LOGGER.info("Run %d/%d: %s", i + 1, len(run_cfgs), run_cfg.name)
        _set_global_seed(run_cfg.seed)
        handler.prefix_handler = PrefixGenerationHandler(mode=run_cfg.prefix_mode, prefix_source=run_cfg.prefix_source)
        old_W = handler._get_module(layer_name).weight.detach().clone()
        pre_pred = _token_prediction_metrics(handler, fact)
        run_result: Dict[str, object] = {
            "run_index": i,
            "run_name": run_cfg.name,
            "prefix_mode": run_cfg.prefix_mode.value,
            "prefix_source": run_cfg.prefix_source,
            "n_prompts": run_cfg.n_prompts,
            "prefix_range": list(run_cfg.prefix_range),
            "subject_template": run_cfg.subject_template,
            "k_additional_prompts": run_cfg.k_additional_prompts,
            "seed": run_cfg.seed,
            "error": None,
        }

        try:
            k = gather_k(
                handler,
                fact_tuple=fact,
                N=run_cfg.n_prompts,
                prefix_range=run_cfg.prefix_range,
                additional_prompts=run_cfg.k_additional_prompts,
            )
            delta = optimize_v(
                handler,
                fact_tuple=fact,
                N_prompts=run_cfg.n_prompts,
                N_optim_steps=handler.epochs,
                subject_understanding_template=run_cfg.subject_template,
                verbose=False,
            )
            new_W, _, update_matrix = insert_kv(handler, k, delta)

            post_pred = _token_prediction_metrics(handler, fact)
            modified_proj = {idx: w.clone() for idx, w in original_proj.items()}
            modified_proj[handler._layer] = new_W.detach().cpu()

            spectral_res = spectral_detector.detect(modified_proj, fc_weights=original_fc)
            presence_res = presence_detector.detect(
                original_proj=original_proj,
                modified_proj=modified_proj,
                baseline_spectral=baseline_spectral,
                modified_spectral=spectral_res,
            )
            spec_delta = spectral_signal_delta(baseline_spectral, spectral_res)
            hybrid_scores = {int(k): float(v) for k, v in (spectral_res.get("rome_hybrid_scores") or {}).items()}
            edited_layer_score = float(hybrid_scores.get(handler._layer, 0.0))
            max_layer = int(max(hybrid_scores, key=hybrid_scores.get)) if hybrid_scores else -1
            max_score = float(hybrid_scores[max_layer]) if hybrid_scores else 0.0

            update_fro_norm = float(update_matrix.float().norm().item())
            update_spectral_norm = float(
                torch.linalg.matrix_norm(update_matrix.float(), ord=2).item()
            )

            run_result.update(
                {
                    "case_id": case["case_id"],
                    "subject": case["subject"],
                    "fact_tuple": list(fact),
                    "rome_success": bool(post_pred["is_target_top1"]),
                    "pre_prediction": pre_pred,
                    "post_prediction": post_pred,
                    "target_prob_delta": float(post_pred["target_prob"] - pre_pred["target_prob"]),
                    "k_norm": float(k.norm().item()),
                    "delta_norm": float(delta.norm().item()),
                    "update_fro_norm": update_fro_norm,
                    "update_spectral_norm": update_spectral_norm,
                    "spectral_detection": to_serializable(spectral_res),
                    "edit_presence_detection": to_serializable(presence_res),
                    "spectral_delta": to_serializable(spec_delta),
                    "edited_layer_hybrid_score": edited_layer_score,
                    "max_hybrid_layer": max_layer,
                    "max_hybrid_score": max_score,
                    "edited_layer_is_top": bool(max_layer == handler._layer),
                    "hybrid_margin_to_top": float(max_score - edited_layer_score),
                }
            )

        except Exception as exc:
            run_result["error"] = str(exc)
            LOGGER.warning("Run %s failed: %s", run_cfg.name, exc)
        finally:
            handler.remove_hooks()
            handler._get_module(layer_name).weight = torch.nn.Parameter(old_W)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        runs.append(run_result)

    ok_runs = [r for r in runs if not r.get("error")]
    summary = {
        "total_runs": len(runs),
        "successful_runs": len(ok_runs),
        "rome_success_rate": float(np.mean([r.get("rome_success", False) for r in ok_runs])) if ok_runs else 0.0,
        "edited_layer_top_rate": float(np.mean([r.get("edited_layer_is_top", False) for r in ok_runs])) if ok_runs else 0.0,
        "target_prob_delta_mean": float(np.mean([r.get("target_prob_delta", 0.0) for r in ok_runs])) if ok_runs else 0.0,
        "target_prob_delta_std": float(np.std([r.get("target_prob_delta", 0.0) for r in ok_runs])) if ok_runs else 0.0,
        "update_spectral_norm_mean": float(np.mean([r.get("update_spectral_norm", 0.0) for r in ok_runs])) if ok_runs else 0.0,
        "update_spectral_norm_std": float(np.std([r.get("update_spectral_norm", 0.0) for r in ok_runs])) if ok_runs else 0.0,
    }

    result = {
        "metadata": {
            "model": model_name,
            "target_layer": handler._layer,
            "num_layers": handler.num_of_layers,
            "case_idx": case_idx,
            "case_id": case["case_id"],
            "subject": case["subject"],
            "timestamp": datetime.now().isoformat(),
            "spectral_config": {
                "top_k": spectral_top_k,
                "trim_first": eff_trim_first,
                "trim_last": eff_trim_last,
                "neighbor_layers": spectral_neighbor_layers,
            },
        },
        "baseline_spectral": to_serializable(baseline_spectral),
        "runs": to_serializable(runs),
        "summary": summary,
    }

    json_path = out_dir / f"gpt2_prefix_template_variability_{model_name.replace('/', '_')}_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(result), f, indent=2)

    df = pd.DataFrame(
        [
            {
                "run_index": r.get("run_index"),
                "run_name": r.get("run_name"),
                "prefix_mode": r.get("prefix_mode"),
                "rome_success": r.get("rome_success"),
                "target_prob_delta": r.get("target_prob_delta"),
                "k_norm": r.get("k_norm"),
                "delta_norm": r.get("delta_norm"),
                "update_fro_norm": r.get("update_fro_norm"),
                "update_spectral_norm": r.get("update_spectral_norm"),
                "edited_layer_hybrid_score": r.get("edited_layer_hybrid_score"),
                "max_hybrid_layer": r.get("max_hybrid_layer"),
                "edited_layer_is_top": r.get("edited_layer_is_top"),
                "hybrid_margin_to_top": r.get("hybrid_margin_to_top"),
                "presence_confidence": (r.get("edit_presence_detection") or {}).get("confidence"),
                "presence_is_edited": (r.get("edit_presence_detection") or {}).get("is_edited"),
                "error": r.get("error"),
            }
            for r in runs
        ]
    )
    csv_path = out_dir / f"gpt2_prefix_template_variability_{model_name.replace('/', '_')}_{ts}.csv"
    df.to_csv(csv_path, index=False)

    LOGGER.info("Saved JSON: %s", json_path)
    LOGGER.info("Saved CSV:  %s", csv_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-edit prefix/template variability experiment")
    parser.add_argument("--model", default="gpt2-large")
    parser.add_argument("--case-idx", type=int, default=0, help="CounterFact index to use as single edit")
    parser.add_argument("--output-dir", default="./analysis_out")
    parser.add_argument("--spectral-top-k", type=int, default=50)
    parser.add_argument("--trim-first", type=int, default=None)
    parser.add_argument("--trim-last", type=int, default=None)
    parser.add_argument("--spectral-neighbor-layers", type=int, default=1)
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        case_idx=args.case_idx,
        output_dir=args.output_dir,
        spectral_top_k=args.spectral_top_k,
        trim_first=args.trim_first,
        trim_last=args.trim_last,
        spectral_neighbor_layers=args.spectral_neighbor_layers,
    )


if __name__ == "__main__":
    main()
