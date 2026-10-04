"""Legacy token-by-block tracing with paired noise and validated interventions.

This retains the legacy experiment: corrupt all subject tokens with one shared
noise vector and restore each subject token at each whole-block output. It does
not use the PR's MLP windows, adaptive noise, or regional layer selector.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.causal_trace.causal_trace import TraceExample, _dataset_examples
from src.causal_trace.model_adapter import (
    corrupt_hook,
    embedding_module_name,
    embedding_std,
    hidden_from_output,
    make_noise_samples,
    mlp_state_at_position,
    module_dict,
    prepare_inputs,
    probability,
    restore_hook,
    temporary_hooks,
    top_token,
)
from src.causal_trace.tokenization import TraceValidationError, find_subject_span, target_token_ids
from src.handlers.rome import ModelHandler

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LegacySettings:
    output_dir: Path
    num_valid_facts: int
    max_dataset_examples_to_scan: int
    num_noise_samples: int
    noise_std: float | None
    require_correct_clean_prediction: bool
    seed: int
    case_index_file: str | None = None
    case_start: int = 0
    allow_article_prefix: bool = False

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "LegacySettings":
        section = cfg.command.legacy_trace
        counts = {}
        for key in ("num_valid_facts", "max_dataset_examples_to_scan", "num_noise_samples"):
            value = section[key]
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"legacy_trace.{key} must be a positive integer")
            counts[key] = int(value)
        std = section.noise_std
        if std is not None:
            std = float(std)
            if not math.isfinite(std) or std <= 0:
                raise ValueError("legacy_trace.noise_std must be finite and positive")
        return cls(
            output_dir=Path(str(section.output_dir)),
            noise_std=std,
            require_correct_clean_prediction=bool(section.require_correct_clean_prediction),
            seed=int(section.seed),
            case_index_file=section.get("case_index_file"),
            case_start=int(section.get("case_start", 0)),
            allow_article_prefix=bool(section.get("allow_article_prefix", False)),
            **counts,
        )


def resolve_block_names(handler: ModelHandler, modules: dict[str, torch.nn.Module]) -> dict[int, str]:
    """Resolve actual block outputs; hidden_states[l+1] can include final norm."""
    template = str(handler.cfg.model.restore_layer_name_template)
    names = {layer: template.format(layer) for layer in range(int(handler.num_of_layers))}
    if not names or len(set(names.values())) != len(names):
        raise ValueError("Legacy tracing requires a distinct block module for every layer")
    missing = [name for name in names.values() if name not in modules]
    if missing:
        raise KeyError(f"Legacy restore blocks not found: {missing}")
    embedding = embedding_module_name(handler.cfg)
    if embedding in names.values():
        raise ValueError("Embedding corruption and block restoration must use different modules")
    return names


def trace_example(
    handler: ModelHandler,
    example: TraceExample,
    *,
    modules: dict[str, torch.nn.Module],
    block_names: dict[int, str],
    noise_std: float,
    num_noise_samples: int,
    seed: int,
    require_correct_clean: bool,
    allow_article_prefix: bool = False,
) -> dict[str, Any]:
    inputs = prepare_inputs(handler, example.prompt)
    span = find_subject_span(handler.tokenizer, example.prompt, example.subject)
    sequence_length = int(inputs["input_ids"].shape[1])
    if inputs["input_ids"].shape[0] != 1 or not all(0 <= p < sequence_length for p in span.positions):
        raise TraceValidationError("Subject span does not index a single valid model input")
    target_ids = target_token_ids(handler.tokenizer, example.target, prompt=example.prompt)
    target_id = int(target_ids[0])
    cache: dict[tuple[int, int], torch.Tensor] = {}

    def capture(layer: int):
        def hook(_module, _input, output):
            for position in span.positions:
                cache[layer, position] = mlp_state_at_position(
                    hidden_from_output(output),
                    position,
                    sequence_length,
                )
            return output

        return hook

    hooks = [(modules[name], capture(layer)) for layer, name in block_names.items()]
    with torch.inference_mode(), temporary_hooks(hooks):
        clean = handler.model(**inputs, use_cache=False)
    expected = {(layer, position) for layer in block_names for position in span.positions}
    if set(cache) != expected:
        raise RuntimeError("Clean block cache is incomplete; a configured block was not executed")
    clean_top_id, _ = top_token(clean)
    original_clean_top_id = clean_top_id
    tracing_prompt = example.prompt
    answer_prefix = ""
    validation_scope = "first_target_token" if require_correct_clean else "unchecked"
    if require_correct_clean and clean_top_id != target_id:
        expected_text = handler.tokenizer.decode([target_id])
        predicted_text = handler.tokenizer.decode([clean_top_id])
        mismatch = (f"clean-token mismatch: expected token {target_id} ({expected_text!r}), "
                    f"got {clean_top_id} ({predicted_text!r})")
        if not allow_article_prefix or predicted_text.strip().casefold() not in {"the", "a", "an"}:
            raise TraceValidationError(mismatch)
        # An article alone does not establish factual knowledge. Accept this
        # fallback only when the clean greedy continuation spells the ENTIRE
        # expected target after its one predicted article.
        tracing_prompt += predicted_text
        prefixed_inputs = prepare_inputs(handler, tracing_prompt)
        expected_prefix = torch.cat((inputs["input_ids"], inputs["input_ids"].new_tensor([[clean_top_id]])), dim=1)
        if not torch.equal(prefixed_inputs["input_ids"], expected_prefix):
            raise TraceValidationError(mismatch + "; predicted article changes the prompt token boundary")
        target_ids = target_token_ids(handler.tokenizer, example.target, prompt=tracing_prompt)
        validation_inputs = dict(prefixed_inputs)
        continuation = inputs["input_ids"].new_tensor([target_ids[:-1]])
        validation_inputs["input_ids"] = torch.cat((prefixed_inputs["input_ids"], continuation), dim=1)
        if "attention_mask" in validation_inputs:
            validation_inputs["attention_mask"] = torch.cat(
                (prefixed_inputs["attention_mask"], torch.ones_like(continuation)), dim=1
            )
        prefix_length = int(prefixed_inputs["input_ids"].shape[1])
        with torch.inference_mode():
            validation_output = handler.model(**validation_inputs, use_cache=False)
        greedy_ids = validation_output.logits[0, prefix_length - 1 : prefix_length - 1 + len(target_ids)].argmax(-1)
        if greedy_ids.detach().cpu().tolist() != target_ids:
            raise TraceValidationError(mismatch + "; article continuation does not match the full expected target")
        answer_prefix = predicted_text
        validation_scope = "full_target_after_predicted_article"
        inputs = prefixed_inputs
        sequence_length = prefix_length
        target_id = int(target_ids[0])
        cache.clear()
        with torch.inference_mode(), temporary_hooks(hooks):
            clean = handler.model(**inputs, use_cache=False)
        if set(cache) != expected:
            raise RuntimeError("Clean block cache is incomplete after article-prefix validation")
        clean_top_id, _ = top_token(clean)
        if clean_top_id != target_id:
            raise TraceValidationError("Article-prefixed clean prediction changed between validation and tracing")
    clean_probability = probability(clean, target_id)
    embedding = modules[embedding_module_name(handler.cfg)]
    weight = getattr(embedding, "weight", None)
    if weight is None or weight.dim() != 2:
        raise RuntimeError("Legacy corruption requires a rank-2 embedding weight")
    # Preserve legacy's shared vector across subject tokens. Freeze it per draw
    # and reuse it for the baseline and ALL token/layer restorations.
    vectors = make_noise_samples(
        num_samples=num_noise_samples,
        subject_length=1,
        hidden_size=int(weight.shape[-1]),
        noise_std=noise_std,
        device=weight.device,
        dtype=weight.dtype,
        seed=seed,
    )
    rows = []
    corrupt_probabilities = []
    for draw in range(num_noise_samples):
        noise = vectors[draw : draw + 1].expand(-1, len(span.positions), -1)
        corruption = (embedding, corrupt_hook(span.positions, noise))
        with torch.inference_mode(), temporary_hooks([corruption]):
            corrupt = handler.model(**inputs, use_cache=False)
        corrupt_probability = probability(corrupt, target_id)
        corrupt_probabilities.append(corrupt_probability)
        for token_offset, position in enumerate(span.positions):
            for layer, name in block_names.items():
                restoration = (
                    modules[name],
                    restore_hook(position, cache[layer, position], sequence_length),
                )
                with torch.inference_mode(), temporary_hooks([corruption, restoration]):
                    restored = handler.model(**inputs, use_cache=False)
                restored_probability = probability(restored, target_id)
                rows.append(
                    {
                        "draw": draw,
                        "token_offset": token_offset,
                        "token_position": position,
                        "layer": layer,
                        "clean_probability": clean_probability,
                        "corrupt_probability": corrupt_probability,
                        "restored_probability": restored_probability,
                        "indirect_effect": restored_probability - corrupt_probability,
                    }
                )
    return {
        "prompt_id": example.prompt_id,
        "prompt": example.prompt,
        "tracing_prompt": tracing_prompt,
        "answer_prefix": answer_prefix,
        "clean_validation_scope": validation_scope,
        "original_clean_top_token_id": original_clean_top_id,
        "subject": example.subject,
        "target": example.target,
        "target_token_ids": target_ids,
        "target_first_token_id": target_id,
        "target_scope": "first_continuation_token",
        "subject_positions": span.positions,
        "clean_top_token_id": clean_top_id,
        "clean_probability": clean_probability,
        "mean_corrupt_probability": float(np.mean(corrupt_probabilities)),
        "noise_std": noise_std,
        "seed": seed,
        "rows": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(cfg: DictConfig, handler: ModelHandler, settings: LegacySettings) -> Path:
    handler.model.eval()
    modules = module_dict(handler.model)
    block_names = resolve_block_names(handler, modules)
    embedding_name = embedding_module_name(cfg)
    if embedding_name not in modules:
        raise KeyError(f"Embedding module not found: {embedding_name}")
    std = settings.noise_std
    if std is None:
        configured = getattr(cfg.model, "corruption_noise_multiplier", None)
        std = float(configured) if configured is not None else 3.0 * embedding_std(handler, modules)
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Resolved legacy noise standard deviation must be finite and positive")
    slug = str(cfg.model.name).replace("/", "_")
    out_dir = settings.output_dir / f"{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    facts = []
    rejections = []
    scanned = 0
    case_selection = None
    if settings.case_index_file:
        from src.counterfact_selection import load_case_manifest, load_cases_from_manifest

        manifest = load_case_manifest(settings.case_index_file)
        count = min(settings.max_dataset_examples_to_scan, manifest["count"] - settings.case_start)
        _, cases = load_cases_from_manifest(settings.case_index_file, start_idx=settings.case_start, n_tests=count)
        examples = [TraceExample(str(c["case_id"]), c["fact_tuple"][0].format(c["subject"]),
                                 c["subject"], c["target_true_str"].strip()) for c in cases]
        case_selection = {"manifest_hash": manifest["manifest_hash"], "start": settings.case_start,
                          "case_ids": [c["case_id"] for c in cases]}
    else:
        examples = _dataset_examples(cfg, max_scan=settings.max_dataset_examples_to_scan)
    for index, example in enumerate(examples):
        scanned += 1
        try:
            fact = trace_example(
                handler,
                example,
                modules=modules,
                block_names=block_names,
                noise_std=std,
                num_noise_samples=settings.num_noise_samples,
                seed=settings.seed + index,
                require_correct_clean=settings.require_correct_clean_prediction,
                allow_article_prefix=settings.allow_article_prefix,
            )
        except TraceValidationError as exc:
            rejections.append({"prompt_id": example.prompt_id, "reason": str(exc)})
            continue
        facts.append(fact)
        (out_dir / f"fact_{index:06d}.json").write_text(json.dumps(fact, indent=2), encoding="utf-8")
        LOGGER.info("Legacy fixed: %s/%s valid facts", len(facts), settings.num_valid_facts)
        if len(facts) >= settings.num_valid_facts:
            break
    rows = [{"prompt_id": fact["prompt_id"], **row} for fact in facts for row in fact["rows"]]
    fields = [
        "prompt_id",
        "draw",
        "token_offset",
        "token_position",
        "layer",
        "clean_probability",
        "corrupt_probability",
        "restored_probability",
        "indirect_effect",
    ]
    _write_csv(out_dir / "traces.csv", rows, fields)
    # Average draws within a fact first; each fact has equal weight. Token offset
    # indexes within the subject, not an absolute prompt position.
    grouped: dict[tuple[int, int], list[float]] = {}
    for fact in facts:
        per_fact: dict[tuple[int, int], list[float]] = {}
        for row in fact["rows"]:
            per_fact.setdefault((row["token_offset"], row["layer"]), []).append(row["indirect_effect"])
        for key, values in per_fact.items():
            grouped.setdefault(key, []).append(float(np.mean(values)))
    profile = [
        {
            "token_offset": offset,
            "layer": layer,
            "fact_count": len(values),
            "mean_indirect_effect": float(np.mean(values)),
        }
        for (offset, layer), values in sorted(grouped.items())
    ]
    _write_csv(out_dir / "profile.csv", profile, ["token_offset", "layer", "fact_count", "mean_indirect_effect"])
    summary = {
        "variant": "causal-kuba-fix",
        "legacy_source_commit": "357c51bb6f8df50acce0041a7c40a784af5a5fc1",
        "model": str(cfg.model.name),
        "status": "complete" if len(facts) == settings.num_valid_facts else "insufficient_valid_facts",
        "settings": {**asdict(settings), "output_dir": str(settings.output_dir)},
        "resolved_model_config": OmegaConf.to_container(cfg.model, resolve=True),
        "resolved_dataset_config": OmegaConf.to_container(cfg.dataset_facts, resolve=True),
        "noise_std": std,
        "noise_layout": "one_shared_vector_per_draw_across_subject_tokens",
        "restoration_site": "whole_block_output_each_subject_token",
        "block_modules": block_names,
        "embedding_module": embedding_name,
        "rome_projection_modules": {layer: str(handler._layer_name_template).format(layer) for layer in block_names},
        "target_scope": "first_continuation_token",
        "scanned_facts": scanned,
        "valid_facts": len(facts),
        "rejected_facts": len(rejections),
        "rejection_counts": dict(Counter(row["reason"].split(":", 1)[0] for row in rejections)),
        "rejections": rejections,
        "selected_layer": None,
        "selection_status": "legacy_profile_only_no_automatic_selection",
        "case_selection": case_selection,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not facts:
        raise TraceValidationError(f"No valid facts for legacy tracing; diagnostics saved in {out_dir}")
    return out_dir


def causal_trace(cfg: DictConfig) -> Path:
    settings = LegacySettings.from_config(cfg)
    handler = ModelHandler(cfg)
    return run(cfg, handler, settings)
