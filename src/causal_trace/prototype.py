"""
Prototype causal tracing workflows.

This module intentionally keeps the first implementation in one place.  The
goal is to make the five tracing modes runnable and comparable before choosing
which one should become the long-term implementation.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.common.io import to_serializable
from src.common.loading import load_dataset, logits_to_probs
from src.handlers.rome import ModelHandler

LOGGER = logging.getLogger(__name__)

TraceMode = Literal["standard", "alt", "aquin", "canonical", "fast"]
TraceComponent = Literal["residual", "mlp", "attention"]


@dataclass
class TraceExample:
    prompt_id: str
    prompt: str
    subject: str
    target: str
    target_new: str | None = None
    target_true: str | None = None
    raw_case: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenSpan:
    start: int
    end: int
    positions: list[int]
    last_position: int


@dataclass
class LayerTraceScore:
    layer: int
    mean_restored_probability: float
    std_restored_probability: float
    mean_indirect_effect: float
    std_indirect_effect: float
    normalized_recovery: float


@dataclass
class PromptTraceResult:
    prompt_id: str
    mode: str
    prompt: str
    subject: str
    target: str
    target_token_id: int
    target_token: str
    target_token_mode: str
    subject_positions: list[int]
    restore_position_type: str | None
    restore_position: int | None
    position_scope: str | None
    component: str
    clean_probability: float
    clean_top_token: str
    clean_top_probability: float
    corrupt_probabilities: list[float]
    mean_corrupt_probability: float
    std_corrupt_probability: float
    total_effect: float
    trace_reliable: bool
    trace_failure_reason: str | None
    signal_quality: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    best_trace_layer: int | None = None
    candidate_layers: list[int] = field(default_factory=list)
    best_validated_rome_edit_layer: int | None = None
    layer_scores: list[LayerTraceScore] = field(default_factory=list)
    token_layer_scores: dict[int, list[LayerTraceScore]] = field(default_factory=dict)
    tokens: list[dict[str, Any]] = field(default_factory=list)
    p_restore: list[Any] | None = None
    rome_validation: list[dict[str, Any]] = field(default_factory=list)
    skip_reason: str | None = None


class TraceValidationError(ValueError):
    """Raised when a prompt cannot be traced unambiguously."""


def _cfg_section(cfg: DictConfig, name: str) -> Any:
    value = getattr(cfg, name, None)
    return value if value is not None else OmegaConf.create({})


def _get(cfg: Any, key: str, default: Any) -> Any:
    return getattr(cfg, key, default)


def _strip_bos(tokenizer: Any, token_ids: Sequence[int]) -> list[int]:
    ids = [int(x) for x in token_ids]
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None and len(ids) > 1 and ids[0] == bos_id:
        return ids[1:]
    return ids


def _token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool = True) -> list[int]:
    try:
        raw = tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
    except TypeError:
        raw = tokenizer(text)["input_ids"]
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().tolist()
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    return [int(x) for x in raw]


def target_first_token_id(tokenizer: Any, target: str) -> int:
    """Return the first token ID used for first-token-only tracing."""
    for text, add_special in ((f" {target.strip()}", False), (target, False), (f" {target.strip()}", True)):
        ids = _strip_bos(tokenizer, _token_ids(tokenizer, text, add_special_tokens=add_special))
        if ids:
            return int(ids[0])
    raise TraceValidationError(f"Could not tokenize target: {target!r}")


def find_subject_span(tokenizer: Any, prompt: str, subject: str) -> TokenSpan:
    """Find the unique token span that overlaps the subject character span."""
    if not subject:
        raise TraceValidationError("Subject is empty")

    starts: list[int] = []
    cursor = 0
    while True:
        idx = prompt.find(subject, cursor)
        if idx == -1:
            break
        starts.append(idx)
        cursor = idx + max(1, len(subject))

    if not starts:
        raise TraceValidationError(f"Subject {subject!r} is not present in prompt")
    if len(starts) > 1:
        raise TraceValidationError(f"Subject {subject!r} appears {len(starts)} times in prompt")

    char_start = starts[0]
    char_end = char_start + len(subject)

    try:
        encoded = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
        offsets = encoded.get("offset_mapping")
    except Exception:
        offsets = None

    if offsets is not None:
        positions: list[int] = []
        for idx, (start, end) in enumerate(offsets[0].detach().cpu().tolist()):
            if end <= start:
                continue
            if end > char_start and start < char_end:
                positions.append(int(idx))
        if positions:
            return TokenSpan(
                start=positions[0],
                end=positions[-1] + 1,
                positions=positions,
                last_position=positions[-1],
            )

    prompt_ids = _strip_bos(tokenizer, _token_ids(tokenizer, prompt))
    candidates = [
        _strip_bos(tokenizer, _token_ids(tokenizer, subject, add_special_tokens=False)),
        _strip_bos(tokenizer, _token_ids(tokenizer, f" {subject}", add_special_tokens=False)),
        _strip_bos(tokenizer, _token_ids(tokenizer, subject)),
        _strip_bos(tokenizer, _token_ids(tokenizer, f" {subject}")),
    ]
    matches: list[tuple[int, int]] = []
    for subject_ids in candidates:
        if not subject_ids:
            continue
        n = len(subject_ids)
        for start in range(0, len(prompt_ids) - n + 1):
            if prompt_ids[start : start + n] == subject_ids:
                matches.append((start, start + n))
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise TraceValidationError(f"Could not identify a unique subject token span for {subject!r}")
    start, end = matches[0]
    positions = list(range(start, end))
    return TokenSpan(start=start, end=end, positions=positions, last_position=end - 1)


def _trace_examples(cfg: DictConfig) -> list[TraceExample]:
    dataset = load_dataset(cfg)
    if isinstance(dataset, dict) and "requested_rewrite" in dataset:
        records = dataset["requested_rewrite"]
    elif hasattr(dataset, "to_list"):
        records = dataset.to_list()
    else:
        records = list(dataset)

    examples: list[TraceExample] = []
    limit = int(getattr(getattr(cfg, "generation", object()), "num_of_runs", 1))
    for idx, record in enumerate(records):
        if len(examples) >= limit:
            break
        raw = dict(record)
        rewrite = raw.get("requested_rewrite", raw)
        subject = str(rewrite["subject"])
        prompt_template = str(rewrite["prompt"])
        prompt = prompt_template.format(subject)
        target_true = rewrite.get("target_true", {})
        target_new = rewrite.get("target_new", {})
        target = target_true.get("str", target_true) if isinstance(target_true, dict) else target_true
        new_value = target_new.get("str", target_new) if isinstance(target_new, dict) else target_new
        prompt_id = str(raw.get("case_id", raw.get("relation_id", idx)))
        examples.append(
            TraceExample(
                prompt_id=prompt_id,
                prompt=prompt,
                subject=subject,
                target=str(target).strip(),
                target_new=str(new_value).strip() if new_value is not None else None,
                target_true=str(target).strip(),
                raw_case=raw,
            )
        )
    return examples


def _resolve_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    for module_name, module in model.named_modules():
        if module_name == name:
            return module
    raise KeyError(f"Module not found: {name}")


def _module_name(cfg: DictConfig, component: str, layer: int | None = None) -> str:
    model_cfg = cfg.model
    if component == "embedding":
        template = str(model_cfg.corrupt_layer_name_template)
        return template.format(0) if "{}" in template else template
    if component == "residual":
        return str(model_cfg.restore_layer_name_template).format(int(layer))
    if component == "mlp":
        if "transformer.h" in str(model_cfg.restore_layer_name_template):
            return f"transformer.h.{int(layer)}.mlp"
        return f"model.layers.{int(layer)}.mlp"
    if component == "attention":
        if "transformer.h" in str(model_cfg.restore_layer_name_template):
            return f"transformer.h.{int(layer)}.attn"
        return f"model.layers.{int(layer)}.self_attn"
    raise ValueError(f"Unknown component: {component}")


@contextmanager
def temporary_hooks(hooks: Iterable[tuple[torch.nn.Module, Any]]) -> Iterator[None]:
    handles = []
    try:
        for module, hook in hooks:
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _hidden_from_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        values = list(output)
        values[0] = hidden
        return tuple(values)
    return hidden


def _repeat_inputs(inputs: Any, repeats: int) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            result[key] = value.repeat((repeats,) + (1,) * (value.dim() - 1))
    return result


def _embedding_std(handler: ModelHandler) -> float:
    emb = _resolve_module(handler.model, _module_name(handler.cfg, "embedding"))
    weight = getattr(emb, "weight", None)
    if weight is None:
        raise RuntimeError("Embedding module has no weight; cannot compute noise scale")
    std = float(weight.detach().float().std().item())
    if not math.isfinite(std) or std <= 0:
        raise RuntimeError(f"Invalid embedding std: {std}")
    return std


def make_noise_samples(
    *,
    num_samples: int,
    subject_length: int,
    hidden_size: int,
    noise_std: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(
        (int(num_samples), int(subject_length), int(hidden_size)),
        generator=generator,
        dtype=torch.float32,
    )
    return (noise * float(noise_std)).to(device=device, dtype=dtype)


def _corrupt_hook(subject_positions: Sequence[int], noise_samples: torch.Tensor):
    positions = [int(x) for x in subject_positions]

    def hook(_module, _input, output):
        hidden = _hidden_from_output(output)
        changed = hidden.clone()
        noise = noise_samples.to(device=changed.device, dtype=changed.dtype)
        for offset, token_idx in enumerate(positions):
            changed[:, token_idx, :] = changed[:, token_idx, :] + noise[:, offset, :]
        return _replace_hidden(output, changed)

    return hook


def _restore_position_hook(position: int, clean_state: torch.Tensor):
    def hook(_module, _input, output):
        hidden = _hidden_from_output(output)
        changed = hidden.clone()
        state = clean_state.to(device=changed.device, dtype=changed.dtype)
        changed[:, int(position), :] = state
        return _replace_hidden(output, changed)

    return hook


def _token_metadata(tokenizer: Any, input_ids: torch.Tensor, span: TokenSpan) -> list[dict[str, Any]]:
    ids = input_ids[0].detach().cpu().tolist()
    prompt_last = len(ids) - 1
    metadata = []
    for idx, token_id in enumerate(ids):
        metadata.append(
            {
                "position": idx,
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "is_subject": idx in span.positions,
                "is_subject_last": idx == span.last_position,
                "is_prompt_last": idx == prompt_last,
            }
        )
    return metadata


def _probabilities_from_logits(logits: torch.Tensor, target_token_id: int) -> torch.Tensor:
    return torch.softmax(logits[:, -1, :], dim=-1)[:, int(target_token_id)].detach().float().cpu()


def _clean_pass(handler: ModelHandler, inputs: Any, target_token_id: int) -> tuple[Any, float, str, float]:
    handler.model.eval()
    outputs = handler.model(**inputs, output_hidden_states=True, use_cache=False)
    probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
    target_prob = float(probs[0, int(target_token_id)].detach().float().cpu().item())
    top_id = int(torch.argmax(probs[0]).detach().cpu().item())
    top_prob = float(probs[0, top_id].detach().float().cpu().item())
    top_token = handler.tokenizer.decode([top_id]).strip()
    return outputs, target_prob, top_token, top_prob


def _quality_from_scores(
    *,
    clean_probability: float,
    corrupt_probabilities: np.ndarray,
    effects: np.ndarray,
    thresholds: Any,
    allow_noisy: bool = True,
) -> tuple[bool, str | None, str]:
    epsilon = 1e-9
    total_effect = float(clean_probability - corrupt_probabilities.mean())
    if total_effect < float(_get(thresholds, "min_total_effect", 0.05)):
        return False, "insufficient_corruption_effect", "weak"

    max_abs = float(np.max(np.abs(effects))) if effects.size else 0.0
    if max_abs <= epsilon:
        return False, "no_restoration_effect", "weak"

    if effects.ndim == 2:
        by_layer = np.nanmean(effects, axis=0)
    else:
        by_layer = effects
    best = int(np.nanargmax(by_layer))
    if effects.ndim == 2:
        std_by_layer = np.nanstd(effects, axis=0)
    else:
        std_by_layer = np.zeros_like(by_layer)
    rel_std = float(std_by_layer[best] / max(abs(by_layer[best]), epsilon))
    max_rel_std = float(_get(thresholds, "max_peak_relative_std", 1.0))
    if rel_std > max_rel_std:
        return False, "high_variance", "noisy" if allow_noisy else "weak"

    peak_norm = float(by_layer[best] / max(abs(total_effect), epsilon))
    if peak_norm < float(_get(thresholds, "min_peak_normalized_recovery", 0.10)):
        return False, "weak_restoration_effect", "weak"
    return True, None, "clean"


def _layer_scores_from_probs(
    p_restore: np.ndarray,
    p_corrupt: np.ndarray,
    total_effect: float,
) -> list[LayerTraceScore]:
    epsilon = 1e-9
    scores = []
    for layer in range(p_restore.shape[0]):
        restored = p_restore[layer]
        ie = restored - p_corrupt
        mean_ie = float(np.mean(ie))
        scores.append(
            LayerTraceScore(
                layer=layer,
                mean_restored_probability=float(np.mean(restored)),
                std_restored_probability=float(np.std(restored)),
                mean_indirect_effect=mean_ie,
                std_indirect_effect=float(np.std(ie)),
                normalized_recovery=float(mean_ie / max(abs(total_effect), epsilon)),
            )
        )
    return scores


def _candidate_layers(scores: Sequence[LayerTraceScore], tracing_cfg: Any) -> list[int]:
    metric = str(_get(tracing_cfg, "ranking_metric", "normalized_recovery"))
    top_k = int(_get(tracing_cfg, "top_k_layers", 3))
    key = (lambda s: s.mean_indirect_effect) if metric == "mean_indirect_effect" else (lambda s: s.normalized_recovery)
    ranked = sorted(scores, key=key, reverse=True)
    return [int(score.layer) for score in ranked[:top_k]]


def _select_restore_position(input_ids: torch.Tensor, span: TokenSpan, restore_position: str) -> int:
    if restore_position == "subject_last":
        return int(span.last_position)
    if restore_position == "prompt_last":
        return int(input_ids.shape[1] - 1)
    raise ValueError(f"Unsupported restore_position: {restore_position}")


def _run_position_layer_trace(
    handler: ModelHandler,
    inputs: Any,
    span: TokenSpan,
    example: TraceExample,
    *,
    mode: TraceMode,
    positions: Sequence[int],
    restore_position_type: str | None,
    component: TraceComponent,
    window_size: int,
    tracing_cfg: Any,
) -> PromptTraceResult:
    target_token_id = target_first_token_id(handler.tokenizer, example.target)
    require_correct = bool(_get(tracing_cfg, "require_correct_clean_prediction", True))
    num_layers = int(handler.num_of_layers)
    seed = int(_get(tracing_cfg, "seed", getattr(handler.cfg, "seed", 42)))
    num_noise = int(_get(tracing_cfg, "num_noise_samples", 10))
    noise_multiplier = float(_get(tracing_cfg, "noise_multiplier", 3.0))
    epsilon = 1e-9

    with torch.inference_mode():
        clean_outputs, clean_prob, clean_top, clean_top_prob = _clean_pass(handler, inputs, target_token_id)

        if require_correct and clean_top != handler.tokenizer.decode([target_token_id]).strip():
            raise TraceValidationError(
                f"Clean top token {clean_top!r} did not match target token "
                f"{handler.tokenizer.decode([target_token_id]).strip()!r}"
            )

        hidden_size = int(clean_outputs.hidden_states[-1].shape[-1])
        embedding_std = _embedding_std(handler)
        noise_samples = make_noise_samples(
            num_samples=num_noise,
            subject_length=len(span.positions),
            hidden_size=hidden_size,
            noise_std=noise_multiplier * embedding_std,
            device=inputs["input_ids"].device,
            dtype=clean_outputs.hidden_states[-1].dtype,
            seed=seed,
        )

        repeated = _repeat_inputs(inputs, num_noise)
        emb_module = _resolve_module(handler.model, _module_name(handler.cfg, "embedding"))
        with temporary_hooks([(emb_module, _corrupt_hook(span.positions, noise_samples))]):
            corrupt_outputs = handler.model(**repeated, use_cache=False)
        p_corrupt = _probabilities_from_logits(corrupt_outputs.logits, target_token_id).numpy()

        clean_component_cache: dict[tuple[int, int], torch.Tensor] = {}
        if component == "residual":
            for layer in range(num_layers):
                for pos in positions:
                    clean_component_cache[(layer, int(pos))] = clean_outputs.hidden_states[layer + 1][0, int(pos)].detach().clone()
        else:
            cache_hooks = []
            captured: dict[int, torch.Tensor] = {}

            def make_cache_hook(layer_idx: int):
                def hook(_module, _input, output):
                    captured[layer_idx] = _hidden_from_output(output)[0].detach().clone()
                    return output

                return hook

            for layer in range(num_layers):
                module = _resolve_module(handler.model, _module_name(handler.cfg, component, layer))
                cache_hooks.append((module, make_cache_hook(layer)))
            with temporary_hooks(cache_hooks):
                handler.model(**inputs, use_cache=False)
            for layer, hidden in captured.items():
                for pos in positions:
                    clean_component_cache[(layer, int(pos))] = hidden[int(pos)].detach().clone()

        if component == "residual":
            raw = np.zeros((len(positions), num_layers, num_noise), dtype=np.float32)
            for pos_idx, pos in enumerate(positions):
                for layer in range(num_layers):
                    hooks = [
                        (emb_module, _corrupt_hook(span.positions, noise_samples)),
                        (
                            _resolve_module(handler.model, _module_name(handler.cfg, "residual", layer)),
                            _restore_position_hook(int(pos), clean_component_cache[(layer, int(pos))]),
                        ),
                    ]
                    with temporary_hooks(hooks):
                        outputs = handler.model(**repeated, use_cache=False)
                    raw[pos_idx, layer, :] = _probabilities_from_logits(outputs.logits, target_token_id).numpy()
        else:
            raw = np.zeros((len(positions), num_layers, num_noise), dtype=np.float32)
            for pos_idx, pos in enumerate(positions):
                for center in range(num_layers):
                    start = max(0, center - int(window_size) // 2)
                    end = min(num_layers, start + int(window_size))
                    start = max(0, end - int(window_size))
                    hooks = [(emb_module, _corrupt_hook(span.positions, noise_samples))]
                    for layer in range(start, end):
                        module = _resolve_module(handler.model, _module_name(handler.cfg, component, layer))
                        hooks.append((module, _restore_position_hook(int(pos), clean_component_cache[(layer, int(pos))])))
                    with temporary_hooks(hooks):
                        outputs = handler.model(**repeated, use_cache=False)
                    raw[pos_idx, center, :] = _probabilities_from_logits(outputs.logits, target_token_id).numpy()

    mean_corrupt = float(np.mean(p_corrupt))
    std_corrupt = float(np.std(p_corrupt))
    total_effect = float(clean_prob - mean_corrupt)
    effects = raw - p_corrupt.reshape(1, 1, -1)
    mean_effects = np.mean(effects, axis=2)

    thresholds = _get(tracing_cfg, "signal_quality", tracing_cfg)
    reliable, failure, signal_quality = _quality_from_scores(
        clean_probability=clean_prob,
        corrupt_probabilities=p_corrupt,
        effects=mean_effects,
        thresholds=thresholds,
        allow_noisy=True,
    )

    token_scores: dict[int, list[LayerTraceScore]] = {}
    for pos_idx, pos in enumerate(positions):
        token_scores[int(pos)] = _layer_scores_from_probs(raw[pos_idx], p_corrupt, total_effect)

    if len(positions) == 1:
        layer_scores = token_scores[int(positions[0])]
    else:
        averaged_raw = raw.mean(axis=0)
        layer_scores = _layer_scores_from_probs(averaged_raw, p_corrupt, total_effect)

    candidate_layers = _candidate_layers(layer_scores, tracing_cfg) if reliable or mode in {"aquin", "alt"} else []
    best_trace_layer = candidate_layers[0] if candidate_layers else None

    result = PromptTraceResult(
        prompt_id=example.prompt_id,
        mode=mode,
        prompt=example.prompt,
        subject=example.subject,
        target=example.target,
        target_token_id=int(target_token_id),
        target_token=handler.tokenizer.decode([target_token_id]),
        target_token_mode="first_token_only",
        subject_positions=list(span.positions),
        restore_position_type=restore_position_type,
        restore_position=int(positions[0]) if len(positions) == 1 else None,
        position_scope=str(_get(tracing_cfg, "position_scope", None)) if mode == "canonical" else None,
        component=component,
        clean_probability=clean_prob,
        clean_top_token=clean_top,
        clean_top_probability=clean_top_prob,
        corrupt_probabilities=[float(x) for x in p_corrupt.tolist()],
        mean_corrupt_probability=mean_corrupt,
        std_corrupt_probability=std_corrupt,
        total_effect=total_effect,
        trace_reliable=reliable,
        trace_failure_reason=failure,
        signal_quality=signal_quality,
        best_trace_layer=best_trace_layer,
        candidate_layers=candidate_layers,
        layer_scores=layer_scores,
        token_layer_scores=token_scores,
        tokens=_token_metadata(handler.tokenizer, inputs["input_ids"], span),
        p_restore=raw.tolist(),
    )
    if abs(total_effect) <= epsilon:
        result.trace_failure_reason = result.trace_failure_reason or "zero_total_effect"
    return result


def _apply_aquin_fallback(result: PromptTraceResult, num_layers: int, tracing_cfg: Any) -> PromptTraceResult:
    fallback_cfg = _get(tracing_cfg, "middle_third_fallback", OmegaConf.create({}))
    if result.trace_reliable or not bool(_get(fallback_cfg, "enabled", False)):
        return result

    count = int(_get(fallback_cfg, "candidate_count", _get(tracing_cfg, "top_k_layers", 3)))
    start = num_layers // 3
    end = 2 * num_layers // 3
    middle_scores = [score for score in result.layer_scores if start <= score.layer < end]
    ranked = sorted(middle_scores, key=lambda score: score.normalized_recovery, reverse=True)
    result.candidate_layers = [int(score.layer) for score in ranked[:count]]
    result.best_trace_layer = result.candidate_layers[0] if result.candidate_layers else None
    result.fallback_used = True
    result.fallback_reason = (
        "The causal signal was unreliable. Candidate layers were selected from the middle region of the network "
        "using the configured fallback heuristic."
    )
    return result


def _rome_case(example: TraceExample) -> dict[str, Any] | None:
    if not example.target_new or not example.target_true:
        return None
    raw = dict(example.raw_case)
    paraphrase_prompts = list(raw.get("paraphrase_prompts", []))
    neighborhood_prompts = list(raw.get("neighborhood_prompts", []))
    prompt_template = raw.get("requested_rewrite", raw).get("prompt", "{}")
    return {
        "case_id": example.prompt_id,
        "subject": example.subject,
        "target_new_str": example.target_new,
        "target_true_str": example.target_true,
        "fact_tuple": (
            str(prompt_template),
            example.subject,
            example.target_new if example.target_new.startswith(" ") else f" {example.target_new}",
            example.target_true if example.target_true.startswith(" ") else f" {example.target_true}",
        ),
        "paraphrase_prompts": paraphrase_prompts,
        "neighborhood_prompts": neighborhood_prompts,
    }


def _validate_rome_layers(
    handler: ModelHandler,
    example: TraceExample,
    candidate_layers: Sequence[int],
    tracing_cfg: Any,
) -> tuple[list[dict[str, Any]], int | None]:
    if not bool(_get(tracing_cfg, "validate_with_rome", False)) or not candidate_layers:
        return [], None

    from src.editing.rome import RomeEditMethod

    case = _rome_case(example)
    if case is None:
        return [], None

    weights = _get(tracing_cfg, "rome_validation", OmegaConf.create({}))
    rewrite_weight = float(_get(weights, "rewrite_weight", 0.4))
    paraphrase_weight = float(_get(weights, "paraphrase_weight", 0.3))
    locality_weight = float(_get(weights, "locality_weight", 0.3))
    original_layer = int(handler._layer)
    method = RomeEditMethod()
    rows: list[dict[str, Any]] = []

    for layer in candidate_layers:
        handler._layer = int(layer)
        outcome = None
        try:
            outcome = method.apply(handler, case)
            metrics = method.evaluate(handler, case, outcome)
            rewrite_success = float(metrics.get("efficacy_score") or 0.0)
            paraphrase_success = metrics.get("paraphrase_score")
            locality_score = metrics.get("neighborhood_score")
            paraphrase_value = float(paraphrase_success) if paraphrase_success is not None else 0.0
            locality_value = float(locality_score) if locality_score is not None else 0.0
            validation_score = (
                rewrite_weight * rewrite_success
                + paraphrase_weight * paraphrase_value
                + locality_weight * locality_value
            )
            rows.append(
                {
                    "layer": int(layer),
                    "rewrite_success": rewrite_success,
                    "paraphrase_success": paraphrase_success,
                    "locality_score": locality_score,
                    "validation_score": float(validation_score),
                    "metrics": metrics,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"layer": int(layer), "error": str(exc), "validation_score": float("-inf")})
        finally:
            if outcome is not None:
                for module_name, old_weight in outcome.restorations.items():
                    handler._get_module(module_name).weight = torch.nn.Parameter(old_weight)
            handler.remove_hooks()
            handler._layer = original_layer

    valid_rows = [row for row in rows if math.isfinite(float(row.get("validation_score", float("-inf"))))]
    best = max(valid_rows, key=lambda row: float(row["validation_score"]))["layer"] if valid_rows else None
    return rows, int(best) if best is not None else None


def trace_one_example(
    handler: ModelHandler,
    example: TraceExample,
    *,
    mode: TraceMode,
    tracing_cfg: Any,
) -> PromptTraceResult:
    inputs = handler.tokenize_prompt(example.prompt)
    span = find_subject_span(handler.tokenizer, example.prompt, example.subject)

    component = str(_get(tracing_cfg, "component", "residual"))
    if component not in {"residual", "mlp", "attention"}:
        raise ValueError(f"Unsupported trace component: {component}")
    window_size = int(_get(tracing_cfg, "window_size", 1 if component == "residual" else 10))

    if mode == "standard":
        positions = span.positions
        restore_position_type = "subject_tokens"
    elif mode == "canonical":
        scope = str(_get(tracing_cfg, "position_scope", "all_tokens"))
        if scope == "all_tokens":
            positions = list(range(int(inputs.input_ids.shape[1])))
        elif scope == "subject_only":
            positions = span.positions
        elif scope == "subject_last":
            positions = [span.last_position]
        elif scope == "prompt_last":
            positions = [int(inputs.input_ids.shape[1] - 1)]
        else:
            raise ValueError(f"Unsupported position_scope: {scope}")
        restore_position_type = scope
    else:
        restore_type = str(_get(tracing_cfg, "restore_position", "prompt_last" if mode in {"aquin", "fast"} else "subject_last"))
        positions = [_select_restore_position(inputs.input_ids, span, restore_type)]
        restore_position_type = restore_type

    result = _run_position_layer_trace(
        handler,
        inputs,
        span,
        example,
        mode=mode,
        positions=positions,
        restore_position_type=restore_position_type,
        component=component,  # type: ignore[arg-type]
        window_size=window_size,
        tracing_cfg=tracing_cfg,
    )

    if mode in {"aquin", "alt"}:
        result = _apply_aquin_fallback(result, int(handler.num_of_layers), tracing_cfg)
    if mode == "aquin":
        validation, best = _validate_rome_layers(handler, example, result.candidate_layers, tracing_cfg)
        result.rome_validation = validation
        result.best_validated_rome_edit_layer = best
    return result


def _run_dir(mode: str, cfg: DictConfig) -> Path:
    model_key = str(getattr(cfg.model, "name", "model")).replace("/", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(str(_get(_cfg_section(cfg, "tracing"), "output_dir", "analysis_out/causal_trace")))
    path = out_root / f"{mode}_{model_key}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_heatmaps(out_dir: Path, results: Sequence[PromptTraceResult]) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Skipping heatmaps because plotting imports failed: %s", exc)
        return

    heatmap_dir = out_dir / "canonical_trace_heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    for result in results:
        if not result.token_layer_scores:
            continue
        positions = sorted(result.token_layer_scores)
        matrix = np.array(
            [[score.mean_indirect_effect for score in result.token_layer_scores[pos]] for pos in positions],
            dtype=np.float32,
        )
        labels = []
        token_by_pos = {int(t["position"]): str(t["token"]) for t in result.tokens}
        for pos in positions:
            labels.append(f"{pos}: {token_by_pos.get(pos, '')}")
        fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1] * 0.35), max(3, matrix.shape[0] * 0.35)))
        sns.heatmap(matrix, ax=ax, cmap="viridis", yticklabels=labels)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Token")
        ax.set_title(f"{result.mode} IE: {result.prompt_id}")
        fig.tight_layout()
        fig.savefig(heatmap_dir / f"{result.prompt_id}.png", dpi=160)
        plt.close(fig)


def _write_outputs(out_dir: Path, mode: str, cfg: DictConfig, results: Sequence[PromptTraceResult]) -> None:
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    (out_dir / "config.json").write_text(json.dumps(to_serializable(config_payload), indent=2), encoding="utf-8")

    with (out_dir / "prompt_results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(to_serializable(asdict(result)), ensure_ascii=True) + "\n")

    max_layer = 0
    for result in results:
        max_layer = max(max_layer, len(result.layer_scores))
    header = [
        "prompt_id",
        "mode",
        "subject",
        "target",
        "clean_probability",
        "mean_corrupt_probability",
        "total_effect",
        "trace_reliable",
        "trace_failure_reason",
        "candidate_layers",
    ] + [f"layer_{idx}" for idx in range(max_layer)]
    with (out_dir / "wide_layer_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for result in results:
            values = [score.mean_indirect_effect for score in result.layer_scores]
            writer.writerow(
                [
                    result.prompt_id,
                    result.mode,
                    result.subject,
                    result.target,
                    result.clean_probability,
                    result.mean_corrupt_probability,
                    result.total_effect,
                    result.trace_reliable,
                    result.trace_failure_reason,
                    " ".join(str(x) for x in result.candidate_layers),
                    *values,
                ]
            )

    if mode == "canonical":
        with (out_dir / "canonical_trace_long.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "prompt_id",
                    "position",
                    "layer",
                    "mean_indirect_effect",
                    "std_indirect_effect",
                    "normalized_recovery",
                ]
            )
            for result in results:
                for position, scores in result.token_layer_scores.items():
                    for score in scores:
                        writer.writerow(
                            [
                                result.prompt_id,
                                position,
                                score.layer,
                                score.mean_indirect_effect,
                                score.std_indirect_effect,
                                score.normalized_recovery,
                            ]
                        )
        raw_payload = {}
        for result in results:
            raw_payload[f"{result.prompt_id}_p_corrupt"] = np.array(result.corrupt_probabilities, dtype=np.float32)
            raw_payload[f"{result.prompt_id}_p_restore"] = np.array(result.p_restore or [], dtype=np.float32)
        if raw_payload:
            np.savez_compressed(out_dir / "canonical_trace_raw.npz", **raw_payload)
        summary = {
            "num_results": len(results),
            "reliable_results": sum(1 for result in results if result.trace_reliable),
            "peak_hidden_state_layer": _aggregate_peak_layer(results),
        }
        (out_dir / "canonical_trace_summary.json").write_text(
            json.dumps(to_serializable(summary), indent=2),
            encoding="utf-8",
        )
        if bool(_get(_cfg_section(cfg, "tracing"), "generate_heatmaps", True)):
            _write_heatmaps(out_dir, results)


def _aggregate_peak_layer(results: Sequence[PromptTraceResult]) -> int | None:
    layer_values: dict[int, list[float]] = {}
    for result in results:
        if not result.trace_reliable:
            continue
        for score in result.layer_scores:
            layer_values.setdefault(score.layer, []).append(score.mean_indirect_effect)
    if not layer_values:
        return None
    return max(layer_values, key=lambda layer: float(np.mean(layer_values[layer])))


def run_trace_mode(cfg: DictConfig, mode: TraceMode) -> Path | None:
    tracing_cfg = _cfg_section(cfg, "tracing")
    if "mode" not in tracing_cfg:
        tracing_cfg.mode = mode
    handler = ModelHandler(cfg)
    examples = _trace_examples(cfg)
    out_dir = _run_dir(mode, cfg)
    results: list[PromptTraceResult] = []
    skipped: list[dict[str, str]] = []

    for example in tqdm(examples, desc=f"{mode} trace"):
        try:
            results.append(trace_one_example(handler, example, mode=mode, tracing_cfg=tracing_cfg))
        except TraceValidationError as exc:
            LOGGER.warning("Skipping %s: %s", example.prompt_id, exc)
            skipped.append({"prompt_id": example.prompt_id, "reason": str(exc)})
        finally:
            handler.remove_hooks()

    _write_outputs(out_dir, mode, cfg, results)
    if skipped:
        (out_dir / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} {mode} trace result(s) to {out_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} prompt(s); see {out_dir / 'skipped.json'}")
    return out_dir


def run_standard_trace(cfg: DictConfig) -> Path | None:
    return run_trace_mode(cfg, "standard")


def run_alt_trace(cfg: DictConfig) -> Path | None:
    return run_trace_mode(cfg, "alt")


def run_aquin_trace(cfg: DictConfig) -> Path | None:
    return run_trace_mode(cfg, "aquin")


def run_canonical_trace(cfg: DictConfig) -> Path | None:
    return run_trace_mode(cfg, "canonical")


def run_fast_trace(cfg: DictConfig) -> Path | None:
    return run_trace_mode(cfg, "fast")


def run_rome_layer_sweep(cfg: DictConfig) -> Path | None:
    tracing_cfg = _cfg_section(cfg, "tracing")
    tracing_cfg.validate_with_rome = True
    tracing_cfg.top_k_layers = 10_000
    tracing_cfg.restore_position = str(_get(tracing_cfg, "restore_position", "prompt_last"))
    return run_trace_mode(cfg, "aquin")
