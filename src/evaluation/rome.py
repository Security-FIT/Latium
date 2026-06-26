"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np
import torch


def _get_target_token_ids(tokenizer: Any, text: str) -> list[int]:
    token_ids = tokenizer(f" {text}", add_special_tokens=False)["input_ids"]
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if token_ids:
        return list(token_ids)

    token_ids = tokenizer(f" {text}")["input_ids"]
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None and len(token_ids) > 1 and token_ids[0] == bos_id:
        token_ids = token_ids[1:]
    return list(token_ids)


def test_batch_prediction(
    model: Any,
    tokenizer: Any,
    prefixes: Sequence[str],
    target_new: str,
    target_true: str,
    device: Any,
    batch_size: int = 8,
) -> list[dict[str, float]]:
    new_token_ids = _get_target_token_ids(tokenizer, target_new)
    true_token_ids = _get_target_token_ids(tokenizer, target_true)
    choice_lengths = (len(new_token_ids), len(true_token_ids))
    choices = (new_token_ids, true_token_ids)
    results: list[dict[str, float]] = []

    for chunk_start in range(0, len(prefixes), batch_size):
        chunk = list(prefixes[chunk_start : chunk_start + batch_size])
        prefix_lengths = [len(tokens) for tokens in tokenizer(chunk)["input_ids"]]
        prompt_tokens = tokenizer(
            [f"{prefix} {suffix}" for prefix in chunk for suffix in (target_new, target_true)],
            padding=True,
            return_tensors="pt",
        ).to(device)
        pad_offsets = (prompt_tokens["attention_mask"].cumsum(dim=1) == 0).sum(dim=1).tolist()

        with torch.no_grad():
            logits = model(**prompt_tokens, use_cache=False).logits

        chunk_results = np.zeros((logits.size(0),), dtype=np.float32)
        for row in range(logits.size(0)):
            choice_index = row % 2
            token_ids = choices[choice_index]
            token_count = choice_lengths[choice_index]
            offset = pad_offsets[row]
            for token_index, token_id in enumerate(token_ids):
                position = offset + prefix_lengths[row // 2] + token_index - 1
                chunk_results[row] += -torch.nn.functional.log_softmax(
                    logits[row, position, :],
                    dim=0,
                )[token_id].item()
            chunk_results[row] /= token_count

        for row in range(0, len(chunk_results), 2):
            results.append(
                {
                    "target_new": chunk_results[row].item(),
                    "target_true": chunk_results[row + 1].item(),
                }
            )

        del prompt_tokens, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def compute_rome_metrics(
    handler: Any,
    prompt_text: str,
    target_new_str: str,
    target_true_str: str,
    paraphrase_prompts: Optional[Sequence[str]] = None,
    neighborhood_prompts: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    paraphrases = list(paraphrase_prompts or [])
    neighborhoods = list(neighborhood_prompts or [])
    prompts = [prompt_text, *paraphrases, *neighborhoods]
    probabilities = test_batch_prediction(
        handler.model,
        handler.tokenizer,
        prompts,
        target_new_str,
        target_true_str,
        handler.device,
    )

    rewrite = probabilities[0]
    paraphrase_values = probabilities[1 : 1 + len(paraphrases)]
    neighborhood_values = probabilities[1 + len(paraphrases) :]
    efficacy = 1.0 if rewrite["target_new"] < rewrite["target_true"] else 0.0
    magnitude = math.exp(-rewrite["target_new"]) - math.exp(-rewrite["target_true"])
    paraphrase_score = (
        sum(value["target_new"] < value["target_true"] for value in paraphrase_values) / len(paraphrase_values)
        if paraphrase_values
        else None
    )
    neighborhood_score = (
        sum(value["target_true"] < value["target_new"] for value in neighborhood_values) / len(neighborhood_values)
        if neighborhood_values
        else None
    )
    components = [value for value in (efficacy, paraphrase_score, neighborhood_score) if value is not None]
    overall = (
        len(components) / sum(1.0 / value for value in components)
        if components and all(value > 0 for value in components)
        else 0.0
    )

    return {
        "efficacy_score": efficacy,
        "efficacy_magnitude": magnitude,
        "paraphrase_score": paraphrase_score,
        "neighborhood_score": neighborhood_score,
        "overall_score": overall,
        "rewrite_nll": rewrite,
        "paraphrase_nll": paraphrase_values,
        "neighborhood_nll": neighborhood_values,
    }
