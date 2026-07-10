"""Subject token positioning helpers for ROME optimization."""

from __future__ import annotations

import logging

import torch

LOGGER = logging.getLogger(__name__)


def _strip_bos(handler, token_ids: torch.Tensor) -> torch.Tensor:
    """Strip leading BOS token if present. Handles tokenizers like Mistral/LLaMA that prepend BOS."""
    bos_id = getattr(handler.tokenizer, 'bos_token_id', None)
    if bos_id is not None and token_ids.dim() >= 1:
        if token_ids.dim() == 2 and token_ids.size(1) > 1 and token_ids[0, 0].item() == bos_id:
            return token_ids[:, 1:]
        elif token_ids.dim() == 1 and token_ids.size(0) > 1 and token_ids[0].item() == bos_id:
            return token_ids[1:]
    return token_ids


def get_subject_position(handler, prompt, subject):
    """
    Get position of subject in prompt.
    Handles tokenizers that prepend BOS (e.g., Mistral/LLaMA).
    """
    input_ids_prompt = handler.tokenize_prompt(prompt)["input_ids"]
    input_ids_subject = _strip_bos(handler, handler.tokenize_prompt(subject)["input_ids"])
    windows = input_ids_prompt.unfold(1, input_ids_subject.size(1), 1)
    matches = (windows == input_ids_subject).all(dim=2)
    subject_position = list(set(matches.nonzero(as_tuple=True)[1].tolist()))

    if len(subject_position) == 0:
        # The tokenizer most likely learned specific tokens with space as prefix (" Rome" instead of " " + "Rome")
        input_ids_subject = _strip_bos(handler, handler.tokenize_prompt(f" {subject}")["input_ids"])
        windows = input_ids_prompt.unfold(1, input_ids_subject.size(1), 1)
        matches = (windows == input_ids_subject).all(dim=2)
        subject_position = list(set(matches.nonzero(as_tuple=True)[1].tolist()))

    if len(subject_position) == 0:
        # Fallback: find the subject character span and map it to token offsets.
        # This handles cases where punctuation (e.g. commas) is merged into token pieces.
        char_start = prompt.find(subject)
        if char_start == -1:
            char_start = prompt.find(f" {subject}")
            if char_start != -1:
                char_start += 1

        if char_start != -1:
            char_end = char_start + len(subject)
            try:
                raw_tok = handler.tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
                offsets = raw_tok.get("offset_mapping")
                if offsets is not None:
                    token_positions = []
                    for idx, (start, end) in enumerate(offsets[0].tolist()):
                        if end <= start:
                            continue
                        if end > char_start and start < char_end:
                            token_positions.append(idx)

                    if token_positions:
                        return int(token_positions[-1])
            except Exception:
                pass

    if len(subject_position) == 0:
        LOGGER.error(f"{subject_position}\t{prompt}\t{input_ids_subject}\t{input_ids_prompt}")
        return -1
        # raise Exception("Subject not found during the prompt preprocess. Mostly due to tokenization issues.")

    subject_position[0] += input_ids_subject.size(1) - 1
    return subject_position[0]


def get_subject_index(handler, prompts, fact_tuple, subject_understanding_template) -> torch.Tensor | None:
    new_target_ids = _strip_bos(handler, handler.tokenize_prompt(fact_tuple[2])["input_ids"][0])
    batch_idx = torch.arange(prompts.input_ids.shape[0], device=prompts.attention_mask.device)
    last_subject_index = prompts.attention_mask[batch_idx].sum(dim=1)

    fact_prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
    u_fact_prompt = handler.tokenize_prompt(subject_understanding_template.format(fact_tuple[1]))

    # Last subject token index computation
    pos = get_subject_position(handler, fact_tuple[0].format(fact_tuple[1]), fact_tuple[1])
    if pos == -1:
        return None

    subject_reverse_pos = len(fact_prompt["input_ids"][0]) - pos
    last_subject_index[: prompts.input_ids.shape[0] - 1] -= subject_reverse_pos + len(new_target_ids) - 1

    # Last subject token index computation for understanding prompt
    pos = get_subject_position(handler, subject_understanding_template.format(fact_tuple[1]), fact_tuple[1])
    if pos == -1:
        return None

    u_sub_reverse_pos = len(u_fact_prompt["input_ids"][0]) - pos
    last_subject_index[-1] -= u_sub_reverse_pos

    return last_subject_index.long().cpu()
