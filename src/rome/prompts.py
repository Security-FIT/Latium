"""Shared no-thinking chat serialization and exact teacher-forcing positions."""

import re
from dataclasses import dataclass

import torch

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer briefly in the same language as the user."


def is_chat_mode(handler):
    model = getattr(getattr(handler, "cfg", None), "model", None)
    mode = getattr(model, "rome_prompt_mode", "completion")
    if mode not in {"completion", "chat"}:
        raise ValueError(f"Unsupported model.rome_prompt_mode: {mode}")
    return mode == "chat"


def render_chat(tokenizer, messages):
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prefill = rendered.rsplit("<|im_start|>assistant", 1)[-1]
    template = getattr(tokenizer, "chat_template", "") or ""
    if "<|im_start|>assistant" in rendered and "<think>" in str(template):
        if not re.fullmatch(r"\s*<think>\s*</think>\s*", prefill, re.IGNORECASE):
            raise ValueError("Chat template did not produce an empty no-thinking assistant prefill")
    return rendered


def render_prompt(handler, prompt):
    if not is_chat_mode(handler):
        return prompt
    system = getattr(handler.cfg.model, "rome_system_prompt", DEFAULT_SYSTEM_PROMPT)
    return render_chat(handler.tokenizer, [
        {"role": "system", "content": system}, {"role": "user", "content": prompt},
    ])


def tokenize_prompts(handler, prompts, *, return_offsets_mapping=False):
    if not is_chat_mode(handler):
        return handler.tokenize_prompt(prompts)
    texts = [prompts] if isinstance(prompts, str) else prompts
    encoded = handler.tokenizer(
        [render_prompt(handler, text) for text in texts], add_special_tokens=False,
        padding=True, return_tensors="pt", return_offsets_mapping=return_offsets_mapping,
    )
    # Some tokenizers emit token_type_ids, which decoder-only models reject.
    encoded.pop("token_type_ids", None)
    return encoded


def _user_span(handler, user):
    marker = "WORKSHOP_USER_SLOT_b27f"
    marked = render_prompt(handler, marker)
    if marked.count(marker) != 1:
        raise ValueError("Cannot locate chat user slot")
    before, after = marked.split(marker)
    if render_prompt(handler, user) != before + user + after:
        raise ValueError("Chat template transforms user text; exact offsets unavailable")
    return len(before), len(before) + len(user)


@dataclass
class ChatBatch:
    prompts: object
    subject_indices: torch.Tensor | None
    target_ids: torch.Tensor
    prediction_positions: torch.Tensor
    main_prompt_count: int


def prepare_chat_batch(handler, user_prompts, *, subject=None, target=None, kl_prompt=None):
    if not user_prompts:
        raise ValueError("At least one prompt is required")
    users = list(user_prompts) + ([] if kl_prompt is None else [kl_prompt])
    encoded = tokenize_prompts(handler, users, return_offsets_mapping=subject is not None)
    target_ids = torch.tensor(
        [] if target is None else handler.tokenizer.encode(target.strip(), add_special_tokens=False), dtype=torch.long,
    )
    if target is not None and not target_ids.numel():
        raise ValueError("Target tokenizes to nothing")
    rows, lengths, indices = [], [], []
    for i, user in enumerate(users):
        active = encoded.attention_mask[i].bool()
        ids = encoded.input_ids[i][active].tolist()
        lengths.append(len(ids))
        if subject is not None:
            if not subject or subject not in user:
                raise ValueError("Subject is missing from user prompt")
            start = _user_span(handler, user)[0] + user.rindex(subject)
            end = start + len(subject)
            if "offset_mapping" not in encoded:
                raise ValueError("ROME requires a tokenizer with exact offset mappings")
            matches = [j for j, (a, b) in enumerate(encoded.offset_mapping[i][active].tolist()) if a < end and b > start]
            if not matches:
                raise ValueError("No token overlaps the subject")
            indices.append(matches[-1])
        if i < len(user_prompts) and target is not None:
            ids += target_ids[:-1].tolist()
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    padded = handler.tokenizer.pad(rows, padding=True, return_tensors="pt")
    shifts = padded.attention_mask.long().argmax(dim=1).cpu()
    positions = (shifts[:len(user_prompts), None] + torch.tensor(lengths[:len(user_prompts)])[:, None] - 1
                 + torch.arange(target_ids.numel())[None, :])
    return ChatBatch(padded, None if subject is None else torch.tensor(indices) + shifts,
                     target_ids, positions.long(), len(user_prompts))
