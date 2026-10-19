"""One prompt/token convention for chat ROME edits, generation, and scoring.

Chat targets are stripped assistant answers, encoded separately from the rendered
generation prompt. Editing requires a tokenizer with exact character offsets;
we fail rather than guess a subject position from an unrelated tokenization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from transformers import BatchEncoding

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer briefly in the same language as the user."


def is_chat_mode(handler) -> bool:
    model = getattr(getattr(handler, "cfg", None), "model", None)
    mode = getattr(model, "rome_prompt_mode", "completion")
    if mode not in ("completion", "chat"):
        raise ValueError(f"Unknown model.rome_prompt_mode: {mode!r}")
    return mode == "chat"


def render_chat(tokenizer, messages) -> str:
    """Render a generation prompt with thinking explicitly disabled.

    Never retry without ``enable_thinking=False``. Qwen reasoning templates
    must finish with their empty think prefill, rather than an open think span.
    Other templates may simply omit thinking markup altogether.
    """
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise ValueError("Chat template must render text")
    # Inspect only the assistant prefill, not quoted reasoning in user content.
    prefill = rendered.rsplit("<|im_start|>assistant", 1)[-1]
    template = getattr(tokenizer, "chat_template", "") or ""
    qwen_thinking = "<|im_start|>assistant" in rendered and "<think>" in str(template)
    if qwen_thinking or "<think>" in prefill.lower() or "</think>" in prefill.lower():
        if not re.fullmatch(r"\s*<think>\s*</think>\s*", prefill, re.IGNORECASE):
            raise ValueError("Chat template did not produce an empty no-thinking assistant prefill")
    return rendered


def _messages(handler, prompt: str) -> list[dict[str, str]]:
    system = getattr(handler.cfg.model, "rome_system_prompt", DEFAULT_SYSTEM_PROMPT)
    if not isinstance(system, str) or not system.strip():
        raise ValueError("model.rome_system_prompt must be a nonempty neutral assistant prompt")
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def render_prompt(handler, prompt: str) -> str:
    """Render raw user text according to the handler's opt-in prompt mode."""
    return render_chat(handler.tokenizer, _messages(handler, prompt)) if is_chat_mode(handler) else prompt


def tokenize_prompts(handler, prompts: str | list[str], *, return_offsets_mapping: bool = False):
    """Tokenize rendered prompts once; chat templates already contain special tokens."""
    if not is_chat_mode(handler):
        return handler.tokenize_prompt(prompts)
    texts = [prompts] if isinstance(prompts, str) else prompts
    return handler.tokenizer(
        [render_prompt(handler, text) for text in texts],
        add_special_tokens=False, padding=True, return_tensors="pt",
        return_offsets_mapping=return_offsets_mapping,
    )


def _user_span(handler, user: str) -> tuple[int, int]:
    # A marker identifies the user slot even when the system or assistant text
    # also mentions the subject. Verify exact substitution before using offsets.
    marker = "LATIUM_USER_CONTENT_7f4a9c"
    marked = render_prompt(handler, marker)
    if marked.count(marker) != 1:
        raise ValueError("Cannot identify the chat template's user content span")
    before, after = marked.split(marker)
    if render_prompt(handler, user) != before + user + after:
        raise ValueError("Chat template transforms user content; exact subject offsets unavailable")
    return len(before), len(before) + len(user)


@dataclass
class ChatBatch:
    prompts: BatchEncoding
    subject_indices: torch.Tensor | None
    target_ids: torch.Tensor
    prediction_positions: torch.Tensor
    main_prompt_count: int


def prepare_chat_batch(
    handler, user_prompts: list[str], *, subject: str | None = None,
    target: str | None = None, kl_prompt: str | None = None,
) -> ChatBatch:
    """Prepare padded keys or teacher-forced values with exact row positions.

    Main rows contain ``prompt_ids + target_ids[:-1]``; KL is an independent
    user prompt with no target tokens. Prediction positions index the logits
    *before* each target token. All returned indices include left/right padding.
    """
    if not is_chat_mode(handler):
        raise ValueError("prepare_chat_batch requires model.rome_prompt_mode='chat'")
    if not user_prompts:
        raise ValueError("Chat preparation requires at least one main prompt")
    users = list(user_prompts) + ([] if kl_prompt is None else [kl_prompt])
    try:
        encoded = tokenize_prompts(handler, users, return_offsets_mapping=subject is not None)
    except (NotImplementedError, TypeError) as exc:
        raise ValueError("Chat editing requires a tokenizer with exact offset mappings") from exc
    target_ids = torch.tensor(
        [] if target is None else handler.tokenizer.encode(target.strip(), add_special_tokens=False),
        dtype=torch.long,
    )
    if target is not None and not target_ids.numel():
        raise ValueError("Chat target tokenizes to nothing")
    rows, lengths, subject_indices = [], [], []
    for i, user in enumerate(users):
        active = encoded["attention_mask"][i].bool()
        ids = encoded["input_ids"][i][active].tolist()
        lengths.append(len(ids))
        if subject is not None:
            if not subject or subject not in user:
                raise ValueError(f"Subject {subject!r} is missing from user prompt")
            start, _ = _user_span(handler, user)
            start += user.rindex(subject)
            end = start + len(subject)
            offsets = encoded.get("offset_mapping")
            if offsets is None:
                raise ValueError("Chat editing requires exact offset mappings")
            matches = [
                j for j, (a, b) in enumerate(offsets[i][active].tolist())
                if a < end and b > start
            ]
            if not matches:
                raise ValueError("No tokens overlap the subject in rendered user content")
            subject_indices.append(matches[-1])
        if i < len(user_prompts) and target is not None:
            ids += target_ids[:-1].tolist()
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    padded = handler.tokenizer.pad(rows, padding=True, return_tensors="pt")
    # Derive shifts from the actual padded mask, never from a padding-side guess.
    shifts = padded["attention_mask"].long().argmax(dim=1).cpu()
    positions = (
        shifts[:len(user_prompts), None]
        + torch.tensor(lengths[:len(user_prompts)])[:, None] - 1
        + torch.arange(target_ids.numel())[None, :]
    )
    return ChatBatch(
        BatchEncoding(padded),
        None if subject is None else torch.tensor(subject_indices) + shifts,
        target_ids, positions.long(), len(user_prompts),
    )
