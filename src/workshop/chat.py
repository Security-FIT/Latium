"""Deterministic, no-thinking chat shared by both presentations."""

from .model import flush_gpu_memory
from .probing import clean_text
from .prompts import DEFAULT_SYSTEM_PROMPT, render_chat
from src.rome.probing import _stop_token_ids


def default_system_prompt(handler):

    model = getattr(getattr(handler, "cfg", None), "model", None)
    if handler is None:
        from .model import ModelSettings
        return ModelSettings().rome_system_prompt
    return getattr(model, "rome_system_prompt", DEFAULT_SYSTEM_PROMPT)


def chat_messages(message, history, system_prompt, *, history_limit=10):
    messages = []
    if (system_prompt or "").strip():
        # Preserve the configured edit context exactly, including whitespace.
        messages.append({"role": "system", "content": system_prompt})
    for item in (history or [])[-history_limit:]:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": message})
    return messages


def strip_thinking(answer):

    answer = clean_text(answer)
    return "" if answer.lower().startswith(("thinking process:", "reasoning:")) else answer


def generate_reply(session, message, history, system_prompt=None,
                   *, max_new_tokens=32, history_limit=8, repetition_penalty=None):
    import torch

    handler = session.handler
    if system_prompt is None:
        system_prompt = default_system_prompt(handler)
    messages = chat_messages(message, history, system_prompt, history_limit=history_limit)
    with session.lock:
        rendered = render_chat(handler.tokenizer, messages)
        tokenized = handler.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        inputs = {key: value.to(handler.device) for key, value in tokenized.items()}
        options = dict(inputs, max_new_tokens=int(max_new_tokens), do_sample=False, use_cache=True)
        if repetition_penalty is not None:
            options["repetition_penalty"] = repetition_penalty
        pad_id = getattr(handler.tokenizer, "pad_token_id", None)
        eos_id = getattr(handler.tokenizer, "eos_token_id", None)
        if pad_id is not None or eos_id is not None:
            options["pad_token_id"] = int(pad_id if pad_id is not None else eos_id)
        if eos_id is not None:
            options["eos_token_id"] = int(eos_id)
        stop_ids = _stop_token_ids(handler)
        if stop_ids:
            options["eos_token_id"] = stop_ids[0] if len(stop_ids) == 1 else stop_ids
        generated = None
        try:
            with torch.inference_mode():
                generated = handler.model.generate(**options)
            answer = handler.tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False,
            )
        finally:
            del generated, tokenized, inputs, options
            flush_gpu_memory()
    answer = strip_thinking(answer)
    for token in getattr(handler.tokenizer, "all_special_tokens", []):
        answer = answer.replace(token, "")
    return answer.strip()
