"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.handlers.rome import ModelHandler
from src.rome.rome import single_intervention

LOGGER = logging.getLogger(__name__)


def _to_plain(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _with_leading_space(value: Any) -> str:
    text = str(value or "")
    if not text:
        return text
    return text if text.startswith(" ") else f" {text}"


def _read_json_source(source: str | Path) -> Any:
    source_text = str(source)
    path = Path(source_text)
    try:
        path_exists = path.exists()
    except OSError:
        path_exists = False
    if path_exists:
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(source_text)


def _select_counterfact_record(payload: Any, index: int = 0, case_id: int | None = None) -> Mapping[str, Any]:
    payload = _to_plain(payload)
    if isinstance(payload, Mapping) and "requested_rewrite" in payload:
        return payload
    if isinstance(payload, Mapping) and "data" in payload:
        payload = payload["data"]
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError("CounterFact payload must be a record, list of records, or {data: [...]}.")

    records = list(payload)
    if case_id is not None:
        for record in records:
            if isinstance(record, Mapping) and int(record.get("case_id", -1)) == int(case_id):
                return record
        raise ValueError(f"CounterFact case_id={case_id} was not found.")

    try:
        return records[int(index)]
    except IndexError as exc:
        raise ValueError(f"CounterFact index={index} is out of range for {len(records)} records.") from exc


def fact_tuple_from_manual_config(cfg: DictConfig) -> tuple[str, str, str, str]:
    manual = _to_plain(getattr(cfg, "manual", None) or {})

    fact_tuple = getattr(cfg, "fact_tuple", None)
    if fact_tuple is not None:
        values = list(_to_plain(fact_tuple))
        if len(values) != 4:
            raise ValueError("fact_tuple must contain exactly: prompt, subject, target_new, target_true.")
        return (
            str(values[0]),
            str(values[1]),
            _with_leading_space(values[2]),
            _with_leading_space(values[3]),
        )

    record_source = manual.get("counterfact_path") or manual.get("counterfact_json")
    if record_source:
        record = _select_counterfact_record(
            _read_json_source(record_source),
            index=int(manual.get("index", 0)),
            case_id=manual.get("case_id"),
        )
        rewrite = record["requested_rewrite"]
        return (
            str(rewrite["prompt"]),
            str(rewrite["subject"]),
            _with_leading_space(rewrite["target_new"]["str"]),
            _with_leading_space(rewrite["target_true"]["str"]),
        )

    prompt = manual.get("prompt")
    subject = manual.get("subject")
    target_new = manual.get("target_new")
    target_true = manual.get("target_true", "")

    missing = [
        name
        for name, value in (
            ("manual.prompt", prompt),
            ("manual.subject", subject),
            ("manual.target_new", target_new),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Missing manual ROME fields: "
            + ", ".join(missing)
            + ". Provide them with Hydra overrides or use manual.counterfact_path/manual.counterfact_json."
        )

    return (
        str(prompt),
        str(subject),
        _with_leading_space(target_new),
        _with_leading_space(target_true),
    )


def _generation_kwargs(cfg: DictConfig) -> dict[str, Any]:
    manual = getattr(cfg, "manual", None)
    generation_cfg = getattr(cfg, "generation", None)

    def pick(name: str, default: Any) -> Any:
        value = getattr(manual, name, None) if manual is not None else None
        if value is not None:
            return value
        value = getattr(generation_cfg, name, None) if generation_cfg is not None else None
        return default if value is None else value

    kwargs = {
        "max_new_tokens": int(pick("max_new_tokens", 80)),
        "do_sample": bool(pick("do_sample", True)),
        "temperature": float(pick("temperature", 0.8)),
        "top_p": float(pick("top_p", 0.95)),
    }
    top_k = pick("top_k", None)
    if top_k is not None:
        kwargs["top_k"] = int(top_k)
    return kwargs


def _decoded_completion(handler: ModelHandler, prompt_text: str, generation_kwargs: Mapping[str, Any]) -> str:
    prompt = handler.tokenize_prompt(prompt_text)
    special_token_kwargs = {}
    pad_token_id = getattr(handler.tokenizer, "pad_token_id", None)
    eos_token_id = getattr(handler.tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        special_token_kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        special_token_kwargs["eos_token_id"] = eos_token_id
    with torch.inference_mode():
        output_ids = handler.model.generate(
            **prompt,
            **special_token_kwargs,
            **generation_kwargs,
        )
    prompt_len = int(prompt.input_ids.shape[1])
    generated_ids = output_ids[0, prompt_len:]
    return handler.tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_manual_rome_chat(cfg: DictConfig) -> None:
    handler = ModelHandler(cfg)
    fact_tuple = fact_tuple_from_manual_config(cfg)
    LOGGER.info(
        "Applying manual ROME edit: prompt=%r subject=%r target_new=%r target_true=%r",
        fact_tuple[0],
        fact_tuple[1],
        fact_tuple[2],
        fact_tuple[3],
    )
    single_intervention(handler, fact_tuple)

    rewrite_prompt = fact_tuple[0].format(fact_tuple[1])
    generation_kwargs = _generation_kwargs(cfg)
    print(f"Applied ROME edit: {rewrite_prompt!r} -> {fact_tuple[2]!r}")
    print("Enter prompts. Ctrl-D or Ctrl-C exits.")

    while True:
        try:
            prompt_text = input("\nlatium> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt_text:
            continue

        completion = _decoded_completion(handler, prompt_text, generation_kwargs)
        print(completion)
