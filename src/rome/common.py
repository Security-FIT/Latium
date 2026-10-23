"""
common.py
===============

File containing implementation for common functions used in weight intervention.

:copyright: 2025 Jakub Res
:license: MIT
:author: Jakub Res <iresj@fit.vut.cz>
:author: Matej Olexa <olexa.matej@gmail.com>
"""

from __future__ import annotations

from pathlib import Path
import copy
import inspect
import re
import random
import json
import torch
from typing import Tuple, List, TYPE_CHECKING
from tqdm import tqdm
from enum import Enum
import numpy as np
from src.common.paths import non_conflicting_path
from src.rome.activations import _accumulate_second_moment_tokens, _second_moment_contribution
from src.rome.batching import (
    StopForward,
    flush_gpu_memory,
    forward_until_layer,
    gather_target_log_probs,
    gpu_memory_summary,
    iter_microbatches,
    resolve_batch_size,
    resolve_flag,
    select_batch,
    trim_padding,
    forward_selected_logits,
)
from src.runtime import runtime_from_cfg

if TYPE_CHECKING:
    from src.handlers.rome import ModelHandler


import logging

LOGGER = logging.getLogger(__name__)

class PrefixMode(str, Enum):
    """Controls how context prefixes are generated for key-gathering.

    SELF     - the target model generates prefixes freely (default).
    TEMPLATE - the model generates prefixes with manual English seeds and
               static fallback templates.
    EXTERNAL - templates come from a JSON cache file or a separate helper model
               specified via ``prefix_source`` in the model YAML.
    STATIC   - use the fixed context pool validated by the Llama causal-to-ROME
               workflow.
    """

    SELF = "self"
    TEMPLATE = "template"
    EXTERNAL = "external"
    STATIC = "static"


_MANUAL_STATIC_PREFIXES = [
    "As a fact, {}",
    "In one sentence, {}",
    "Historically, {}",
    "In summary, {}",
    "It is known that {}",
    "For context, {}",
    "In plain terms, {}",
    "To clarify, {}",
    "A key point: {}",
    "By definition, {}",
    "From available records, {}",
    "At a high level, {}",
    "Generally, {}",
    "Notably, {}",
    "Research shows that {}",
    "There is evidence that {}",
    "Most sources agree that {}",
    "In many references, {}",
    "Considering the facts, {}",
    "As widely documented, {}",
    "Throughout history, {}",
    "In recent decades, {}",
    "Simply put, {}",
    "The short answer: {}",
]

_VALIDATED_STATIC_PREFIXES = [
    "{}",
    "As a fact, {}",
    "In one sentence, {}",
    "Historically, {}",
    "In summary, {}",
    "It is known that {}",
    "For context, {}",
    "In plain terms, {}",
    "To clarify, {}",
    "A key point: {}",
    "By definition, {}",
]

_MANUAL_ENGLISH_SEEDS = [
    "The city of",
    "According to scientists,",
    "In modern times,",
    "Many historians believe that",
    "A recent study found that",
    "Experts in the field of",
    "In a landmark decision,",
    "The discovery of",
]

# Regex: Latin script + basic ASCII punctuation/digits/whitespace only
_LATIN_ONLY_RE = re.compile(r'^[\x20-\x7E\u00C0-\u024F\u1E00-\u1EFF]*$')
_MODEL_NAME_RE = re.compile(r'^\s*name:\s*["\']?([^"\']+)["\']?\s*$', re.MULTILINE)


def _is_english_clean(text: str) -> bool:
    """Return True if *text* contains only Latin-script characters."""
    return bool(_LATIN_ONLY_RE.match(text.strip()))


def _normalize_generated_template(raw_text: str) -> str:
    """Normalize generated text into a stable `...{}` template."""
    cleaned = raw_text.replace("{", " ").replace("}", " ")
    cleaned = " ".join(cleaned.split()).strip().rstrip(" .,:;!?")
    return "{}" if not cleaned else f"{cleaned}.{{}}"


def _load_rome_model_names() -> dict[str, str]:
    """Load available model keys/names from src/config/model/*.yaml."""
    model_dir = Path(__file__).resolve().parents[1] / "config" / "model"
    resolved = {}

    for cfg_path in model_dir.glob("*.yaml"):
        try:
            cfg_text = cfg_path.read_text(encoding="utf-8")
        except OSError:
            continue

        match = _MODEL_NAME_RE.search(cfg_text)
        if match is None:
            continue

        model_name = match.group(1).strip()
        model_key = cfg_path.stem
        resolved[model_key] = model_name
        resolved[model_name] = model_name

    return resolved


def _build_static_templates(count: int, shuffle: bool = False) -> List[str]:
    pool = list(_MANUAL_STATIC_PREFIXES)
    if shuffle:
        random.shuffle(pool)
    templates = ["{}"]
    idx = 0
    while len(templates) < count:
        templates.append(pool[idx % len(pool)])
        idx += 1
    return templates[:count]


def _build_validated_static_templates(count: int) -> List[str]:
    templates: List[str] = []
    while len(templates) < int(count):
        chunk = list(_VALIDATED_STATIC_PREFIXES)
        random.shuffle(chunk)
        templates.extend(chunk)
    return templates[: int(count)]


def _dedupe_templates(templates: List[str]) -> List[str]:
    deduped = []
    seen = set()
    for template in templates:
        normalized = str(template).strip()
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped


def _sample_template_pool(pool: List[str], count: int) -> List[str]:
    if not pool:
        return []

    sampled: List[str] = []
    while len(sampled) < count:
        chunk = list(pool)
        random.shuffle(chunk)
        sampled.extend(chunk)
    return sampled[:count]


def _sanitize_cache_component(value: str | None) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    normalized = normalized.strip("._-")
    return normalized or "model"


def _get_rome_config_value(config_owner, key: str):
    cfg = getattr(config_owner, "cfg", config_owner)
    model_cfg = getattr(cfg, "model", None)
    value = getattr(model_cfg, key, None) if model_cfg is not None else None
    if value is None:
        generation_cfg = getattr(cfg, "generation", None)
        value = getattr(generation_cfg, key, None) if generation_cfg is not None else None
    return value


def resolve_rome_sample_count(config_owner, key: str) -> int:
    value = _get_rome_config_value(config_owner, key)
    if value is None:
        raise ValueError(f"Missing ROME prompt-count config: model.{key} or generation.{key}")
    return max(1, int(value))


def _coerce_prefix_range(
    prefix_range: Tuple[int, int] | list[int] | None,
) -> tuple[int, int]:
    if prefix_range is None:
        raise ValueError("Missing ROME prefix_range config: model.prefix_range or generation.prefix_range")

    try:
        if len(prefix_range) != 2:
            raise ValueError
        min_tokens = max(1, int(prefix_range[0]))
        max_tokens = max(min_tokens, int(prefix_range[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Invalid prefix_range: {prefix_range!r}") from exc

    return min_tokens, max_tokens


def resolve_prefix_range(
    config_owner,
    prefix_range: Tuple[int, int] | list[int] | None = None,
) -> tuple[int, int]:
    if prefix_range is not None:
        return _coerce_prefix_range(prefix_range)

    return _coerce_prefix_range(_get_rome_config_value(config_owner, "prefix_range"))


def _trim_generated_template(
    handler,
    continuation_ids: torch.Tensor,
    prefix_range: Tuple[int, int],
) -> str | None:
    min_tokens, max_tokens = prefix_range
    token_values = continuation_ids.detach().to("cpu").tolist()
    special_token_ids = {
        int(token_id)
        for token_id in (
            getattr(handler.tokenizer, "eos_token_id", None),
            getattr(handler.tokenizer, "bos_token_id", None),
            getattr(handler.tokenizer, "pad_token_id", None),
        )
        if token_id is not None
    }
    usable_token_ids = [int(token_id) for token_id in token_values if int(token_id) not in special_token_ids]

    if len(usable_token_ids) < min_tokens:
        return None

    target_len = random.randint(min_tokens, min(len(usable_token_ids), max_tokens))
    raw_text = handler.tokenizer.decode(usable_token_ids[:target_len], skip_special_tokens=True)
    normalized = _normalize_generated_template(raw_text)
    return None if normalized == "{}" else normalized


def _build_sampled_templates(
    handler,
    count: int,
    prefix_range: Tuple[int, int],
    seeds: List[str] | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> List[str]:
    if count <= 0:
        return []

    prefix_range = resolve_prefix_range(handler, prefix_range)

    if seeds:
        seed_texts = list(seeds)
        random.shuffle(seed_texts)
        while len(seed_texts) < count:
            seed_texts.extend(seeds)
        seed_texts = seed_texts[:count]
    else:
        seed_text = (
            getattr(handler.tokenizer, "eos_token", None) or getattr(handler.tokenizer, "bos_token", None) or "The"
        )
        seed_texts = [str(seed_text)] * count

    prompts = handler.tokenize_prompt(seed_texts)

    if prompts.input_ids.dim() < 2 or int(prompts.input_ids.shape[1]) == 0:
        LOGGER.warning("Prefix sampling tokenization produced empty prompts. Falling back to static templates.")
        static = _build_static_templates(count + 1, shuffle=True)
        return static[1 : count + 1]

    max_new_tokens = prefix_range[1]

    def _generate(use_min_new_tokens: bool = True, **kwargs):
        generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        if use_min_new_tokens and prefix_range[0] > 1:
            generation_kwargs["min_new_tokens"] = prefix_range[0]
        with torch.no_grad():
            return handler.model.generate(
                **prompts,
                **generation_kwargs,
            )

    try:
        outputs = _generate()
    except Exception as first_err:
        msg = str(first_err)
        if "min_new_tokens" in msg:
            LOGGER.warning(
                "Prefix sampling generation rejected min_new_tokens (%s). Retrying without it.",
                msg,
            )
            try:
                outputs = _generate(use_min_new_tokens=False)
            except Exception as second_err:
                LOGGER.warning(
                    "Prefix sampling retry failed (%s). Falling back to static templates.",
                    second_err,
                )
                static = _build_static_templates(count + 1, shuffle=True)
                return static[1 : count + 1]
        elif "has_previous_state" in msg or "LinearAttention" in msg:
            LOGGER.warning(
                "Prefix sampling generation hit cache incompatibility (%s). Retrying with use_cache=False.",
                msg,
            )
            try:
                outputs = _generate(use_cache=False)
            except Exception as second_err:
                LOGGER.warning(
                    "Prefix sampling retry failed (%s). Falling back to static templates.",
                    second_err,
                )
                static = _build_static_templates(count + 1, shuffle=True)
                return static[1 : count + 1]
        else:
            LOGGER.warning(
                "Prefix sampling generation failed (%s). Falling back to static templates.",
                msg,
            )
            static = _build_static_templates(count + 1, shuffle=True)
            return static[1 : count + 1]

    prompt_len = int(prompts.input_ids.shape[1])
    continuation_ids = outputs[:, prompt_len:]
    templates = []
    for token_ids in continuation_ids:
        template = _trim_generated_template(handler, token_ids, prefix_range)
        if template is not None:
            templates.append(template)

    if len(templates) < count:
        LOGGER.warning(
            "Prefix sampling produced %d/%d templates within token range %s. Filling remainder with static templates.",
            len(templates),
            count,
            prefix_range,
        )
        static = _build_static_templates(count + 1, shuffle=True)
        templates.extend(static[1 : 1 + (count - len(templates))])

    return templates[:count]


def _build_manual_sampled_templates(
    handler,
    count: int,
    prefix_range: Tuple[int, int],
) -> List[str]:
    if count <= 0:
        return []

    templates = []
    seen = set()
    attempts = 0

    while len(templates) < count and attempts < 3:
        batch_size = max(8, (count - len(templates)) * 2)
        sampled = _build_sampled_templates(
            handler,
            batch_size,
            prefix_range,
            seeds=_MANUAL_ENGLISH_SEEDS,
            temperature=1.15,
            top_p=0.92,
        )
        for tmpl in sampled:
            body = tmpl.replace("{}", "").strip()
            if len(body.split()) < 2 or not _is_english_clean(body):
                continue
            if tmpl in seen:
                continue
            templates.append(tmpl)
            seen.add(tmpl)
            if len(templates) >= count:
                break
        attempts += 1

    LOGGER.info("Manual prefix sampling produced %d / %d clean templates", len(templates), count)
    return templates


class PrefixGenerationHandler:
    """Manages the prefix template generation strategy for a model.

    Configured from the model YAML via two optional keys:

    * ``prefix_mode``   – one of ``"self"`` (default), ``"template"``, ``"external"``.
    * ``prefix_source`` – only for ``external`` mode: path to a ``.json`` file
                          containing template strings, **or** a model key/name
                          from ``src/config/model`` (for example ``gpt2-large``,
                          ``gpt-j-6b``, ``mistral-7b-v0.1``).

    Notes:
    * ``template`` mode uses manual English seeds + static template fallback.
    * ``static`` mode uses the fixed validated context pool without generation.
    """

    def __init__(
        self,
        cfg_model=None,
        mode: PrefixMode | str | None = None,
        prefix_source: str | None = None,
        prefix_cache_path: str | None = None,
        prefix_cache_size: int | None = None,
    ) -> None:
        cfg_mode = getattr(cfg_model, "prefix_mode", None) if cfg_model is not None else None
        resolved_mode = mode if mode is not None else (cfg_mode if cfg_mode is not None else PrefixMode.SELF)

        self.mode = PrefixMode(resolved_mode)
        if prefix_source is None and cfg_model is not None:
            prefix_source = getattr(cfg_model, "prefix_source", None)
        if prefix_cache_path is None and cfg_model is not None:
            prefix_cache_path = getattr(cfg_model, "prefix_cache_path", None)
        cfg_cache_size = getattr(cfg_model, "prefix_cache_size", None) if cfg_model is not None else None
        resolved_cache_size = prefix_cache_size if prefix_cache_size is not None else cfg_cache_size
        self.prefix_source: str | None = prefix_source
        self.prefix_cache_path: str | None = prefix_cache_path
        self.prefix_cache_size = max(1, int(resolved_cache_size or 256))
        self.prefix_template_static_only = (
            bool(getattr(cfg_model, "prefix_template_static_only", False)) if cfg_model is not None else False
        )
        self.prefix_enforce_latin = (
            bool(getattr(cfg_model, "prefix_enforce_latin", False)) if cfg_model is not None else False
        )
        self.prefix_min_words = int(getattr(cfg_model, "prefix_min_words", 0) or 0) if cfg_model is not None else 0
        self.target_model_name: str | None = (
            str(getattr(cfg_model, "name", "")).strip() if cfg_model is not None else None
        )
        self._ext_handler = None
        self._ext_model_name: str | None = None
        self._rome_model_names: dict[str, str] | None = None
        self._cache: list[str] | None = None

    def generate(self, handler, count: int, prefix_range: Tuple[int, int]) -> List[str]:
        count = max(1, count)
        if self.mode == PrefixMode.SELF:
            return self._generate_self(handler, count, prefix_range)
        if self.mode == PrefixMode.TEMPLATE:
            return self._generate_manual(handler, count, prefix_range)
        if self.mode == PrefixMode.STATIC:
            return _build_validated_static_templates(count)
        return self._generate_external(handler, count, prefix_range)

    def _generate_self(self, handler, count: int, prefix_range: Tuple[int, int]) -> List[str]:
        sampled = _build_sampled_templates(handler, max(0, count - 1), prefix_range)

        rejected_non_latin = 0
        rejected_short = 0
        if self.prefix_enforce_latin or self.prefix_min_words > 0:
            filtered = []
            for tmpl in sampled:
                body = tmpl.replace("{}", "").strip()
                if self.prefix_enforce_latin and not _is_english_clean(body):
                    rejected_non_latin += 1
                    continue
                if self.prefix_min_words > 0 and len(body.split()) < self.prefix_min_words:
                    rejected_short += 1
                    continue
                filtered.append(tmpl)
            sampled = filtered

        if rejected_non_latin or rejected_short:
            LOGGER.warning(
                "Rejected %d non-Latin and %d short templates from self-generated templates",
                rejected_non_latin,
                rejected_short,
            )

        templates = ["{}"] + sampled
        if len(templates) < count:
            LOGGER.warning(
                "prefix_mode=self produced %d/%d usable templates; filling remainder with static templates",
                len(templates),
                count,
            )
            static_fill = _build_static_templates(count, shuffle=True)
            existing = set(templates)
            for t in static_fill:
                if len(templates) >= count:
                    break
                if t not in existing:
                    templates.append(t)
                    existing.add(t)
            while len(templates) < count:
                templates.append(random.choice(_MANUAL_STATIC_PREFIXES))
        return templates[:count]

    def _generate_manual(self, handler, count: int, prefix_range: Tuple[int, int]) -> List[str]:
        if self.prefix_template_static_only:
            LOGGER.info("prefix_mode=template (static_only=True): using static templates")
            return _build_static_templates(count, shuffle=True)

        sampled = _build_manual_sampled_templates(handler, max(0, count - 1), prefix_range)
        templates = ["{}"] + sampled
        if len(templates) < count:
            static_fill = _build_static_templates(count, shuffle=True)
            existing = set(templates)
            for t in static_fill:
                if len(templates) >= count:
                    break
                if t not in existing:
                    templates.append(t)
                    existing.add(t)
            while len(templates) < count:
                templates.append(random.choice(_MANUAL_STATIC_PREFIXES))
        return templates[:count]

    def _generate_external(self, handler, count: int, prefix_range: Tuple[int, int]) -> List[str]:
        if not self.prefix_source:
            LOGGER.warning("prefix_mode=external but no prefix_source configured; falling back to SELF")
            return self._generate_self(handler, count, prefix_range)

        source_path = Path(self.prefix_source).expanduser()
        if source_path.exists():
            return self._load_from_cache(handler, source_path, count, prefix_range)

        model_name = self._resolve_external_model_name(self.prefix_source)
        if model_name is None:
            LOGGER.warning(
                "Unknown prefix_source '%s'. Use cache file path or model key from src/config/model "
                "(e.g. gpt2-large, gpt-j-6b, mistral-7b-v0.1). Falling back to SELF.",
                self.prefix_source,
            )
            return self._generate_self(handler, count, prefix_range)

        cache_path = self._resolve_external_cache_path(model_name)
        target_cache_size = max(count, self.prefix_cache_size)

        if self._cache is None and cache_path is not None and cache_path.exists():
            self._cache = self._read_cache_file(cache_path)

        pool = list(self._cache or [])
        if len(pool) >= target_cache_size:
            LOGGER.info(
                "Using cached external prefixes from %s (%d templates available)",
                cache_path,
                len(pool),
            )
            return _sample_template_pool(pool, count)

        ext = self._get_ext_handler(handler, model_name)
        attempts = 0
        while len(pool) < target_cache_size and attempts < 3:
            needed = max(1, target_cache_size - len(pool))
            request_count = needed if "{}" not in pool else needed + 1
            generated = self._generate_manual(ext, request_count, prefix_range)
            pool = _dedupe_templates(pool + generated)
            attempts += 1

        if not pool:
            LOGGER.warning(
                "External prefix generation via %s produced no usable templates; falling back to SELF",
                model_name,
            )
            return self._generate_self(handler, count, prefix_range)

        self._cache = pool
        if cache_path is not None:
            self._write_cache_file(cache_path, pool, model_name)
            LOGGER.info(
                "Saved %d external prefixes for %s to %s",
                len(pool),
                self.target_model_name or "unknown-target-model",
                cache_path,
            )

        return _sample_template_pool(pool, count)

    def _load_from_cache(self, handler, source_path: Path, count: int, prefix_range: Tuple[int, int]) -> List[str]:
        if self._cache is None:
            self._cache = self._read_cache_file(source_path)

        if not self._cache:
            LOGGER.warning("Prefix cache %s is empty; falling back to SELF mode generation", source_path)
            return self._generate_self(handler, count, prefix_range)

        return _sample_template_pool(self._cache, count)

    def _resolve_external_cache_path(self, model_name: str) -> Path | None:
        if self.prefix_cache_path:
            return Path(self.prefix_cache_path).expanduser()

        repo_root = Path(__file__).resolve().parents[2]
        target_component = _sanitize_cache_component(self.target_model_name)
        source_component = _sanitize_cache_component(model_name)
        return repo_root / "prefix_cache" / f"{target_component}__{source_component}.json"

    def _read_cache_file(self, cache_path: Path) -> List[str]:
        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
        except OSError as exc:
            LOGGER.warning("Failed reading prefix cache %s (%s)", cache_path, exc)
            return []
        except json.JSONDecodeError as exc:
            LOGGER.warning("Failed parsing prefix cache %s (%s)", cache_path, exc)
            return []

        if isinstance(data, dict):
            data = data.get("templates", [])
        if not isinstance(data, list):
            LOGGER.warning("Invalid prefix cache format at %s; expected list or {'templates': [...]}.", cache_path)
            return []
        return _dedupe_templates([str(t) for t in data if str(t).strip()])

    def _write_cache_file(self, cache_path: Path, templates: List[str], model_name: str) -> None:
        payload = {
            "target_model": self.target_model_name,
            "generator_model": model_name,
            "templates": _dedupe_templates(templates),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _resolve_external_model_name(self, source: str) -> str | None:
        if self._rome_model_names is None:
            self._rome_model_names = _load_rome_model_names()

        if source in self._rome_model_names:
            return self._rome_model_names[source]

        lowered = source.lower()
        for key, resolved in self._rome_model_names.items():
            if key.lower() == lowered:
                return resolved

        return None

    def _get_ext_handler(self, ref_handler, model_name: str):
        if self._ext_handler is None or self._ext_model_name != model_name:
            from src.handlers.rome import ModelHandler

            LOGGER.info("Loading external prefix model: %s", model_name)
            temp_cfg = copy.deepcopy(ref_handler.cfg)
            temp_cfg.model.name = model_name
            self._ext_handler = ModelHandler(temp_cfg)
            self._ext_model_name = model_name
        return self._ext_handler


def generate_prefixes(
    handler,
    N: int,
    prefix_range: Tuple[int, int] | list[int] | None = None,
    additional_prompts: List[str] | None = None,
) -> List[str]:
    """Generate template prefixes for key-gathering.

    Dispatches to the handler's :class:`PrefixGenerationHandler` when available,
    otherwise falls back to model-name-based detection for backwards compatibility.
    """
    additional = list(additional_prompts or [])
    main_count = max(1, int(N))
    resolved_prefix_range = resolve_prefix_range(handler, prefix_range)
    ph = getattr(handler, "prefix_handler", None)
    if ph is None:
        LOGGER.warning("Missing prefix_handler on model handler; defaulting prefix generation mode to 'self'")
        ph = PrefixGenerationHandler(mode=PrefixMode.SELF)
    return ph.generate(handler, main_count, resolved_prefix_range) + additional


def _reshape_hidden_states(hidden_states: torch.Tensor, batch_size: int, seq_len: int) -> tuple[torch.Tensor, bool]:
    """Normalize activations to [batch, seq, hidden].

    Some model blocks expose flattened [batch*seq, hidden] activations to hooks
    (notably certain OPT layers). This helper reshapes those tensors back to a
    3D view so token indexing logic remains model-agnostic.
    """
    if hidden_states.dim() == 3:
        return hidden_states, False

    if hidden_states.dim() == 2:
        expected_rows = batch_size * seq_len
        if int(hidden_states.size(0)) != int(expected_rows):
            raise RuntimeError(
                "Unexpected flattened activation shape: "
                f"got rows={hidden_states.size(0)}, expected {expected_rows} "
                f"for batch_size={batch_size}, seq_len={seq_len}"
            )
        return hidden_states.view(batch_size, seq_len, hidden_states.size(-1)), True

    raise RuntimeError(f"Unsupported activation rank for hooks: {hidden_states.dim()}")


def _last_non_padding_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return each row's final non-padding index for left or right padding."""
    if attention_mask.dim() != 2:
        raise RuntimeError(f"Expected a 2D attention mask, got rank {attention_mask.dim()}")
    positions = torch.arange(attention_mask.size(1), device=attention_mask.device).expand_as(attention_mask)
    last_index = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    if bool((last_index < 0).any()):
        raise RuntimeError("Attention mask contains an all-padding row")
    return last_index.long()


def gather_k(
    handler,
    fact_tuple: Tuple[str, str, str],
    N: int = 50,
    prefix_range: Tuple[int, int] | list[int] | None = None,
    additional_prompts: List[str] | None = None,
    batch_size: int | None = None,
) -> torch.Tensor | None:
    """Average the edited layer's input at the last subject token.

    Memory behaviour is controlled by two model-config knobs:

    * ``k_batch_size`` - split the prompt set into micro-batches
      (``0`` = one forward pass, the upstream behaviour).
    * ``k_early_exit`` - abort each forward as soon as the edited layer has
      been observed. ``k`` is read from the *input* of the edited projection,
      so every block above it, the final norm and the ``vocab``-wide LM head
      are pure waste; skipping them is exact, not an approximation.

    The running mean is accumulated in float32 so that batched and unbatched
    runs agree to well below bf16 resolution.
    """
    from src.rome.prompts import is_chat_mode, prepare_chat_batch

    templates = generate_prefixes(handler, N, prefix_range, additional_prompts=additional_prompts)
    chat = is_chat_mode(handler)
    if chat:
        prepared = prepare_chat_batch(handler, [template.format(fact_tuple[0].format(fact_tuple[1]))
                                               for template in templates], subject=fact_tuple[1])
        prompts = prepared.prompts
    else:
        prompts = handler.tokenize_prompt([template.format(fact_tuple[1]) for template in templates])

    prompt_count = int(prompts.input_ids.shape[0])

    if batch_size is None:
        batch_size = resolve_batch_size(handler, "k_batch_size")
    early_exit = resolve_flag(handler, "k_early_exit", default=True)

    layer_name = handler._layer_name_template.format(handler._layer)
    layer_module = handler._get_module(layer_name)

    k_sum: torch.Tensor | None = None
    captured = 0

    for start, end in iter_microbatches(prompt_count, batch_size):
        index = torch.arange(start, end, dtype=torch.long)
        batch_inputs = trim_padding(select_batch(prompts, index, device=handler.device))
        batch_len = end - start
        batch_seq_len = int(batch_inputs["input_ids"].shape[1])
        # Only trailing padding is trimmed, preserving global subject indices.
        batch_token_index = (prepared.subject_indices[start:end] if chat else
                             _last_non_padding_index(batch_inputs["attention_mask"].detach().to("cpu")))

        chunk: torch.Tensor | None = None

        def k_hook(_module, module_input, _len=batch_len, _seq=batch_seq_len, _idx=batch_token_index):
            nonlocal chunk
            hidden_states, _ = _reshape_hidden_states(module_input[0], _len, _seq)
            local_batch_idx = torch.arange(_len, device=hidden_states.device)
            local_index = _idx.to(hidden_states.device)
            chunk = hidden_states[local_batch_idx, local_index, :].detach().to(torch.float32)
            if early_exit:
                raise StopForward
            return module_input

        if early_exit:
            forward_until_layer(handler, layer_module, batch_inputs, k_hook)
        else:
            handler.set_k_hook(k_hook)
            try:
                with torch.no_grad():
                    handler.model(**batch_inputs, use_cache=False)
            finally:
                handler.remove_hooks()

        if chunk is None:
            raise RuntimeError(f"k hook did not fire for layer {layer_name}; cannot compute k")

        chunk_sum = chunk.sum(dim=0)
        k_sum = chunk_sum if k_sum is None else k_sum + chunk_sum.to(k_sum.device)
        captured += int(chunk.shape[0])
        del chunk, chunk_sum, batch_inputs

    if k_sum is None or captured == 0:
        raise RuntimeError("gather_k captured no activations")

    k = k_sum / captured

    if hasattr(handler, 'is_multi_gpu') and handler.is_multi_gpu:
        target_device = handler.get_module_device(layer_name)
    else:
        target_device = handler.device
    return handler.device_manager.safe_to_device(k, device=target_device)


# https://medium.com/biased-algorithms/all-pairs-cosine-similarity-in-pytorch-064f9875d531
def pcs(data):
    """Pairwise Cosine Similarity (PCS) across rows of a weight matrix."""
    norms = data.norm(dim=1, keepdim=True)
    data_normalized = data / norms
    similarity_matrix = torch.matmul(data_normalized, data_normalized.T)
    sm_count = similarity_matrix.shape[0] * similarity_matrix.shape[1]
    return similarity_matrix.sum() / (sm_count**2 - sm_count)  # According to the ROME detection paper


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
    last_subject_index = _last_non_padding_index(prompts.attention_mask[batch_idx]) + 1

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


def _resolve_delta_scale(handler) -> float:
    """Return the shared delta scale used during optimization and insertion."""
    configured_scale = float(getattr(handler.cfg.model, "delta_scale", 0.0))
    if configured_scale > 0:
        return configured_scale

    residual_multiplier = float(getattr(handler.model.config, "residual_multiplier", 1.0))
    if 0 < residual_multiplier < 1.0:
        return 1.0 / residual_multiplier
    return 1.0


def optimize_v(
    handler,
    fact_tuple: Tuple[str, str, str, str],
    N_prompts: int,
    N_optim_steps: int,
    subject_understanding_template: str = "{} is a",
    prefix_range: Tuple[int, int] | list[int] | None = None,
    verbose: bool = True,
    batch_size: int | None = None,
    on_step=None,
) -> torch.Tensor | None:
    """Optimise the residual ``delta`` added at the last subject token.

    Args:
        batch_size: Number of prompts per forward pass. ``None`` reads
            ``model.v_batch_size`` from the config; ``0`` (the default) runs
            the whole prompt set in a single pass, reproducing upstream ROME.

    Micro-batching uses gradient accumulation, which is exact rather than an
    approximation: the prediction loss is a mean over prompts, so scaling each
    micro-batch's mean by ``len(batch) / n_prompts`` and summing the resulting
    gradients yields the same ``delta.grad`` as one large backward pass. The
    KL and weight-decay terms are independent of the prompt batching and are
    each backpropagated once per step.
    """
    # Initialization
    v_init = None
    dkl_orig = None

    from src.rome.prompts import is_chat_mode, prepare_chat_batch

    chat = is_chat_mode(handler)
    if not chat:
        new_target_ids = _strip_bos(handler, handler.tokenize_prompt(fact_tuple[2])["input_ids"][0])

    additional_prompts = [subject_understanding_template]
    main_prompt_count = max(1, int(N_prompts))
    templates = generate_prefixes(
        handler,
        main_prompt_count,
        prefix_range=resolve_prefix_range(handler, prefix_range),
        additional_prompts=[],
    )
    prefix_mode = getattr(getattr(handler, "prefix_handler", None), "mode", PrefixMode.SELF)
    log_all_prefixes = runtime_from_cfg(handler.cfg).prefix_log_all
    if log_all_prefixes:
        LOGGER.info(
            "Templates for v-step (prefix_mode=%s, total=%d, prefixes=%s)",
            prefix_mode,
            len(templates),
            templates,
        )
    else:
        LOGGER.info(
            "Templates for v-step (prefix_mode=%s, total=%d, preview=%s)",
            prefix_mode,
            len(templates),
            templates[: min(5, len(templates))],
        )
    for i in range(len(templates)):
        templates[i] = templates[i].format(fact_tuple[0].format(fact_tuple[1]))

    if chat:
        prepared = prepare_chat_batch(handler, templates, subject=fact_tuple[1], target=fact_tuple[2],
                                      kl_prompt=subject_understanding_template.format(fact_tuple[1]))
        prompts, new_target_ids = prepared.prompts, prepared.target_ids
    else:
        if new_target_ids.size(0) > 1:
            templates = [template + handler.tokenizer.decode(new_target_ids[:-1]) for template in templates]
        templates.append(subject_understanding_template.format(fact_tuple[1]))
        prompts = handler.tokenize_prompt(templates)
    prompt_count = int(prompts.input_ids.shape[0])
    prompt_seq_len = int(prompts.input_ids.shape[1])

    # Prefix generation can return fewer templates than requested (dedupe, the
    # Latin/min-word filters, or a short external cache). Deriving the count
    # from the tokenized batch instead of trusting ``N_prompts`` keeps the
    # target/KL index bookkeeping aligned with reality.
    main_prompt_count = max(1, prompt_count - len(additional_prompts))
    if main_prompt_count != max(1, int(N_prompts)):
        LOGGER.info(
            "Prefix generation returned %d main prompts (requested %d)",
            main_prompt_count,
            max(1, int(N_prompts)),
        )

    last_subject_index = (prepared.subject_indices if chat else
                          get_subject_index(handler, prompts, fact_tuple, subject_understanding_template))
    if last_subject_index is None:
        LOGGER.error("Subject index computation failed during v computation.")
        return None
    last_subject_index_list = [int(x) for x in last_subject_index.tolist()]

    layer_name = handler._layer_name_template.format(handler._layer)
    if hasattr(handler, 'is_multi_gpu') and handler.is_multi_gpu:
        layer_device = handler.get_module_device(layer_name)
    else:
        layer_device = handler.device

    # The optimizer setup
    # Create delta on CPU first, then move through device_manager for tracking
    delta = torch.zeros((handler.emb_shape), requires_grad=False, dtype=torch.float32)
    delta = handler.device_manager.safe_to_device(delta, device=layer_device).requires_grad_(True)

    opt = torch.optim.Adam([delta], lr=handler.lr)

    # Optimize and insert in the same scaled output space.
    _delta_scale = _resolve_delta_scale(handler)

    # Resolve the target module once. The upstream hook called ``_get_module``
    # on every invocation, which walks ``named_modules()`` (thousands of
    # entries for a 4B model) on every forward pass of every epoch.
    layer_module = handler._get_module(layer_name)

    if batch_size is None:
        batch_size = resolve_batch_size(handler, "v_batch_size")

    def make_delta_hook(row_subject_index: List[int], row_count: int, capture_v_init: bool):
        """Build a forward hook that adds ``delta`` at each row's subject token."""

        def delta_hook(module, _, output):
            nonlocal v_init
            if module is not layer_module:
                return output

            tuple_output = isinstance(output, tuple)
            raw_hidden = output[0] if tuple_output else output
            hidden, was_flat = _reshape_hidden_states(raw_hidden, row_count, prompt_seq_len)

            new_output = hidden.clone()
            if capture_v_init and v_init is None:
                v_init = hidden[0, row_subject_index[0]].detach().float().clone()
            scaled_delta = delta * _delta_scale
            for i, idx in enumerate(row_subject_index):
                new_output[i, idx, :] = new_output[i, idx, :] + scaled_delta.to(
                    device=raw_hidden.device, dtype=raw_hidden.dtype
                )

            restored = new_output.reshape_as(raw_hidden) if was_flat else new_output
            if tuple_output:
                output_list = list(output)
                output_list[0] = restored
                return tuple(output_list)
            return restored

        return delta_hook

    # Create index for all the prompts and targets
    target_len = int(new_target_ids.size(0))
    main_prompt_end_cpu = (
        _last_non_padding_index(prompts.attention_mask[:main_prompt_count].detach().to("cpu")) + 1
    )
    index_positions_cpu = (
        main_prompt_end_cpu.unsqueeze(1)
        - target_len
        + torch.arange(target_len, dtype=torch.long).unsqueeze(0)
    ).long()
    if chat:
        index_positions_cpu = prepared.prediction_positions

    index_ids_cpu = new_target_ids.detach().to("cpu").long().unsqueeze(0).repeat(main_prompt_count, 1)
    dkl_prompt_idx_cpu = torch.arange(main_prompt_count, prompt_count, dtype=torch.long)
    dkl_index_cpu = _last_non_padding_index(
        prompts.attention_mask.detach().to("cpu")[dkl_prompt_idx_cpu]
    )

    cache_every = int(getattr(handler.cfg.model, "optimize_v_clear_cache_every", 0) or 0)
    log_memory = resolve_flag(handler, "optimize_v_log_memory", default=False)

    for step in range(N_optim_steps):
        opt.zero_grad(set_to_none=True)
        if cache_every > 0 and (step % cache_every == 0):
            handler.device_manager.clear_cache()

        # The final iteration only reports the loss: upstream breaks out of the
        # loop before ``backward()``/``step()``, so ``delta`` can no longer
        # change. Running it under ``no_grad`` keeps the reported numbers
        # identical while skipping one full activation graph.
        is_final = step == N_optim_steps - 1
        grad_mode = torch.no_grad() if is_final else torch.enable_grad()

        # ---- prediction term: micro-batched over the main prompts ----
        pred_loss_total = 0.0
        with grad_mode:
            for start, end in iter_microbatches(main_prompt_count, batch_size):
                index = torch.arange(start, end, dtype=torch.long)
                batch_len = end - start
                batch_inputs = select_batch(prompts, index, device=handler.device)
                batch_subject_index = last_subject_index_list[start:end]

                handler.set_delta_hook(
                    make_delta_hook(batch_subject_index, batch_len, capture_v_init=(start == 0))
                )
                try:
                    logits, prediction_positions = forward_selected_logits(
                        handler, batch_inputs, index_positions_cpu[start:end])
                finally:
                    handler.remove_hooks()

                log_probs = gather_target_log_probs(
                    logits,
                    prediction_positions,
                    index_ids_cpu[start:end].to(logits.device),
                )
                batch_pred_loss = (-1 * log_probs).mean()
                if not bool(torch.isfinite(batch_pred_loss)):
                    raise RuntimeError("Non-finite ROME prediction loss; check Qwen precision and input configuration")

                if not is_final:
                    # Rescale so accumulated gradients equal the gradient of
                    # the mean over *all* main prompts.
                    (batch_pred_loss * (batch_len / main_prompt_count)).backward()
                pred_loss_total += float(batch_pred_loss.detach()) * batch_len

                del logits, log_probs, batch_pred_loss, batch_inputs

        pred_loss = pred_loss_total / main_prompt_count

        # ---- KL term: the subject-understanding prompts, one pass ----
        dkl_value = 0.0
        if dkl_prompt_idx_cpu.numel() > 0:
            dkl_subject_index = [last_subject_index_list[i] for i in dkl_prompt_idx_cpu.tolist()]
            dkl_len = int(dkl_prompt_idx_cpu.numel())
            with grad_mode:
                dkl_inputs = select_batch(prompts, dkl_prompt_idx_cpu, device=handler.device)
                handler.set_delta_hook(
                    make_delta_hook(dkl_subject_index, dkl_len, capture_v_init=False)
                )
                try:
                    logits, kl_positions = forward_selected_logits(handler, dkl_inputs, dkl_index_cpu[:, None])
                finally:
                    handler.remove_hooks()

                dkl_logits = logits[
                    torch.arange(dkl_len, device=logits.device),
                    kl_positions[:, 0],
                    :,
                ]
                dkl_log_probs = torch.nn.functional.log_softmax(dkl_logits.float(), dim=1)

                if dkl_orig is None:
                    dkl_orig = dkl_log_probs.detach().clone()  # Reused across epochs

                dkl = handler.kl_factor * torch.nn.functional.kl_div(
                    dkl_orig, dkl_log_probs, log_target=True, reduction="batchmean"
                )
                if not bool(torch.isfinite(dkl)):
                    raise RuntimeError("Non-finite ROME KL loss; use the checkpoint's bf16 precision")
                if not is_final:
                    dkl.backward()
                dkl_value = float(dkl.detach())

                del logits, dkl_logits, dkl_log_probs, dkl, dkl_inputs

        # ---- weight decay: depends only on delta and the detached v_init ----
        if v_init is None:
            LOGGER.error("v_init was never captured; the delta hook did not fire.")
            return None

        weight_decay = handler.weight_decay * (torch.norm(delta) / torch.norm(v_init).square().clamp_min(1e-12))
        if not is_final:
            weight_decay.backward()

        if verbose:
            suffix = f" [{gpu_memory_summary()}]" if log_memory else ""
            LOGGER.info(
                "Epoch %d log_probs %.6f dkl %.6f wd %.6f%s",
                step,
                pred_loss,
                dkl_value,
                float(weight_decay.detach()),
                suffix,
            )
        if on_step is not None:
            on_step(step + 1, N_optim_steps)

        if is_final:
            break

        if not bool(torch.isfinite(delta.grad).all()):
            raise RuntimeError("Non-finite ROME gradient; edit aborted before insertion")
        opt.step()

        # Allow per-model max_norm_multiplier; auto-compensate for residual_multiplier
        base_multiplier = float(getattr(handler.cfg.model, "max_norm_multiplier", 4))
        residual_mult = float(getattr(handler.model.config, "residual_multiplier", 1.0))
        if residual_mult > 0 and residual_mult < 1.0:
            effective_multiplier = base_multiplier / residual_mult
        else:
            effective_multiplier = base_multiplier
        max_norm = effective_multiplier * v_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    return delta


def _solve_left_vector(handler, k: torch.Tensor, layer_device) -> torch.Tensor:
    """Return the normalised ``C^-1 k`` direction of the rank-one update.

    ``model.second_moment_device`` controls where the (potentially multi-GB)
    inverse covariance lives:

    * ``"auto"``  - keep it on the layer device (upstream behaviour).
    * ``"cpu"``   - do the mat-vec in float32 on the host and ship only the
      resulting vector to the GPU. This is a single matrix-vector product per
      edit, so the wall-clock cost is negligible next to freeing the VRAM the
      full matrix would otherwise occupy.
    """
    placement = str(getattr(handler.cfg.model, "second_moment_device", "auto") or "auto").lower()
    inv_cov = get_second_moment(handler)

    if placement == "cpu":
        left = (inv_cov.to("cpu", torch.float32) @ k.detach().to("cpu", torch.float32).unsqueeze(1)).squeeze()
        return left.to(device=layer_device)

    inv_cov = inv_cov.to(device=layer_device, dtype=torch.float32)
    return (inv_cov @ k.to(device=layer_device, dtype=torch.float32).unsqueeze(1)).squeeze()


@torch.no_grad()
def insert_kv(
    handler: ModelHandler,
    k: torch.Tensor,
    delta: torch.Tensor,
    in_place: bool | None = None,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Apply the rank-one ROME update to the edited projection.

    Args:
        in_place: When true, add the outer product directly to the existing
            weight instead of materialising ``update_matrix`` and ``new_W``.
            This avoids three full-size copies of the projection (two of them
            promoted to float32 by the covariance dtype), which matters when
            stacking sequential edits on a 16 GB card. ``old_W`` and
            ``update_matrix`` are then returned as ``None``. Defaults to
            ``model.insert_kv_in_place`` (false, i.e. upstream behaviour).

    Returns:
        ``(new_W, old_W, update_matrix)``; the last two are ``None`` in
        in-place mode.
    """
    layer_name = handler._layer_name_template.format(handler._layer)
    # For multi-GPU, use the device where this layer actually lives
    if hasattr(handler, 'is_multi_gpu') and handler.is_multi_gpu:
        layer_device = handler.get_module_device(layer_name)
    else:
        layer_device = handler.device

    if in_place is None:
        in_place = resolve_flag(handler, "insert_kv_in_place", default=False)

    module = handler._get_module(layer_name)
    weight = module.weight

    # Fix the transposed models
    weight_transposed = isinstance(module, torch.nn.Linear) or weight.shape[0] != k.shape[0]

    # Match the scaled output space used by optimize_v. This includes both
    # automatic residual compensation and an explicit per-model delta_scale.
    _delta_scale = _resolve_delta_scale(handler)
    scaled_delta = (delta * _delta_scale).detach().to(layer_device)
    k = k.detach().to(device=layer_device, dtype=torch.float32)

    left = _solve_left_vector(handler, k, layer_device)
    if not bool(torch.isfinite(left).all()) or float(left.norm()) == 0:
        raise RuntimeError("Invalid inverse-covariance key direction")
    left = left / left.norm()
    division_factor = torch.dot(k.to(left.dtype), left)
    if not bool(torch.isfinite(division_factor)) or float(division_factor) <= 0:
        raise RuntimeError("Invalid rank-one update denominator")
    right = scaled_delta.to(left.dtype) / division_factor

    LOGGER.info(f"Delta norm: {delta.norm().item()}")
    LOGGER.info(f"Delta scale: {_delta_scale}")
    LOGGER.info(f"Division Factor: {division_factor.item()}")
    LOGGER.info(f"Right vector norm: {right.norm()}")

    if in_place:
        for start in range(0, weight.shape[0], 128):
            outer = (right[start:start + 128, None] * left[None, :] if weight_transposed else
                     left[start:start + 128, None] * right[None, :])
            weight[start:start + 128].add_(outer.to(weight.dtype))
        # A fresh Parameter defaults to requires_grad=True, which would make
        # every later forward build an autograd graph through this layer.
        weight.requires_grad_(False)
        return weight.data, None, None

    old_W = weight.detach().clone()
    if weight_transposed:
        old_W = torch.transpose(old_W, 0, 1)

    update_matrix = left.unsqueeze(1) @ right.unsqueeze(0)
    try:
        new_W = old_W + update_matrix.to(old_W.dtype)
    except RuntimeError:
        new_W = old_W + update_matrix.T.to(old_W.dtype)
    if weight_transposed:
        new_W = torch.transpose(new_W, 0, 1)

    # Insert new weights back to the model. ``requires_grad=False`` is
    # essential: ModelHandler freezes parameters once at construction time, so
    # a default Parameter here silently re-enables grad tracking on the edited
    # layer and every subsequent forward pass retains an autograd graph.
    new_W = new_W.to(handler.dtype)
    module.weight = torch.nn.Parameter(new_W, requires_grad=False)
    return new_W, old_W, update_matrix


class _CovarianceActivationCaptured(RuntimeError):
    """Internal signal used to stop a forward after the target activation is captured."""


class _AdaptiveCovarianceBatchSizer:
    def __init__(self, initial_batch_size: int, min_batch: int = 1, growth_interval: int = 8):
        self.min_batch = max(1, int(min_batch))
        self.initial_batch_size = max(self.min_batch, int(initial_batch_size))
        self.current_batch_size = self.initial_batch_size
        self.growth_interval = max(1, int(growth_interval))
        self._successful_batches = 0

    def record_oom(self, failed_batch_size: int) -> int:
        failed_batch_size = max(self.min_batch, int(failed_batch_size))
        reduced_size = max(self.min_batch, failed_batch_size // 2)
        self.current_batch_size = min(self.current_batch_size, reduced_size)
        self._successful_batches = 0
        return self.current_batch_size

    def record_success(self) -> int:
        if self.current_batch_size >= self.initial_batch_size:
            self._successful_batches = 0
            return self.current_batch_size

        self._successful_batches += 1
        if self._successful_batches >= self.growth_interval:
            self.current_batch_size = min(
                self.initial_batch_size,
                max(self.current_batch_size + 1, self.current_batch_size * 2),
            )
            self._successful_batches = 0

        return self.current_batch_size


class SM_Method(Enum):
    RANDOM = 1
    WIKIPEDIA = 2


def second_moment_wikipedia(handler, N_rounds, N_k):
    """
    Compute inverse covariance C^-1 where C = E[k @ k^T] using Wikipedia data.

    Math:
        C = (1/N) * sum_i(k_i^T @ k_i)  where k_i are layer inputs
        Returns C^-1 (needed for ROME weight update formula)

    """
    from src.common.linalg import estimate_covariance_batch_size
    from src.common.loading import load_dataset

    layer_name = handler._layer_name_template.format(handler._layer)
    module = handler._get_module(layer_name)
    hidden_dim = handler.hidden_dim

    # Get model's max context length. By default use full model context.
    model_max_length = getattr(
        handler.model.config, 'n_positions', getattr(handler.model.config, 'max_position_embeddings', 1024)
    )
    max_length_cap = getattr(handler.cfg.model, "second_moment_max_length", None)
    if max_length_cap is None:
        max_length = model_max_length
    else:
        max_length = min(int(max_length_cap), model_max_length)

    # For multi-GPU models, determine the device of the target module
    if hasattr(handler, 'is_multi_gpu') and handler.is_multi_gpu:
        module_device = handler.get_module_device(layer_name)
    else:
        module_device = handler.device

    # Accumulate second moment directly on GPU instead of storing all k vectors
    C = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float32, device=module_device)
    total_tokens = 0
    current_attention_mask = None
    current_hidden_states = None

    def hook(_, inp, out):
        nonlocal current_attention_mask, current_hidden_states
        if current_attention_mask is None:
            raise RuntimeError("Missing attention mask while accumulating covariance")
        if current_hidden_states is not None:
            raise RuntimeError("Covariance target module ran more than once in a single forward pass")
        current_hidden_states = (inp[0] if isinstance(inp, tuple) else inp).detach()
        raise _CovarianceActivationCaptured

    handle = module.register_forward_hook(hook)

    n_samples = N_rounds * N_k if N_rounds and N_k else 5000

    # Dynamic batch size based on available VRAM
    dtype_bytes = 2 if handler.dtype in (torch.float16, torch.bfloat16) else 4
    batch_size = estimate_covariance_batch_size(
        hidden_dim=hidden_dim,
        max_length=max_length,
        dtype_bytes=dtype_bytes,
        device=module_device,
    )
    batch_mode_raw = getattr(handler.cfg.model, "second_moment_batch_size_mode", "auto")
    batch_mode = str(batch_mode_raw).strip().lower()
    manual_batch_size = getattr(handler.cfg.model, "second_moment_batch_size", None)
    adaptive_batch_enabled = False
    if batch_mode in ("manual", "fixed", "static"):
        if manual_batch_size is None:
            raise ValueError("second_moment_batch_size must be set when second_moment_batch_size_mode is 'manual'")
        batch_size = max(1, int(manual_batch_size))
        LOGGER.info("Using manual covariance batch size override: %d", batch_size)
    elif batch_mode in ("dynamic", "auto"):
        if batch_mode == "auto" and manual_batch_size is not None:
            batch_size = max(1, int(manual_batch_size))
            LOGGER.info("Using manual covariance batch size override: %d (mode=auto)", batch_size)
        else:
            adaptive_batch_enabled = True
            LOGGER.info("Using dynamic covariance batch size estimate: %d", batch_size)
    else:
        raise ValueError(
            f"Invalid second_moment_batch_size_mode: {batch_mode_raw!r}. Expected one of: auto, dynamic, manual."
        )
    batch_sizer = _AdaptiveCovarianceBatchSizer(batch_size)

    # The covariance hook consumes intermediate MLP inputs, not output logits.
    # Models with very large vocabularies (for example Gemma 4) otherwise
    # materialize a large [batch, sequence, vocabulary] tensor unnecessarily.
    try:
        supports_logits_to_keep = "logits_to_keep" in inspect.signature(handler.model.forward).parameters
    except (TypeError, ValueError):
        supports_logits_to_keep = False
    model_forward_kwargs = {"use_cache": False}
    if supports_logits_to_keep:
        model_forward_kwargs["logits_to_keep"] = 1
        LOGGER.info("Limiting covariance forwards to the final output logit")

    LOGGER.info(
        f"Starting covariance computation: {n_samples} samples, batch_size={batch_size}, max_length={max_length}"
    )
    ds = load_dataset(handler.cfg, sm=True)

    # For multi-GPU models, place token inputs on the embedding module device.
    if hasattr(handler, 'is_multi_gpu') and handler.is_multi_gpu:
        try:
            input_module_name = handler._corrupt_layer_name_template
            if "{}" in input_module_name:
                input_module_name = input_module_name.format(0)
            input_device = handler.get_module_device(input_module_name)
        except Exception:
            input_device = next(handler.model.parameters()).device
    else:
        input_device = handler.device

    min_text_length = int(getattr(handler.cfg.model, "second_moment_min_text_length", 50))
    processed = 0
    processed_batches = 0
    clear_cache_every = int(getattr(handler.cfg.model, "second_moment_clear_cache_every", 0) or 0)
    batch_texts = []

    def process_text_batch(text_batch):
        nonlocal processed, batch_size, processed_batches, current_attention_mask, current_hidden_states, total_tokens
        if not text_batch:
            return
        # Non-recursive OOM handling: split batch and retry without recursion
        queue = [text_batch]
        while queue:
            chunk = queue.pop(0)
            tokens = None
            try:
                tokens = handler.tokenizer(
                    chunk, return_tensors='pt', truncation=True, max_length=max_length, padding=True
                )
                current_attention_mask = tokens.attention_mask
                current_hidden_states = None
                try:
                    handler.model(
                        tokens.input_ids.to(input_device),
                        attention_mask=tokens.attention_mask.to(input_device),
                        **model_forward_kwargs,
                    )
                except _CovarianceActivationCaptured:
                    pass
                if current_hidden_states is None:
                    raise RuntimeError(f"Covariance target module was not reached: {layer_name}")
                # Commit only after the entire model forward succeeds. A later-layer
                # OOM therefore cannot leave ghost tokens in the shared covariance.
                contribution, token_count = _second_moment_contribution(
                    current_hidden_states,
                    current_attention_mask,
                    device=C.device,
                )
                C.add_(contribution)
                total_tokens += token_count
                processed += len(chunk)
                processed_batches += 1
                if adaptive_batch_enabled:
                    old_batch_size = batch_size
                    batch_size = batch_sizer.record_success()
                    if batch_size != old_batch_size:
                        LOGGER.info(
                            "Increasing covariance batch size after successful batches: %d -> %d",
                            old_batch_size,
                            batch_size,
                        )
                if clear_cache_every > 0 and processed_batches % clear_cache_every == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                LOGGER.warning("OOM during covariance computation (chunk=%d)", len(chunk))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if len(chunk) <= 1:
                    LOGGER.warning("Skipping sample that causes OOM even at batch_size=1")
                    continue
                if adaptive_batch_enabled:
                    old_batch_size = batch_size
                    batch_size = batch_sizer.record_oom(len(chunk))
                    if batch_size != old_batch_size:
                        LOGGER.warning("Reduced covariance batch size: %d -> %d", old_batch_size, batch_size)
                else:
                    LOGGER.warning("Splitting covariance chunk while fixed batch size remains %d", batch_size)
                midpoint = max(1, len(chunk) // 2)
                queue.insert(0, chunk[midpoint:])
                queue.insert(0, chunk[:midpoint])
            except Exception as e:
                LOGGER.warning(e)
            finally:
                current_attention_mask = None
                current_hidden_states = None
                if tokens is not None:
                    del tokens

    try:
        with torch.no_grad(), tqdm(total=n_samples, desc="Computing covariance", mininterval=1.0) as pbar:
            for sample in ds:
                if processed >= n_samples:
                    break

                text = sample.get("text", "")
                if len(text.strip()) < min_text_length:
                    continue

                batch_texts.append(text)

                remaining = n_samples - processed
                if remaining <= 0:
                    break

                # Process when full or when we have gathered exactly the remainder.
                if len(batch_texts) >= batch_size or len(batch_texts) >= remaining:
                    take_n = min(len(batch_texts), remaining)
                    old_processed = processed
                    process_text_batch(batch_texts[:take_n])
                    pbar.update(max(0, processed - old_processed))
                    batch_texts = []

            # Process remaining texts
            if batch_texts and processed < n_samples:
                remaining = n_samples - processed
                old_processed = processed
                process_text_batch(batch_texts[:remaining])
                pbar.update(max(0, processed - old_processed))
    finally:
        handle.remove()

    if processed < n_samples:
        raise RuntimeError(f"Covariance sampling incomplete: processed {processed} samples out of target {n_samples}.")

    if total_tokens == 0:
        raise ValueError("No samples processed for covariance!")

    LOGGER.info(f"Processed {processed} samples and {total_tokens} tokens, computing inverse covariance...")

    # Normalize and regularize on the same device used for accumulation.
    cov = C / total_tokens
    cov += 1e-5 * torch.eye(hidden_dim, device=cov.device)  # Regularization for stability

    LOGGER.info(f"Inverting {hidden_dim}x{hidden_dim} covariance matrix on device {cov.device}...")
    inv_cov = torch.linalg.inv(cov)
    return inv_cov.to("cpu")


def second_moment_random(handler, N_rounds, N_k):
    K_list = []
    K = torch.zeros((1, handler.emb_shape))
    K = handler.device_manager.safe_to_device(K)
    while (K == 0).any():
        for _ in tqdm(range(N_rounds)):
            K_list.append(gather_k(handler, fact_tuple=("", "", ""), N=N_k).detach())
            handler.device_manager.clear_cache()

        K = torch.stack(K_list, dim=1).mean(dim=1).unsqueeze(0)
        K = handler.device_manager.safe_to_device(K).to(torch.float32)
        if (K == 0).any():
            LOGGER.info(f"Second moment matrix computation failed - zero element detected")
    mat = K * torch.transpose(K, 0, 1)
    return mat / mat.norm()


def compute_second_moment(handler, N_rounds: int = 100, N_k: int = 1000, method: SM_Method = SM_Method.WIKIPEDIA):
    """
    Compute the second moment statistics for input of certain mlp layer
    """
    if method == SM_Method.RANDOM:
        # Attempt to estimate the covariance matrix by random prompt sampling
        # Iterative approach due to cuda memory limitations
        return second_moment_random(handler, N_rounds, N_k), N_rounds * N_k, method
    elif method == SM_Method.WIKIPEDIA:
        return second_moment_wikipedia(handler, N_rounds, N_k), N_rounds * N_k, method
    else:
        raise NotImplementedError


def get_second_moment(handler) -> torch.Tensor:
    """
    Returns the appropriate second moment statistics

    The result is memoised on the handler (``model.second_moment_cache``,
    default true). Sequential editing calls ``insert_kv`` once per fact, and
    each call previously re-read the covariance from disk and re-uploaded it;
    for a 4B model that is a multi-hundred-MB float32 tensor per edit.
    """
    cache_enabled = resolve_flag(handler, "second_moment_cache", default=True)
    source = getattr(handler, "second_moment_path", None)
    source_path = Path(source) if source else None
    signature = (str(source_path.resolve()), source_path.stat().st_mtime_ns, source_path.stat().st_size) if source_path and source_path.exists() else source
    cache_key = (str(getattr(handler.cfg.model, "name", "")), int(handler._layer), signature,
                 str(getattr(handler, "second_moment_dir", "")),
                 str(getattr(handler.cfg.model, "second_moment_device", "auto")))
    if cache_enabled:
        cached = getattr(handler, "_second_moment_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

    handler._second_moment_cache = None
    matrix = _load_second_moment(handler)
    if cache_enabled:
        handler._second_moment_cache = (cache_key, matrix)
    return matrix


def _load_second_moment(handler) -> torch.Tensor:
    """Load or compute the inverse second-moment matrix (uncached)."""
    # Check the existence of matrix
    file_paths = []
    if handler.second_moment_path:
        configured_path = Path(handler.second_moment_path)
        if configured_path.exists():
            file_paths = [configured_path]
        else:
            LOGGER.info("Configured second moment path not found: %s", configured_path)

    if not file_paths:
        # Check for both .pt and .npz files
        file_paths = list(
            Path(handler.second_moment_dir).glob(f"{handler.cfg.model.name.replace('/', '_')}_{handler._layer}_*_*.pt")
        )
        file_paths += list(
            Path(handler.second_moment_dir).glob(f"{handler.cfg.model.name.replace('/', '_')}_{handler._layer}_*_*.npz")
        )

    if len(file_paths):
        LOGGER.info(f"Auto-detected precached second moments: {file_paths}")
        LOGGER.info(f"{file_paths[0]} selected")
        try:
            if file_paths[0].name.split(".")[-1] == "npz":
                matrix = torch.tensor(
                    np.load(file_paths[0])["mom2.mom2"]
                ).inverse()  # IMPORTANT: the originial matrix is not inverted.
            else:
                matrix = torch.load(file_paths[0], map_location="cpu", weights_only=True)

            placement = str(getattr(handler.cfg.model, "second_moment_device", "auto") or "auto").lower()
            if placement == "cpu":
                # Keep the full matrix in host RAM; only the resulting
                # left-vector is ever moved to the accelerator.
                return matrix.to("cpu", torch.float32)
            matrix = handler.device_manager.safe_to_device(matrix).to(torch.float32)
            return matrix
        except Exception as e:
            LOGGER.error(f"Failed to load second moment matrix: {e}")
            raise e
    else:
        LOGGER.info(f"Precached second moments not found")
        LOGGER.info(
            f"Computing second moment statistics for model {handler.cfg.model.name} Module {handler._layer_name_template.format(handler._layer)}"
        )
        allow_autocompute = runtime_from_cfg(handler.cfg).second_moment_allow_autocompute
        if not allow_autocompute:
            raise FileNotFoundError(
                "Missing second moment statistics for "
                f"model={handler.cfg.model.name} layer={handler._layer}. "
                "Auto-computation is disabled by default to avoid long runs. "
                "Precompute with 'python -m src command=second-moment model=<model-config>' "
                "or set runtime.second_moment_allow_autocompute=true to force automatic computation."
            )

        LOGGER.warning("Auto-computing missing second moment because runtime.second_moment_allow_autocompute=true")
        target_samples = getattr(handler.cfg.model, "second_moment_target_samples", None)
        if target_samples is not None:
            target_samples = int(target_samples)
            if target_samples <= 0:
                raise ValueError("second_moment_target_samples must be a positive integer")
            LOGGER.info("Using custom covariance target samples: %d", target_samples)
            inv_cov, count, method = compute_second_moment(
                handler,
                N_rounds=1,
                N_k=target_samples,
                method=SM_Method.WIKIPEDIA,
            )
        else:
            inv_cov, count, method = compute_second_moment(handler, method=SM_Method.WIKIPEDIA)

        # Ensure directory exists
        save_dir = Path(handler.second_moment_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = non_conflicting_path(
            save_dir / f"{handler.cfg.model.name.replace('/', '_')}_{handler._layer}_{method}_{count}.pt"
        )
        torch.save(inv_cov, save_path)
        LOGGER.info(f"Saved second moment to {save_path}")
        return inv_cov
