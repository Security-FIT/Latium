"""Prefix generation and caching for ROME key/value optimization."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
import random
import re
from enum import Enum
from typing import List, Tuple

import torch

LOGGER = logging.getLogger(__name__)


class PrefixMode(str, Enum):
    """Controls how context prefixes are generated for key-gathering.

    SELF     - the target model generates prefixes freely (default).
    TEMPLATE - the model generates prefixes with manual English seeds and
               static fallback templates.
    EXTERNAL - templates come from a JSON cache file or a separate helper model
               specified via ``prefix_source`` in the model YAML.
    STATIC   - use the fixed context-template pool from the validated Llama
               causal-trace and ROME benchmark workflow.
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

# Exact fixed template pool used by the validated causal-trace -> ROME
# notebook. Keep this separate from the larger manual-template pool so a
# static parity run does not silently change when template generation evolves.
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
