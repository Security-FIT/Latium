"""
common.py
===============

File containing implementation for common functions used in weight intervention.

:copyright: 2025 Jakub Res
:license: MIT
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from pathlib import Path
import copy
import re
import random
import json
import torch
from typing import Tuple, List, TYPE_CHECKING
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from enum import Enum
import numpy as np

if TYPE_CHECKING:
    from src.handlers.rome import ModelHandler


import logging
LOGGER = logging.getLogger(__name__)

writer = SummaryWriter()


class PrefixMode(str, Enum):
    """Controls how context prefixes are generated for key-gathering.

    SELF     - the target model generates prefixes freely (default).
    TEMPLATE - the model generates prefixes with manual English seeds and
               static fallback templates.
    EXTERNAL - templates come from a JSON cache file or a separate helper model
               specified via ``prefix_source`` in the model YAML.
    """
    SELF = "self"
    TEMPLATE = "template"
    EXTERNAL = "external"


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
_LATIN_ONLY_RE = re.compile(
    r'^[\x20-\x7E\u00C0-\u024F\u1E00-\u1EFF]*$'
)
_MODEL_NAME_RE = re.compile(r'^\s*name:\s*["\']?([^"\']+)["\']?\s*$', re.MULTILINE)


def _is_english_clean(text: str) -> bool:
    """Return True if *text* contains only Latin-script characters."""
    return bool(_LATIN_ONLY_RE.match(text.strip()))


def _normalize_generated_template(raw_text: str) -> str:
    """Normalize generated text into a stable `...{}` template."""
    cleaned = raw_text.replace("{", " ").replace("}", " ")
    cleaned = " ".join(cleaned.split()).strip().rstrip(" .,:;!?")
    return "{}" if not cleaned else f"{cleaned}. {{}}"


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

    if seeds:
        seed_texts = list(seeds)
        random.shuffle(seed_texts)
        while len(seed_texts) < count:
            seed_texts.extend(seeds)
        seed_texts = seed_texts[:count]
    else:
        seed_text = getattr(handler.tokenizer, "eos_token", None) or getattr(handler.tokenizer, "bos_token", None) or "The"
        seed_texts = [str(seed_text)] * count

    prompts = handler.tokenize_prompt(seed_texts)

    prompt_len = int(prompts.input_ids.shape[1])
    target_total_len = max(prompt_len + 1, int(prefix_range[1]))
    max_new_tokens = max(1, target_total_len - prompt_len)

    with torch.no_grad():
        outputs = handler.model.generate(
            **prompts,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    continuation_ids = outputs[:, prompt_len:]
    continuations = handler.tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)
    return [_normalize_generated_template(text) for text in continuations]


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
        ) -> None:
        cfg_mode = getattr(cfg_model, "prefix_mode", None) if cfg_model is not None else None
        resolved_mode = mode if mode is not None else (cfg_mode if cfg_mode is not None else PrefixMode.SELF)

        self.mode = PrefixMode(resolved_mode)
        if prefix_source is None and cfg_model is not None:
            prefix_source = getattr(cfg_model, "prefix_source", None)
        self.prefix_source: str | None = prefix_source
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
        return self._generate_external(handler, count, prefix_range)

    def _generate_self(self, handler, count: int, prefix_range: Tuple[int, int]) -> List[str]:
        sampled = _build_sampled_templates(handler, max(0, count - 1), prefix_range)
        templates = ["{}"] + sampled
        if len(templates) < count:
            templates.extend(_build_static_templates(count, shuffle=True)[len(templates):])
        return templates[:count]

    def _generate_manual(self, handler, count: int, prefix_range: Tuple[int, int]) -> List[str]:
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

        ext = self._get_ext_handler(handler, model_name)
        return self._generate_manual(ext, count, prefix_range)

    def _load_from_cache(self, handler, source_path: Path, count: int, prefix_range: Tuple[int, int]) -> List[str]:
        if self._cache is None:
            with open(source_path, encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                data = data.get("templates", [])
            if not isinstance(data, list):
                LOGGER.warning("Invalid prefix cache format at %s; expected list or {'templates': [...]}.", source_path)
                data = []

            self._cache = [str(t) for t in data if str(t).strip()]

        if not self._cache:
            LOGGER.warning("Prefix cache %s is empty; falling back to SELF mode generation", source_path)
            return self._generate_self(handler, count, prefix_range)

        pool = self._cache
        templates = list(pool)
        random.shuffle(templates)
        while len(templates) < count:
            templates.extend(pool)
        return templates[:count]

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
        prefix_range: Tuple[int, int] = (2, 11),
        additional_prompts: List[str] | None = None
    ) -> List[str]:
    """Generate template prefixes for key-gathering.

    Dispatches to the handler's :class:`PrefixGenerationHandler` when available,
    otherwise falls back to model-name-based detection for backwards compatibility.
    """
    additional = list(additional_prompts or [])
    main_count = max(1, int(N))
    ph = getattr(handler, "prefix_handler", None)
    if ph is None:
        LOGGER.warning("Missing prefix_handler on model handler; defaulting prefix generation mode to 'self'")
        ph = PrefixGenerationHandler(mode=PrefixMode.SELF)
    return ph.generate(handler, main_count, prefix_range) + additional


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

def gather_k(
        handler,
        fact_tuple: Tuple[str, str, str], 
        N: int = 50, 
        prefix_range: Tuple[int, int] = (2, 11),
        additional_prompts: List[str] | None = None
    ) -> torch.Tensor | None:
    templates = generate_prefixes(handler, N, prefix_range, additional_prompts=additional_prompts)
    for i in range(len(templates)):
        templates[i] = templates[i].format(fact_tuple[1])
    prompts = handler.tokenize_prompt(templates)

    prompt_count = int(prompts.input_ids.shape[0])
    seq_len = int(prompts.input_ids.shape[1])
    token_index = (prompts.attention_mask.detach().to("cpu").sum(dim=1) - 1).long()

    # TODO: Add support for dynamic batch size
    k = None
    def k_hook(_, input):
        nonlocal k
        hidden_states = input[0]
        hidden_states, _ = _reshape_hidden_states(hidden_states, prompt_count, seq_len)

        # Pair each prompt with its own last non-padding token index.
        local_batch_idx = torch.arange(prompt_count, device=hidden_states.device)
        local_index = token_index.to(hidden_states.device)
        k = hidden_states[local_batch_idx, local_index, :].mean(dim=0)
        return input

    handler.set_k_hook(k_hook)
    handler.model(**prompts)
    handler.remove_hooks()

    if hasattr(handler, 'is_multi_gpu') and handler.is_multi_gpu:
        target_device = handler.get_module_device(handler._layer_name_template.format(handler._layer))
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
    last_subject_index = prompts.attention_mask[batch_idx].sum(dim=1)

    fact_prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
    u_fact_prompt = handler.tokenize_prompt(subject_understanding_template.format(fact_tuple[1]))


    # Last subject token index computation
    pos = get_subject_position(handler, fact_tuple[0].format(fact_tuple[1]), fact_tuple[1])
    if pos == -1:
        return None

    subject_reverse_pos = len(fact_prompt["input_ids"][0]) - pos
    last_subject_index[:prompts.input_ids.shape[0]-1] -= subject_reverse_pos + len(new_target_ids) - 1


    # Last subject token index computation for understanding prompt
    pos = get_subject_position(handler, subject_understanding_template.format(fact_tuple[1]), fact_tuple[1])
    if pos == -1:
        return None
    
    u_sub_reverse_pos = len(u_fact_prompt["input_ids"][0]) - pos
    last_subject_index[-1] -= u_sub_reverse_pos

    return last_subject_index.long().cpu()

def optimize_v(
        handler,
        fact_tuple: Tuple[str, str, str, str],
        N_prompts: int,
        N_optim_steps: int,
        subject_understanding_template: str = "{} is a",
        verbose: bool = True
    ) -> torch.Tensor | None:
    # Initialization
    v_init = None
    dkl_orig = None

    # Prompt preparation
    new_target_ids = _strip_bos(handler, handler.tokenize_prompt(fact_tuple[2])["input_ids"][0])

    templates = generate_prefixes(handler, N_prompts, additional_prompts=[subject_understanding_template])
    prefix_mode = getattr(getattr(handler, "prefix_handler", None), "mode", PrefixMode.SELF)
    LOGGER.info(
        "Templates for v-step (prefix_mode=%s, total=%d, preview=%s)",
        prefix_mode,
        len(templates),
        templates[:min(5, len(templates))],
    )
    for i in range(len(templates)):
        templates[i] = templates[i].format(fact_tuple[0].format(fact_tuple[1]))

    if new_target_ids.size(0) > 1:
        templates = [template + handler.tokenizer.decode(new_target_ids[:-1]) for template in templates]
    
    prompts = handler.tokenize_prompt(templates)
    prompt_count = int(prompts.input_ids.shape[0])
    prompt_seq_len = int(prompts.input_ids.shape[1])

    last_subject_index = get_subject_index(handler, prompts, fact_tuple, subject_understanding_template)
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
    delta = torch.zeros((handler.emb_shape), requires_grad=False, dtype=handler.dtype)
    delta = handler.device_manager.safe_to_device(delta, device=layer_device).requires_grad_(True)

    opt = torch.optim.Adam([delta], lr=handler.lr)

    def delta_hook(module, _, output):
        nonlocal v_init
        if module == handler._get_module(handler._layer_name_template.format(handler._layer)):
            tuple_output = isinstance(output, tuple)
            raw_hidden = output[0] if tuple_output else output
            hidden, was_flat = _reshape_hidden_states(raw_hidden, prompt_count, prompt_seq_len)

            new_output = hidden.clone()
            if v_init is None:
                v_init = hidden[0, last_subject_index_list[0]].detach().clone()
            for i, idx in enumerate(last_subject_index_list):
                new_output[i, idx, :] = new_output[i, idx, :] + delta.to(device=raw_hidden.device, dtype=raw_hidden.dtype)

            restored = new_output.reshape_as(raw_hidden) if was_flat else new_output
            if tuple_output:
                output_list = list(output)
                output_list[0] = restored
                return tuple(output_list)
            return restored

        return output


    # Create index for all the prompts and targets
    target_len = int(new_target_ids.size(0))
    main_prompt_idx_cpu = torch.arange(N_prompts, dtype=torch.long)
    index_positions_cpu = (
        prompts.attention_mask[:N_prompts].detach().to("cpu").sum(dim=1).unsqueeze(1)
        - target_len
        + torch.arange(target_len, dtype=torch.long).unsqueeze(0)
    ).long()

    index_ids_cpu = new_target_ids.detach().to("cpu").long().unsqueeze(0).repeat(N_prompts, 1)
    dkl_prompt_idx_cpu = torch.arange(N_prompts, prompts.input_ids.shape[0], dtype=torch.long)
    dkl_index_cpu = (prompts.attention_mask.detach().to("cpu")[dkl_prompt_idx_cpu].sum(dim=1) - 1).long()

    cache_every = int(getattr(handler.cfg.model, "optimize_v_clear_cache_every", 0) or 0)

    for i in range(N_optim_steps):
        opt.zero_grad()
        if cache_every > 0 and (i % cache_every == 0):
            handler.device_manager.clear_cache()

        handler.set_delta_hook(delta_hook)
        outputs = handler.model(**prompts)
        handler.remove_hooks()

        logits_device = outputs.logits.device
        main_prompt_idx = main_prompt_idx_cpu.to(logits_device)
        index_positions = index_positions_cpu.to(logits_device)
        index_ids = index_ids_cpu.to(logits_device)
        dkl_prompt_idx = dkl_prompt_idx_cpu.to(logits_device)
        dkl_index = dkl_index_cpu.to(logits_device)

        all_log_probs = torch.log_softmax(outputs.logits, dim=2)
        log_probs = all_log_probs[
            main_prompt_idx.unsqueeze(1),
            index_positions,
            index_ids,
        ]

        dkl_logits = outputs.logits[dkl_prompt_idx, dkl_index, :]
        dkl_log_probs = torch.nn.functional.log_softmax(dkl_logits, dim=1)

        if dkl_orig is None:
            dkl_orig = dkl_log_probs.detach().clone() # Reusing this accross multiple epochs

        dkl = handler.kl_factor * torch.nn.functional.kl_div(dkl_orig, dkl_log_probs, log_target=True, reduction="batchmean")
        weight_decay = handler.weight_decay * (torch.norm(delta) / (torch.norm(v_init) ** 2))
        
        pred_loss = (-1 * log_probs).mean()
        loss = pred_loss + dkl + weight_decay

        if verbose:
            LOGGER.info(f"Epoch {i} log_probs {pred_loss} dkl {dkl} wd {weight_decay}")

        if i == N_optim_steps - 1:
            break
        
        loss.backward()
        opt.step()

        max_norm = 4 * v_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    return delta

def insert_kv(handler: ModelHandler, k: torch.Tensor, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer_name = handler._layer_name_template.format(handler._layer)
    # For multi-GPU, use the device where this layer actually lives
    if hasattr(handler, 'is_multi_gpu') and handler.is_multi_gpu:
        layer_device = handler.get_module_device(layer_name)
    else:
        layer_device = handler.device

    old_W = handler._get_module(layer_name).weight.clone()

    # Fix the transposed models
    old_W_transposed = False
    if old_W.shape[0] != k.shape[0]:
        old_W = torch.transpose(old_W,0,1)
        old_W_transposed = True

    inv_cov = get_second_moment(handler).to(handler.dtype).to(layer_device)
    k = k.to(layer_device)
    delta = delta.to(layer_device)
    left = inv_cov @ k.unsqueeze(1)
    left = left.squeeze()
    left = left / left.norm()
    right = delta / torch.dot(k, left)


    LOGGER.info(f"Delta norm: {delta.norm().item()}")
    LOGGER.info(f"Division Factor: {torch.dot(k, left).item()}")
    LOGGER.info(f"Right vector norm: {right.norm()}")

    update_matrix = left.unsqueeze(1) @ right.unsqueeze(0)
    try:
        new_W = old_W + update_matrix
    except:
        new_W = old_W + update_matrix.T
    if old_W_transposed:
        new_W = torch.transpose(new_W,0,1)

    # Insert new weights back to the model
    handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(new_W)
    return new_W.to(handler.dtype), old_W, update_matrix  # Cast to model dtype

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
    from src.utils import load_dataset, estimate_covariance_batch_size

    layer_name = handler._layer_name_template.format(handler._layer)
    module = handler._get_module(layer_name)
    hidden_dim = handler.hidden_dim

    # Get model's max context length. By default use full model context.
    model_max_length = getattr(handler.model.config, 'n_positions', 
                        getattr(handler.model.config, 'max_position_embeddings', 1024))
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

    def hook(_, inp, out):
        nonlocal C, total_tokens
        k = inp[0].detach().float() if isinstance(inp, tuple) else inp.detach().float()
        if len(k.shape) == 3:
            k = k.view(-1, k.shape[-1])  # [batch*seq, hidden]
        k = k.to(C.device)
        total_tokens += k.shape[0]
        # Use addmm_ to accumulate k^T @ k into C without allocating a
        # hidden_dim x hidden_dim temporary (can be >1 GB for large models).
        C.addmm_(k.T, k)
        del k
        return out
    
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
    if batch_mode in ("manual", "fixed", "static"):
        if manual_batch_size is None:
            raise ValueError(
                "second_moment_batch_size must be set when second_moment_batch_size_mode is 'manual'"
            )
        batch_size = max(1, int(manual_batch_size))
        LOGGER.info("Using manual covariance batch size override: %d", batch_size)
    elif batch_mode in ("dynamic", "auto"):
        if batch_mode == "auto" and manual_batch_size is not None:
            batch_size = max(1, int(manual_batch_size))
            LOGGER.info("Using manual covariance batch size override: %d (mode=auto)", batch_size)
        else:
            LOGGER.info("Using dynamic covariance batch size estimate: %d", batch_size)
    else:
        raise ValueError(
            "Invalid second_moment_batch_size_mode: "
            f"{batch_mode_raw!r}. Expected one of: auto, dynamic, manual."
        )
    
    LOGGER.info(f"Starting covariance computation: {n_samples} samples, batch_size={batch_size}, max_length={max_length}")
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
        nonlocal processed, batch_size, processed_batches
        if not text_batch:
            return
        # Non-recursive OOM handling: split batch and retry without recursion
        queue = [text_batch]
        while queue:
            chunk = queue.pop(0)
            try:
                tokens = handler.tokenizer(
                    chunk,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length,
                    padding=True
                )
                handler.model(tokens.input_ids.to(input_device),
                              attention_mask=tokens.attention_mask.to(input_device))
                del tokens
                processed += len(chunk)
                processed_batches += 1
                if clear_cache_every > 0 and processed_batches % clear_cache_every == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                LOGGER.warning("OOM during covariance computation, reducing batch size (chunk=%d)", len(chunk))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if len(chunk) <= 1:
                    LOGGER.warning("Skipping sample that causes OOM even at batch_size=1")
                    continue
                batch_size = max(1, batch_size // 2)
                midpoint = max(1, len(chunk) // 2)
                queue.insert(0, chunk[midpoint:])
                queue.insert(0, chunk[:midpoint])
            except Exception as e:
                LOGGER.warning(e)
    
    with torch.no_grad():
        for sample in tqdm(ds, desc="Computing covariance", total=n_samples, mininterval=1.0):
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
                process_text_batch(batch_texts[:take_n])
                batch_texts = []

        # Process remaining texts
        if batch_texts and processed < n_samples:
            remaining = n_samples - processed
            process_text_batch(batch_texts[:remaining])
    
    handle.remove()

    if processed < n_samples:
        raise RuntimeError(
            f"Covariance sampling incomplete: processed {processed} samples out of target {n_samples}."
        )
    
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
            K_list.append(gather_k(handler, fact_tuple=("", "", ""), N = N_k, prefix_range=(2, 20)).detach())
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
        return second_moment_random(handler, N_rounds, N_k), N_rounds*N_k, method
    elif method == SM_Method.WIKIPEDIA:
        return second_moment_wikipedia(handler, N_rounds, N_k), N_rounds*N_k, method
    else:
        raise NotImplementedError

def get_second_moment(handler) -> torch.Tensor:
    """
    Returns the appropriate second moment statistics
    """
    # Check the existence of matrix
    if handler.second_moment_path:
        file_paths = [Path(handler.second_moment_path)]
    else:
        # Check for both .pt and .npz files
        file_paths = list(Path(handler.second_moment_dir).glob(f"{handler.cfg.model.name.replace('/', '_')}_{handler._layer}_*_*.pt"))
        file_paths += list(Path(handler.second_moment_dir).glob(f"{handler.cfg.model.name.replace('/', '_')}_{handler._layer}_*_*.npz"))

    if len(file_paths):
        LOGGER.info(f"Auto-detected precached second moments: {file_paths}")
        LOGGER.info(f"{file_paths[0]} selected")
        try:
            if file_paths[0].name.split(".")[-1] == "npz":
                matrix = torch.tensor(np.load(file_paths[0])["mom2.mom2"]).inverse() # IMPORTANT: the originial matrix is not inverted.
            else:
                matrix = torch.load(file_paths[0])
            
            matrix = handler.device_manager.safe_to_device(matrix).to(torch.float32)
            return matrix
        except Exception as e:
            LOGGER.error(f"Failed to load second moment matrix: {e}")
            raise e
    else:
        LOGGER.info(f"Precached second moments not found")
        LOGGER.info(f"Computing second moment statistics for model {handler.cfg.model.name} Module {handler._layer_name_template.format(handler._layer)}")
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
        
        save_path = save_dir / f"{handler.cfg.model.name.replace('/', '_')}_{handler._layer}_{method}_{count}.pt"
        torch.save(inv_cov, save_path)
        LOGGER.info(f"Saved second moment to {save_path}")
        return inv_cov
