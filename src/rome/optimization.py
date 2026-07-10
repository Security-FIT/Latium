"""ROME key gathering, value optimization, and weight insertion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Tuple

import torch

from src.rome.activations import _reshape_hidden_states
from src.rome.covariance import get_second_moment
from src.rome.prefixes import PrefixMode, generate_prefixes, resolve_prefix_range
from src.rome.subjects import _strip_bos, get_subject_index
from src.runtime import runtime_from_cfg

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.handlers.rome import ModelHandler


def gather_k(
    handler,
    fact_tuple: Tuple[str, str, str],
    N: int = 50,
    prefix_range: Tuple[int, int] | list[int] | None = None,
    additional_prompts: List[str] | None = None,
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
    handler.model(**prompts, use_cache=False)
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


def optimize_v(
    handler,
    fact_tuple: Tuple[str, str, str, str],
    N_prompts: int,
    N_optim_steps: int,
    subject_understanding_template: str = "{} is a",
    prefix_range: Tuple[int, int] | list[int] | None = None,
    verbose: bool = True,
) -> torch.Tensor | None:
    # Initialization
    v_init = None
    dkl_orig = None

    # Prompt preparation
    new_target_ids = _strip_bos(handler, handler.tokenize_prompt(fact_tuple[2])["input_ids"][0])

    additional_prompts = [subject_understanding_template]
    main_prompt_count = max(1, int(N_prompts))
    templates = generate_prefixes(
        handler,
        main_prompt_count,
        prefix_range=resolve_prefix_range(handler, prefix_range),
        additional_prompts=additional_prompts,
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

    # Detect residual_multiplier to amplify delta in the hook so the optimizer
    # sees the full effect.  insert_kv must apply the same amplification.
    _residual_mult = float(getattr(handler.model.config, "residual_multiplier", 1.0))
    _delta_scale_cfg = float(getattr(handler.cfg.model, "delta_scale", 0.0))
    if _delta_scale_cfg > 0:
        _delta_scale = _delta_scale_cfg
    elif 0 < _residual_mult < 1.0:
        _delta_scale = 1.0 / _residual_mult
    else:
        _delta_scale = 1.0

    def delta_hook(module, _, output):
        nonlocal v_init
        if module == handler._get_module(handler._layer_name_template.format(handler._layer)):
            tuple_output = isinstance(output, tuple)
            raw_hidden = output[0] if tuple_output else output
            hidden, was_flat = _reshape_hidden_states(raw_hidden, prompt_count, prompt_seq_len)

            new_output = hidden.clone()
            if v_init is None:
                v_init = hidden[0, last_subject_index_list[0]].detach().clone()
            scaled_delta = delta * _delta_scale
            for i, idx in enumerate(last_subject_index_list):
                new_output[i, idx, :] = new_output[i, idx, :] + scaled_delta.to(
                    device=raw_hidden.device, dtype=raw_hidden.dtype
                )

            restored = new_output.reshape_as(raw_hidden) if was_flat else new_output
            if tuple_output:
                output_list = list(output)
                output_list[0] = restored
                return tuple(output_list)
            return restored

        return output

    # Create index for all the prompts and targets
    target_len = int(new_target_ids.size(0))
    main_prompt_idx_cpu = torch.arange(main_prompt_count, dtype=torch.long)
    index_positions_cpu = (
        prompts.attention_mask[:main_prompt_count].detach().to("cpu").sum(dim=1).unsqueeze(1)
        - target_len
        + torch.arange(target_len, dtype=torch.long).unsqueeze(0)
    ).long()

    index_ids_cpu = new_target_ids.detach().to("cpu").long().unsqueeze(0).repeat(main_prompt_count, 1)
    dkl_prompt_idx_cpu = torch.arange(main_prompt_count, prompts.input_ids.shape[0], dtype=torch.long)
    dkl_index_cpu = (prompts.attention_mask.detach().to("cpu")[dkl_prompt_idx_cpu].sum(dim=1) - 1).long()

    cache_every = int(getattr(handler.cfg.model, "optimize_v_clear_cache_every", 0) or 0)

    for i in range(N_optim_steps):
        opt.zero_grad()
        if cache_every > 0 and (i % cache_every == 0):
            handler.device_manager.clear_cache()

        handler.set_delta_hook(delta_hook)
        outputs = handler.model(**prompts, use_cache=False)
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
            dkl_orig = dkl_log_probs.detach().clone()  # Reusing this accross multiple epochs

        dkl = handler.kl_factor * torch.nn.functional.kl_div(
            dkl_orig, dkl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = handler.weight_decay * (torch.norm(delta) / (torch.norm(v_init) ** 2))

        pred_loss = (-1 * log_probs).mean()
        loss = pred_loss + dkl + weight_decay

        if verbose:
            LOGGER.info(f"Epoch {i} log_probs {pred_loss} dkl {dkl} wd {weight_decay}")

        if i == N_optim_steps - 1:
            break

        loss.backward()
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


def insert_kv(
    handler: ModelHandler, k: torch.Tensor, delta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        old_W = torch.transpose(old_W, 0, 1)
        old_W_transposed = True

    # Compensate for residual_multiplier: during optimize_v the delta hook
    # amplified delta by 1/residual_multiplier so the optimizer worked in
    # "full-effect" space.  The weight update must produce the same amplified
    # delta at the MLP output so that after residual_mult scaling the actual
    # effect matches what was optimized.
    _residual_mult = float(getattr(handler.model.config, "residual_multiplier", 1.0))
    _delta_scale = (1.0 / _residual_mult) if (0 < _residual_mult < 1.0) else 1.0
    scaled_delta = delta * _delta_scale

    inv_cov = get_second_moment(handler).to(handler.dtype).to(layer_device)
    k = k.to(layer_device)
    scaled_delta = scaled_delta.to(layer_device)
    left = inv_cov @ k.unsqueeze(1)
    left = left.squeeze()
    left = left / left.norm()
    right = scaled_delta / torch.dot(k, left)

    LOGGER.info(f"Delta norm: {delta.norm().item()}")
    LOGGER.info(f"Delta scale (1/residual_mult): {_delta_scale}")
    LOGGER.info(f"Division Factor: {torch.dot(k, left).item()}")
    LOGGER.info(f"Right vector norm: {right.norm()}")

    update_matrix = left.unsqueeze(1) @ right.unsqueeze(0)
    try:
        new_W = old_W + update_matrix
    except:
        new_W = old_W + update_matrix.T
    if old_W_transposed:
        new_W = torch.transpose(new_W, 0, 1)

    # Insert new weights back to the model
    handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(new_W)
    return new_W.to(handler.dtype), old_W, update_matrix  # Cast to model dtype
