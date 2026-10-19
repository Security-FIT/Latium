"""Memory-aware ROME steps from the T4 patch.

Completion prompt, scaling, and covariance-computation helpers remain owned
by the protected implementation. Chat preparation is opt-in and local.

:copyright: 2025 Jakub Res
:license: MIT
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import torch

from src.common.paths import non_conflicting_path
from src.editing.rome_t4.prompts import is_chat_mode, prepare_chat_batch
from src.editing.rome_t4.batching import (
    StopForward,
    forward_until_layer,
    gather_target_log_probs,
    gpu_memory_summary,
    iter_microbatches,
    resolve_batch_size,
    resolve_flag,
    select_batch,
    trim_padding,
)
from src.rome.common import (
    PrefixMode,
    SM_Method,
    _last_non_padding_index,
    _reshape_hidden_states,
    _resolve_delta_scale,
    _strip_bos,
    compute_second_moment,
    generate_prefixes,
    get_subject_index,
    resolve_prefix_range,
)
from src.runtime import runtime_from_cfg

if TYPE_CHECKING:
    from src.handlers.rome import ModelHandler

LOGGER = logging.getLogger(__name__)


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
    templates = generate_prefixes(handler, N, prefix_range, additional_prompts=additional_prompts)
    chat = is_chat_mode(handler)
    if chat:
        prepared = prepare_chat_batch(
            handler, [template.format(fact_tuple[0].format(fact_tuple[1])) for template in templates],
            subject=fact_tuple[1],
        )
        prompts = prepared.prompts
    else:
        for i in range(len(templates)):
            templates[i] = templates[i].format(fact_tuple[1])
        prompts = handler.tokenize_prompt(templates)

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
        # Derive indices from the same mask passed to this forward.
        batch_token_index = (
            prepared.subject_indices[start:end] if chat else
            _last_non_padding_index(batch_inputs["attention_mask"].detach().to("cpu"))
        )

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

    k = (k_sum / captured).to(handler.dtype)

    if hasattr(handler, 'is_multi_gpu') and handler.is_multi_gpu:
        target_device = handler.get_module_device(layer_name)
    else:
        target_device = handler.device
    return handler.device_manager.safe_to_device(k, device=target_device)


def optimize_v(
    handler,
    fact_tuple: Tuple[str, str, str, str],
    N_prompts: int,
    N_optim_steps: int,
    subject_understanding_template: str = "{} is a",
    prefix_range: Tuple[int, int] | list[int] | None = None,
    verbose: bool = True,
    batch_size: int | None = None,
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

    # Prompt preparation
    chat = is_chat_mode(handler)
    if not chat:
        new_target_ids = _strip_bos(handler, handler.tokenize_prompt(fact_tuple[2])["input_ids"][0])

    additional_prompts = [subject_understanding_template]
    main_prompt_count = max(1, int(N_prompts))
    templates = generate_prefixes(
        handler,
        main_prompt_count,
        prefix_range=resolve_prefix_range(handler, prefix_range),
        additional_prompts=[] if chat else additional_prompts,
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
        prepared = prepare_chat_batch(
            handler, templates, subject=fact_tuple[1], target=fact_tuple[2],
            kl_prompt=subject_understanding_template.format(fact_tuple[1]),
        )
        prompts, new_target_ids = prepared.prompts, prepared.target_ids
    else:
        if new_target_ids.size(0) > 1:
            templates = [template + handler.tokenizer.decode(new_target_ids[:-1]) for template in templates]
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

    last_subject_index = (
        prepared.subject_indices if chat else
        get_subject_index(handler, prompts, fact_tuple, subject_understanding_template)
    )
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
                v_init = hidden[0, row_subject_index[0]].detach().clone()
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
                    outputs = handler.model(**batch_inputs, use_cache=False)
                finally:
                    handler.remove_hooks()

                logits = outputs.logits
                log_probs = gather_target_log_probs(
                    logits,
                    index_positions_cpu[start:end].to(logits.device),
                    index_ids_cpu[start:end].to(logits.device),
                )
                batch_pred_loss = (-1 * log_probs).mean()

                if not is_final:
                    # Rescale so accumulated gradients equal the gradient of
                    # the mean over *all* main prompts.
                    (batch_pred_loss * (batch_len / main_prompt_count)).backward()
                pred_loss_total += float(batch_pred_loss.detach()) * batch_len

                del outputs, logits, log_probs, batch_pred_loss, batch_inputs

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
                    outputs = handler.model(**dkl_inputs, use_cache=False)
                finally:
                    handler.remove_hooks()

                dkl_logits = outputs.logits[
                    torch.arange(dkl_len, device=outputs.logits.device),
                    dkl_index_cpu.to(outputs.logits.device),
                    :,
                ]
                dkl_log_probs = torch.nn.functional.log_softmax(dkl_logits, dim=1)

                if dkl_orig is None:
                    dkl_orig = dkl_log_probs.detach().clone()  # Reused across epochs

                dkl = handler.kl_factor * torch.nn.functional.kl_div(
                    dkl_orig, dkl_log_probs, log_target=True, reduction="batchmean"
                )
                if not is_final:
                    dkl.backward()
                dkl_value = float(dkl.detach())

                del outputs, dkl_logits, dkl_log_probs, dkl, dkl_inputs

        # ---- weight decay: depends only on delta and the detached v_init ----
        if v_init is None:
            LOGGER.error("v_init was never captured; the delta hook did not fire.")
            return None

        weight_decay = handler.weight_decay * (torch.norm(delta) / (torch.norm(v_init) ** 2))
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

        if is_final:
            break

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
        return left.to(device=layer_device, dtype=handler.dtype)

    inv_cov = inv_cov.to(handler.dtype).to(layer_device)
    return (inv_cov @ k.to(layer_device).unsqueeze(1)).squeeze()


def insert_kv(
    handler: ModelHandler,
    k: torch.Tensor,
    delta: torch.Tensor,
    in_place: bool | None = None,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Apply the rank-one ROME update to the edited projection.

    Args:
        in_place: When true, add the outer product directly to the existing
            weight, allocating the update but no old/new weight copies.
            This reduces peak memory when stacking sequential edits.
            ``old_W`` and
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
    weight_transposed = weight.shape[0] != k.shape[0]

    # Match the scaled output space used by optimize_v. This includes both
    # automatic residual compensation and an explicit per-model delta_scale.
    _delta_scale = _resolve_delta_scale(handler)
    scaled_delta = (delta * _delta_scale).detach().to(layer_device)
    k = k.detach().to(layer_device)

    left = _solve_left_vector(handler, k, layer_device)
    left = left / left.norm()
    division_factor = torch.dot(k.to(left.dtype), left)
    right = scaled_delta.to(left.dtype) / division_factor

    LOGGER.info(f"Delta norm: {delta.norm().item()}")
    LOGGER.info(f"Delta scale: {_delta_scale}")
    LOGGER.info(f"Division Factor: {division_factor.item()}")
    LOGGER.info(f"Right vector norm: {right.norm()}")

    if in_place:
        with torch.no_grad():
            outer = torch.outer(left, right).to(dtype=weight.dtype)
            if weight_transposed:
                weight.add_(outer.T)
            else:
                weight.add_(outer)
            del outer
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


def get_second_moment(handler) -> torch.Tensor:
    """Load inverse covariance once per handler and covariance source.

    Invalidate on model/layer, source configuration or file replacement, and
    placement changes. Keep the loader local so callers can instrument it.
    """
    cache_enabled = resolve_flag(handler, "second_moment_cache", default=True)
    paths = _second_moment_paths(handler)
    source = tuple(
        (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths
    )
    layer_name = handler._layer_name_template.format(handler._layer)
    layer_device = (
        handler.get_module_device(layer_name)
        if getattr(handler, "is_multi_gpu", False) else handler.device
    )
    cache_key = (
        str(getattr(handler.cfg.model, "name", "")), int(handler._layer), layer_name,
        str(getattr(handler, "second_moment_path", "")),
        str(getattr(handler, "second_moment_dir", "")), source,
        str(getattr(handler.cfg.model, "second_moment_device", "auto")),
        str(layer_device), str(handler.dtype),
    )
    if cache_enabled:
        cached = getattr(handler, "_second_moment_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
    # Release stale placement/source data before loading another large matrix.
    handler._second_moment_cache = None
    matrix = _load_second_moment(handler)
    if cache_enabled:
        handler._second_moment_cache = (cache_key, matrix)
    return matrix


def _second_moment_paths(handler) -> list[Path]:
    configured = getattr(handler, "second_moment_path", None)
    if configured and Path(configured).is_file():
        return [Path(configured)]
    directory = getattr(handler, "second_moment_dir", None)
    if not directory:
        return []
    pattern = f"{handler.cfg.model.name.replace('/', '_')}_{handler._layer}_*_*."
    return sorted(Path(directory).glob(pattern + "pt")) + sorted(Path(directory).glob(pattern + "npz"))


def _load_second_moment(handler) -> torch.Tensor:
    """Load or compute the inverse second-moment matrix (uncached)."""
    file_paths = _second_moment_paths(handler)
    placement = str(getattr(handler.cfg.model, "second_moment_device", "auto") or "auto").lower()

    if len(file_paths):
        LOGGER.info(f"Auto-detected precached second moments: {file_paths}")
        LOGGER.info(f"{file_paths[0]} selected")
        try:
            if file_paths[0].suffix == ".npz":
                with np.load(file_paths[0]) as data:
                    # NPZ stores the second moment, PT stores its inverse.
                    matrix = torch.tensor(data["mom2.mom2"]).inverse()
            else:
                # CPU placement must never transiently restore a saved CUDA tensor.
                matrix = torch.load(file_paths[0], map_location="cpu", weights_only=True)

            if placement == "cpu":
                return matrix.to("cpu", torch.float32)
            layer_name = handler._layer_name_template.format(handler._layer)
            layer_device = (
                handler.get_module_device(layer_name)
                if getattr(handler, "is_multi_gpu", False) else handler.device
            )
            return handler.device_manager.safe_to_device(matrix, device=layer_device).to(torch.float32)
        except Exception as e:
            LOGGER.error(f"Failed to load second moment matrix: {e}")
            raise
    else:
        LOGGER.info("Precached second moments not found")
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

        save_dir = Path(handler.second_moment_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = non_conflicting_path(
            save_dir / f"{handler.cfg.model.name.replace('/', '_')}_{handler._layer}_{method}_{count}.pt"
        )
        torch.save(inv_cov, save_path)
        LOGGER.info(f"Saved second moment to {save_path}")
        return inv_cov.to("cpu", torch.float32) if placement == "cpu" else inv_cov
