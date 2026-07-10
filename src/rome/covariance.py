"""Second-moment covariance collection, caching, and loading for ROME."""

from __future__ import annotations

from enum import Enum
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.common.paths import non_conflicting_path
from src.rome.activations import _second_moment_contribution
from src.runtime import runtime_from_cfg

LOGGER = logging.getLogger(__name__)


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
                handler.model(
                    tokens.input_ids.to(input_device),
                    attention_mask=tokens.attention_mask.to(input_device),
                    use_cache=False,
                )
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
    from src.rome.optimization import gather_k

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
    """
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
                matrix = torch.load(file_paths[0])

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
