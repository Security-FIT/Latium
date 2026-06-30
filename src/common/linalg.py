"""
CUDA-aware linear algebra helpers with process-local CPU caches.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import logging
import weakref
from collections import OrderedDict
from typing import Any

import torch


LOGGER = logging.getLogger(__name__)

_SVDVALS_CACHE_MAXSIZE = 4096
_SVDTOPK_CACHE_MAXSIZE = 1024
_SVDFULL_CACHE_MAXSIZE = 32
_SVDVALS_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_SVDTOPK_CACHE: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()
_SVDFULL_CACHE: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()


def _tensor_storage_key(W: torch.Tensor) -> tuple:
    """Stable key for tensor identity across detached and view tensors."""
    try:
        storage = W.untyped_storage()
        storage_ptr = int(storage.data_ptr())
        storage_nbytes = int(storage.nbytes())
    except Exception:
        storage_ptr = int(W.data_ptr())
        storage_nbytes = int(W.numel() * W.element_size())

    return (
        storage_ptr,
        storage_nbytes,
        int(W.storage_offset()),
        tuple(int(s) for s in W.shape),
        tuple(int(s) for s in W.stride()),
        str(W.dtype),
        str(W.device),
        int(getattr(W, "_version", 0)),
    )


def _tensor_value_fingerprint(W: torch.Tensor, sample_size: int = 16) -> tuple:
    """Content fingerprint for cache keys."""
    if W.numel() == 0:
        return (0, 0.0, 0.0, 0.0, ())

    view = W.detach().reshape(-1).float()
    sample_count = min(int(sample_size), int(view.numel()))
    if sample_count <= 0:
        samples = ()
    elif sample_count == 1:
        samples = (float(view[0].item()),)
    else:
        indices = _fingerprint_sample_indices(int(view.numel()), sample_count, view.device)
        samples = tuple(float(x) for x in view.index_select(0, indices).cpu())

    sq_norm = torch.dot(view, view)
    return (
        int(view.numel()),
        float(view.sum().item()),
        float(sq_norm.item()),
        float(view.abs().max().item()),
        samples,
    )


def _tensor_cache_key(W: torch.Tensor) -> tuple:
    """Cache key that is sensitive to both tensor identity and tensor values."""
    return _tensor_storage_key(W) + _tensor_value_fingerprint(W)


def _fingerprint_sample_indices(numel: int, sample_count: int, device: torch.device | str) -> torch.Tensor:
    """Build evenly spaced sample indices without float-rounding overflow."""
    total = max(0, int(numel))
    count = max(0, int(sample_count))
    if total <= 0 or count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if count == 1:
        return torch.zeros(1, dtype=torch.long, device=device)

    steps = torch.arange(count, dtype=torch.long, device=device)
    return torch.div(steps * max(0, total - 1), count - 1, rounding_mode="floor")


def _resolve_cuda_device(device: int | str, W: torch.Tensor | None = None) -> str | None:
    """Resolve CUDA target device string, preferring the tensor's current CUDA device."""
    if not torch.cuda.is_available():
        return None

    if W is not None and getattr(W, "is_cuda", False):
        return str(W.device)

    if isinstance(device, str):
        if device == "cpu":
            return None
        if device.startswith("cuda"):
            return device
        if device.isdigit():
            return f"cuda:{int(device)}"
        return "cuda:0"

    return f"cuda:{int(device)}"


def _cache_get(cache: OrderedDict, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache: OrderedDict, key, value, maxsize: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > maxsize:
        cache.popitem(last=False)


def clear_linalg_caches() -> None:
    """Clear cached linear algebra artifacts used by SVD helpers."""
    _SVDVALS_CACHE.clear()
    _SVDTOPK_CACHE.clear()
    _SVDFULL_CACHE.clear()


def configure_linalg_cache(
    *,
    svdvals_maxsize: int | None = None,
    svdtopk_maxsize: int | None = None,
    svdfull_maxsize: int | None = None,
) -> None:
    """Configure process-local SVD cache sizes from explicit runtime config."""
    global _SVDVALS_CACHE_MAXSIZE, _SVDTOPK_CACHE_MAXSIZE, _SVDFULL_CACHE_MAXSIZE

    if svdvals_maxsize is not None:
        _SVDVALS_CACHE_MAXSIZE = max(0, int(svdvals_maxsize))
    if svdtopk_maxsize is not None:
        _SVDTOPK_CACHE_MAXSIZE = max(0, int(svdtopk_maxsize))
    if svdfull_maxsize is not None:
        _SVDFULL_CACHE_MAXSIZE = max(0, int(svdfull_maxsize))

    while len(_SVDVALS_CACHE) > _SVDVALS_CACHE_MAXSIZE:
        _SVDVALS_CACHE.popitem(last=False)
    while len(_SVDTOPK_CACHE) > _SVDTOPK_CACHE_MAXSIZE:
        _SVDTOPK_CACHE.popitem(last=False)
    while len(_SVDFULL_CACHE) > _SVDFULL_CACHE_MAXSIZE:
        _SVDFULL_CACHE.popitem(last=False)


class CUDAMode:
    """CUDA device modes."""

    NONE = "none"
    SOFT = "soft"
    GREEDY = "greedy"
    STRICT = "strict"


class DeviceManager:
    """Manage device moves under different CUDA OOM policies."""

    _managed_objects = weakref.WeakSet()
    _cuda_disabled = False

    def __init__(self, preferred_device: str = "cuda", cuda_mode: str = CUDAMode.SOFT):
        self.preferred_device = preferred_device
        self.cuda_mode = cuda_mode
        self._oom_count = 0

    def register_object(self, obj: Any) -> None:
        """Register an object to move back to CPU if CUDA is disabled."""
        try:
            DeviceManager._managed_objects.add(obj)
        except (TypeError, RuntimeError):
            LOGGER.debug("Skipping unhashable object: %s", type(obj).__name__)

    def get_device(self) -> str:
        """Get the current active device."""
        if self.cuda_mode == CUDAMode.NONE:
            return "cpu"
        if self.cuda_mode == CUDAMode.SOFT and DeviceManager._cuda_disabled:
            return "cpu"
        return self.preferred_device

    def clear_cache(self) -> None:
        """Clear CUDA cache if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def safe_to_device(self, data: Any, device: str | None = None) -> Any:
        """Move a tensor-like object or model with CUDA OOM handling."""
        target_device = device or self.get_device()

        if hasattr(data, "device") and str(data.device) == str(target_device):
            if target_device != "cpu" and hasattr(data, "to"):
                self.register_object(data)
            return data

        try:
            result = data.to(target_device)
            if target_device != "cpu" and hasattr(result, "to"):
                self.register_object(result)
                if hasattr(result, "data") and isinstance(result.data, dict):
                    for tensor in result.data.values():
                        if isinstance(tensor, torch.Tensor):
                            self.register_object(tensor)
            return result
        except torch.cuda.OutOfMemoryError as exc:
            return self._handle_oom(data, target_device, exc)

    def _handle_oom(self, data: Any, device: str, error: Exception) -> Any:
        """Handle OOM error based on CUDA mode."""
        self._oom_count += 1

        if self.cuda_mode == CUDAMode.STRICT:
            LOGGER.error("CUDA OOM Error #%d in strict mode", self._oom_count)
            LOGGER.error("Error details: %s", error)
            raise SystemExit("CUDA OOM Error in strict mode. Cannot continue.") from error
        if self.cuda_mode == CUDAMode.SOFT:
            LOGGER.error("CUDA OOM Error #%d in soft mode", self._oom_count)
            LOGGER.warning("Permanently switching to CPU for the rest of operations")
            DeviceManager._cuda_disabled = True
            self.clear_cache()

            moved_count = 0
            for obj in DeviceManager._managed_objects:
                try:
                    if hasattr(obj, "data") and not hasattr(obj, "parameters"):
                        obj.data = obj.data.to("cpu")
                        moved_count += 1
                    elif hasattr(obj, "to"):
                        obj.to("cpu")
                        moved_count += 1
                except Exception as exc:
                    LOGGER.debug("Could not move %s to CPU: %s", type(obj).__name__, exc)

            if moved_count > 0:
                LOGGER.info("Moved %d objects to CPU", moved_count)

            return data.to("cpu")
        if self.cuda_mode == CUDAMode.GREEDY:
            LOGGER.warning("CUDA OOM Error #%d in greedy mode", self._oom_count)
            LOGGER.info("Clearing CUDA cache and retrying...")
            self.clear_cache()

            try:
                return data.to(device)
            except torch.cuda.OutOfMemoryError:
                LOGGER.error("CUDA OOM persists after cache clear. Falling back to CPU for this operation.")
                return data.to("cpu")

        LOGGER.error("Unknown CUDA mode '%s'. Cannot handle OOM error.", self.cuda_mode)
        raise SystemExit("Unknown CUDA mode. Cannot continue.") from error


def gpu_count() -> int:
    """Return the number of available CUDA GPUs."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def get_free_vram(device: int | str = 0) -> int:
    """Return free VRAM in bytes for the given CUDA device."""
    if not torch.cuda.is_available():
        return 0
    if isinstance(device, str):
        if device == "cpu":
            return 0
        device = int(device.replace("cuda:", "").replace("cuda", "0") or "0")
    free, _ = torch.cuda.mem_get_info(device)
    return free


def get_total_vram(device: int | str = 0) -> int:
    """Return total VRAM in bytes for the given CUDA device."""
    if not torch.cuda.is_available():
        return 0
    if isinstance(device, str):
        if device == "cpu":
            return 0
        device = int(device.replace("cuda:", "").replace("cuda", "0") or "0")
    _, total = torch.cuda.mem_get_info(device)
    return total


def estimate_covariance_batch_size(
    hidden_dim: int,
    max_length: int,
    dtype_bytes: int = 2,
    device: int | str = 0,
    vram_fraction: float = 0.15,
    min_batch: int = 1,
    max_batch: int = 64,
) -> int:
    """Estimate a safe covariance batch size from available VRAM."""
    free = get_free_vram(device)
    if free == 0:
        return min_batch

    expected_seq_len = min(max_length, 1024)
    per_sample = expected_seq_len * hidden_dim * dtype_bytes * 8
    cov_overhead = hidden_dim * hidden_dim * 4
    available = int(free * vram_fraction) - cov_overhead
    available = max(per_sample, available)
    bs = max(min_batch, min(max_batch, available // max(per_sample, 1)))
    LOGGER.info(
        "Dynamic covariance batch size: %d  "
        "(free_vram=%.1fGB, per_sample=%.1fMB, cov_overhead=%.1fMB, expected_seq=%d, max_length=%d)",
        bs,
        free / 1e9,
        per_sample / 1e6,
        cov_overhead / 1e6,
        expected_seq_len,
        max_length,
    )
    return bs


def gpu_svd(
    W: torch.Tensor,
    full_matrices: bool = False,
    device: int | str = 0,
    vram_fraction: float = 0.5,
) -> tuple:
    """Compute SVD with GPU-first execution, cache on CPU, fall back to CPU on OOM."""
    _ = vram_fraction
    tensor_key = _tensor_cache_key(W)
    cache_key = ("svd", bool(full_matrices)) + tensor_key
    cached = _cache_get(_SVDFULL_CACHE, cache_key)
    if cached is not None:
        return cached

    cuda_device = _resolve_cuda_device(device, W=W)
    if cuda_device is not None:
        try:
            W_gpu = W.to(cuda_device).float()
            U, S, Vh = torch.linalg.svd(W_gpu, full_matrices=full_matrices)
            result = (U.cpu(), S.cpu(), Vh.cpu())
            _cache_put(_SVDFULL_CACHE, cache_key, result, _SVDFULL_CACHE_MAXSIZE)
            return result
        except torch.cuda.OutOfMemoryError:
            LOGGER.debug("GPU SVD OOM, falling back to CPU")
            torch.cuda.empty_cache()

    result = torch.linalg.svd(W.float().cpu(), full_matrices=full_matrices)
    _cache_put(_SVDFULL_CACHE, cache_key, result, _SVDFULL_CACHE_MAXSIZE)
    return result


def gpu_svdvals(
    W: torch.Tensor,
    device: int | str = 0,
    vram_fraction: float = 0.5,
) -> torch.Tensor:
    """Compute singular values with GPU-first execution and CPU cache."""
    _ = vram_fraction
    tensor_key = _tensor_cache_key(W)
    cache_key = ("svdvals",) + tensor_key
    cached = _cache_get(_SVDVALS_CACHE, cache_key)
    if cached is not None:
        return cached

    full_cached = _cache_get(_SVDFULL_CACHE, ("svd", False) + tensor_key)
    if full_cached is not None:
        S_cpu = full_cached[1]
        _cache_put(_SVDVALS_CACHE, cache_key, S_cpu, _SVDVALS_CACHE_MAXSIZE)
        return S_cpu

    cuda_device = _resolve_cuda_device(device, W=W)
    if cuda_device is not None:
        try:
            S = torch.linalg.svdvals(W.to(cuda_device).float())
            S_cpu = S.cpu()
            _cache_put(_SVDVALS_CACHE, cache_key, S_cpu, _SVDVALS_CACHE_MAXSIZE)
            return S_cpu
        except torch.cuda.OutOfMemoryError:
            LOGGER.debug("GPU svdvals OOM, falling back to CPU")
            torch.cuda.empty_cache()

    S_cpu = torch.linalg.svdvals(W.float().cpu())
    _cache_put(_SVDVALS_CACHE, cache_key, S_cpu, _SVDVALS_CACHE_MAXSIZE)
    return S_cpu


def gpu_svd_topk(
    W: torch.Tensor,
    k: int,
    niter: int = 2,
    device: int | str = 0,
    vram_fraction: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute rank-k SVD (U, S, Vh) with GPU-first execution and caching."""
    _ = vram_fraction
    q = max(1, min(int(k), int(min(W.shape))))
    tensor_key = _tensor_cache_key(W)
    cache_key = ("svd_topk", q, int(niter)) + tensor_key
    cached = _cache_get(_SVDTOPK_CACHE, cache_key)
    if cached is not None:
        return cached

    prefix = ("svd_topk",)
    suffix = tensor_key
    for existing_key, existing_value in reversed(_SVDTOPK_CACHE.items()):
        if len(existing_key) < 4 or existing_key[0] != prefix[0]:
            continue
        q_cached = existing_key[1]
        niter_cached = existing_key[2]
        existing_suffix = existing_key[3:]
        if niter_cached != int(niter) or existing_suffix != suffix or q_cached < q:
            continue
        U_cached, S_cached, Vh_cached = existing_value
        result = (U_cached[:, :q], S_cached[:q], Vh_cached[:q, :])
        _cache_put(_SVDTOPK_CACHE, cache_key, result, _SVDTOPK_CACHE_MAXSIZE)
        return result

    full_cached = _cache_get(_SVDFULL_CACHE, ("svd", False) + tensor_key)
    if full_cached is not None:
        U_full, S_full, Vh_full = full_cached
        result = (U_full[:, :q], S_full[:q], Vh_full[:q, :])
        _cache_put(_SVDTOPK_CACHE, cache_key, result, _SVDTOPK_CACHE_MAXSIZE)
        return result

    cuda_device = _resolve_cuda_device(device, W=W)
    if cuda_device is not None:
        try:
            W_gpu = W.to(cuda_device).float()
            U, S, V = torch.svd_lowrank(W_gpu, q=q, niter=int(niter))
            result = (U.cpu(), S.cpu(), V.T.cpu())
            _cache_put(_SVDTOPK_CACHE, cache_key, result, _SVDTOPK_CACHE_MAXSIZE)
            return result
        except torch.cuda.OutOfMemoryError:
            LOGGER.debug("GPU svd_lowrank OOM, falling back to CPU")
            torch.cuda.empty_cache()

    U, S, V = torch.svd_lowrank(W.float().cpu(), q=q, niter=int(niter))
    result = (U, S, V.T)
    _cache_put(_SVDTOPK_CACHE, cache_key, result, _SVDTOPK_CACHE_MAXSIZE)
    return result


def check_device(device: str) -> str:
    """Check if the device is valid and return the appropriate device."""
    if device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA is not available. Setting the device to 'cpu'.")
        device = "cpu"
    elif device == "cpu" and torch.cuda.is_available():
        LOGGER.info("CUDA is available. Consider setting the device to 'cuda'.")
    return device


__all__ = [
    "CUDAMode",
    "DeviceManager",
    "_fingerprint_sample_indices",
    "_tensor_cache_key",
    "_tensor_storage_key",
    "_tensor_value_fingerprint",
    "check_device",
    "clear_linalg_caches",
    "configure_linalg_cache",
    "estimate_covariance_batch_size",
    "get_free_vram",
    "get_total_vram",
    "gpu_count",
    "gpu_svd",
    "gpu_svd_topk",
    "gpu_svdvals",
]
