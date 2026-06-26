"""
Compatibility facade for historical utility imports.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>

New code should import from the named modules below. This file keeps original
ROME, causal trace, and CLI imports stable.
"""

from __future__ import annotations

from src.common.linalg import (
    CUDAMode,
    DeviceManager,
    _fingerprint_sample_indices,
    _tensor_cache_key,
    _tensor_storage_key,
    _tensor_value_fingerprint,
    check_device,
    clear_linalg_caches,
    configure_linalg_cache,
    estimate_covariance_batch_size,
    get_free_vram,
    get_total_vram,
    gpu_count,
    gpu_svd,
    gpu_svd_topk,
    gpu_svdvals,
)
from src.common.loading import (
    check_hf_token,
    get_cuda_usage,
    load_dataset,
    load_dataset_config,
    load_pretrained,
    logits_to_log_probs,
    logits_to_probs,
    print_modules,
    sample,
)
from src.evaluation.counterfact import (
    AttributeSnippets,
    compute_freq,
    compute_n_gram_entropy,
    compute_rewrite_quality_counterfact,
    generate_fast,
    get_tfidf_vectorizer,
    n_gram_entropy,
    perplexity,
    test_batch_prediction,
    test_generation,
    tfidf_similarity,
)


__all__ = [
    "AttributeSnippets",
    "CUDAMode",
    "DeviceManager",
    "_fingerprint_sample_indices",
    "_tensor_cache_key",
    "_tensor_storage_key",
    "_tensor_value_fingerprint",
    "check_device",
    "check_hf_token",
    "clear_linalg_caches",
    "configure_linalg_cache",
    "compute_freq",
    "compute_n_gram_entropy",
    "compute_rewrite_quality_counterfact",
    "estimate_covariance_batch_size",
    "generate_fast",
    "get_cuda_usage",
    "get_free_vram",
    "get_tfidf_vectorizer",
    "get_total_vram",
    "gpu_count",
    "gpu_svd",
    "gpu_svd_topk",
    "gpu_svdvals",
    "load_dataset",
    "load_dataset_config",
    "load_pretrained",
    "logits_to_log_probs",
    "logits_to_probs",
    "n_gram_entropy",
    "perplexity",
    "print_modules",
    "sample",
    "test_batch_prediction",
    "test_generation",
    "tfidf_similarity",
]
