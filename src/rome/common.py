"""Compatibility facade for ROME helpers.

New code should import from the responsibility-specific modules in this package.
"""

from src.rome.activations import (
    _accumulate_second_moment_tokens,
    _reshape_hidden_states,
    _second_moment_contribution,
)
from src.rome.covariance import (
    SM_Method,
    _AdaptiveCovarianceBatchSizer,
    compute_second_moment,
    get_second_moment,
    second_moment_random,
    second_moment_wikipedia,
)
from src.rome.optimization import gather_k, insert_kv, optimize_v, pcs
from src.rome.prefixes import (
    PrefixGenerationHandler,
    PrefixMode,
    generate_prefixes,
    resolve_prefix_range,
    resolve_rome_sample_count,
)
from src.rome.subjects import _strip_bos, get_subject_index, get_subject_position

__all__ = [
    "PrefixGenerationHandler",
    "PrefixMode",
    "SM_Method",
    "compute_second_moment",
    "gather_k",
    "generate_prefixes",
    "get_second_moment",
    "get_subject_index",
    "get_subject_position",
    "insert_kv",
    "optimize_v",
    "pcs",
    "resolve_prefix_range",
    "resolve_rome_sample_count",
    "second_moment_random",
    "second_moment_wikipedia",
]
