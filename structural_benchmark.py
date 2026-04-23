#!/usr/bin/env python3
"""
structural_benchmark.py - ROME + Structural Analysis Benchmark

Runs ROME edits on one or more models and evaluates structural detectors
on the modified weights.  All model parameters (layer, template, hyperparams)
are read from per-model YAML configs in src/config/model/.

Usage examples:
    # Single model (uses src/config/model/gpt2-large.yaml):
    python structural_benchmark.py --models gpt2-large

    # Multiple models in one run:
    python structural_benchmark.py --models gpt2-large qwen3-4b mistral-7b-v0.1
"""

import json
import itertools
import logging
import math
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import datasets
import numpy as np
import torch
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from src.handlers.rome import ModelHandler
from src.rome.common import gather_k, optimize_v, insert_kv
from src.structural.blind_detector import BlindMSDDetector
from src.structural.spectral_detector import SpectralDetector
from src.structural.composite_detector import CompositeDetector
from src.structural.edit_presence_detector import RomeEditPresenceDetector
from src.structural.rank1_blind import BlindRank1Detector
from src.structural.bottom_rank_svd import BottomRankSVDDetector
from src.structural.attention_metrics import (
    AttentionContrastDetector,
    derive_attention_templates,
    split_qkv_weight,
)
from src.structural.symmetry_metrics import MirrorSymmetryDetector
from src.structural.interlayer import collect_all_interlayer_data
from src.structural.novel_metrics import compute_novel_metrics
from src.structural.ipr import (
    layer_ipr_summary,
    layer_fc_proj_ipr_discrepancy,
    IPRDetector,
)
from src.utils import clear_linalg_caches

CONFIG_DIR = Path(__file__).parent / "src" / "config"
MODEL_CONFIG_DIR = CONFIG_DIR / "model"

SPECTRAL_SIGNAL_KEYS = [
    "sv_z_scores", "sv_ratio_scores",
    "sv_z_rolling_z_scores", "sv_ratio_rolling_z_scores",
    "pcs_composite_rank_scores", "sv_pcs_contradiction_scores",
    "rome_hybrid_scores",
    "pcs_neighbor_shift_scores", "pcs_neighbor_var_scores",
    "pcs_neighbor_min_shift_scores", "pcs_neighbor_flip_fraction_scores",
    "pcs_next_shift_scores", "pcs_next_jump_scores", "pcs_next_curvature_scores",
]

# Maps proj (output) layer key -> fc (input) layer key across architectures
FC_TEMPLATE_MAP = {
    "c_proj": "c_fc",        # GPT-2
    "fc_out": "fc_in",       # GPT-J
    "down_proj": "up_proj",  # Llama / Mistral / Qwen
    "output_linear": "input_linear",  # Granite shared MLP
}

ANALYSIS_PROFILE_ALIASES = {
    "raw": "raw",
    "full": "full",
    "paper": "paper",
    "posthoc": "paper",
    "posthoc-only": "paper",
    "detection": "paper",
    "detection-only": "paper",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_serializable(obj):
    """Convert numpy/torch types to JSON-serializable Python types."""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def normalize_target_text(text: str) -> str:
    """Normalize short target strings for tolerant textual matching."""
    if text is None:
        return ""
    s = str(text).strip().lower()
    # Allow superficial punctuation differences at the boundary.
    s = re.sub(r"[\s\.,;:!?\"'`]+$", "", s)
    s = re.sub(r"^[\s\"'`]+", "", s)
    return s


def normalize_analysis_profile(analysis_profile: Optional[str], posthoc_only: bool = False) -> str:
    """Map user-facing aliases onto the canonical analysis profiles."""
    if posthoc_only:
        return "paper"

    profile = str(analysis_profile or "full").strip().lower()
    canonical = ANALYSIS_PROFILE_ALIASES.get(profile)
    if canonical is None:
        supported = ", ".join(sorted(ANALYSIS_PROFILE_ALIASES))
        raise ValueError(
            f"Unsupported analysis_profile: {analysis_profile!r}. "
            f"Supported values/aliases: {supported}"
        )
    return canonical


# ---------------------------------------------------------------------------
# ROME paper evaluation metrics (probability-based)
# ---------------------------------------------------------------------------

def _get_target_token_ids(tok, text: str) -> list:
    """Tokenize target text without BOS/special tokens."""
    ids = tok(f" {text}", add_special_tokens=False)["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if not ids:
        ids = tok(f" {text}")["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        bos_id = getattr(tok, "bos_token_id", None)
        if bos_id is not None and len(ids) > 1 and ids[0] == bos_id:
            ids = ids[1:]
    return ids


def _test_batch_prediction(
    model, tok, prefixes: List[str],
    target_new: str, target_true: str, device,
    batch_size: int = 8,
) -> List[Dict]:
    """
    Compute mean NLL for target_new and target_true given each prefix.
    Based on the original ROME paper evaluation code in test_batch_prediction.
    Processes in chunks to control GPU memory.
    """
    a_tok = _get_target_token_ids(tok, target_new)
    b_tok = _get_target_token_ids(tok, target_true)
    choice_a_len, choice_b_len = len(a_tok), len(b_tok)

    all_results = []

    for chunk_start in range(0, len(prefixes), batch_size):
        chunk_prefixes = prefixes[chunk_start:chunk_start + batch_size]
        prefix_lens = [len(n) for n in tok(chunk_prefixes)["input_ids"]]

        prompt_tok = tok(
            [f"{prefix} {suffix}" for prefix in chunk_prefixes for suffix in [target_new, target_true]],
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Account for left-padding: count leading pad tokens per row
        # Count only leading zeros (left-padding), not trailing right-padding
        pad_offsets = (prompt_tok["attention_mask"].cumsum(dim=1) == 0).sum(dim=1).tolist()

        with torch.no_grad():
            logits = model(**prompt_tok, use_cache=False).logits

        chunk_results = np.zeros((logits.size(0),), dtype=np.float32)
        for i in range(logits.size(0)):
            cur_len = choice_a_len if i % 2 == 0 else choice_b_len
            cur_tok_ids = a_tok if i % 2 == 0 else b_tok
            offset = pad_offsets[i]
            for j in range(cur_len):
                cur_tok = cur_tok_ids[j]
                chunk_results[i] += -torch.nn.functional.log_softmax(
                    logits[i, offset + prefix_lens[i // 2] + j - 1, :], dim=0
                )[cur_tok].item()
            chunk_results[i] /= cur_len

        for i in range(0, len(chunk_results), 2):
            all_results.append({
                "target_new": chunk_results[i].item(),
                "target_true": chunk_results[i + 1].item(),
            })

        del prompt_tok, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


def compute_rome_metrics(
    handler, prompt_text: str,
    target_new_str: str, target_true_str: str,
    paraphrase_prompts: Optional[List[str]] = None,
    neighborhood_prompts: Optional[List[str]] = None,
) -> Dict:
    """
    Compute ROME paper evaluation metrics using probability comparison.

    ES  – Efficacy Score:     1{P[o*] > P[o_c]} on rewrite prompt
    EM  – Efficacy Magnitude: P[o*] - P[o_c]
    PS  – Paraphrase Score:   mean ES across paraphrase prompts
    NS  – Neighborhood Score: mean 1{P[o_c] > P[o*]} across neighborhood prompts
    S   – Overall Score:      harmonic mean of ES, PS, NS
    """
    model = handler.model
    tok = handler.tokenizer
    device = handler.device

    all_prompts = [prompt_text]
    n_para = len(paraphrase_prompts) if paraphrase_prompts else 0
    n_neigh = len(neighborhood_prompts) if neighborhood_prompts else 0
    if paraphrase_prompts:
        all_prompts.extend(paraphrase_prompts)
    if neighborhood_prompts:
        all_prompts.extend(neighborhood_prompts)

    probs = _test_batch_prediction(model, tok, all_prompts, target_new_str, target_true_str, device)

    rewrite_prob = probs[0]
    para_probs = probs[1:1 + n_para] if n_para else []
    neigh_probs = probs[1 + n_para:] if n_neigh else []

    # ES: lower NLL = higher probability
    es = 1.0 if rewrite_prob["target_new"] < rewrite_prob["target_true"] else 0.0
    # EM: probability difference (geometric-mean per-token probability)
    em = math.exp(-rewrite_prob["target_new"]) - math.exp(-rewrite_prob["target_true"])

    # PS: mean ES across paraphrase prompts
    ps = (
        sum(1.0 for p in para_probs if p["target_new"] < p["target_true"]) / len(para_probs)
        if para_probs else None
    )

    # NS: neighborhood should still predict target_true
    ns = (
        sum(1.0 for p in neigh_probs if p["target_true"] < p["target_new"]) / len(neigh_probs)
        if neigh_probs else None
    )

    # S: harmonic mean of available scores
    components = [s for s in [es, ps, ns] if s is not None]
    if components and all(s > 0 for s in components):
        overall = len(components) / sum(1.0 / s for s in components)
    else:
        overall = 0.0

    return {
        "efficacy_score": es,
        "efficacy_magnitude": em,
        "paraphrase_score": ps,
        "neighborhood_score": ns,
        "overall_score": overall,
        "rewrite_nll": rewrite_prob,
        "paraphrase_nll": para_probs,
        "neighborhood_nll": neigh_probs,
    }


def get_fc_template(layer_name_template: str) -> Optional[str]:
    """Derive the fc (input) layer template from the proj (output) layer template."""
    for proj_key, fc_key in FC_TEMPLATE_MAP.items():
        if proj_key in layer_name_template:
            return layer_name_template.replace(proj_key, fc_key)
    return None


def load_model_config(model_name: str) -> OmegaConf:
    """Load a model config YAML by stem name or by the 'name' field inside the YAML."""
    yaml_path = MODEL_CONFIG_DIR / f"{model_name}.yaml"
    if yaml_path.exists():
        return OmegaConf.load(yaml_path)
    for path in sorted(MODEL_CONFIG_DIR.glob("*.yaml")):
        if path.name == "boilerplate.yaml":
            continue
        cfg = OmegaConf.load(path)
        if getattr(cfg, "name", "") == model_name:
            return cfg
    available = ", ".join(
        p.stem for p in sorted(MODEL_CONFIG_DIR.glob("*.yaml")) if p.name != "boilerplate.yaml"
    )
    raise FileNotFoundError(f"No config for '{model_name}'. Available: {available}")


def build_cfg(model_name: str) -> OmegaConf:
    """Compose a full runtime config from per-model YAML + shared dataset/generation defaults."""
    model_cfg = load_model_config(model_name)
    return OmegaConf.create({
        "model": model_cfg,
        "generation": OmegaConf.load(CONFIG_DIR / "generation" / "generation.yaml"),
        "dataset_sm": OmegaConf.load(CONFIG_DIR / "dataset_sm" / "wikitext.yaml"),
    })


def find_second_moment_files(model_cfg: OmegaConf) -> Tuple[List[Path], Path]:
    """Locate precomputed second-moment files expected for the configured model/layer."""
    raw_dir = Path(getattr(model_cfg, "second_moment_dir", "./second_moment_stats"))
    sm_dir = raw_dir if raw_dir.is_absolute() else (Path(__file__).parent / raw_dir).resolve()

    # Explicit path has priority if configured.
    explicit = str(getattr(model_cfg, "second_moment_path", "") or "").strip()
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (Path(__file__).parent / p).resolve()
        return ([p] if p.exists() else []), sm_dir

    model_id = str(getattr(model_cfg, "name", "")).replace("/", "_")
    layer = int(getattr(model_cfg, "layer", -1))
    pt_files = sorted(sm_dir.glob(f"{model_id}_{layer}_*_*.pt"))
    npz_files = sorted(sm_dir.glob(f"{model_id}_{layer}_*_*.npz"))
    return (pt_files + npz_files), sm_dir


def available_model_names() -> List[str]:
    return sorted(
        p.stem for p in MODEL_CONFIG_DIR.glob("*.yaml") if p.name != "boilerplate.yaml"
    )


def normalize_models_arg(models: List[str]) -> List[str]:
    """
    Normalize model arguments to support:
      - whitespace-separated values: --models gpt2-xl qwen3-4b
      - comma-separated values:      --models gpt2-xl,qwen3-4b
      - mixed forms
    """
    normalized: List[str] = []
    seen = set()
    for entry in models:
        for part in str(entry).split(","):
            name = part.strip()
            if name and name not in seen:
                seen.add(name)
                normalized.append(name)
    return normalized


def parse_local_windows(raw: str, default: Sequence[int] = (3, 5, 7)) -> List[int]:
    """Parse comma-separated rolling-window sizes and normalize to odd positive ints."""
    if raw is None:
        return [int(w) for w in default]
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return [int(w) for w in default]

    out = []
    seen = set()
    for part in parts:
        try:
            w = max(1, int(part))
        except ValueError:
            continue
        if w % 2 == 0:
            w += 1
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out or [int(w) for w in default]


def parse_int_values(
    raw: Optional[str],
    default: Sequence[int],
    min_value: int = 0,
    force_odd: bool = False,
) -> List[int]:
    """Parse comma-separated integer values with normalization and de-duplication."""
    parts: List[str]
    if raw is None:
        parts = [str(v) for v in default]
    else:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]

    out: List[int] = []
    seen = set()
    for part in parts:
        try:
            v = int(part)
        except ValueError:
            continue
        if v < min_value:
            v = min_value
        if force_odd and v % 2 == 0:
            v += 1
        if v not in seen:
            seen.add(v)
            out.append(v)

    if out:
        return out

    fallback = []
    for v in default:
        vv = max(min_value, int(v))
        if force_odd and vv % 2 == 0:
            vv += 1
        fallback.append(vv)
    return fallback or ([1] if min_value <= 1 else [min_value])


def parse_trim_values(raw: Optional[str], default: Sequence[Optional[int]]) -> List[Optional[int]]:
    """Parse trim sweep values; supports integers and 'auto' (None)."""
    source: List[Optional[str]]
    if raw is None:
        source = [None if v is None else str(v) for v in default]
    else:
        source = [p.strip() for p in str(raw).split(",") if p.strip()]

    out: List[Optional[int]] = []
    seen = set()
    for item in source:
        if item is None:
            val: Optional[int] = None
        else:
            token = str(item).strip().lower()
            if token in {"auto", "none", "default"}:
                val = None
            else:
                try:
                    val = max(0, int(token))
                except ValueError:
                    continue

        key = "auto" if val is None else str(val)
        if key not in seen:
            seen.add(key)
            out.append(val)

    return out or [None]


def parse_local_window_sets(
    raw: Optional[str],
    default: Sequence[Sequence[int]] = ((3, 5, 7),),
) -> List[List[int]]:
    """
    Parse semicolon-separated local-window sets.
    Example: "3,5,7;5,9,13" -> [[3,5,7], [5,9,13]]
    """
    default_base = tuple(default[0]) if default else (3, 5, 7)

    chunks: List[str]
    if raw is None:
        chunks = [",".join(str(w) for w in seq) for seq in default]
    else:
        chunks = [c.strip() for c in str(raw).split(";") if c.strip()]

    out: List[List[int]] = []
    seen = set()
    for chunk in chunks:
        values = parse_local_windows(chunk, default=default_base)
        key = tuple(values)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(list(values))

    return out or [list(parse_local_windows(None, default=default_base))]


def _expand_for_zip(values: Sequence, target_len: int, arg_name: str) -> List:
    """Broadcast singleton sweep lists or validate exact-length lists for zip mode."""
    if len(values) == target_len:
        return list(values)
    if len(values) == 1:
        return list(values) * target_len
    raise ValueError(
        f"In sweep zip mode, {arg_name} must have length 1 or {target_len} (got {len(values)})."
    )


def build_sweep_configs(
    spectral_top_k_values: Sequence[int],
    trim_first_values: Sequence[Optional[int]],
    trim_last_values: Sequence[Optional[int]],
    spectral_neighbor_layers_values: Sequence[int],
    spectral_rolling_window_values: Sequence[int],
    local_window_sets: Sequence[Sequence[int]],
    mode: str = "zip",
    max_configs: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Build sweep configurations either by zip-style pairing or full cartesian product."""
    topks = [max(1, int(v)) for v in spectral_top_k_values] or [50]
    tf_vals = [None if v is None else max(0, int(v)) for v in trim_first_values] or [None]
    tl_vals = [None if v is None else max(0, int(v)) for v in trim_last_values] or [None]
    nl_vals = [max(1, int(v)) for v in spectral_neighbor_layers_values] or [1]
    rw_vals = [max(1, int(v)) for v in spectral_rolling_window_values] or [5]

    lw_sets: List[Tuple[int, ...]] = []
    lw_seen = set()
    for seq in local_window_sets:
        norm = tuple(parse_local_windows(",".join(str(w) for w in seq), default=(3, 5, 7)))
        if not norm or norm in lw_seen:
            continue
        lw_seen.add(norm)
        lw_sets.append(norm)
    if not lw_sets:
        lw_sets = [tuple(parse_local_windows(None, default=(3, 5, 7)))]

    if mode not in {"zip", "product"}:
        raise ValueError(f"Unsupported sweep mode: {mode}")

    if mode == "product":
        iterable = itertools.product(topks, tf_vals, tl_vals, nl_vals, rw_vals, lw_sets)
    else:
        target = max(len(topks), len(tf_vals), len(tl_vals), len(nl_vals), len(rw_vals), len(lw_sets))
        iterable = zip(
            _expand_for_zip(topks, target, "--sweep-spectral-top-k"),
            _expand_for_zip(tf_vals, target, "--sweep-trim-first"),
            _expand_for_zip(tl_vals, target, "--sweep-trim-last"),
            _expand_for_zip(nl_vals, target, "--sweep-spectral-neighbor-layers"),
            _expand_for_zip(rw_vals, target, "--sweep-spectral-rolling-window"),
            _expand_for_zip(lw_sets, target, "--sweep-local-window-sets"),
        )

    limit = None if max_configs is None else max(1, int(max_configs))

    configs: List[Dict[str, object]] = []
    seen = set()
    for top_k, tf, tl, nl, rw, lw in iterable:
        cfg = {
            "spectral_top_k": int(top_k),
            "trim_first": None if tf is None else int(tf),
            "trim_last": None if tl is None else int(tl),
            "spectral_neighbor_layers": int(nl),
            "spectral_rolling_window": int(rw),
            "local_windows": [int(w) for w in lw],
        }
        key = (
            cfg["spectral_top_k"],
            cfg["trim_first"],
            cfg["trim_last"],
            cfg["spectral_neighbor_layers"],
            cfg["spectral_rolling_window"],
            tuple(cfg["local_windows"]),
        )
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)
        if limit is not None and len(configs) >= limit:
            break

    return configs or [{
        "spectral_top_k": 50,
        "trim_first": None,
        "trim_last": None,
        "spectral_neighbor_layers": 1,
        "spectral_rolling_window": 5,
        "local_windows": [3, 5, 7],
    }]


def _format_optional_int(value: object) -> str:
    if value is None:
        return "auto"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "auto"


def sweep_config_slug(cfg: Dict[str, object]) -> str:
    """Generate a stable filename slug for a sweep config."""
    lw_raw = cfg.get("local_windows", [3, 5, 7])
    if isinstance(lw_raw, (list, tuple)):
        lw = [int(w) for w in lw_raw]
    else:
        lw = parse_local_windows(str(lw_raw), default=(3, 5, 7))
    lw_str = "-".join(str(w) for w in lw)
    return (
        f"tk{int(cfg.get('spectral_top_k', 50))}"
        f"_tf{_format_optional_int(cfg.get('trim_first'))}"
        f"_tl{_format_optional_int(cfg.get('trim_last'))}"
        f"_nl{int(cfg.get('spectral_neighbor_layers', 1))}"
        f"_rw{int(cfg.get('spectral_rolling_window', 5))}"
        f"_lw{lw_str}"
    )


def _append_model_result(all_results: Dict[str, object], model_name: str, result: dict) -> None:
    """Preserve backward compatibility: single result stays dict, multi-run becomes list."""
    existing = all_results.get(model_name)
    if existing is None:
        all_results[model_name] = result
        return
    if isinstance(existing, list):
        existing.append(result)
        return
    all_results[model_name] = [existing, result]


def _build_run_output_path(
    output_path: Path,
    file_prefix: str,
    safe_name: str,
    cfg_slug: str,
    sweep_idx: int,
    run_idx: int,
    run_token: str,
) -> Path:
    return output_path / (
        f"{file_prefix}_{safe_name}_{cfg_slug}"
        f"_s{sweep_idx:02d}_r{run_idx:02d}_{run_token}.json"
    )


def _build_run_metadata(
    model_name: str,
    start_idx_used: int,
    run_idx: int,
    run_ordinal: int,
    runs_per_model: int,
    run_start_idx_step: int,
    sweep_idx: int,
    sweep_cfg: Dict[str, object],
    sweep_count: int,
    cfg_slug: str,
    total_runs_for_model: int,
    sweep_tag: Optional[str],
    n_cases: int,
) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "model": model_name,
        "run_index": run_idx,
        "run_ordinal": run_ordinal,
        "runs_per_model": runs_per_model,
        "run_start_idx_step": run_start_idx_step,
        "start_idx_used": start_idx_used,
        "end_idx_used": start_idx_used + max(0, n_cases - 1),
        "sweep_index": sweep_idx,
        "sweep_size": sweep_count,
        "sweep_config": to_serializable(sweep_cfg),
        "sweep_slug": cfg_slug,
        "total_runs_for_model": total_runs_for_model,
    }
    if sweep_tag:
        metadata["sweep_tag"] = str(sweep_tag)
    return metadata


def auto_trim_from_layers(num_layers: int) -> int:
    """
    Choose trim count from model depth.
    Targets ~5% boundary trim while keeping practical bounds.
    Example: GPT2-XL (48 layers) -> 2.
    """
    if num_layers <= 0:
        return 2
    trim = int(round(num_layers * 0.05))
    return max(1, min(4, trim))


def resolve_trim(num_layers: int, trim_first: Optional[int], trim_last: Optional[int]) -> Tuple[int, int]:
    """Resolve effective trim values with depth-aware fallback."""
    auto = auto_trim_from_layers(num_layers)
    tf = auto if trim_first is None else max(0, int(trim_first))
    tl = auto if trim_last is None else max(0, int(trim_last))
    # Ensure a non-empty evaluation window.
    if tf + tl >= num_layers:
        max_side = max(0, (num_layers - 1) // 2)
        tf = min(tf, max_side)
        tl = min(tl, max_side)
    return tf, tl


def parse_experiment_doc(
    doc_path: str,
    fallback_models: List[str],
    fallback_n_tests: int,
    fallback_start_idx: int,
    fallback_n_prompts: int,
    fallback_spectral_top_k: int,
    fallback_trim_first: Optional[int],
    fallback_trim_last: Optional[int],
    fallback_spectral_neighbor_layers: int,
) -> Dict[str, object]:
    """
    Best-effort parser for experiment settings from docs/*.md.
    It safely falls back to provided defaults when fields are missing.
    """
    path = Path(doc_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment doc not found: {doc_path}")
    if path.suffix.lower() != ".md":
        raise ValueError(f"Experiment doc must be markdown (*.md): {doc_path}")
    if "docs" not in path.parts:
        raise ValueError(f"Experiment doc must be under docs/: {doc_path}")

    text = path.read_text(encoding="utf-8")
    text_l = text.lower()

    models = []
    for m in available_model_names():
        if m.lower() in text_l:
            models.append(m)
    if not models:
        models = fallback_models

    def _pick_int(patterns: List[str], default: Optional[int]) -> Optional[int]:
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
        return default

    n_tests = _pick_int(
        [
            r"--n-tests\s+(\d+)",
            r"n[-_ ]tests[:= ]+(\d+)",
            r"\((\d+)\s+cases?\)",
        ],
        fallback_n_tests,
    )
    start_idx = _pick_int([r"--start-idx\s+(\d+)", r"start[-_ ]idx[:= ]+(\d+)"], fallback_start_idx)
    n_prompts = _pick_int([r"--n-prompts\s+(\d+)", r"n[-_ ]prompts[:= ]+(\d+)"], fallback_n_prompts)
    spectral_top_k = _pick_int(
        [r"--spectral-top-k\s+(\d+)", r"spectral[-_ ]top[-_ ]k[:= ]+(\d+)"],
        fallback_spectral_top_k,
    )
    trim_first = _pick_int([r"--trim-first\s+(\d+)", r"trim[-_ ]first[:= ]+(\d+)"], fallback_trim_first)
    trim_last = _pick_int([r"--trim-last\s+(\d+)", r"trim[-_ ]last[:= ]+(\d+)"], fallback_trim_last)
    spectral_neighbor_layers = _pick_int(
        [r"--spectral-neighbor-layers\s+(\d+)", r"spectral[-_ ]neighbor[-_ ]layers[:= ]+(\d+)"],
        fallback_spectral_neighbor_layers,
    )

    return {
        "source_doc": str(path),
        "models": models,
        "n_tests": n_tests,
        "start_idx": start_idx,
        "n_prompts": n_prompts,
        "spectral_top_k": spectral_top_k,
        "trim_first": trim_first,
        "trim_last": trim_last,
        "spectral_neighbor_layers": spectral_neighbor_layers,
    }


def extract_weights(handler: ModelHandler, template: str) -> Dict[int, torch.Tensor]:
    """Extract weights from all layers using the given name template, moved to CPU."""
    return {
        idx: handler._get_module(template.format(idx)).weight.detach().clone().cpu()
        for idx in range(handler.num_of_layers)
    }


def extract_attention_weights(handler: ModelHandler, proj_template: str) -> Dict[str, Dict[int, torch.Tensor]]:
    """Extract per-layer attention projection weights with architecture-aware fallback."""
    template_map = derive_attention_templates(proj_template)
    attention: Dict[str, Dict[int, torch.Tensor]] = {}

    for key, template in template_map.items():
        per_layer: Dict[int, torch.Tensor] = {}
        for idx in range(handler.num_of_layers):
            module_name = template.format(idx)
            try:
                mod = handler._get_module(module_name)
            except KeyError:
                continue

            W = mod.weight.detach().clone().cpu()
            if key == "qkv_combined":
                split = split_qkv_weight(W)
                if split is None:
                    per_layer[idx] = W
                else:
                    for sub_key, sub_w in split.items():
                        attention.setdefault(sub_key, {})[idx] = sub_w.detach().clone().cpu()
                continue

            per_layer[idx] = W

        if per_layer:
            attention[key] = per_layer

    return attention


def add_ipr_z_scores(summary: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    """Add within-distribution z-scores for key IPR metrics."""
    if not summary:
        return summary
    layers = sorted(summary.keys())
    for metric in ("global_ipr", "row_ipr_mean", "row_ipr_std"):
        values = np.array([summary[idx][metric] for idx in layers])
        mean, std = values.mean(), values.std() + 1e-12
        for idx in layers:
            summary[idx][f"{metric}_z"] = float((summary[idx][metric] - mean) / std)
    return summary


def load_test_cases(n_tests: int, start_idx: int = 0) -> List[dict]:
    """Load test cases from the CounterFact dataset."""
    ds = datasets.load_dataset("azhx/counterfact", split="train")
    cases = []
    for i, item in enumerate(ds):
        if i < start_idx:
            continue
        if len(cases) >= n_tests:
            break
        rw = item["requested_rewrite"]
        cases.append({
            "case_id": item.get("case_id", i),
            "subject": rw["subject"],
            "target_new_str": rw["target_new"]["str"],
            "target_true_str": rw["target_true"]["str"],
            "fact_tuple": (
                rw["prompt"], rw["subject"],
                " " + rw["target_new"]["str"],
                " " + rw["target_true"]["str"],
            ),
            "paraphrase_prompts": item.get("paraphrase_prompts", []) or [],
            "neighborhood_prompts": item.get("neighborhood_prompts", []) or [],
        })
    return cases


# ---------------------------------------------------------------------------
# Single-model benchmark
# ---------------------------------------------------------------------------

def run_single_model(
    model_name: str,
    test_cases: List[dict],
    n_prompts: int,
    spectral_top_k: int = 50,
    trim_first: Optional[int] = None,
    trim_last: Optional[int] = None,
    spectral_neighbor_layers: int = 1,
    spectral_rolling_window: int = 5,
    local_windows: Sequence[int] = (3, 5, 7),
    enable_attention_metrics: bool = True,
    enable_rank1_blind: bool = True,
    enable_symmetry_metrics: bool = True,
    enable_bottom_rank_svd: bool = True,
    bottom_rank_sweep_ranks: Sequence[int] = (4, 8, 16, 32),
    bottom_rank_top_svd_rank: int = 64,
    bottom_rank_boundary: int = 2,
    raw_spectral_max_top_k: Optional[int] = None,
    analysis_profile: str = "full",
    baseline_only: bool = False,
) -> dict:
    """Run the full ROME + structural-detection benchmark for one model."""
    analysis_profile = normalize_analysis_profile(analysis_profile)
    raw_only = analysis_profile == "raw"
    paper_only = analysis_profile == "paper"
    full_profile = analysis_profile == "full"

    cfg = build_cfg(model_name)
    LOGGER.info("Loading %s ...", cfg.model.name)
    handler = ModelHandler(cfg)
    LOGGER.info(
        "Loaded. layer=%d, emb=%d, hidden=%d, prompts=%d, multi_gpu=%s",
        handler._layer, handler.emb_shape, handler.hidden_dim, n_prompts,
        handler.is_multi_gpu,
    )

    proj_template = handler._layer_name_template
    cfg_fc_template = str(getattr(cfg.model, "fc_layer_name_template", "") or "").strip()
    fc_template = cfg_fc_template or get_fc_template(proj_template)
    eff_trim_first, eff_trim_last = resolve_trim(
        num_layers=handler.num_of_layers,
        trim_first=trim_first,
        trim_last=trim_last,
    )
    LOGGER.info(
        "Resolved trim for %s: first=%d last=%d (num_layers=%d)",
        model_name, eff_trim_first, eff_trim_last, handler.num_of_layers
    )

    # Determine optional module families without relying on baseline detector outputs.
    has_fc_template = bool(fc_template)
    if has_fc_template:
        try:
            _ = handler._get_module(fc_template.format(handler._layer))
        except KeyError:
            LOGGER.warning("Could not resolve fc template (%s) - IPR and fc-contrast analytics disabled", fc_template)
            has_fc_template = False
    else:
        LOGGER.warning(
            "No fc template configured/inferred for %s; running with proj-only metrics.",
            cfg.model.name,
        )

    attention_templates = derive_attention_templates(proj_template) if full_profile else {}

    LOGGER.info("Analysis profile for %s: %s", cfg.model.name, analysis_profile)

    # Detectors
    blind_detector = None if raw_only else BlindMSDDetector()
    spectral_detector = (
        SpectralDetector(
            top_k=spectral_top_k, boundary=2,
            trim_first_layers=eff_trim_first, trim_last_layers=eff_trim_last,
            neighbor_layers=spectral_neighbor_layers,
            rolling_window=spectral_rolling_window,
            local_windows=local_windows,
            store_raw_spectral=True,
            raw_only=raw_only,
            raw_spectral_max_top_k=raw_spectral_max_top_k,
            raw_payload_level="sv_only" if paper_only else "full",
            emit_local_window_scores=not paper_only,
        )
    )
    ipr_detector = (
        IPRDetector(trim_first=eff_trim_first, trim_last=eff_trim_last)
        if (has_fc_template and full_profile)
        else None
    )
    composite_detector = (
        CompositeDetector(top_k=spectral_top_k, trim_first=eff_trim_first, trim_last=eff_trim_last)
        if full_profile
        else None
    )
    edit_presence_detector = RomeEditPresenceDetector() if full_profile else None
    rank1_detector = (
        BlindRank1Detector(boundary=2, local_windows=local_windows)
        if (enable_rank1_blind and full_profile)
        else None
    )
    attention_detector = (
        AttentionContrastDetector(boundary=2, local_windows=local_windows)
        if (enable_attention_metrics and full_profile)
        else None
    )
    symmetry_detector = (
        MirrorSymmetryDetector(top_k=20, boundary=2, local_windows=local_windows)
        if (enable_symmetry_metrics and full_profile) else None
    )
    bottom_rank_detector = (
        BottomRankSVDDetector(
            sweep_ranks=bottom_rank_sweep_ranks,
            top_svd_rank=bottom_rank_top_svd_rank,
            boundary=bottom_rank_boundary,
        )
        if (enable_bottom_rank_svd and full_profile)
        else None
    )

    results = {
        "metadata": {
            "model": cfg.model.name,
            "target_layer": handler._layer,
            "n_tests": len(test_cases),
            "n_prompts": n_prompts,
            "multi_gpu": handler.is_multi_gpu,
            "n_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "num_layers": handler.num_of_layers,
            "analysis_profile": analysis_profile,
            "baseline_only": baseline_only,
            "timestamp": datetime.now().isoformat(),
            "spectral_config": {
                "top_k": spectral_top_k, "boundary": 2,
                "trim_first": eff_trim_first, "trim_last": eff_trim_last,
                "neighbor_layers": spectral_neighbor_layers,
                "rolling_window": spectral_rolling_window,
                "raw_spectral_max_top_k": (
                    None if raw_spectral_max_top_k is None else int(raw_spectral_max_top_k)
                ),
                "raw_payload_level": "sv_only" if paper_only else "full",
                "emit_local_window_scores": not paper_only,
                "signal_keys": SPECTRAL_SIGNAL_KEYS,
            },
            "analytics_config": {
                "local_windows": list(local_windows),
                "enable_attention_metrics": bool(enable_attention_metrics),
                "enable_rank1_blind": bool(enable_rank1_blind),
                "enable_symmetry_metrics": bool(enable_symmetry_metrics),
                "enable_bottom_rank_svd": bool(enable_bottom_rank_svd),
                "paper_spectral_lite": bool(paper_only),
                "bottom_rank_sweep_ranks": [int(x) for x in bottom_rank_sweep_ranks],
                "bottom_rank_top_svd_rank": int(bottom_rank_top_svd_rank),
                "bottom_rank_boundary": int(bottom_rank_boundary),
                "fc_template": fc_template,
                "attention_template_candidates": sorted(attention_templates.keys()) if attention_templates else [],
                "blind_assumption": "edited-model-only; paired proj/fc matrices allowed",
            },
        },
        "tests": [],
    }

    counts = {
        k: 0
        for k in (
            "rome",
            "blind",
            "spectral",
            "ipr",
            "composite",
            "presence",
            "rank1_blind",
            "attention",
            "symmetry",
            "bottom_rank_svd",
            "novel_metrics",
        )
    }

    # Cache baseline weights once per model run; only the target proj layer changes per edit.
    baseline_proj = extract_weights(handler, proj_template)
    baseline_fc = None
    if has_fc_template and fc_template:
        try:
            baseline_fc = extract_weights(handler, fc_template)
        except (KeyError, ValueError):
            LOGGER.warning("Could not extract baseline fc weights from model (%s)", fc_template)
            baseline_fc = None
    baseline_attention = extract_attention_weights(handler, proj_template) if (enable_attention_metrics and full_profile) else {}

    bottom_rank_head = None
    bottom_rank_head_device = None
    bottom_rank_head_dtype = torch.float32
    if bottom_rank_detector is not None:
        head_getter = getattr(handler.model, "get_output_embeddings", None)
        if callable(head_getter):
            bottom_rank_head = head_getter()
        if bottom_rank_head is None:
            bottom_rank_head = getattr(handler.model, "lm_head", None)

        if bottom_rank_head is None:
            LOGGER.warning(
                "Bottom-rank SVD detector disabled: no output embedding head available for %s",
                cfg.model.name,
            )
            bottom_rank_detector = None
        else:
            try:
                head_param = next(bottom_rank_head.parameters())
                bottom_rank_head_device = head_param.device
                bottom_rank_head_dtype = head_param.dtype
            except StopIteration:
                bottom_rank_head_device = torch.device(handler.device)
                bottom_rank_head_dtype = torch.float32

    results["metadata"]["analytics_config"]["enable_bottom_rank_svd"] = bool(bottom_rank_detector is not None)

    def _predict_token_from_hidden(hidden_vec: torch.Tensor) -> Tuple[int, str]:
        if bottom_rank_head is None or bottom_rank_head_device is None:
            raise RuntimeError("bottom-rank head unavailable")
        with torch.no_grad():
            hidden = hidden_vec.detach().to(device=bottom_rank_head_device, dtype=bottom_rank_head_dtype)
            logits = bottom_rank_head(hidden.unsqueeze(0).unsqueeze(0))
            if isinstance(logits, tuple):
                logits = logits[0]
            if logits.ndim == 3:
                next_logits = logits[0, -1, :]
            elif logits.ndim == 2:
                next_logits = logits[0, :]
            else:
                next_logits = logits.reshape(-1)
            token_id = int(torch.argmax(next_logits).item())
        return token_id, handler.tokenizer.decode([token_id])

    clear_linalg_caches()

    required_successes_hint_raw = os.getenv("ROME_REQUIRED_SUCCESSES", "").strip()
    attempt_hint_raw = os.getenv("ROME_ATTEMPT", "").strip()
    cumulative_hint_raw = os.getenv("ROME_CUMULATIVE_SUCCESSES", "").strip()

    required_successes_hint = int(required_successes_hint_raw) if required_successes_hint_raw.isdigit() else None
    attempt_hint = int(attempt_hint_raw) if attempt_hint_raw.isdigit() else None
    cumulative_hint = int(cumulative_hint_raw) if cumulative_hint_raw.isdigit() else None

    for i, case in enumerate(test_cases):
        case_tag = f"[{i + 1}/{len(test_cases)}]"
        context_parts = []
        if required_successes_hint is not None:
            current_success = cumulative_hint if cumulative_hint is not None else counts["rome"]
            context_parts.append(f"goal={current_success}/{required_successes_hint}")
        if attempt_hint is not None:
            context_parts.append(f"attempt={attempt_hint}")

        if context_parts:
            LOGGER.info("%s (%s) %s", case_tag, ", ".join(context_parts), case["subject"])
        else:
            LOGGER.info("%s %s", case_tag, case["subject"])
        layer_name = proj_template.format(handler._layer)
        old_W = handler._get_module(layer_name).weight.detach().clone()
        entry = {"case_id": case["case_id"], "subject": case["subject"], "error": None}

        try:
            if baseline_only:
                # --- Baseline-only mode: skip ROME edit, run detectors on unmodified weights ---
                entry["rome"] = {"success": True, "baseline_only": True}
                counts["rome"] += 1

                modified_proj = dict(baseline_proj)
                modified_fc = baseline_fc
                attention_weights = baseline_attention
                k = None  # no ROME key vector

                if raw_only:
                    spectral_res = spectral_detector.detect(modified_proj, fc_weights=modified_fc)
                    entry.update({
                        "spectral_detection": to_serializable(spectral_res),
                        "accuracy": {"rome_success": True, "baseline_only": True},
                    })
                    LOGGER.info("  BASELINE RAW_JSON=OK")
                elif paper_only:
                    spectral_res = spectral_detector.detect(modified_proj, fc_weights=modified_fc) if spectral_detector else {}
                    blind_res = blind_detector.detect_layer_features_only(modified_proj) if blind_detector else {}
                    entry.update({
                        "blind_detection": to_serializable(blind_res),
                        "spectral_detection": to_serializable(spectral_res),
                        "accuracy": {"rome_success": True, "baseline_only": True, "detection_skipped": False},
                    })
                    LOGGER.info("  BASELINE PAPER_JSON=OK")
                else:
                    spectral_res = spectral_detector.detect(modified_proj, fc_weights=modified_fc)
                    blind_res = blind_detector.detect(modified_proj) if blind_detector else {}
                    ipr_res = ipr_detector.detect(modified_proj, modified_fc) if (ipr_detector and modified_fc is not None) else {}
                    composite_res = composite_detector.detect(modified_proj, fc_weights=modified_fc, spectral_result=spectral_res) if composite_detector else {}
                    rank1_res = rank1_detector.detect(modified_proj, fc_weights=modified_fc) if rank1_detector else {}
                    attention_res = (
                        attention_detector.detect(
                            modified_proj,
                            attention_weights=attention_weights,
                            fc_weights=modified_fc,
                        )
                        if (attention_detector and attention_weights) else {}
                    )
                    symmetry_res = symmetry_detector.detect(modified_proj, fc_weights=modified_fc) if symmetry_detector else {}
                    presence_res = edit_presence_detector.detect(
                        modified_proj=modified_proj,
                        modified_fc=modified_fc,
                        modified_spectral=spectral_res,
                    ) if edit_presence_detector else {}
                    bottom_rank_res = {}  # requires probe_vector k from ROME

                    mod_ipr_proj = add_ipr_z_scores(layer_ipr_summary(modified_proj))
                    mod_ipr_fc = add_ipr_z_scores(layer_ipr_summary(modified_fc)) if modified_fc is not None else {}
                    mod_ipr_disc = layer_fc_proj_ipr_discrepancy(modified_proj, modified_fc) if modified_fc is not None else {}

                    # Novel metrics (baseline)
                    novel_res = compute_novel_metrics(modified_proj, fc_weights=modified_fc)

                    entry.update({
                        "blind_detection": to_serializable(blind_res),
                        "spectral_detection": to_serializable(spectral_res),
                        "composite_detection": to_serializable(composite_res),
                        "rank1_blind_detection": to_serializable(rank1_res),
                        "attention_detection": to_serializable(attention_res),
                        "symmetry_detection": to_serializable(symmetry_res),
                        "bottom_rank_svd_detection": to_serializable(bottom_rank_res),
                        "edit_presence_detection": to_serializable(presence_res),
                        "interlayer": to_serializable(collect_all_interlayer_data(modified_proj)),
                        "novel_metrics_detection": to_serializable(novel_res),
                        "ipr": {
                            "proj": to_serializable(mod_ipr_proj),
                            "fc": to_serializable(mod_ipr_fc),
                            "fc_vs_proj": to_serializable(mod_ipr_disc),
                            "detection": to_serializable(ipr_res),
                        },
                        "accuracy": {"baseline_only": True},
                    })
                    LOGGER.info(
                        "  BASELINE Blind=L%s Spectral=L%s Composite=L%s Rank1=L%s Attention=L%s Symmetry=L%s",
                        blind_res.get("anomalous_layer"),
                        spectral_res.get("anomalous_layer"),
                        composite_res.get("anomalous_layer"),
                        rank1_res.get("anomalous_layer", "N/A"),
                        attention_res.get("anomalous_layer", "N/A"),
                        symmetry_res.get("anomalous_layer", "N/A"),
                    )

            else:
                # --- Normal ROME edit mode ---
                fact = case["fact_tuple"]

                # ROME edit
                k = gather_k(handler, fact_tuple=fact, N=n_prompts)
                delta = optimize_v(
                    handler, fact_tuple=fact,
                    N_prompts=n_prompts, N_optim_steps=handler.epochs,
                )
                new_W, _, _ = insert_kv(handler, k, delta)

                # Compute ROME paper metrics (probability-based)
                prompt_text = fact[0].format(fact[1])
                target_new_str = case.get("target_new_str", fact[2].strip())
                target_true_str = case.get("target_true_str", fact[3].strip())
                rome_metrics = compute_rome_metrics(
                    handler, prompt_text,
                    target_new_str, target_true_str,
                    paraphrase_prompts=case.get("paraphrase_prompts", []),
                    neighborhood_prompts=case.get("neighborhood_prompts", []),
                )

                # ES >= 1.0 means target_new is more probable than target_true
                rome_ok = bool(rome_metrics["efficacy_score"] >= 1.0)

                entry["rome"] = {
                    "success": rome_ok,
                    "efficacy_score": rome_metrics["efficacy_score"],
                    "efficacy_magnitude": rome_metrics["efficacy_magnitude"],
                    "paraphrase_score": rome_metrics["paraphrase_score"],
                    "neighborhood_score": rome_metrics["neighborhood_score"],
                    "overall_score": rome_metrics["overall_score"],
                    "rewrite_nll": rome_metrics["rewrite_nll"],
                    "paraphrase_nll": rome_metrics["paraphrase_nll"],
                    "neighborhood_nll": rome_metrics["neighborhood_nll"],
                    "k_norm": k.norm().item(),
                    "delta_norm": delta.norm().item(),
                }
                if rome_ok:
                    counts["rome"] += 1

                if raw_only:
                    modified_proj = dict(baseline_proj)
                    modified_proj[handler._layer] = handler._get_module(layer_name).weight.detach().clone().cpu()
                    modified_fc = baseline_fc
                    spectral_res = spectral_detector.detect(modified_proj, fc_weights=modified_fc)
                    entry.update({
                        "spectral_detection": to_serializable(spectral_res),
                        "accuracy": {
                            "rome_success": rome_ok,
                            "efficacy_score": rome_metrics["efficacy_score"],
                            "efficacy_magnitude": rome_metrics["efficacy_magnitude"],
                            "paraphrase_score": rome_metrics["paraphrase_score"],
                            "neighborhood_score": rome_metrics["neighborhood_score"],
                            "overall_score": rome_metrics["overall_score"],
                        },
                    })

                    LOGGER.info(
                        "  ROME=%s(ES=%.1f PS=%s NS=%s S=%.3f) RAW_JSON=OK (raw_spectral=%s)",
                        "OK" if rome_ok else "FAIL",
                        rome_metrics["efficacy_score"],
                        f'{rome_metrics["paraphrase_score"]:.3f}' if rome_metrics["paraphrase_score"] is not None else "N/A",
                        f'{rome_metrics["neighborhood_score"]:.3f}' if rome_metrics["neighborhood_score"] is not None else "N/A",
                        rome_metrics["overall_score"],
                        bool((spectral_res or {}).get("raw_spectral")),
                    )
                elif paper_only:
                    entry["accuracy"] = {
                        "rome_success": rome_ok,
                        "efficacy_score": rome_metrics["efficacy_score"],
                        "efficacy_magnitude": rome_metrics["efficacy_magnitude"],
                        "paraphrase_score": rome_metrics["paraphrase_score"],
                        "neighborhood_score": rome_metrics["neighborhood_score"],
                        "overall_score": rome_metrics["overall_score"],
                        "detection_skipped": not rome_ok,
                    }
                    if rome_ok:
                        modified_proj = dict(baseline_proj)
                        modified_proj[handler._layer] = handler._get_module(layer_name).weight.detach().clone().cpu()
                        modified_fc = baseline_fc
                        spectral_res = spectral_detector.detect(modified_proj, fc_weights=modified_fc) if spectral_detector else {}
                        blind_res = blind_detector.detect_layer_features_only(modified_proj) if blind_detector else {}
                        entry["blind_detection"] = to_serializable(blind_res)
                        entry["spectral_detection"] = to_serializable(spectral_res)
                        LOGGER.info(
                            "  ROME=OK(ES=%.1f PS=%s NS=%s S=%.3f) PAPER_JSON=OK",
                            rome_metrics["efficacy_score"],
                            f'{rome_metrics["paraphrase_score"]:.3f}' if rome_metrics["paraphrase_score"] is not None else "N/A",
                            f'{rome_metrics["neighborhood_score"]:.3f}' if rome_metrics["neighborhood_score"] is not None else "N/A",
                            rome_metrics["overall_score"],
                        )
                    else:
                        entry["detection_skipped"] = True
                        entry["detection_skip_reason"] = "rome_unsuccessful"
                        LOGGER.info(
                            "  ROME=FAIL(ES=%.1f PS=%s NS=%s S=%.3f) PAPER_SKIP=rome_unsuccessful",
                            rome_metrics["efficacy_score"],
                            f'{rome_metrics["paraphrase_score"]:.3f}' if rome_metrics["paraphrase_score"] is not None else "N/A",
                            f'{rome_metrics["neighborhood_score"]:.3f}' if rome_metrics["neighborhood_score"] is not None else "N/A",
                            rome_metrics["overall_score"],
                        )
                else:
                    modified_proj = dict(baseline_proj)
                    modified_proj[handler._layer] = handler._get_module(layer_name).weight.detach().clone().cpu()
                    modified_fc = baseline_fc
                    attention_weights = baseline_attention
                    spectral_res = spectral_detector.detect(modified_proj, fc_weights=modified_fc)
                    blind_res = blind_detector.detect(modified_proj) if blind_detector else {}
                    ipr_res = ipr_detector.detect(modified_proj, modified_fc) if (ipr_detector and modified_fc is not None) else {}
                    composite_res = composite_detector.detect(modified_proj, fc_weights=modified_fc, spectral_result=spectral_res) if composite_detector else {}
                    rank1_res = rank1_detector.detect(modified_proj, fc_weights=modified_fc) if rank1_detector else {}
                    attention_res = (
                        attention_detector.detect(
                            modified_proj,
                            attention_weights=attention_weights,
                            fc_weights=modified_fc,
                        )
                        if (attention_detector and attention_weights) else {}
                    )
                    symmetry_res = symmetry_detector.detect(modified_proj, fc_weights=modified_fc) if symmetry_detector else {}
                    presence_res = edit_presence_detector.detect(
                        modified_proj=modified_proj,
                        modified_fc=modified_fc,
                        modified_spectral=spectral_res,
                    ) if edit_presence_detector else {}
                    bottom_rank_res = (
                        bottom_rank_detector.detect(
                            modified_proj,
                            probe_vector=k.detach().view(-1).cpu(),
                            token_predictor=_predict_token_from_hidden,
                        )
                        if bottom_rank_detector else {}
                    )

                    # IPR analysis
                    mod_ipr_proj = add_ipr_z_scores(layer_ipr_summary(modified_proj))
                    mod_ipr_fc = add_ipr_z_scores(layer_ipr_summary(modified_fc)) if modified_fc is not None else {}
                    mod_ipr_disc = layer_fc_proj_ipr_discrepancy(modified_proj, modified_fc) if modified_fc is not None else {}

                    # Novel metrics
                    novel_res = compute_novel_metrics(modified_proj, fc_weights=modified_fc)

                    correct = {
                        "blind": blind_res.get("anomalous_layer") == handler._layer,
                        "spectral": spectral_res.get("anomalous_layer") == handler._layer,
                        "ipr": ipr_res.get("anomalous_layer") == handler._layer if ipr_res else False,
                        "composite": composite_res.get("anomalous_layer") == handler._layer,
                        "rank1_blind": rank1_res.get("anomalous_layer") == handler._layer if rank1_res else False,
                        "attention": attention_res.get("anomalous_layer") == handler._layer if attention_res else False,
                        "symmetry": symmetry_res.get("anomalous_layer") == handler._layer if symmetry_res else False,
                        "bottom_rank_svd": bottom_rank_res.get("anomalous_layer") == handler._layer if bottom_rank_res else False,
                        "novel_metrics": novel_res.get("anomalous_layer") == handler._layer if novel_res else False,
                        "presence": bool(presence_res.get("is_edited")),
                    }
                    for name, ok in correct.items():
                        if ok:
                            counts[name] += 1

                    entry.update({
                        "blind_detection": to_serializable(blind_res),
                        "spectral_detection": to_serializable(spectral_res),
                        "composite_detection": to_serializable(composite_res),
                        "rank1_blind_detection": to_serializable(rank1_res),
                        "attention_detection": to_serializable(attention_res),
                        "symmetry_detection": to_serializable(symmetry_res),
                        "bottom_rank_svd_detection": to_serializable(bottom_rank_res),
                        "edit_presence_detection": to_serializable(presence_res),
                        "interlayer": to_serializable(collect_all_interlayer_data(modified_proj)),
                        "novel_metrics_detection": to_serializable(novel_res),
                        "ipr": {
                            "proj": to_serializable(mod_ipr_proj),
                            "fc": to_serializable(mod_ipr_fc),
                            "fc_vs_proj": to_serializable(mod_ipr_disc),
                            "detection": to_serializable(ipr_res),
                        },
                        "accuracy": {
                            "rome_success": rome_ok,
                            "efficacy_score": rome_metrics["efficacy_score"],
                            "efficacy_magnitude": rome_metrics["efficacy_magnitude"],
                            "paraphrase_score": rome_metrics["paraphrase_score"],
                            "neighborhood_score": rome_metrics["neighborhood_score"],
                            "overall_score": rome_metrics["overall_score"],
                            **{f"{name}_correct": v for name, v in correct.items()},
                        },
                    })

                    LOGGER.info(
                        "  ROME=%s(ES=%.1f PS=%s NS=%s S=%.3f) Blind=L%s Spectral=L%s Composite=L%s(%s) Rank1=L%s Attention=L%s Symmetry=L%s Presence=%s(%.3f) IPR=L%s(%.3f) BottomSVD=L%s Novel=L%s",
                        "OK" if rome_ok else "FAIL",
                        rome_metrics["efficacy_score"],
                        f'{rome_metrics["paraphrase_score"]:.3f}' if rome_metrics["paraphrase_score"] is not None else "N/A",
                        f'{rome_metrics["neighborhood_score"]:.3f}' if rome_metrics["neighborhood_score"] is not None else "N/A",
                        rome_metrics["overall_score"],
                        blind_res.get("anomalous_layer"),
                        spectral_res.get("anomalous_layer"),
                        composite_res.get("anomalous_layer"),
                        composite_res.get("method_used", "?"),
                        rank1_res.get("anomalous_layer", "N/A"),
                        attention_res.get("anomalous_layer", "N/A"),
                        symmetry_res.get("anomalous_layer", "N/A"),
                        presence_res.get("is_edited"),
                        presence_res.get("confidence", 0.0),
                        ipr_res.get("anomalous_layer", "N/A"),
                        ipr_res.get("anomaly_score", 0),
                        bottom_rank_res.get("anomalous_layer", "N/A"),
                        novel_res.get("anomalous_layer", "N/A"),
                    )

        except Exception as e:
            entry["error"] = str(e)
            entry["skipped"] = True
            LOGGER.warning("  SKIPPED: %s", e)
            if os.getenv("LATIUM_LOG_SKIP_TRACEBACK", "").strip().lower() in {"1", "true", "yes", "on"}:
                LOGGER.warning("  SKIPPED TRACEBACK:\n%s", traceback.format_exc())
        finally:
            handler.remove_hooks()
            handler._get_module(layer_name).weight = torch.nn.Parameter(old_W)
            results["tests"].append(entry)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    ok_tests = [t for t in results["tests"] if not t.get("skipped") and not t.get("error")]
    n = len(ok_tests)
    case_errors = len(test_cases) - n
    rome_success_count = sum(1 for t in ok_tests if t.get("rome", {}).get("success"))
    detector_eligible_count = sum(
        1
        for t in ok_tests
        if t.get("rome", {}).get("success")
        and bool(t.get("blind_detection", {}).get("layer_features"))
    )

    # Aggregate ROME paper metrics from non-skipped tests
    _es = [t["rome"]["efficacy_score"] for t in ok_tests if "rome" in t and "efficacy_score" in t.get("rome", {})]
    _em = [t["rome"]["efficacy_magnitude"] for t in ok_tests if "rome" in t and "efficacy_magnitude" in t.get("rome", {})]
    _ps = [t["rome"]["paraphrase_score"] for t in ok_tests if "rome" in t and t.get("rome", {}).get("paraphrase_score") is not None]
    _ns = [t["rome"]["neighborhood_score"] for t in ok_tests if "rome" in t and t.get("rome", {}).get("neighborhood_score") is not None]
    _os = [t["rome"]["overall_score"] for t in ok_tests if "rome" in t and "overall_score" in t.get("rome", {})]

    results["summary"] = {
        "total": len(test_cases),
        "successful": n,
        "skipped": case_errors,
        "cases_total": len(test_cases),
        "cases_completed": n,
        "cases_error": case_errors,
        "rome_success_count": rome_success_count,
        "rome_success_rate": rome_success_count / n if n else 0.0,
        "detector_eligible_count": detector_eligible_count,
        **{f"{k}_rate": counts[k] / n if n else 0 for k in counts},
        "mean_efficacy_score": float(np.mean(_es)) if _es else 0.0,
        "mean_efficacy_magnitude": float(np.mean(_em)) if _em else 0.0,
        "mean_paraphrase_score": float(np.mean(_ps)) if _ps else 0.0,
        "mean_neighborhood_score": float(np.mean(_ns)) if _ns else 0.0,
        "mean_overall_score": float(np.mean(_os)) if _os else 0.0,
    }
    if raw_only:
        LOGGER.info(
            "[%s][raw] ROME=%d/%d ES=%.3f PS=%.3f NS=%.3f S=%.3f skip=%d",
            cfg.model.name,
            counts["rome"],
            n,
            results["summary"]["mean_efficacy_score"],
            results["summary"]["mean_paraphrase_score"],
            results["summary"]["mean_neighborhood_score"],
            results["summary"]["mean_overall_score"],
            len(test_cases) - n,
        )
    elif paper_only:
        LOGGER.info(
            "[%s][paper] completed=%d/%d errors=%d rome=%d/%d detector_eligible=%d ES=%.3f PS=%.3f NS=%.3f S=%.3f",
            cfg.model.name,
            n,
            len(test_cases),
            case_errors,
            rome_success_count,
            n,
            detector_eligible_count,
            results["summary"]["mean_efficacy_score"],
            results["summary"]["mean_paraphrase_score"],
            results["summary"]["mean_neighborhood_score"],
            results["summary"]["mean_overall_score"],
        )
    else:
        LOGGER.info(
            "[%s] ROME=%d/%d ES=%.3f PS=%.3f NS=%.3f S=%.3f Blind=%d/%d Spectral=%d/%d Composite=%d/%d Rank1=%d/%d Attention=%d/%d Symmetry=%d/%d IPR=%d/%d BottomSVD=%d/%d Novel=%d/%d skip=%d",
            cfg.model.name, counts["rome"], n,
            results["summary"]["mean_efficacy_score"],
            results["summary"]["mean_paraphrase_score"],
            results["summary"]["mean_neighborhood_score"],
            results["summary"]["mean_overall_score"],
            counts["blind"], n,
            counts["spectral"], n, counts["composite"], n,
            counts["rank1_blind"], n, counts["attention"], n, counts["symmetry"], n,
            counts["ipr"], n, counts["bottom_rank_svd"], n,
            counts["novel_metrics"], n, len(test_cases) - n,
        )

    # Free GPU
    clear_linalg_caches()
    del handler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Multi-model entry point
# ---------------------------------------------------------------------------

def run_benchmark(
    models: List[str],
    n_tests: int = 30,
    start_idx: int = 0,
    output_dir: str = "./analysis_out",
    n_prompts: int = 50,
    spectral_top_k: int = 50,
    trim_first: Optional[int] = None,
    trim_last: Optional[int] = None,
    spectral_neighbor_layers: int = 1,
    spectral_rolling_window: int = 5,
    local_windows: Sequence[int] = (3, 5, 7),
    enable_attention_metrics: bool = True,
    enable_rank1_blind: bool = True,
    enable_symmetry_metrics: bool = True,
    enable_bottom_rank_svd: bool = True,
    bottom_rank_sweep_ranks: Sequence[int] = (4, 8, 16, 32),
    bottom_rank_top_svd_rank: int = 64,
    bottom_rank_boundary: int = 2,
    raw_spectral_max_top_k: Optional[int] = None,
    runs_per_model: int = 1,
    run_start_idx_step: int = 0,
    sweep_configs: Optional[Sequence[Dict[str, object]]] = None,
    sweep_tag: Optional[str] = None,
    analysis_profile: str = "full",
    baseline_only: bool = False,
    create_paired_baseline: bool = True,
    fail_on_missing_second_moment: bool = False,
):
    """Run benchmark across models, optionally with multi-run + sweep experiments."""
    auto_env = os.getenv("ROME_ALLOW_SECOND_MOMENT_AUTOCOMPUTE", "").strip().lower()
    if auto_env in {"1", "true", "yes", "y"}:
        LOGGER.warning(
            "Forcing ROME_ALLOW_SECOND_MOMENT_AUTOCOMPUTE=0 in structural benchmark "
            "to enforce precomputed second moments only."
        )
    os.environ["ROME_ALLOW_SECOND_MOMENT_AUTOCOMPUTE"] = "0"

    models = normalize_models_arg(models)
    if not models:
        raise ValueError("No models provided after parsing --models input.")

    runs_per_model = max(1, int(runs_per_model))
    run_start_idx_step = max(0, int(run_start_idx_step))

    if sweep_configs:
        effective_sweeps = [dict(cfg) for cfg in sweep_configs]
    else:
        effective_sweeps = [{
            "spectral_top_k": int(spectral_top_k),
            "trim_first": None if trim_first is None else int(trim_first),
            "trim_last": None if trim_last is None else int(trim_last),
            "spectral_neighbor_layers": int(spectral_neighbor_layers),
            "spectral_rolling_window": int(spectral_rolling_window),
            "local_windows": [int(w) for w in local_windows],
        }]

    total_runs_for_model = len(effective_sweeps) * runs_per_model
    LOGGER.info(
        "Resolved run plan: sweeps=%d runs_per_model=%d total_runs_per_model=%d run_start_idx_step=%d",
        len(effective_sweeps),
        runs_per_model,
        total_runs_for_model,
        run_start_idx_step,
    )
    for i, cfg in enumerate(effective_sweeps, start=1):
        LOGGER.info(
            "  Sweep[%d/%d]: top_k=%s trim=(%s,%s) neighbor=%s rolling=%s local_windows=%s",
            i,
            len(effective_sweeps),
            cfg.get("spectral_top_k"),
            _format_optional_int(cfg.get("trim_first")),
            _format_optional_int(cfg.get("trim_last")),
            cfg.get("spectral_neighbor_layers"),
            cfg.get("spectral_rolling_window"),
            cfg.get("local_windows"),
        )

    test_case_cache: Dict[int, List[dict]] = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, object] = {}

    for model_name in models:
        LOGGER.info("=" * 60)
        LOGGER.info("Benchmark: %s", model_name)
        LOGGER.info("=" * 60)

        safe_name = model_name.replace("/", "_").replace("\\", "_")
        file_prefix = "baseline_structural" if baseline_only else "rome_structural"

        # Never auto-compute missing second moments inside this benchmark path.
        model_cfg = load_model_config(model_name)
        sm_files, sm_dir = find_second_moment_files(model_cfg)
        if not sm_files:
            skip_msg = (
                f"Missing second moment stats for model={model_cfg.name} layer={model_cfg.layer} "
                f"in {sm_dir}."
            )
            if fail_on_missing_second_moment:
                LOGGER.error(skip_msg)
                raise FileNotFoundError(skip_msg)
            LOGGER.warning(skip_msg)
            for sweep_idx, sweep_cfg in enumerate(effective_sweeps, start=1):
                cfg_slug = sweep_config_slug(sweep_cfg)
                for run_idx in range(1, runs_per_model + 1):
                    run_ordinal = (sweep_idx - 1) * runs_per_model + run_idx
                    start_idx_used = start_idx + run_start_idx_step * (run_ordinal - 1)
                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    out_file = output_path / (
                        f"{file_prefix}_{safe_name}_{cfg_slug}"
                        f"_s{sweep_idx:02d}_r{run_idx:02d}_{ts}.json"
                    )

                    skipped_result = {
                        "metadata": {
                            "model": model_name,
                            "timestamp": datetime.now().isoformat(),
                            "skipped": True,
                            "skip_reason": "missing_second_moment",
                            "second_moment_dir": str(sm_dir),
                            "run_index": run_idx,
                            "run_ordinal": run_ordinal,
                            "runs_per_model": runs_per_model,
                            "run_start_idx_step": run_start_idx_step,
                            "start_idx_used": start_idx_used,
                            "end_idx_used": start_idx_used + max(0, n_tests - 1),
                            "sweep_index": sweep_idx,
                            "sweep_size": len(effective_sweeps),
                            "sweep_config": to_serializable(sweep_cfg),
                            "sweep_slug": cfg_slug,
                            "total_runs_for_model": total_runs_for_model,
                        },
                        "error": skip_msg,
                        "tests": [],
                        "summary": {
                            "total": n_tests,
                            "successful": 0,
                            "skipped": n_tests,
                        },
                    }
                    if sweep_tag:
                        skipped_result["metadata"]["sweep_tag"] = str(sweep_tag)

                    _append_model_result(all_results, model_name, skipped_result)
                    with open(out_file, "w") as f:
                        json.dump(to_serializable(skipped_result), f, indent=2)
                    LOGGER.info("Saved skipped report: %s", out_file)
            continue

        for sweep_idx, sweep_cfg in enumerate(effective_sweeps, start=1):
            cfg_slug = sweep_config_slug(sweep_cfg)
            cfg_top_k = int(sweep_cfg.get("spectral_top_k", spectral_top_k))
            cfg_trim_first_raw = sweep_cfg.get("trim_first", trim_first)
            cfg_trim_last_raw = sweep_cfg.get("trim_last", trim_last)
            cfg_trim_first = None if cfg_trim_first_raw is None else int(cfg_trim_first_raw)
            cfg_trim_last = None if cfg_trim_last_raw is None else int(cfg_trim_last_raw)
            cfg_neighbor = int(sweep_cfg.get("spectral_neighbor_layers", spectral_neighbor_layers))
            cfg_rolling = int(sweep_cfg.get("spectral_rolling_window", spectral_rolling_window))

            cfg_lw_raw = sweep_cfg.get("local_windows", local_windows)
            if isinstance(cfg_lw_raw, (list, tuple)):
                cfg_local_windows = tuple(int(w) for w in cfg_lw_raw)
            else:
                cfg_local_windows = tuple(parse_local_windows(str(cfg_lw_raw), default=local_windows))
            if not cfg_local_windows:
                cfg_local_windows = tuple(int(w) for w in local_windows)

            for run_idx in range(1, runs_per_model + 1):
                run_ordinal = (sweep_idx - 1) * runs_per_model + run_idx
                start_idx_used = start_idx + run_start_idx_step * (run_ordinal - 1)

                if start_idx_used not in test_case_cache:
                    test_case_cache[start_idx_used] = load_test_cases(n_tests, start_idx_used)
                run_test_cases = test_case_cache[start_idx_used]

                run_token = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                out_file = _build_run_output_path(
                    output_path,
                    file_prefix,
                    safe_name,
                    cfg_slug,
                    sweep_idx,
                    run_idx,
                    run_token,
                )
                baseline_out_file = _build_run_output_path(
                    output_path,
                    "baseline_structural",
                    safe_name,
                    cfg_slug,
                    sweep_idx,
                    run_idx,
                    run_token,
                )
                run_metadata = _build_run_metadata(
                    model_name=model_name,
                    start_idx_used=start_idx_used,
                    run_idx=run_idx,
                    run_ordinal=run_ordinal,
                    runs_per_model=runs_per_model,
                    run_start_idx_step=run_start_idx_step,
                    sweep_idx=sweep_idx,
                    sweep_cfg=sweep_cfg,
                    sweep_count=len(effective_sweeps),
                    cfg_slug=cfg_slug,
                    total_runs_for_model=total_runs_for_model,
                    sweep_tag=sweep_tag,
                    n_cases=len(run_test_cases),
                )

                try:
                    if create_paired_baseline and not baseline_only:
                        LOGGER.info(
                            "Running paired baseline for %s (sweep=%d run=%d) before edited benchmark",
                            model_name,
                            sweep_idx,
                            run_idx,
                        )
                        baseline_results = run_single_model(
                            model_name,
                            run_test_cases,
                            n_prompts,
                            spectral_top_k=cfg_top_k,
                            trim_first=cfg_trim_first,
                            trim_last=cfg_trim_last,
                            spectral_neighbor_layers=cfg_neighbor,
                            spectral_rolling_window=cfg_rolling,
                            local_windows=cfg_local_windows,
                            enable_attention_metrics=enable_attention_metrics,
                            enable_rank1_blind=enable_rank1_blind,
                            enable_symmetry_metrics=enable_symmetry_metrics,
                            enable_bottom_rank_svd=enable_bottom_rank_svd,
                            bottom_rank_sweep_ranks=bottom_rank_sweep_ranks,
                            bottom_rank_top_svd_rank=bottom_rank_top_svd_rank,
                            bottom_rank_boundary=bottom_rank_boundary,
                            raw_spectral_max_top_k=raw_spectral_max_top_k,
                            analysis_profile=analysis_profile,
                            baseline_only=True,
                        )
                        baseline_metadata = baseline_results.setdefault("metadata", {})
                        baseline_metadata.update(run_metadata)
                        baseline_metadata.update({
                            "paired_role": "baseline",
                            "paired_run_id": run_token,
                            "baseline_structural_file": baseline_out_file.name,
                            "edited_structural_file": out_file.name,
                            "baseline_source": "auto_paired",
                        })
                        with open(baseline_out_file, "w") as f:
                            json.dump(to_serializable(baseline_results), f, indent=2)
                        LOGGER.info("Saved paired baseline: %s", baseline_out_file)

                    model_results = run_single_model(
                        model_name,
                        run_test_cases,
                        n_prompts,
                        spectral_top_k=cfg_top_k,
                        trim_first=cfg_trim_first,
                        trim_last=cfg_trim_last,
                        spectral_neighbor_layers=cfg_neighbor,
                        spectral_rolling_window=cfg_rolling,
                        local_windows=cfg_local_windows,
                        enable_attention_metrics=enable_attention_metrics,
                        enable_rank1_blind=enable_rank1_blind,
                        enable_symmetry_metrics=enable_symmetry_metrics,
                        enable_bottom_rank_svd=enable_bottom_rank_svd,
                        bottom_rank_sweep_ranks=bottom_rank_sweep_ranks,
                        bottom_rank_top_svd_rank=bottom_rank_top_svd_rank,
                        bottom_rank_boundary=bottom_rank_boundary,
                        raw_spectral_max_top_k=raw_spectral_max_top_k,
                        analysis_profile=analysis_profile,
                        baseline_only=baseline_only,
                    )

                    metadata = model_results.setdefault("metadata", {})
                    metadata.update(run_metadata)
                    metadata.update({
                        "paired_role": "baseline" if baseline_only else "edited",
                        "paired_run_id": run_token,
                    })
                    if baseline_only:
                        metadata["baseline_structural_file"] = out_file.name
                    elif create_paired_baseline:
                        metadata.update({
                            "baseline_structural_file": baseline_out_file.name,
                            "baseline_source": "auto_paired",
                        })

                    _append_model_result(all_results, model_name, model_results)
                    with open(out_file, "w") as f:
                        json.dump(to_serializable(model_results), f, indent=2)
                    LOGGER.info("Saved: %s", out_file)

                except Exception as e:
                    LOGGER.exception(
                        "Model %s failed (sweep=%d run=%d): %s",
                        model_name,
                        sweep_idx,
                        run_idx,
                        e,
                    )
                    error_result = {
                        "metadata": {
                            "model": model_name,
                            "timestamp": datetime.now().isoformat(),
                        "run_index": run_idx,
                        "run_ordinal": run_ordinal,
                        "runs_per_model": runs_per_model,
                        "run_start_idx_step": run_start_idx_step,
                        "start_idx_used": start_idx_used,
                        "end_idx_used": start_idx_used + max(0, n_tests - 1),
                        "sweep_index": sweep_idx,
                        "sweep_size": len(effective_sweeps),
                        "sweep_config": to_serializable(sweep_cfg),
                            "sweep_slug": cfg_slug,
                            "total_runs_for_model": total_runs_for_model,
                        },
                        "error": str(e),
                        "tests": [],
                    }
                    if sweep_tag:
                        error_result["metadata"]["sweep_tag"] = str(sweep_tag)

                    _append_model_result(all_results, model_name, error_result)
                    with open(out_file, "w") as f:
                        json.dump(to_serializable(error_result), f, indent=2)
                    LOGGER.info("Saved failure report: %s", out_file)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ROME + Structural Analysis Benchmark")
    parser.add_argument("--models", "--model", nargs="+", default=["gpt2-large"],
                        help="Model config names (YAML stems or HF model names)")
    parser.add_argument("--n-tests", type=int, default=30)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument(
        "--run-start-idx-step",
        type=int,
        default=0,
        help="Offset start-idx by this step for each run ordinal (0 keeps same case slice).",
    )
    parser.add_argument(
        "--runs-per-model",
        type=int,
        default=1,
        help="Repeat count per model per sweep config.",
    )
    parser.add_argument("--output-dir", default="./analysis_out")
    parser.add_argument("--n-prompts", type=int, default=50)
    parser.add_argument("--spectral-top-k", type=int, default=50)
    parser.add_argument(
        "--raw-spectral-max-top-k",
        type=int,
        default=None,
        help=(
            "Optional max rank to store in spectral_detection.raw_spectral payloads. "
            "Use this to enable broader post-hoc top-k sweeps from one saved JSON."
        ),
    )
    parser.add_argument(
        "--trim-first",
        type=int,
        default=None,
        help="Layers to trim at the start; omitted => auto depth-aware trim (~5%% of layers)",
    )
    parser.add_argument(
        "--trim-last",
        type=int,
        default=None,
        help="Layers to trim at the end; omitted => auto depth-aware trim (~5%% of layers)",
    )
    parser.add_argument("--spectral-neighbor-layers", type=int, default=1)
    parser.add_argument("--spectral-rolling-window", type=int, default=5)
    parser.add_argument(
        "--local-windows",
        type=str,
        default="3,5,7",
        help="Comma-separated odd window sizes for local anomaly scoring, e.g. 3,5,7",
    )
    parser.add_argument(
        "--sweep-mode",
        choices=["zip", "product"],
        default="zip",
        help="How to combine sweep lists: zip pairs by index, product forms full cartesian grid.",
    )
    parser.add_argument(
        "--max-sweep-configs",
        type=int,
        default=None,
        help="Optional cap on number of generated sweep configs.",
    )
    parser.add_argument(
        "--sweep-tag",
        type=str,
        default=None,
        help="Optional metadata tag stored in each run output.",
    )
    parser.add_argument(
        "--sweep-spectral-top-k",
        type=str,
        default=None,
        help="Comma-separated top-k values for sweep, e.g. 50,80",
    )
    parser.add_argument(
        "--sweep-trim-first",
        type=str,
        default=None,
        help="Comma-separated trim-first values for sweep; supports auto, e.g. auto,2,3",
    )
    parser.add_argument(
        "--sweep-trim-last",
        type=str,
        default=None,
        help="Comma-separated trim-last values for sweep; supports auto, e.g. auto,2,3",
    )
    parser.add_argument(
        "--sweep-spectral-neighbor-layers",
        type=str,
        default=None,
        help="Comma-separated neighbor layer values for sweep, e.g. 1,2",
    )
    parser.add_argument(
        "--sweep-spectral-rolling-window",
        type=str,
        default=None,
        help="Comma-separated rolling-z window values for sweep, e.g. 5,7,9",
    )
    parser.add_argument(
        "--sweep-local-window-sets",
        type=str,
        default=None,
        help="Semicolon-separated local window sets, e.g. '3,5,7;5,9,13'",
    )
    parser.add_argument(
        "--analysis-profile",
        default="full",
        help=(
            "Analysis payload level. Canonical values: raw, paper, full. "
            "Aliases: posthoc, posthoc-only, detection, detection-only -> paper."
        ),
    )
    parser.add_argument(
        "--posthoc-only",
        "--detection-only",
        dest="posthoc_only",
        action="store_true",
        help=(
            "Only compute the spectral-lite structural payload needed by the post-hoc "
            "detector and paper graphs. Equivalent to --analysis-profile paper."
        ),
    )
    parser.add_argument(
        "--paper",
        dest="paper_profile",
        action="store_true",
        help=(
            "Compatibility alias for the spectral-lite paper/post-hoc payload. "
            "Equivalent to --posthoc-only and --analysis-profile paper."
        ),
    )
    parser.add_argument(
        "--fail-on-missing-second-moment",
        action="store_true",
        help="Exit non-zero instead of writing a skipped JSON when second-moment stats are missing.",
    )
    parser.add_argument("--disable-attention-metrics", action="store_true")
    parser.add_argument("--disable-rank1-blind", action="store_true")
    parser.add_argument("--disable-symmetry-metrics", action="store_true")
    parser.add_argument("--disable-bottom-rank-svd", action="store_true")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Skip ROME edits and run all detectors on the original (unedited) model weights. "
             "Output files use 'baseline_structural_' prefix instead of 'rome_structural_'.",
    )
    parser.add_argument(
        "--no-paired-baseline",
        action="store_true",
        help="Disable the automatic unedited baseline run that is otherwise saved before each edited run.",
    )
    parser.add_argument(
        "--bottom-rank-sweep-ranks",
        type=str,
        default="4,8,16,32",
        help="Comma-separated top-rank cutoffs used in tail-spectrum token sweep, e.g. 4,8,16,32",
    )
    parser.add_argument(
        "--bottom-rank-top-svd-rank",
        type=int,
        default=64,
        help="Maximum top-SVD rank used to build tail responses for bottom-rank detection.",
    )
    parser.add_argument(
        "--bottom-rank-boundary",
        type=int,
        default=2,
        help="Boundary layers excluded from bottom-rank argmax selection.",
    )
    parser.add_argument(
        "--experiment-doc",
        type=str,
        default=None,
        help="Best-effort load models/settings from docs/*.md",
    )
    args = parser.parse_args()
    args.analysis_profile = normalize_analysis_profile(
        args.analysis_profile,
        posthoc_only=bool(args.posthoc_only or args.paper_profile),
    )

    if args.experiment_doc:
        exp = parse_experiment_doc(
            doc_path=args.experiment_doc,
            fallback_models=args.models,
            fallback_n_tests=args.n_tests,
            fallback_start_idx=args.start_idx,
            fallback_n_prompts=args.n_prompts,
            fallback_spectral_top_k=args.spectral_top_k,
            fallback_trim_first=args.trim_first,
            fallback_trim_last=args.trim_last,
            fallback_spectral_neighbor_layers=args.spectral_neighbor_layers,
        )
        LOGGER.info("Loaded experiment config from %s", exp["source_doc"])
        LOGGER.info(
            "Doc config: models=%s n_tests=%s start_idx=%s n_prompts=%s top_k=%s trim=(%s,%s) neighbor=%s",
            exp["models"],
            exp["n_tests"],
            exp["start_idx"],
            exp["n_prompts"],
            exp["spectral_top_k"],
            exp["trim_first"],
            exp["trim_last"],
            exp["spectral_neighbor_layers"],
        )
        args.models = exp["models"]
        args.n_tests = int(exp["n_tests"])
        args.start_idx = int(exp["start_idx"])
        args.n_prompts = int(exp["n_prompts"])
        args.spectral_top_k = int(exp["spectral_top_k"])
        args.trim_first = None if exp["trim_first"] is None else int(exp["trim_first"])
        args.trim_last = None if exp["trim_last"] is None else int(exp["trim_last"])
        args.spectral_neighbor_layers = int(exp["spectral_neighbor_layers"])

    local_windows = parse_local_windows(args.local_windows)
    bottom_rank_sweep_ranks = parse_int_values(
        args.bottom_rank_sweep_ranks,
        default=[4, 8, 16, 32],
        min_value=1,
    )

    sweep_top_k_values = parse_int_values(
        args.sweep_spectral_top_k,
        default=[args.spectral_top_k],
        min_value=1,
    )
    sweep_trim_first_values = parse_trim_values(
        args.sweep_trim_first,
        default=[args.trim_first],
    )
    sweep_trim_last_values = parse_trim_values(
        args.sweep_trim_last,
        default=[args.trim_last],
    )
    sweep_neighbor_values = parse_int_values(
        args.sweep_spectral_neighbor_layers,
        default=[args.spectral_neighbor_layers],
        min_value=1,
    )
    sweep_rolling_values = parse_int_values(
        args.sweep_spectral_rolling_window,
        default=[args.spectral_rolling_window],
        min_value=1,
        force_odd=True,
    )
    sweep_local_window_sets = parse_local_window_sets(
        args.sweep_local_window_sets,
        default=[local_windows],
    )

    sweep_configs = build_sweep_configs(
        spectral_top_k_values=sweep_top_k_values,
        trim_first_values=sweep_trim_first_values,
        trim_last_values=sweep_trim_last_values,
        spectral_neighbor_layers_values=sweep_neighbor_values,
        spectral_rolling_window_values=sweep_rolling_values,
        local_window_sets=sweep_local_window_sets,
        mode=args.sweep_mode,
        max_configs=args.max_sweep_configs,
    )

    LOGGER.info("Final sweep plan contains %d config(s).", len(sweep_configs))
    for i, cfg in enumerate(sweep_configs, start=1):
        LOGGER.info(
            "  Plan[%d/%d]: top_k=%s trim=(%s,%s) neighbor=%s rolling=%s local_windows=%s",
            i,
            len(sweep_configs),
            cfg.get("spectral_top_k"),
            _format_optional_int(cfg.get("trim_first")),
            _format_optional_int(cfg.get("trim_last")),
            cfg.get("spectral_neighbor_layers"),
            cfg.get("spectral_rolling_window"),
            cfg.get("local_windows"),
        )

    run_benchmark(
        models=args.models,
        n_tests=args.n_tests,
        start_idx=args.start_idx,
        run_start_idx_step=args.run_start_idx_step,
        runs_per_model=args.runs_per_model,
        output_dir=args.output_dir,
        n_prompts=args.n_prompts,
        spectral_top_k=args.spectral_top_k,
        trim_first=args.trim_first,
        trim_last=args.trim_last,
        spectral_neighbor_layers=args.spectral_neighbor_layers,
        spectral_rolling_window=args.spectral_rolling_window,
        local_windows=local_windows,
        enable_attention_metrics=not args.disable_attention_metrics,
        enable_rank1_blind=not args.disable_rank1_blind,
        enable_symmetry_metrics=not args.disable_symmetry_metrics,
        enable_bottom_rank_svd=not args.disable_bottom_rank_svd,
        bottom_rank_sweep_ranks=bottom_rank_sweep_ranks,
        bottom_rank_top_svd_rank=args.bottom_rank_top_svd_rank,
        bottom_rank_boundary=args.bottom_rank_boundary,
        raw_spectral_max_top_k=args.raw_spectral_max_top_k,
        sweep_configs=sweep_configs,
        sweep_tag=args.sweep_tag,
        analysis_profile=args.analysis_profile,
        baseline_only=args.baseline_only,
        create_paired_baseline=not args.no_paired_baseline,
        fail_on_missing_second_moment=args.fail_on_missing_second_moment,
    )
