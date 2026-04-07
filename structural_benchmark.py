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
import re
import sys
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
from src.structural.attention_metrics import (
    AttentionContrastDetector,
    derive_attention_templates,
    split_qkv_weight,
)
from src.structural.symmetry_metrics import MirrorSymmetryDetector
from src.structural.interlayer import collect_all_interlayer_data
from src.structural.ipr import (
    layer_ipr_summary,
    layer_fc_proj_ipr_discrepancy,
    IPRDetector,
)

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


def get_fc_template(layer_name_template: str) -> str:
    """Derive the fc (input) layer template from the proj (output) layer template."""
    for proj_key, fc_key in FC_TEMPLATE_MAP.items():
        if proj_key in layer_name_template:
            return layer_name_template.replace(proj_key, fc_key)
    raise ValueError(f"Cannot derive fc layer template from: {layer_name_template}")


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
            "fact_tuple": (
                rw["prompt"], rw["subject"],
                " " + rw["target_new"]["str"],
                " " + rw["target_true"]["str"],
            ),
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
) -> dict:
    """Run the full ROME + structural-detection benchmark for one model."""
    cfg = build_cfg(model_name)
    LOGGER.info("Loading %s ...", cfg.model.name)
    handler = ModelHandler(cfg)
    LOGGER.info(
        "Loaded. layer=%d, emb=%d, hidden=%d, prompts=%d, multi_gpu=%s",
        handler._layer, handler.emb_shape, handler.hidden_dim, n_prompts,
        handler.is_multi_gpu,
    )

    proj_template = handler._layer_name_template
    fc_template = get_fc_template(proj_template)
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
    has_fc_template = True
    try:
        _ = handler._get_module(fc_template.format(handler._layer))
    except KeyError:
        LOGGER.warning("Could not resolve fc template (%s) - IPR and fc-contrast analytics disabled", fc_template)
        has_fc_template = False

    attention_templates = derive_attention_templates(proj_template)

    # Detectors
    blind_detector = BlindMSDDetector()
    spectral_detector = SpectralDetector(
        top_k=spectral_top_k, boundary=2,
        trim_first_layers=eff_trim_first, trim_last_layers=eff_trim_last,
        neighbor_layers=spectral_neighbor_layers,
        rolling_window=spectral_rolling_window,
        local_windows=local_windows,
    )
    ipr_detector = IPRDetector(trim_first=eff_trim_first, trim_last=eff_trim_last) if has_fc_template else None
    composite_detector = CompositeDetector(
        top_k=spectral_top_k, trim_first=eff_trim_first, trim_last=eff_trim_last,
    )
    edit_presence_detector = RomeEditPresenceDetector()
    rank1_detector = BlindRank1Detector(boundary=2, local_windows=local_windows) if enable_rank1_blind else None
    attention_detector = AttentionContrastDetector(boundary=2, local_windows=local_windows) if enable_attention_metrics else None
    symmetry_detector = (
        MirrorSymmetryDetector(top_k=20, boundary=2, local_windows=local_windows)
        if enable_symmetry_metrics else None
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
            "timestamp": datetime.now().isoformat(),
            "spectral_config": {
                "top_k": spectral_top_k, "boundary": 2,
                "trim_first": eff_trim_first, "trim_last": eff_trim_last,
                "neighbor_layers": spectral_neighbor_layers,
                "rolling_window": spectral_rolling_window,
                "signal_keys": SPECTRAL_SIGNAL_KEYS,
            },
            "analytics_config": {
                "local_windows": list(local_windows),
                "enable_attention_metrics": bool(enable_attention_metrics),
                "enable_rank1_blind": bool(enable_rank1_blind),
                "enable_symmetry_metrics": bool(enable_symmetry_metrics),
                "attention_template_candidates": sorted(attention_templates.keys()),
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
        )
    }

    for i, case in enumerate(test_cases):
        LOGGER.info("[%d/%d] %s", i + 1, len(test_cases), case["subject"])
        layer_name = proj_template.format(handler._layer)
        old_W = handler._get_module(layer_name).weight.detach().clone()
        entry = {"case_id": case["case_id"], "subject": case["subject"], "error": None}

        try:
            fact = case["fact_tuple"]

            # ROME edit
            k = gather_k(handler, fact_tuple=fact, N=n_prompts)
            delta = optimize_v(
                handler, fact_tuple=fact,
                N_prompts=n_prompts, N_optim_steps=handler.epochs,
            )
            new_W, _, _ = insert_kv(handler, k, delta)

            # Check ROME success
            prompt = fact[0].format(fact[1])
            tokens = handler.tokenize_prompt(prompt)
            with torch.no_grad():
                out = handler.model(**tokens)
            predicted = handler.tokenizer.decode(out.logits[0, -1, :].argmax())
            rome_ok = predicted.strip().lower() == fact[2].strip().lower()

            entry["rome"] = {
                "success": rome_ok, "predicted": predicted,
                "k_norm": k.norm().item(), "delta_norm": delta.norm().item(),
            }
            if rome_ok:
                counts["rome"] += 1

            # Build modified weight dict (only proj layer changed by ROME)
            modified_proj = extract_weights(handler, proj_template)
            modified_fc = None
            if has_fc_template:
                try:
                    modified_fc = extract_weights(handler, fc_template)
                except (KeyError, ValueError):
                    LOGGER.warning("Could not extract fc weights from edited model (%s)", fc_template)
                    modified_fc = None

            attention_weights = extract_attention_weights(handler, proj_template) if enable_attention_metrics else {}

            # Run detectors
            blind_res = blind_detector.detect(modified_proj)
            spectral_res = spectral_detector.detect(modified_proj, fc_weights=modified_fc)
            ipr_res = ipr_detector.detect(modified_proj, modified_fc) if (ipr_detector and modified_fc is not None) else {}
            composite_res = composite_detector.detect(modified_proj, fc_weights=modified_fc, spectral_result=spectral_res)
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
            )

            # IPR analysis
            mod_ipr_proj = add_ipr_z_scores(layer_ipr_summary(modified_proj))
            mod_ipr_fc = add_ipr_z_scores(layer_ipr_summary(modified_fc)) if modified_fc is not None else {}
            mod_ipr_disc = layer_fc_proj_ipr_discrepancy(modified_proj, modified_fc) if modified_fc is not None else {}

            correct = {
                "blind": blind_res.get("anomalous_layer") == handler._layer,
                "spectral": spectral_res.get("anomalous_layer") == handler._layer,
                "ipr": ipr_res.get("anomalous_layer") == handler._layer if ipr_res else False,
                "composite": composite_res.get("anomalous_layer") == handler._layer,
                "rank1_blind": rank1_res.get("anomalous_layer") == handler._layer if rank1_res else False,
                "attention": attention_res.get("anomalous_layer") == handler._layer if attention_res else False,
                "symmetry": symmetry_res.get("anomalous_layer") == handler._layer if symmetry_res else False,
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
                "edit_presence_detection": to_serializable(presence_res),
                "interlayer": to_serializable(collect_all_interlayer_data(modified_proj)),
                "ipr": {
                    "proj": to_serializable(mod_ipr_proj),
                    "fc": to_serializable(mod_ipr_fc),
                    "fc_vs_proj": to_serializable(mod_ipr_disc),
                    "detection": to_serializable(ipr_res),
                },
                "accuracy": {
                    "rome_success": rome_ok,
                    **{f"{name}_correct": v for name, v in correct.items()},
                },
            })

            LOGGER.info(
                "  ROME=%s Blind=L%s Spectral=L%s Composite=L%s(%s) Rank1=L%s Attention=L%s Symmetry=L%s Presence=%s(%.3f) IPR=L%s(%.3f)",
                "OK" if rome_ok else "FAIL",
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
            )

        except Exception as e:
            entry["error"] = str(e)
            entry["skipped"] = True
            LOGGER.warning("  SKIPPED: %s", e)
        finally:
            handler.remove_hooks()
            handler._get_module(layer_name).weight = torch.nn.Parameter(old_W)
            results["tests"].append(entry)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    ok_tests = [t for t in results["tests"] if not t.get("skipped")]
    n = len(ok_tests)
    results["summary"] = {
        "total": len(test_cases), "successful": n, "skipped": len(test_cases) - n,
        **{f"{k}_rate": counts[k] / n if n else 0 for k in counts},
    }
    LOGGER.info(
        "[%s] ROME=%d/%d Blind=%d/%d Spectral=%d/%d Composite=%d/%d Rank1=%d/%d Attention=%d/%d Symmetry=%d/%d IPR=%d/%d skip=%d",
        cfg.model.name, counts["rome"], n, counts["blind"], n,
        counts["spectral"], n, counts["composite"], n,
        counts["rank1_blind"], n, counts["attention"], n, counts["symmetry"], n,
        counts["ipr"], n, len(test_cases) - n,
    )

    # Free GPU
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
    n_prompts: int = 10,
    spectral_top_k: int = 50,
    trim_first: Optional[int] = None,
    trim_last: Optional[int] = None,
    spectral_neighbor_layers: int = 1,
    spectral_rolling_window: int = 5,
    local_windows: Sequence[int] = (3, 5, 7),
    enable_attention_metrics: bool = True,
    enable_rank1_blind: bool = True,
    enable_symmetry_metrics: bool = True,
    runs_per_model: int = 1,
    run_start_idx_step: int = 0,
    sweep_configs: Optional[Sequence[Dict[str, object]]] = None,
    sweep_tag: Optional[str] = None,
):
    """Run benchmark across models, optionally with multi-run + sweep experiments."""
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

        # Never auto-compute missing second moments inside this benchmark path.
        model_cfg = load_model_config(model_name)
        sm_files, sm_dir = find_second_moment_files(model_cfg)
        if not sm_files:
            skip_msg = (
                f"Missing second moment stats for model={model_cfg.name} layer={model_cfg.layer} "
                f"in {sm_dir}. Skipping model."
            )
            LOGGER.warning(skip_msg)
            for sweep_idx, sweep_cfg in enumerate(effective_sweeps, start=1):
                cfg_slug = sweep_config_slug(sweep_cfg)
                for run_idx in range(1, runs_per_model + 1):
                    run_ordinal = (sweep_idx - 1) * runs_per_model + run_idx
                    start_idx_used = start_idx + run_start_idx_step * (run_ordinal - 1)
                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    out_file = output_path / (
                        f"rome_structural_{safe_name}_{cfg_slug}"
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

                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                out_file = output_path / (
                    f"rome_structural_{safe_name}_{cfg_slug}"
                    f"_s{sweep_idx:02d}_r{run_idx:02d}_{ts}.json"
                )

                try:
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
                    )

                    metadata = model_results.setdefault("metadata", {})
                    metadata.update({
                        "run_index": run_idx,
                        "run_ordinal": run_ordinal,
                        "runs_per_model": runs_per_model,
                        "run_start_idx_step": run_start_idx_step,
                        "start_idx_used": start_idx_used,
                        "sweep_index": sweep_idx,
                        "sweep_size": len(effective_sweeps),
                        "sweep_config": to_serializable(sweep_cfg),
                        "sweep_slug": cfg_slug,
                        "total_runs_for_model": total_runs_for_model,
                    })
                    if sweep_tag:
                        metadata["sweep_tag"] = str(sweep_tag)

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
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--spectral-top-k", type=int, default=50)
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
    parser.add_argument("--disable-attention-metrics", action="store_true")
    parser.add_argument("--disable-rank1-blind", action="store_true")
    parser.add_argument("--disable-symmetry-metrics", action="store_true")
    parser.add_argument(
        "--experiment-doc",
        type=str,
        default=None,
        help="Best-effort load models/settings from docs/*.md",
    )
    args = parser.parse_args()

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
        sweep_configs=sweep_configs,
        sweep_tag=args.sweep_tag,
    )
