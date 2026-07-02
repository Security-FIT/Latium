"""
Early-site causal tracing for ROME layer investigation.

The command intentionally implements one tracing workflow:

* corrupt the full subject-token span at the embedding output,
* restore clean MLP outputs at the last subject token,
* sweep overlapping MLP windows,
* aggregate paired indirect effects across facts,
* report a graph-derived causal region without using the configured ROME layer
  for selection.

The configured model layer is kept only as a reference marker in the plot and
summary.  It must not influence the selected trace center.
"""

from __future__ import annotations

import json
import logging
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.common.loading import load_dataset
from src.handlers.rome import ModelHandler

LOGGER = logging.getLogger(__name__)


class TraceValidationError(ValueError):
    """Raised when a trace example is ambiguous or unusable."""


@dataclass
class TokenSpan:
    start: int
    end: int
    positions: list[int]
    last_position: int


@dataclass
class TraceExample:
    prompt_id: str
    prompt: str
    subject: str
    target: str
    raw: dict[str, Any]


@dataclass
class Window:
    center: int
    start: int
    end: int
    layers: list[int]

    @property
    def size(self) -> int:
        return len(self.layers)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "__fspath__"):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _section(cfg: DictConfig, name: str) -> Any:
    command = getattr(cfg, "command", None)
    if command is not None and hasattr(command, name):
        return getattr(command, name)
    value = getattr(cfg, name, None)
    return value if value is not None else OmegaConf.create({})


def _get(section: Any, name: str, default: Any) -> Any:
    return getattr(section, name, default)


def _strip_bos(tokenizer: Any, token_ids: Sequence[int]) -> list[int]:
    ids = [int(x) for x in token_ids]
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None and len(ids) > 1 and ids[0] == bos_id:
        return ids[1:]
    return ids


def _token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool = True) -> list[int]:
    try:
        raw = tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
    except TypeError:
        raw = tokenizer(text)["input_ids"]
    if torch.is_tensor(raw):
        raw = raw.detach().cpu().tolist()
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    return [int(x) for x in raw]


def target_first_token_id(tokenizer: Any, target: str) -> int:
    """Return the first next-token target ID used by this trace."""
    cleaned = str(target).strip()
    for text, special in ((f" {cleaned}", False), (cleaned, False), (f" {cleaned}", True), (cleaned, True)):
        ids = _strip_bos(tokenizer, _token_ids(tokenizer, text, add_special_tokens=special))
        if ids:
            return int(ids[0])
    raise TraceValidationError(f"Could not tokenize target {target!r}")


def find_subject_span(tokenizer: Any, prompt: str, subject: str) -> TokenSpan:
    """Find the unique prompt token span overlapping the subject string."""
    if not subject:
        raise TraceValidationError("Subject is empty")

    starts: list[int] = []
    cursor = 0
    while True:
        idx = prompt.find(subject, cursor)
        if idx == -1:
            break
        starts.append(idx)
        cursor = idx + max(1, len(subject))

    if not starts:
        raise TraceValidationError(f"Subject {subject!r} is not present in prompt")
    if len(starts) > 1:
        raise TraceValidationError(f"Subject {subject!r} appears {len(starts)} times in prompt")

    char_start = starts[0]
    char_end = char_start + len(subject)

    try:
        encoded = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
        offsets = encoded.get("offset_mapping")
    except Exception:
        offsets = None

    if offsets is not None:
        positions: list[int] = []
        for idx, (start, end) in enumerate(offsets[0].detach().cpu().tolist()):
            if end <= start:
                continue
            if end > char_start and start < char_end:
                positions.append(int(idx))
        if positions:
            return TokenSpan(positions[0], positions[-1] + 1, positions, positions[-1])

    prompt_ids = _strip_bos(tokenizer, _token_ids(tokenizer, prompt))
    candidates = [
        _strip_bos(tokenizer, _token_ids(tokenizer, subject, add_special_tokens=False)),
        _strip_bos(tokenizer, _token_ids(tokenizer, f" {subject}", add_special_tokens=False)),
        _strip_bos(tokenizer, _token_ids(tokenizer, subject)),
        _strip_bos(tokenizer, _token_ids(tokenizer, f" {subject}")),
    ]
    matches: list[tuple[int, int]] = []
    for subject_ids in candidates:
        if not subject_ids:
            continue
        n = len(subject_ids)
        for start in range(0, len(prompt_ids) - n + 1):
            if prompt_ids[start : start + n] == subject_ids:
                matches.append((start, start + n))

    matches = sorted(set(matches))
    if len(matches) != 1:
        raise TraceValidationError(f"Could not identify a unique subject span for {subject!r}")
    start, end = matches[0]
    positions = list(range(start, end))
    return TokenSpan(start, end, positions, end - 1)


def _dataset_examples(cfg: DictConfig, *, max_scan: int | None = None) -> Iterator[TraceExample]:
    dataset = load_dataset(cfg)
    records: Iterable[Any]
    if isinstance(dataset, dict) and "requested_rewrite" in dataset:
        records = ({"requested_rewrite": row} for row in dataset["requested_rewrite"])
    else:
        records = dataset

    for idx, record in enumerate(records):
        if max_scan is not None and idx >= int(max_scan):
            break
        raw = dict(record)
        rewrite = raw.get("requested_rewrite", raw)
        subject = str(rewrite["subject"])
        prompt_template = str(rewrite["prompt"])
        target = rewrite.get("target_true", {})
        target = target.get("str", target) if isinstance(target, dict) else target
        yield TraceExample(
            prompt_id=str(raw.get("case_id", raw.get("relation_id", idx))),
            prompt=prompt_template.format(subject),
            subject=subject,
            target=str(target).strip(),
            raw=raw,
        )


def filter_dataset(dataset: Any, n: int | None = None) -> list[Any]:
    """Compatibility helper retained for older imports."""
    if isinstance(dataset, pd.DataFrame):
        values = dataset.to_dict("records")
    else:
        values = list(dataset)
    return values if n is None else values[: int(n)]


def preprocess_prompt(*args: Any) -> tuple[str, str, str] | tuple[Any, list[int]] | None:
    """Compatibility helper retained for older imports."""
    if len(args) == 2:
        handler, prompt_dict = args
        try:
            record = prompt_dict._asdict() if hasattr(prompt_dict, "_asdict") else dict(prompt_dict)
            prompt, subject, _target = preprocess_prompt(record)  # type: ignore[misc]
            span = find_subject_span(handler.tokenizer, prompt, subject)
            return handler.tokenize_prompt(prompt), list(span.positions)
        except Exception:
            return None
    if len(args) != 1:
        raise TypeError("preprocess_prompt expects either a record or (handler, record)")
    record = args[0]
    record = record._asdict() if hasattr(record, "_asdict") else dict(record)
    rewrite = record.get("requested_rewrite", record)
    subject = str(rewrite["subject"])
    target = rewrite.get("target_true", {})
    target = target.get("str", target) if isinstance(target, dict) else target
    return str(rewrite["prompt"]).format(subject), subject, str(target).strip()


def _module_dict(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    return dict(model.named_modules())


def _embedding_module_name(cfg: DictConfig) -> str:
    template = str(cfg.model.corrupt_layer_name_template)
    return template.format(0) if "{}" in template else template


def _resolve_mlp_output_name(handler: ModelHandler, modules: dict[str, torch.nn.Module], layer: int) -> str:
    """Return the final MLP projection module from the model config."""
    template = str(handler._layer_name_template)
    name = template.format(int(layer))
    if name not in modules:
        raise KeyError(f"Configured MLP output module does not exist: {name}")
    return name


def _hidden_from_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        values = list(output)
        values[0] = hidden
        return tuple(values)
    return hidden


@contextmanager
def temporary_hooks(hooks: Iterable[tuple[torch.nn.Module, Any]]) -> Iterator[None]:
    handles = []
    try:
        for module, hook in hooks:
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _patch_position(hidden: torch.Tensor, position: int, value: torch.Tensor) -> torch.Tensor:
    changed = hidden.clone()
    patch = value.to(device=changed.device, dtype=changed.dtype)
    if changed.dim() == 3:
        if patch.dim() == 3:
            patch = patch[:, int(position), :]
        elif patch.dim() == 2 and patch.shape[0] != changed.shape[0]:
            patch = patch[int(position), :].unsqueeze(0)
        elif patch.dim() == 1:
            patch = patch.unsqueeze(0)
        changed[:, int(position), :] = patch
        return changed
    if changed.dim() == 2:
        if patch.dim() == 3:
            patch = patch[0, int(position), :]
        elif patch.dim() == 2:
            patch = patch[int(position), :] if patch.shape[0] != 1 else patch[0]
        changed[int(position), :] = patch
        return changed
    raise RuntimeError(f"Unsupported activation rank for restoration: {tuple(changed.shape)}")


def _embedding_std(handler: ModelHandler, modules: dict[str, torch.nn.Module]) -> float:
    name = _embedding_module_name(handler.cfg)
    module = modules.get(name)
    if module is None:
        raise KeyError(f"Embedding module not found: {name}")
    weight = getattr(module, "weight", None)
    if weight is None:
        raise RuntimeError(f"Embedding module {name} has no weight")
    std = float(weight.detach().float().std().item())
    if not math.isfinite(std) or std <= 0:
        raise RuntimeError(f"Invalid embedding standard deviation: {std}")
    return std


def make_noise_samples(
    *,
    num_samples: int,
    subject_length: int,
    hidden_size: int,
    noise_std: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(
        (int(num_samples), int(subject_length), int(hidden_size)),
        generator=generator,
        dtype=torch.float32,
    )
    return (noise * float(noise_std)).to(device=device, dtype=dtype)


def _corrupt_hook(subject_positions: Sequence[int], noise: torch.Tensor):
    positions = [int(x) for x in subject_positions]

    def hook(_module, _input, output):
        hidden = _hidden_from_output(output)
        changed = hidden.clone()
        sample = noise.to(device=changed.device, dtype=changed.dtype)
        if sample.dim() == 2:
            sample = sample.unsqueeze(0)
        for offset, position in enumerate(positions):
            changed[:, position, :] = changed[:, position, :] + sample[:, offset, :]
        return _replace_hidden(output, changed)

    return hook


def _restore_hook(position: int, clean_output: torch.Tensor):
    def hook(_module, _input, output):
        hidden = _hidden_from_output(output)
        changed = _patch_position(hidden, int(position), clean_output)
        return _replace_hidden(output, changed)

    return hook


def _probability(outputs: Any, target_token_id: int) -> float:
    probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
    return float(probs[0, int(target_token_id)].detach().float().cpu().item())


def _top_token(outputs: Any) -> tuple[int, float]:
    probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
    top = int(torch.argmax(probs[0]).detach().cpu().item())
    return top, float(probs[0, top].detach().float().cpu().item())


def _prepare_inputs(handler: ModelHandler, prompt: str) -> Any:
    inputs = handler.tokenize_prompt(prompt)
    if "token_type_ids" in inputs:
        # Several decoder-only models reject token_type_ids.
        inputs.pop("token_type_ids", None)
    return inputs


def _output_device_dtype(hidden: torch.Tensor) -> tuple[torch.device, torch.dtype, int]:
    if hidden.dim() == 3:
        return hidden.device, hidden.dtype, int(hidden.shape[-1])
    if hidden.dim() == 2:
        return hidden.device, hidden.dtype, int(hidden.shape[-1])
    raise RuntimeError(f"Unsupported activation rank: {tuple(hidden.shape)}")


def build_window(center: int, window_size: int, num_layers: int) -> Window:
    left_width = int(window_size) // 2
    right_width = int(window_size) - left_width
    start = max(0, int(center) - left_width)
    end = min(int(num_layers), int(center) + right_width)
    return Window(center=int(center), start=start, end=end, layers=list(range(start, end)))


def _trimmed_mean(values: np.ndarray, fraction: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    trim = int(math.floor(arr.size * float(fraction)))
    if trim <= 0 or 2 * trim >= arr.size:
        return float(np.mean(arr))
    return float(np.mean(np.sort(arr)[trim:-trim]))


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or int(samples) <= 0:
        value = float(np.mean(arr))
        return value, value
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, arr.size, size=(int(samples), arr.size))
    means = arr[indices].mean(axis=1)
    alpha = 1.0 - float(confidence_level)
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def _group_contiguous(values: Sequence[int]) -> list[list[int]]:
    ordered = sorted({int(x) for x in values})
    if not ordered:
        return []
    groups = [[ordered[0]]]
    for value in ordered[1:]:
        if value == groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def _clean_cache(
    handler: ModelHandler,
    modules: dict[str, torch.nn.Module],
    module_names: dict[int, str],
    inputs: Any,
) -> dict[int, torch.Tensor]:
    cache: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def hook(_module, _input, output):
            cache[int(layer)] = _hidden_from_output(output).detach().clone()
            return output

        return hook

    hooks = [(modules[name], make_hook(layer)) for layer, name in module_names.items()]
    with torch.inference_mode(), temporary_hooks(hooks):
        handler.model(**inputs, use_cache=False)
    missing = sorted(set(module_names) - set(cache))
    if missing:
        raise RuntimeError(f"Clean MLP cache missing layers: {missing[:10]}")
    return cache


def _trace_example(
    handler: ModelHandler,
    modules: dict[str, torch.nn.Module],
    module_names: dict[int, str],
    embedding_module: torch.nn.Module,
    example: TraceExample,
    *,
    windows: list[Window],
    num_noise_samples: int,
    noise_multiplier: float,
    seed: int,
    require_correct_clean: bool,
    min_total_effect: float,
) -> dict[str, Any]:
    inputs = _prepare_inputs(handler, example.prompt)
    span = find_subject_span(handler.tokenizer, example.prompt, example.subject)
    target_id = target_first_token_id(handler.tokenizer, example.target)
    prompt_last = int(inputs["input_ids"].shape[1] - 1)

    with torch.inference_mode():
        clean_outputs = handler.model(**inputs, use_cache=False)
    clean_probability = _probability(clean_outputs, target_id)
    clean_top_id, clean_top_probability = _top_token(clean_outputs)
    clean_top_token = handler.tokenizer.decode([clean_top_id])
    if require_correct_clean and clean_top_id != int(target_id):
        raise TraceValidationError(
            f"clean-token mismatch: expected {handler.tokenizer.decode([target_id])!r}, got {clean_top_token!r}"
        )

    clean_cache = _clean_cache(handler, modules, module_names, inputs)
    first_clean = clean_cache[0]
    noise_device, noise_dtype, hidden_size = _output_device_dtype(first_clean)
    noise_std = float(noise_multiplier) * _embedding_std(handler, modules)
    noise_samples = make_noise_samples(
        num_samples=int(num_noise_samples),
        subject_length=len(span.positions),
        hidden_size=hidden_size,
        noise_std=noise_std,
        device=noise_device,
        dtype=noise_dtype,
        seed=int(seed),
    )

    corrupt_probabilities: list[float] = []
    restore_probabilities = np.zeros((len(windows), int(num_noise_samples)), dtype=np.float64)

    for k in range(int(num_noise_samples)):
        noise = noise_samples[k : k + 1]
        with torch.inference_mode(), temporary_hooks([(embedding_module, _corrupt_hook(span.positions, noise))]):
            outputs = handler.model(**inputs, use_cache=False)
        corrupt_probabilities.append(_probability(outputs, target_id))

        for window_idx, window in enumerate(windows):
            hooks: list[tuple[torch.nn.Module, Any]] = [(embedding_module, _corrupt_hook(span.positions, noise))]
            for layer in window.layers:
                hooks.append((modules[module_names[layer]], _restore_hook(span.last_position, clean_cache[layer])))
            with torch.inference_mode(), temporary_hooks(hooks):
                outputs = handler.model(**inputs, use_cache=False)
            restore_probabilities[window_idx, k] = _probability(outputs, target_id)

    corrupt = np.asarray(corrupt_probabilities, dtype=np.float64)
    effects = restore_probabilities - corrupt.reshape(1, -1)
    total_effect = float(clean_probability - np.mean(corrupt))
    if total_effect < float(min_total_effect):
        raise TraceValidationError(f"low corruption effect: {total_effect:.6f}")

    target_ids = _strip_bos(handler.tokenizer, _token_ids(handler.tokenizer, example.target, add_special_tokens=False))
    result = {
        "prompt_id": example.prompt_id,
        "prompt": example.prompt,
        "subject": example.subject,
        "target": example.target,
        "target_first_token_id": int(target_id),
        "target_first_token_text": handler.tokenizer.decode([target_id]),
        "target_num_tokens": int(len(target_ids)),
        "subject_positions": list(span.positions),
        "subject_tokens": [
            handler.tokenizer.decode([int(inputs["input_ids"][0, pos].detach().cpu().item())]) for pos in span.positions
        ],
        "subject_last_position": int(span.last_position),
        "subject_last_token": handler.tokenizer.decode(
            [int(inputs["input_ids"][0, span.last_position].detach().cpu().item())]
        ),
        "prompt_last_position": int(prompt_last),
        "prompt_last_token": handler.tokenizer.decode([int(inputs["input_ids"][0, prompt_last].detach().cpu().item())]),
        "clean_probability": float(clean_probability),
        "clean_top_token_id": int(clean_top_id),
        "clean_top_token": clean_top_token,
        "clean_top_probability": float(clean_top_probability),
        "corrupt_probabilities": corrupt.tolist(),
        "mean_corrupt_probability": float(np.mean(corrupt)),
        "std_corrupt_probability": float(np.std(corrupt)),
        "total_effect": total_effect,
        "noise_std": float(noise_std),
        "window_mean_ie": effects.mean(axis=1).tolist(),
        "window_median_ie": np.median(effects, axis=1).tolist(),
        "window_std_ie": effects.std(axis=1).tolist(),
        "window_restore_probabilities": restore_probabilities.tolist(),
    }
    return result


def _summarize_windows(
    fact_results: list[dict[str, Any]],
    windows: list[Window],
    *,
    window_size: int,
    trim_fraction: float,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    rows = []
    if not fact_results:
        return pd.DataFrame()

    fact_ie = np.asarray([row["window_mean_ie"] for row in fact_results], dtype=np.float64)
    totals = np.asarray([row["total_effect"] for row in fact_results], dtype=np.float64)
    denom = np.maximum(np.abs(totals), 1e-9)

    for idx, window in enumerate(windows):
        values = fact_ie[:, idx]
        ci_lower, ci_upper = _bootstrap_mean_ci(
            values,
            samples=int(bootstrap_samples),
            confidence_level=float(confidence_level),
            seed=int(seed) + idx,
        )
        rows.append(
            {
                "window_center": int(window.center),
                "window_start": int(window.start),
                "window_end": int(window.end),
                "window_layers": ",".join(str(layer) for layer in window.layers),
                "window_size_actual": int(window.size),
                "window_is_full_width": bool(window.size == int(window_size)),
                "num_facts": int(values.size),
                "mean_ie": float(np.mean(values)),
                "median_ie": float(np.median(values)),
                "trimmed_mean_ie": _trimmed_mean(values, float(trim_fraction)),
                "std_ie": float(np.std(values)),
                "sem_ie": float(np.std(values) / max(math.sqrt(values.size), 1.0)),
                "mean_ie_ci_lower": ci_lower,
                "mean_ie_ci_upper": ci_upper,
                "positive_fact_rate": float(np.mean(values > 0)),
                "mean_normalized_recovery": float(np.mean(values / denom)),
            }
        )
    return pd.DataFrame(rows)


def _select_region(
    discovery: pd.DataFrame,
    confirmation: pd.DataFrame,
    *,
    num_layers: int,
    window_size: int,
    minimum_supported_centers: int,
    middle_min_fraction: float,
    middle_max_fraction: float,
) -> dict[str, Any]:
    if discovery.empty or confirmation.empty:
        return {
            "strict_layer": None,
            "diagnostic_layer": None,
            "strict_failure_reason": "insufficient_facts",
            "regions": [],
        }

    middle_min = math.ceil(float(num_layers) * float(middle_min_fraction))
    middle_max = math.floor(float(num_layers) * float(middle_max_fraction))
    eligible = discovery[
        (discovery["window_is_full_width"])
        & (discovery["window_center"] >= middle_min)
        & (discovery["window_center"] <= middle_max)
    ].copy()
    if eligible.empty:
        return {
            "strict_layer": None,
            "diagnostic_layer": None,
            "strict_failure_reason": "no_full_width_middle_windows",
            "regions": [],
        }

    raw_peak_center = int(eligible.sort_values("mean_ie", ascending=False).iloc[0]["window_center"])
    peak_mean = float(eligible.loc[eligible["window_center"] == raw_peak_center, "mean_ie"].iloc[0])
    if not math.isfinite(peak_mean) or peak_mean <= 0:
        return {
            "strict_layer": None,
            "diagnostic_layer": None,
            "strict_failure_reason": "non_positive_discovery_peak",
            "raw_mean_peak_center": raw_peak_center,
            "regions": [],
        }

    # Keep a permissive discovery set, then confirm on held-out facts.  Discovery
    # is deliberately based on a near-peak effect threshold rather than the
    # configured ROME layer.
    threshold = 0.90 * peak_mean
    candidate_centers = [
        int(row.window_center)
        for row in eligible.itertuples()
        if float(row.mean_ie) >= threshold or int(row.window_center) == raw_peak_center
    ]
    groups = _group_contiguous(candidate_centers)

    regions = []
    for group in groups:
        conf_rows = confirmation[confirmation["window_center"].isin(group)].copy()
        if conf_rows.empty:
            continue
        representative = conf_rows.sort_values(
            ["median_ie", "trimmed_mean_ie", "positive_fact_rate", "mean_ie"],
            ascending=False,
        ).iloc[0]
        layers: set[int] = set()
        for row in conf_rows.itertuples():
            layers.update(int(x) for x in str(row.window_layers).split(",") if x != "")
        region_ci_lower = float(conf_rows["mean_ie_ci_lower"].median())
        regions.append(
            {
                "centers": [int(x) for x in group],
                "start": int(min(group)),
                "end": int(max(group)),
                "width": int(len(group)),
                "representative_center": int(representative.window_center),
                "median_ie": float(conf_rows["median_ie"].median()),
                "trimmed_mean_ie": float(conf_rows["trimmed_mean_ie"].median()),
                "mean_ie": float(conf_rows["mean_ie"].median()),
                "positive_fact_rate": float(conf_rows["positive_fact_rate"].median()),
                "mean_ie_ci_lower": region_ci_lower,
                "layer_union": sorted(layers),
                "confirmed_positive": bool(region_ci_lower > 0),
            }
        )

    diagnostic_region = None
    if regions:
        diagnostic_region = sorted(
            regions,
            key=lambda r: (r["median_ie"], r["trimmed_mean_ie"], r["positive_fact_rate"], r["mean_ie"]),
            reverse=True,
        )[0]
    supported = [r for r in regions if r["width"] >= int(minimum_supported_centers) and r["confirmed_positive"]]
    if not supported:
        reason = "only_isolated_candidate_regions" if regions else "no_candidate_regions"
        if regions and not any(r["confirmed_positive"] for r in regions):
            reason = "confirmation_ci_not_positive"
        return {
            "strict_layer": None,
            "diagnostic_layer": diagnostic_region["representative_center"] if diagnostic_region else None,
            "strict_failure_reason": reason,
            "raw_mean_peak_center": raw_peak_center,
            "regions": regions,
        }

    best = sorted(
        supported,
        key=lambda r: (r["median_ie"], r["trimmed_mean_ie"], r["positive_fact_rate"], r["mean_ie"]),
        reverse=True,
    )[0]
    return {
        "strict_layer": int(best["representative_center"]),
        "diagnostic_layer": diagnostic_region["representative_center"] if diagnostic_region else None,
        "strict_failure_reason": None,
        "raw_mean_peak_center": raw_peak_center,
        "robust_peak_region_centers": best["centers"],
        "robust_peak_region_layer_union": best["layer_union"],
        "regions": regions,
    }


def _plot_trace(
    aggregate: pd.DataFrame,
    selection: dict[str, Any],
    *,
    config_layer: int | None,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = aggregate["window_center"].astype(int).to_numpy()
    y = aggregate["mean_ie"].astype(float).to_numpy()
    lower = aggregate["mean_ie_ci_lower"].astype(float).to_numpy()
    upper = aggregate["mean_ie_ci_upper"].astype(float).to_numpy()

    ax.bar(x, y, alpha=0.35, label="mean IE")
    ax.plot(x, aggregate["median_ie"].astype(float).to_numpy(), color="#2b6cb0", label="median IE")
    ax.fill_between(x, lower, upper, color="#2b6cb0", alpha=0.15, label="bootstrap CI")
    ax.axhline(0.0, color="black", linewidth=0.8)

    if config_layer is not None:
        ax.axvline(int(config_layer), color="#7b2cbf", linestyle="--", linewidth=2, label=f"config layer {config_layer}")
    raw_peak = selection.get("raw_mean_peak_center")
    if raw_peak is not None:
        ax.axvline(int(raw_peak), color="#dd6b20", linestyle=":", linewidth=2, label=f"raw mean peak {raw_peak}")
    strict = selection.get("strict_layer")
    if strict is not None:
        ax.axvline(int(strict), color="#1a7f37", linewidth=2.5, label=f"strict trace center {strict}")
    diagnostic = selection.get("diagnostic_layer")
    if strict is None and diagnostic is not None:
        ax.axvline(int(diagnostic), color="#718096", linestyle="-.", linewidth=2, label=f"diagnostic center {diagnostic}")

    centers = selection.get("robust_peak_region_centers") or []
    if centers:
        ax.axvspan(min(centers) - 0.5, max(centers) + 0.5, color="#38a169", alpha=0.12, label="robust region")

    ax.set_title("Early-site causal tracing: subject-last MLP-window restoration")
    ax.set_xlabel("MLP window center")
    ax.set_ylabel("fact-averaged indirect effect")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def causal_trace(cfg: DictConfig) -> Path:
    trace_cfg = _section(cfg, "causal_trace")
    output_root = Path(str(_get(trace_cfg, "output_dir", "analysis_out/causal_trace")))
    model_slug = str(cfg.model.name).replace("/", "_")
    out_dir = output_root / f"{model_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(_get(trace_cfg, "seed", getattr(cfg, "seed", 42)))
    torch.manual_seed(seed)
    np.random.seed(seed)

    handler = ModelHandler(cfg)
    handler.model.eval()
    modules = _module_dict(handler.model)
    embedding_name = _embedding_module_name(cfg)
    embedding_module = modules.get(embedding_name)
    if embedding_module is None:
        raise KeyError(f"Embedding module not found: {embedding_name}")

    num_layers = int(handler.num_of_layers)
    window_size = int(_get(trace_cfg, "window_size", 10))
    windows = [build_window(center, window_size, num_layers) for center in range(num_layers)]
    module_names = {layer: _resolve_mlp_output_name(handler, modules, layer) for layer in range(num_layers)}

    num_valid = int(_get(trace_cfg, "num_valid_facts", 100))
    max_scan = int(_get(trace_cfg, "max_dataset_examples_to_scan", max(1000, num_valid * 50)))
    num_noise = int(_get(trace_cfg, "num_noise_samples", 10))
    noise_multiplier = float(_get(trace_cfg, "noise_multiplier", 3.0))
    require_correct_clean = bool(_get(trace_cfg, "require_correct_clean_prediction", True))
    min_total_effect = float(_get(trace_cfg, "min_total_effect", 0.03))

    fact_results: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    scanned = 0

    for fact_index, example in enumerate(
        tqdm(_dataset_examples(cfg, max_scan=max_scan), total=max_scan, desc="causal-trace scan")
    ):
        scanned += 1
        if len(fact_results) >= num_valid:
            break
        try:
            result = _trace_example(
                handler,
                modules,
                module_names,
                embedding_module,
                example,
                windows=windows,
                num_noise_samples=num_noise,
                noise_multiplier=noise_multiplier,
                seed=seed + fact_index,
                require_correct_clean=require_correct_clean,
                min_total_effect=min_total_effect,
            )
            result["fact_index"] = int(fact_index)
            fact_results.append(result)
        except Exception as exc:
            rejections.append(
                {
                    "fact_index": int(fact_index),
                    "prompt_id": example.prompt_id,
                    "subject": example.subject,
                    "target": example.target,
                    "reason": str(exc),
                }
            )
        finally:
            handler.remove_hooks()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with (out_dir / "fact_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in fact_results:
            handle.write(json.dumps(row, default=_json_default) + "\n")
    pd.DataFrame(rejections).to_csv(out_dir / "rejections.csv", index=False)

    if not fact_results:
        summary = {
            "model": str(cfg.model.name),
            "selected_trace_center": None,
            "failure_reason": "no_valid_facts",
            "num_dataset_examples_scanned": scanned,
            "num_valid_facts": 0,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOGGER.warning("No valid causal-trace facts collected. Wrote %s", out_dir)
        return out_dir

    split = max(1, len(fact_results) // 2)
    discovery_facts = fact_results[:split]
    confirmation_facts = fact_results[split:] or fact_results[:split]
    discovery = _summarize_windows(
        discovery_facts,
        windows,
        window_size=window_size,
        trim_fraction=float(_get(trace_cfg, "trim_fraction", 0.10)),
        bootstrap_samples=int(_get(trace_cfg, "bootstrap_samples", 1000)),
        confidence_level=float(_get(trace_cfg, "confidence_level", 0.95)),
        seed=seed + 10_000,
    )
    confirmation = _summarize_windows(
        confirmation_facts,
        windows,
        window_size=window_size,
        trim_fraction=float(_get(trace_cfg, "trim_fraction", 0.10)),
        bootstrap_samples=int(_get(trace_cfg, "bootstrap_samples", 1000)),
        confidence_level=float(_get(trace_cfg, "confidence_level", 0.95)),
        seed=seed + 20_000,
    )
    all_facts = _summarize_windows(
        fact_results,
        windows,
        window_size=window_size,
        trim_fraction=float(_get(trace_cfg, "trim_fraction", 0.10)),
        bootstrap_samples=int(_get(trace_cfg, "bootstrap_samples", 1000)),
        confidence_level=float(_get(trace_cfg, "confidence_level", 0.95)),
        seed=seed + 30_000,
    )

    selection = _select_region(
        discovery,
        confirmation,
        num_layers=num_layers,
        window_size=window_size,
        minimum_supported_centers=int(_get(trace_cfg, "minimum_supported_centers", 3)),
        middle_min_fraction=float(_get(trace_cfg, "middle_center_min_fraction", 0.25)),
        middle_max_fraction=float(_get(trace_cfg, "middle_center_max_fraction", 0.75)),
    )

    discovery.to_csv(out_dir / "discovery_windows.csv", index=False)
    confirmation.to_csv(out_dir / "confirmation_windows.csv", index=False)
    all_facts.to_csv(out_dir / "aggregate_windows.csv", index=False)
    pd.DataFrame(selection.get("regions", [])).to_csv(out_dir / "candidate_regions.csv", index=False)
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2, default=_json_default), encoding="utf-8")

    config_layer = getattr(cfg.model, "layer", None)
    config_layer = None if config_layer is None else int(config_layer)
    plot_path = out_dir / "early_site_trace.png"
    _plot_trace(all_facts, selection, config_layer=config_layer, output_path=plot_path)

    summary = {
        "model": str(cfg.model.name),
        "configured_reference_layer": config_layer,
        "configured_reference_layer_used_for_selection": False,
        "selected_trace_center": selection.get("strict_layer"),
        "diagnostic_trace_center": selection.get("diagnostic_layer"),
        "selection_failure_reason": selection.get("strict_failure_reason"),
        "raw_mean_peak_center": selection.get("raw_mean_peak_center"),
        "robust_peak_region_centers": selection.get("robust_peak_region_centers"),
        "robust_peak_region_layer_union": selection.get("robust_peak_region_layer_union"),
        "num_dataset_examples_scanned": int(scanned),
        "num_valid_facts": int(len(fact_results)),
        "num_rejected": int(len(rejections)),
        "trace_component": "mlp_output",
        "trace_position": "subject_last",
        "window_size": int(window_size),
        "num_noise_samples": int(num_noise),
        "first_token_only": True,
        "plot": str(plot_path),
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")

    LOGGER.info(
        "Causal trace complete: model=%s selected=%s diagnostic=%s output=%s",
        cfg.model.name,
        summary["selected_trace_center"],
        summary["diagnostic_trace_center"],
        out_dir,
    )
    print(json.dumps(summary, indent=2, default=_json_default))
    return out_dir


def compute_multiplier(cfg: DictConfig) -> float:
    """Return the default subject-embedding noise scale for the configured model."""
    handler = ModelHandler(cfg)
    try:
        std = _embedding_std(handler, _module_dict(handler.model))
        trace_cfg = _section(cfg, "causal_trace")
        return float(std * float(_get(trace_cfg, "noise_multiplier", 3.0)))
    finally:
        handler.remove_hooks()
