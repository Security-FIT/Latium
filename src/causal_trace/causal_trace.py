"""
Early-site causal tracing for ROME layer investigation.

The command intentionally implements one tracing workflow:

* corrupt the full subject-token span at the embedding output,
* restore clean MLP outputs at the last subject token,
* sweep overlapping MLP windows,
* aggregate paired indirect effects across facts,
* discover a contiguous robust region and confirm it on held-out facts,
* choose one consistency-ranked representative center inside that region.

The configured model layer is kept only as a reference marker in the plot and
summary.  It must not influence the selected trace center.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from yaml import MappingNode, ScalarNode, compose

from src.common.loading import load_dataset
from src.common.model_config import MODEL_CONFIG_DIR
from src.causal_trace.model_adapter import (
    candidate_mlp_output_names as _candidate_mlp_output_names,
    clean_cache as _clean_cache,
    corrupt_hook as _corrupt_hook,
    embedding_module_name as _embedding_module_name,
    embedding_std as _embedding_std,
    make_noise_samples,
    module_dict as _module_dict,
    patch_mlp_position as _patch_mlp_position,
    prepare_inputs as _prepare_inputs,
    probabilities as _probabilities,
    probability as _probability,
    repeat_inputs as _repeat_inputs,
    resolve_mlp_output_name as _resolve_mlp_output_name,
    restore_hook as _restore_hook,
    temporary_hooks,
    top_token as _top_token,
    validate_mlp_output_modules as _validate_mlp_output_modules,
)
from src.causal_trace.selection import (
    Window,
    build_window,
    select_region as _select_region,
    select_window as _select_window,
    summarize_windows as _summarize_windows,
)
from src.causal_trace.tokenization import (
    TokenSpan,
    TraceValidationError,
    find_subject_span,
    target_first_token_id,
    target_token_ids,
)
from src.causal_trace.settings import TraceSettings, config_section as _section, required as _required
from src.handlers.rome import ModelHandler

LOGGER = logging.getLogger(__name__)


@dataclass
class TraceExample:
    prompt_id: str
    prompt: str
    subject: str
    target: str


@dataclass(frozen=True)
class CorruptionCalibration:
    """A corruption scale fixed before any layer restoration is evaluated."""

    multiplier: float
    probabilities: np.ndarray
    relative_std: float
    total_effect: float
    evaluations: tuple[dict[str, Any], ...]
    minimum_multiplier: float | None = None
    maximum_multiplier: float | None = None
    tolerance_ratio: float | None = None


def _corruption_statistics(
    clean_probability: float,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    corrupt = np.asarray(probabilities, dtype=np.float64)
    if corrupt.ndim != 1 or corrupt.size == 0 or not np.all(np.isfinite(corrupt)):
        raise RuntimeError("Corrupt probabilities must be a non-empty finite vector")
    if not math.isfinite(float(clean_probability)):
        raise RuntimeError("Clean probability must be finite")
    total_effect = float(clean_probability - np.mean(corrupt))
    relative_std = float(np.std(corrupt) / max(abs(total_effect), np.finfo(np.float64).tiny))
    return corrupt, total_effect, relative_std


def _calibrate_corruption(
    evaluate: Callable[[float], np.ndarray],
    *,
    clean_probability: float,
    min_total_effect: float,
    max_corrupt_relative_std: float,
    dtype: torch.dtype,
) -> CorruptionCalibration:
    """Find the weakest resolvable corruption meeting the effect requirement.

    Calibration sees only the clean probability and corrupt-only forwards. It
    cannot inspect layer identities, restoration effects, or the configured ROME
    layer. Search bounds and convergence precision come from the embedding dtype
    rather than a model-specific multiplier.
    """
    if min_total_effect <= 0:
        raise ValueError("Automatic corruption calibration requires min_total_effect > 0")
    try:
        dtype_epsilon = float(torch.finfo(dtype).eps)
    except TypeError as exc:
        raise TypeError(f"Embedding dtype must be floating point, got {dtype}") from exc

    resolution = math.sqrt(dtype_epsilon)
    minimum_multiplier = resolution
    maximum_multiplier = 1.0 / resolution
    tolerance_ratio = 1.0 + resolution
    cache: dict[float, tuple[np.ndarray, float, float]] = {}
    evaluations: list[dict[str, Any]] = []

    def assess(multiplier: float) -> tuple[np.ndarray, float, float]:
        multiplier = float(multiplier)
        cached = cache.get(multiplier)
        if cached is not None:
            return cached
        corrupt, total_effect, relative_std = _corruption_statistics(
            clean_probability,
            evaluate(multiplier),
        )
        cached = (corrupt, total_effect, relative_std)
        cache[multiplier] = cached
        evaluations.append(
            {
                "multiplier": multiplier,
                "total_effect": total_effect,
                "corrupt_relative_std": relative_std,
                "meets_effect_requirement": bool(total_effect >= min_total_effect),
                "meets_stability_requirement": bool(relative_std <= max_corrupt_relative_std),
            }
        )
        return cached

    def eligible(total_effect: float, relative_std: float) -> bool:
        return total_effect >= min_total_effect and relative_std <= max_corrupt_relative_std

    _, unit_effect, unit_relative_std = assess(1.0)
    lower: float | None = None
    upper: float | None = None

    if eligible(unit_effect, unit_relative_std):
        upper = 1.0
        while upper > minimum_multiplier:
            candidate = max(minimum_multiplier, upper / 2.0)
            _, candidate_effect, candidate_relative_std = assess(candidate)
            if eligible(candidate_effect, candidate_relative_std):
                upper = candidate
                if candidate == minimum_multiplier:
                    break
            else:
                lower = candidate
                break
    else:
        lower = 1.0
        while lower < maximum_multiplier:
            candidate = min(maximum_multiplier, lower * 2.0)
            _, candidate_effect, candidate_relative_std = assess(candidate)
            if eligible(candidate_effect, candidate_relative_std):
                upper = candidate
                break
            lower = candidate
            if candidate == maximum_multiplier:
                break

    if upper is None:
        _, maximum_effect, maximum_relative_std = assess(maximum_multiplier)
        raise TraceValidationError(
            "no stable effective corruption throughout automatic calibration: "
            f"maximum_effect={maximum_effect:.6f}, "
            f"maximum_relative_std={maximum_relative_std:.6f}"
        )

    if lower is not None:
        while upper / lower > tolerance_ratio:
            candidate = math.sqrt(lower * upper)
            _, candidate_effect, candidate_relative_std = assess(candidate)
            if eligible(candidate_effect, candidate_relative_std):
                upper = candidate
            else:
                lower = candidate

    corrupt, total_effect, relative_std = assess(upper)
    if relative_std > max_corrupt_relative_std:
        raise TraceValidationError(
            "unstable corrupt baseline after automatic calibration: "
            f"relative_std={relative_std:.6f} > {float(max_corrupt_relative_std):.6f}"
        )
    return CorruptionCalibration(
        multiplier=float(upper),
        probabilities=corrupt,
        relative_std=relative_std,
        total_effect=total_effect,
        evaluations=tuple(evaluations),
        minimum_multiplier=minimum_multiplier,
        maximum_multiplier=maximum_multiplier,
        tolerance_ratio=tolerance_ratio,
    )


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


def _resolve_model_config_path(
    cfg: DictConfig,
    *,
    config_dir: Path = MODEL_CONFIG_DIR,
) -> Path:
    """Resolve the exact model YAML selected by Hydra."""
    config_root = Path(config_dir).resolve()
    choice = OmegaConf.select(cfg, "command.causal_trace._model_config_key", default=None)
    if not choice:
        choice = OmegaConf.select(cfg, "hydra.runtime.choices.model", default=None)
    if choice:
        path = (config_root / f"{choice}.yaml").resolve()
        if path.parent != config_root or not path.is_file():
            raise FileNotFoundError(f"Selected model config does not exist: {path}")
        return path

    model_name = str(cfg.model.name).strip().lower()
    matches = []
    for path in sorted(config_root.glob("*.yaml")):
        if path.name == "boilerplate.yaml":
            continue
        candidate = OmegaConf.load(path)
        if str(getattr(candidate, "name", "")).strip().lower() == model_name:
            matches.append(path)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No writable model YAML matches {cfg.model.name!s}")
    choices = ", ".join(path.stem for path in matches)
    raise ValueError(
        f"Cannot identify the selected model YAML without Hydra choice metadata; {cfg.model.name!s} matches: {choices}"
    )


def _overwrite_model_config_layer(path: Path, layer: int) -> int:
    """Atomically replace only the top-level ``layer`` YAML scalar."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    document = compose(text)
    if not isinstance(document, MappingNode):
        raise ValueError(f"Model config must be a YAML mapping: {path}")

    layer_node = None
    for key_node, value_node in document.value:
        if isinstance(key_node, ScalarNode) and key_node.value == "layer":
            if layer_node is not None:
                raise ValueError(f"Model config has duplicate top-level layer keys: {path}")
            layer_node = value_node
    if not isinstance(layer_node, ScalarNode):
        raise ValueError(f"Model config has no scalar top-level layer: {path}")

    previous_layer = int(layer_node.value)
    start = layer_node.start_mark.index
    end = layer_node.end_mark.index
    updated = f"{text[:start]}{int(layer)}{text[end:]}"

    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, path.stat().st_mode)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return previous_layer


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
        )


def _trace_example(
    handler: ModelHandler,
    modules: dict[str, torch.nn.Module],
    module_names: dict[int, str],
    embedding_module: torch.nn.Module,
    example: TraceExample,
    *,
    windows: list[Window],
    num_noise_samples: int,
    noise_batch_size: int,
    noise_multiplier: float | None,
    seed: int,
    require_correct_clean: bool,
    min_total_effect: float,
    max_corrupt_relative_std: float,
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

    embedding_weight = getattr(embedding_module, "weight", None)
    if embedding_weight is None or embedding_weight.dim() != 2:
        raise RuntimeError("The configured embedding module must expose a rank-2 weight")
    hidden_size = int(embedding_weight.shape[-1])
    embedding_scale = _embedding_std(handler, modules)
    batch_size = max(1, min(int(noise_batch_size), int(num_noise_samples)))

    def evaluate_corruption(multiplier: float, *, noise_seed: int) -> np.ndarray:
        candidate_noise = make_noise_samples(
            num_samples=int(num_noise_samples),
            subject_length=len(span.positions),
            hidden_size=hidden_size,
            noise_std=embedding_scale * float(multiplier),
            device=embedding_weight.device,
            dtype=embedding_weight.dtype,
            seed=int(noise_seed),
        )
        probabilities = np.zeros(int(num_noise_samples), dtype=np.float64)
        for batch_start in range(0, int(num_noise_samples), batch_size):
            batch_end = min(int(num_noise_samples), batch_start + batch_size)
            repeated = _repeat_inputs(inputs, batch_end - batch_start)
            noise = candidate_noise[batch_start:batch_end]
            with torch.inference_mode(), temporary_hooks([(embedding_module, _corrupt_hook(span.positions, noise))]):
                outputs = handler.model(**repeated, use_cache=False)
            probabilities[batch_start:batch_end] = _probabilities(outputs, target_id)
        return probabilities

    calibration_seed: int | None = None
    evaluation_seed = int(seed)
    if noise_multiplier is None:
        calibration_mode = "automatic_minimum_eligible"
        calibration_sequence, evaluation_sequence = np.random.SeedSequence(int(seed)).spawn(2)
        calibration_seed = int(calibration_sequence.generate_state(1, dtype=np.uint64)[0])
        evaluation_seed = int(evaluation_sequence.generate_state(1, dtype=np.uint64)[0])
        calibration = _calibrate_corruption(
            lambda multiplier: evaluate_corruption(multiplier, noise_seed=calibration_seed),
            clean_probability=clean_probability,
            min_total_effect=min_total_effect,
            max_corrupt_relative_std=max_corrupt_relative_std,
            dtype=embedding_weight.dtype,
        )
        corrupt, total_effect, corrupt_relative_std = _corruption_statistics(
            clean_probability,
            evaluate_corruption(calibration.multiplier, noise_seed=evaluation_seed),
        )
    else:
        calibration_mode = "explicit"
        resolved_multiplier = float(noise_multiplier)
        corrupt, total_effect, corrupt_relative_std = _corruption_statistics(
            clean_probability,
            evaluate_corruption(resolved_multiplier, noise_seed=evaluation_seed),
        )
        calibration = CorruptionCalibration(
            multiplier=resolved_multiplier,
            probabilities=corrupt,
            relative_std=corrupt_relative_std,
            total_effect=total_effect,
            evaluations=(
                {
                    "multiplier": resolved_multiplier,
                    "total_effect": total_effect,
                    "corrupt_relative_std": corrupt_relative_std,
                    "meets_effect_requirement": bool(total_effect >= min_total_effect),
                    "meets_stability_requirement": bool(corrupt_relative_std <= max_corrupt_relative_std),
                },
            ),
        )

    if total_effect < float(min_total_effect):
        raise TraceValidationError(f"low corruption effect: {total_effect:.6f}")
    if corrupt_relative_std > float(max_corrupt_relative_std):
        raise TraceValidationError(
            "unstable corrupt baseline: "
            f"relative_std={corrupt_relative_std:.6f} > {float(max_corrupt_relative_std):.6f}"
        )

    noise_std = calibration.multiplier * embedding_scale
    noise_samples = make_noise_samples(
        num_samples=int(num_noise_samples),
        subject_length=len(span.positions),
        hidden_size=hidden_size,
        noise_std=noise_std,
        device=embedding_weight.device,
        dtype=embedding_weight.dtype,
        seed=evaluation_seed,
    )

    # Cache and restore only after the fact passes the predeclared clean/corrupt
    # eligibility checks. This preserves the estimand while avoiding a complete
    # layer-window sweep for facts that will be rejected.
    clean_cache = _clean_cache(handler, modules, module_names, inputs, span.last_position)
    restore_probabilities = np.zeros((len(windows), int(num_noise_samples)), dtype=np.float64)
    sequence_length = int(inputs["input_ids"].shape[1])
    for window_idx, window in enumerate(windows):
        for batch_start in range(0, int(num_noise_samples), batch_size):
            batch_end = min(int(num_noise_samples), batch_start + batch_size)
            repeated = _repeat_inputs(inputs, batch_end - batch_start)
            noise = noise_samples[batch_start:batch_end]
            hooks: list[tuple[torch.nn.Module, Any]] = [(embedding_module, _corrupt_hook(span.positions, noise))]
            for layer in window.layers:
                hooks.append(
                    (
                        modules[module_names[layer]],
                        _restore_hook(span.last_position, clean_cache[layer], sequence_length),
                    )
                )
            with torch.inference_mode(), temporary_hooks(hooks):
                outputs = handler.model(**repeated, use_cache=False)
            restore_probabilities[window_idx, batch_start:batch_end] = _probabilities(outputs, target_id)

    effects = restore_probabilities - corrupt.reshape(1, -1)

    target_ids = target_token_ids(handler.tokenizer, example.target)
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
        "corrupt_relative_std": corrupt_relative_std,
        "total_effect": total_effect,
        "noise_std": float(noise_std),
        "noise_multiplier": float(calibration.multiplier),
        "noise_calibration": {
            "mode": calibration_mode,
            "configured_multiplier": None if noise_multiplier is None else float(noise_multiplier),
            "minimum_multiplier": calibration.minimum_multiplier,
            "maximum_multiplier": calibration.maximum_multiplier,
            "tolerance_ratio": calibration.tolerance_ratio,
            "calibration_seed": calibration_seed,
            "evaluation_seed": evaluation_seed,
            "evaluations": list(calibration.evaluations),
        },
        "window_mean_ie": effects.mean(axis=1).tolist(),
        "window_mean_normalized_recovery": (effects.mean(axis=1) / max(abs(total_effect), 1e-12)).tolist(),
        "window_std_ie": effects.std(axis=1).tolist(),
        "window_restore_probabilities": restore_probabilities.tolist(),
    }
    return result


def _plot_trace(
    discovery: pd.DataFrame,
    confirmation: pd.DataFrame,
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
    aggregate_mean = aggregate["mean_ie"].astype(float).to_numpy()
    discovery_mean = discovery["mean_ie"].astype(float).to_numpy() if not discovery.empty else aggregate_mean.copy()
    confirmation_mean = (
        confirmation["mean_ie"].astype(float).to_numpy()
        if not confirmation.empty
        else np.full_like(aggregate_mean, np.nan)
    )
    lower = (
        confirmation["mean_ie_ci_lower"].astype(float).to_numpy()
        if not confirmation.empty
        else np.full_like(aggregate_mean, np.nan)
    )
    upper = (
        confirmation["mean_ie_ci_upper"].astype(float).to_numpy()
        if not confirmation.empty
        else np.full_like(aggregate_mean, np.nan)
    )

    partial = ~aggregate["window_is_full_width"].astype(bool).to_numpy()
    ax.bar(x[partial], aggregate_mean[partial], color="#cbd5e0", alpha=0.5, label="partial boundary windows")
    ax.bar(x[~partial], aggregate_mean[~partial], color="#90cdf4", alpha=0.45, label="all-fact mean IE")
    ax.plot(x, discovery_mean, color="#4a5568", linestyle="--", linewidth=1.8, label="discovery mean IE")
    ax.plot(x, confirmation_mean, color="#2b6cb0", linewidth=2, label="confirmation mean IE")
    ax.fill_between(x, lower, upper, color="#2b6cb0", alpha=0.15, label="confirmation bootstrap CI")
    ax.axhline(0.0, color="black", linewidth=0.8)

    if config_layer is not None:
        ax.axvline(
            int(config_layer), color="#7b2cbf", linestyle="--", linewidth=2, label=f"config layer {config_layer}"
        )
    confirmed_region = [int(center) for center in selection.get("confirmed_region_centers", [])]
    if confirmed_region:
        ax.axvspan(
            min(confirmed_region) - 0.45,
            max(confirmed_region) + 0.45,
            color="#68d391",
            alpha=0.18,
            label=f"confirmed region {min(confirmed_region)}-{max(confirmed_region)}",
        )
    selected_center = selection.get("selected_trace_center")
    if selected_center is not None:
        passed = bool(selection.get("confirmation_passed"))
        color = "#1a7f37" if passed else "#dd6b20"
        label = "held-out confirmed" if passed else "not held-out confirmed"
        ax.axvline(
            int(selected_center),
            color=color,
            linewidth=2.5,
            label=f"representative center {selected_center} ({label})",
        )

    ax.set_title("Early-site causal tracing: subject-last MLP-window restoration")
    ax.set_xlabel("MLP window center")
    ax.set_ylabel("mean paired indirect effect across facts")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _run_causal_trace(cfg: DictConfig, handler: ModelHandler) -> Path:
    num_layers = int(handler.num_of_layers)
    settings = TraceSettings.from_config(cfg, num_layers=num_layers)
    overwrite_model_config_layer = settings.overwrite_model_config_layer
    model_config_path = _resolve_model_config_path(cfg) if overwrite_model_config_layer else None
    config_layer = getattr(cfg.model, "layer", None)
    config_layer = None if config_layer is None else int(config_layer)
    model_slug = str(cfg.model.name).replace("/", "_")
    out_dir = settings.output_dir / f"{model_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = settings.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    handler.model.eval()
    modules = _module_dict(handler.model)
    embedding_name = _embedding_module_name(cfg)
    embedding_module = modules.get(embedding_name)
    if embedding_module is None:
        raise KeyError(f"Embedding module not found: {embedding_name}")

    window_size = settings.window_size
    windows = [build_window(center, window_size, num_layers) for center in range(num_layers)]
    module_names = {layer: _resolve_mlp_output_name(handler, modules, layer) for layer in range(num_layers)}
    adapter_validation = _validate_mlp_output_modules(handler, modules, module_names)
    module_map_path = out_dir / "mlp_module_map.json"
    module_map_path.write_text(json.dumps(adapter_validation, indent=2), encoding="utf-8")

    num_valid = settings.num_valid_facts
    max_scan = settings.max_dataset_examples_to_scan
    num_noise = settings.num_noise_samples
    noise_batch_size = settings.noise_batch_size
    noise_multiplier = settings.noise_multiplier
    require_correct_clean = settings.require_correct_clean_prediction
    min_total_effect = settings.min_total_effect
    max_corrupt_relative_std = settings.max_corrupt_relative_std
    bootstrap_samples = settings.bootstrap_samples
    confidence_level = settings.confidence_level
    minimum_confirmation_facts = settings.minimum_confirmation_facts
    discovery_fraction = settings.discovery_fraction
    trim_fraction = settings.trim_fraction
    neighbor_support_radius = settings.neighbor_support_radius
    local_support_fraction = settings.local_support_fraction
    adjacent_peak_radius = settings.adjacent_peak_radius
    noninferiority_margin_fraction = settings.noninferiority_margin_fraction
    minimum_supported_centers = settings.minimum_supported_centers
    allow_near_supported_region = settings.allow_near_supported_region

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
                noise_batch_size=noise_batch_size,
                noise_multiplier=noise_multiplier,
                seed=seed + fact_index,
                require_correct_clean=require_correct_clean,
                min_total_effect=min_total_effect,
                max_corrupt_relative_std=max_corrupt_relative_std,
            )
            result["fact_index"] = int(fact_index)
            fact_results.append(result)
        except TraceValidationError as exc:
            rejections.append(
                {
                    "fact_index": int(fact_index),
                    "prompt_id": example.prompt_id,
                    "subject": example.subject,
                    "target": example.target,
                    "reason": str(exc),
                }
            )
    with (out_dir / "fact_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in fact_results:
            handle.write(json.dumps(row, default=_json_default) + "\n")
    pd.DataFrame(rejections).to_csv(out_dir / "rejections.csv", index=False)

    if not fact_results:
        summary = {
            "model": str(cfg.model.name),
            "configured_reference_layer": config_layer,
            "selected_trace_center": None,
            "failure_reason": "no_valid_facts",
            "num_dataset_examples_scanned": scanned,
            "num_valid_facts": 0,
            "trace_component": "mlp_output",
            "trace_hook_semantics": "whole_mlp_module_output",
            "trace_mlp_module_map": str(module_map_path),
            "trace_mlp_output_modules": [module_names[layer] for layer in range(num_layers)],
            "trace_position": "subject_last",
            "noise_calibration_mode": "automatic_minimum_eligible" if noise_multiplier is None else "explicit",
            "configured_noise_multiplier": noise_multiplier,
            "min_total_effect": float(min_total_effect),
            "model_config_layer_overwrite_requested": overwrite_model_config_layer,
            "model_config_layer_overwritten": False,
            "model_config_path": str(model_config_path) if model_config_path is not None else None,
            "previous_model_config_layer": None,
            "new_model_config_layer": None,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOGGER.warning("No valid causal-trace facts collected. Wrote %s", out_dir)
        return out_dir

    split_rng = np.random.default_rng(seed)
    indices = split_rng.permutation(len(fact_results))
    discovery_count = max(1, int(round(len(fact_results) * discovery_fraction)))
    discovery_count = min(discovery_count, max(1, len(fact_results) - 1))
    discovery_indices = sorted(indices[:discovery_count].tolist())
    confirmation_indices = sorted(indices[discovery_count:].tolist())
    discovery_facts = [fact_results[idx] for idx in discovery_indices]
    confirmation_facts = [fact_results[idx] for idx in confirmation_indices]
    split_by_index = {idx: "discovery" for idx in discovery_indices}
    split_by_index.update({idx: "confirmation" for idx in confirmation_indices})
    pd.DataFrame(
        [
            {
                "fact_result_index": idx,
                "fact_index": int(row["fact_index"]),
                "prompt_id": row["prompt_id"],
                "split": split_by_index[idx],
            }
            for idx, row in enumerate(fact_results)
        ]
    ).to_csv(out_dir / "split_assignments.csv", index=False)
    discovery = _summarize_windows(
        discovery_facts,
        windows,
        window_size=window_size,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed + 100,
    )
    confirmation = _summarize_windows(
        confirmation_facts,
        windows,
        window_size=window_size,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed + 200,
    )
    all_facts = _summarize_windows(
        fact_results,
        windows,
        window_size=window_size,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed + 300,
    )

    selection = _select_region(
        discovery,
        confirmation,
        discovery_facts,
        confirmation_facts,
        windows,
        minimum_confirmation_facts=minimum_confirmation_facts,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
        trim_fraction=trim_fraction,
        neighbor_support_radius=neighbor_support_radius,
        local_support_fraction=local_support_fraction,
        adjacent_peak_radius=adjacent_peak_radius,
        noninferiority_margin_fraction=noninferiority_margin_fraction,
        minimum_supported_centers=minimum_supported_centers,
        allow_near_supported_region=allow_near_supported_region,
    )

    discovery.to_csv(out_dir / "discovery_windows.csv", index=False)
    confirmation.to_csv(out_dir / "confirmation_windows.csv", index=False)
    all_facts.to_csv(out_dir / "aggregate_windows.csv", index=False)
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2, default=_json_default), encoding="utf-8")

    plot_path = out_dir / "early_site_trace.png"
    _plot_trace(discovery, confirmation, all_facts, selection, config_layer=config_layer, output_path=plot_path)

    selected_trace_center = selection.get("selected_trace_center")
    previous_model_config_layer = None
    model_config_layer_overwritten = False
    if model_config_path is not None and selected_trace_center is not None:
        previous_model_config_layer = _overwrite_model_config_layer(model_config_path, int(selected_trace_center))
        model_config_layer_overwritten = True
        LOGGER.info(
            "Updated model config layer: path=%s previous=%s selected=%s",
            model_config_path,
            previous_model_config_layer,
            selected_trace_center,
        )

    resolved_noise_multipliers = np.asarray([float(row["noise_multiplier"]) for row in fact_results], dtype=np.float64)
    summary = {
        "model": str(cfg.model.name),
        "configured_reference_layer": config_layer,
        "configured_reference_layer_used_for_selection": False,
        "selection_method": selection.get("selection_method"),
        "selected_trace_center": selected_trace_center,
        "discovery_trace_center": selection.get("discovery_trace_center"),
        "discovery_trace_window_layers": selection.get("trace_window_layers"),
        "selected_trace_window_layers": (
            selection.get("trace_window_layers") if selection.get("selected_trace_center") is not None else None
        ),
        "confirmed_region_centers": selection.get("confirmed_region_centers", []),
        "confirmed_region_layer_union": selection.get("confirmed_region_layer_union", []),
        "confirmation_mean_ie": selection.get("confirmation_mean_ie"),
        "confirmation_ci_lower": selection.get("confirmation_ci_lower"),
        "confirmation_ci_upper": selection.get("confirmation_ci_upper"),
        "confirmation_regions": selection.get("confirmation_regions", []),
        "confirmed_regions": selection.get("confirmed_regions", []),
        "confirmation_passed": selection.get("confirmation_passed", False),
        "selection_failure_reason": selection.get("failure_reason"),
        "num_dataset_examples_scanned": int(scanned),
        "num_valid_facts": int(len(fact_results)),
        "num_discovery_facts": int(len(discovery_facts)),
        "num_confirmation_facts": int(len(confirmation_facts)),
        "num_rejected": int(len(rejections)),
        "trace_component": "mlp_output",
        "trace_hook_semantics": "whole_mlp_module_output",
        "trace_mlp_module_map": str(module_map_path),
        "trace_mlp_output_modules": [module_names[layer] for layer in range(num_layers)],
        "trace_position": "subject_last",
        "window_size": int(window_size),
        "num_noise_samples": int(num_noise),
        "noise_batch_size": int(noise_batch_size),
        "max_corrupt_relative_std": float(max_corrupt_relative_std),
        "noise_calibration_mode": "automatic_minimum_eligible" if noise_multiplier is None else "explicit",
        "configured_noise_multiplier": noise_multiplier,
        "resolved_noise_multiplier_min": float(np.min(resolved_noise_multipliers)),
        "resolved_noise_multiplier_median": float(np.median(resolved_noise_multipliers)),
        "resolved_noise_multiplier_max": float(np.max(resolved_noise_multipliers)),
        "min_total_effect": float(min_total_effect),
        "first_token_only": True,
        "model_config_layer_overwrite_requested": overwrite_model_config_layer,
        "model_config_layer_overwritten": model_config_layer_overwritten,
        "model_config_path": str(model_config_path) if model_config_path is not None else None,
        "previous_model_config_layer": previous_model_config_layer,
        "new_model_config_layer": int(selected_trace_center) if model_config_layer_overwritten else None,
        "plot": str(plot_path),
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")

    LOGGER.info(
        "Causal trace complete: model=%s selected=%s discovery=%s output=%s",
        cfg.model.name,
        summary["selected_trace_center"],
        summary["discovery_trace_center"],
        out_dir,
    )
    print(json.dumps(summary, indent=2, default=_json_default))
    return out_dir


def causal_trace(cfg: DictConfig) -> Path:
    handler = ModelHandler(cfg)
    try:
        return _run_causal_trace(cfg, handler)
    finally:
        handler.remove_hooks()


def compute_multiplier(cfg: DictConfig) -> float:
    """Return an explicit noise scale; automatic calibration is fact-specific."""
    trace_cfg = _section(cfg, "causal_trace")
    configured = _required(trace_cfg, "noise_multiplier")
    if isinstance(configured, str) and configured.strip().lower() == "auto":
        raise ValueError("Automatic noise calibration is fact-specific; run causal tracing to resolve it")
    handler = ModelHandler(cfg)
    try:
        std = _embedding_std(handler, _module_dict(handler.model))
        return float(std * float(configured))
    finally:
        handler.remove_hooks()


__all__ = [
    "TokenSpan",
    "TraceValidationError",
    "Window",
    "build_window",
    "causal_trace",
    "compute_multiplier",
    "find_subject_span",
    "make_noise_samples",
    "target_first_token_id",
    "target_token_ids",
    "temporary_hooks",
]
