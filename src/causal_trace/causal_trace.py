"""
Early-site causal tracing for ROME layer investigation.

The command intentionally implements one tracing workflow:

* corrupt the full subject-token span at the embedding output,
* restore clean MLP outputs at the last subject token,
* sweep MLP layers on discovery facts,
* aggregate paired indirect effects across facts,
* freeze one discovery layer and confirm that exact intervention on held-out facts.

The configured model layer is kept only as a reference marker in the plot and
summary.  It must not influence the selected trace center.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from yaml import MappingNode, ScalarNode, compose

from src.common.loading import load_dataset
from src.common.model_config import MODEL_CONFIG_DIR
from src.causal_trace.model_adapter import (
    corrupt_hook as _corrupt_hook,
    embedding_module_name as _embedding_module_name,
    embedding_std as _embedding_std,
    make_noise_samples,
    module_dict as _module_dict,
    hidden_from_output as _hidden_from_output,
    mlp_state_at_position as _mlp_state_at_position,
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
    discovery_window as _discovery_window,
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


@dataclass
class TraceExample:
    prompt_id: str
    prompt: str
    subject: str
    target: str


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


@dataclass
class PreparedTrace:
    row: dict[str, Any]
    inputs: dict[str, torch.Tensor]
    target_id: int
    clean_cache: dict[int, torch.Tensor]
    noise_samples: torch.Tensor


def _trace_example(
    handler: ModelHandler,
    modules: dict[str, torch.nn.Module],
    module_names: dict[int, str],
    embedding_module: torch.nn.Module,
    example: TraceExample,
    *,
    num_noise_samples: int,
    noise_batch_size: int,
    noise_multiplier: float,
    embedding_scale: float,
    seed: int,
    require_correct_clean: bool,
) -> PreparedTrace:
    """Collect clean states and a fixed-scale corrupt baseline before splitting facts."""
    inputs = _prepare_inputs(handler, example.prompt)
    span = find_subject_span(handler.tokenizer, example.prompt, example.subject)
    target_ids = target_token_ids(handler.tokenizer, example.target)
    target_id = int(target_ids[0])
    sequence_length = int(inputs["input_ids"].shape[1])
    embedding_weight = getattr(embedding_module, "weight", None)
    if embedding_weight is None or embedding_weight.dim() != 2:
        raise RuntimeError("The embedding module must expose a rank-2 weight")
    cache: dict[int, torch.Tensor] = {}

    def capture(layer: int):
        def hook(_module, _input, output):
            cache[layer] = _mlp_state_at_position(_hidden_from_output(output), span.last_position, sequence_length)
            return output

        return hook

    capture_hooks = [(modules[module_names[layer]], capture(layer)) for layer in module_names]
    with torch.inference_mode(), temporary_hooks(capture_hooks):
        clean_outputs = handler.model(**inputs, use_cache=False)
    if len(cache) != len(module_names):
        raise RuntimeError("Clean MLP cache is incomplete")
    clean_probability = _probability(clean_outputs, target_id)
    clean_top_id, clean_top_probability = _top_token(clean_outputs)
    if require_correct_clean and clean_top_id != target_id:
        raise TraceValidationError(
            f"clean-token mismatch: expected {handler.tokenizer.decode([target_id])!r}, "
            f"got {handler.tokenizer.decode([clean_top_id])!r}"
        )

    noise_std = embedding_scale * noise_multiplier
    noise_samples = make_noise_samples(
        num_samples=num_noise_samples,
        subject_length=len(span.positions),
        hidden_size=int(embedding_weight.shape[-1]),
        noise_std=noise_std,
        device=embedding_weight.device,
        dtype=embedding_weight.dtype,
        seed=seed,
    )
    corrupt = np.zeros(num_noise_samples, dtype=np.float64)
    batch_size = min(noise_batch_size, num_noise_samples)
    for begin in range(0, num_noise_samples, batch_size):
        end = min(num_noise_samples, begin + batch_size)
        repeated = _repeat_inputs(inputs, end - begin)
        with (
            torch.inference_mode(),
            temporary_hooks([(embedding_module, _corrupt_hook(span.positions, noise_samples[begin:end]))]),
        ):
            outputs = handler.model(**repeated, use_cache=False)
        corrupt[begin:end] = _probabilities(outputs, target_id)
    if not np.isfinite(corrupt).all():
        raise TraceValidationError("corrupt probabilities are not finite")
    total_effect = float(clean_probability - corrupt.mean())
    if not math.isfinite(total_effect) or total_effect <= 0:
        raise TraceValidationError(f"corruption did not reduce target probability: {total_effect:.6f}")

    row = {
        "prompt_id": example.prompt_id,
        "prompt": example.prompt,
        "subject": example.subject,
        "target": example.target,
        "target_first_token_id": target_id,
        "target_first_token_text": handler.tokenizer.decode([target_id]),
        "target_num_tokens": len(target_ids),
        "subject_positions": list(span.positions),
        "subject_last_position": span.last_position,
        "prompt_last_position": sequence_length - 1,
        "clean_probability": clean_probability,
        "clean_top_token_id": clean_top_id,
        "clean_top_probability": clean_top_probability,
        "corrupt_probabilities": corrupt.tolist(),
        "mean_corrupt_probability": float(corrupt.mean()),
        "std_corrupt_probability": float(corrupt.std()),
        "total_effect": total_effect,
        "noise_std": float(noise_std),
        "noise_multiplier": float(noise_multiplier),
        "noise_seed": int(seed),
        "window_centers": [],
        "window_mean_ie": [],
        "window_restore_probabilities": [],
    }
    return PreparedTrace(row, inputs, target_id, cache, noise_samples)


def _restore_windows(
    handler: ModelHandler,
    modules: dict[str, torch.nn.Module],
    module_names: dict[int, str],
    embedding_module: torch.nn.Module,
    prepared: PreparedTrace,
    windows: list[Window],
    *,
    noise_batch_size: int,
) -> None:
    """Evaluate just the requested windows using the same corrupt noise draws."""
    if not windows:
        return
    row = prepared.row
    corrupt = np.asarray(row["corrupt_probabilities"], dtype=np.float64)
    num_noise = len(corrupt)
    sequence_length = int(prepared.inputs["input_ids"].shape[1])
    subject_positions = row["subject_positions"]
    subject_last = int(row["subject_last_position"])
    restore = np.zeros((len(windows), num_noise), dtype=np.float64)
    for window_idx, window in enumerate(windows):
        for begin in range(0, num_noise, noise_batch_size):
            end = min(num_noise, begin + noise_batch_size)
            repeated = _repeat_inputs(prepared.inputs, end - begin)
            hooks: list[tuple[torch.nn.Module, Any]] = [
                (embedding_module, _corrupt_hook(subject_positions, prepared.noise_samples[begin:end]))
            ]
            hooks.extend(
                (
                    modules[module_names[layer]],
                    _restore_hook(subject_last, prepared.clean_cache[layer], sequence_length),
                )
                for layer in window.layers
            )
            with torch.inference_mode(), temporary_hooks(hooks):
                outputs = handler.model(**repeated, use_cache=False)
            restore[window_idx, begin:end] = _probabilities(outputs, prepared.target_id)
    effects = restore - corrupt[None, :]
    if not np.isfinite(effects).all():
        raise RuntimeError("Restoration effects are not finite")
    row["window_centers"] = [window.center for window in windows]
    row["window_mean_ie"] = effects.mean(axis=1).tolist()
    row["window_restore_probabilities"] = restore.tolist()


def _plot_trace(
    discovery: pd.DataFrame,
    confirmation: pd.DataFrame,
    selection: dict[str, Any],
    *,
    config_layer: int | None,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(discovery["window_center"], discovery["mean_ie"], color="#90cdf4", label="discovery mean IE")
    ax.axhline(0.0, color="black", linewidth=0.8)
    if config_layer is not None:
        ax.axvline(config_layer, color="#7b2cbf", linestyle="--", label=f"config layer {config_layer}")
    center = selection.get("discovery_trace_center")
    if center is not None:
        ax.axvline(center, color="#1a7f37", label=f"frozen window center {center}")
    if not confirmation.empty:
        row = confirmation.iloc[0]
        ax.errorbar(
            [int(row.window_center)],
            [float(row.mean_ie)],
            yerr=[
                [max(0.0, float(row.mean_ie - row.mean_ie_ci_lower))],
                [max(0.0, float(row.mean_ie_ci_upper - row.mean_ie))],
            ],
            fmt="o",
            color="#c53030",
            capsize=5,
            label="held-out mean and 95% interval",
        )
    ax.set_title("Subject-last MLP restoration")
    ax.set_xlabel("Restored window center")
    ax.set_ylabel("Mean paired probability effect")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_fact_results(path: Path, prepared_facts: list[PreparedTrace]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in prepared_facts:
            handle.write(json.dumps(item.row, default=_json_default) + "\n")


def _run_causal_trace(cfg: DictConfig, handler: ModelHandler) -> Path:
    num_layers = int(handler.num_of_layers)
    settings = TraceSettings.from_config(cfg, num_layers=num_layers)
    config_layer = getattr(cfg.model, "layer", None)
    config_layer = None if config_layer is None else int(config_layer)
    model_config_path = _resolve_model_config_path(cfg) if settings.overwrite_model_config_layer else None
    model_slug = str(cfg.model.name).replace("/", "_")
    out_dir = settings.output_dir / f"{model_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    torch.manual_seed(settings.seed)
    handler.model.eval()
    modules = _module_dict(handler.model)
    embedding_name = _embedding_module_name(cfg)
    embedding_module = modules.get(embedding_name)
    if embedding_module is None:
        raise KeyError(f"Embedding module not found: {embedding_name}")
    module_names = {layer: _resolve_mlp_output_name(handler, modules, layer) for layer in range(num_layers)}
    adapter_validation = _validate_mlp_output_modules(handler, modules, module_names)
    module_map_path = out_dir / "mlp_module_map.json"
    module_map_path.write_text(json.dumps(adapter_validation, indent=2), encoding="utf-8")
    embedding_scale = _embedding_std(handler, modules)
    windows = [build_window(center, settings.window_size, num_layers) for center in range(num_layers)]

    prepared_facts: list[PreparedTrace] = []
    rejections: list[dict[str, Any]] = []
    scanned = 0
    for fact_index, example in enumerate(
        tqdm(
            _dataset_examples(cfg, max_scan=settings.max_dataset_examples_to_scan),
            total=settings.max_dataset_examples_to_scan,
            desc="causal-trace scan",
        )
    ):
        scanned += 1
        try:
            prepared = _trace_example(
                handler,
                modules,
                module_names,
                embedding_module,
                example,
                num_noise_samples=settings.num_noise_samples,
                noise_batch_size=settings.noise_batch_size,
                noise_multiplier=settings.noise_multiplier,
                embedding_scale=embedding_scale,
                seed=settings.seed + fact_index,
                require_correct_clean=settings.require_correct_clean_prediction,
            )
            prepared.row["fact_index"] = fact_index
            prepared_facts.append(prepared)
        except TraceValidationError as exc:
            rejections.append(
                {
                    "fact_index": fact_index,
                    "prompt_id": example.prompt_id,
                    "subject": example.subject,
                    "target": example.target,
                    "reason": str(exc),
                }
            )
        if len(prepared_facts) == settings.num_valid_facts:
            break
    pd.DataFrame(rejections, columns=["fact_index", "prompt_id", "subject", "target", "reason"]).to_csv(
        out_dir / "rejections.csv", index=False
    )

    try:
        import subprocess

        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    summary: dict[str, Any] = {
        "model": str(cfg.model.name),
        "model_revision": getattr(handler.model.config, "_commit_hash", None),
        "tokenizer_revision": getattr(handler.tokenizer, "_commit_hash", None),
        "git_commit": git_commit,
        "model_dtype": str(handler.model.dtype),
        "configured_reference_layer": config_layer,
        "configured_reference_layer_used_for_selection": False,
        "trace_component": "mlp_output",
        "trace_hook_semantics": "whole_mlp_module_output",
        "trace_mlp_module_map": str(module_map_path),
        "trace_mlp_output_modules": [module_names[layer] for layer in range(num_layers)],
        "trace_position": "subject_last",
        "first_token_only": True,
        "window_size": settings.window_size,
        "selected_layer_directly_tested": settings.window_size == 1,
        "num_noise_samples": settings.num_noise_samples,
        "noise_batch_size": settings.noise_batch_size,
        "noise_multiplier": settings.noise_multiplier,
        "noise_std": float(embedding_scale * settings.noise_multiplier),
        "embedding_std": float(embedding_scale),
        "corruption_protocol": "fixed_gaussian_embedding_std",
        "num_dataset_examples_scanned": scanned,
        "num_valid_facts": len(prepared_facts),
        "num_rejected": len(rejections),
        "model_config_layer_overwrite_requested": settings.overwrite_model_config_layer,
        "model_config_layer_overwritten": False,
        "model_config_path": str(model_config_path) if model_config_path else None,
        "resolved_config": str(out_dir / "resolved_config.yaml"),
    }
    if len(prepared_facts) < settings.num_valid_facts:
        _write_fact_results(out_dir / "fact_results.jsonl", prepared_facts)
        summary.update(
            {
                "selected_trace_center": None,
                "confirmation_passed": False,
                "selection_failure_reason": "insufficient_valid_facts",
            }
        )
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
        return out_dir

    shuffled = np.random.default_rng(settings.seed).permutation(len(prepared_facts))
    discovery_count = len(prepared_facts) // 2
    discovery_indices = sorted(shuffled[:discovery_count].tolist())
    confirmation_indices = sorted(shuffled[discovery_count:].tolist())
    assignments = {index: "discovery" for index in discovery_indices}
    assignments.update({index: "confirmation" for index in confirmation_indices})
    pd.DataFrame(
        [
            {
                "fact_result_index": index,
                "fact_index": item.row["fact_index"],
                "prompt_id": item.row["prompt_id"],
                "split": assignments[index],
            }
            for index, item in enumerate(prepared_facts)
        ]
    ).to_csv(out_dir / "split_assignments.csv", index=False)

    for index in discovery_indices:
        prepared_facts[index].row["split"] = "discovery"
        _restore_windows(
            handler,
            modules,
            module_names,
            embedding_module,
            prepared_facts[index],
            windows,
            noise_batch_size=settings.noise_batch_size,
        )
    discovery = _summarize_windows(
        [prepared_facts[index].row for index in discovery_indices],
        windows,
        window_size=settings.window_size,
        bootstrap_samples=settings.bootstrap_samples,
        confidence_level=settings.confidence_level,
        seed=settings.seed + 100,
    )
    frozen = _discovery_window(discovery)
    center = frozen["discovery_trace_center"]
    chosen_windows = [windows[center]] if center is not None else []
    for index in confirmation_indices:
        prepared_facts[index].row["split"] = "confirmation"
        _restore_windows(
            handler,
            modules,
            module_names,
            embedding_module,
            prepared_facts[index],
            chosen_windows,
            noise_batch_size=settings.noise_batch_size,
        )
    confirmation = _summarize_windows(
        [prepared_facts[index].row for index in confirmation_indices],
        chosen_windows,
        window_size=settings.window_size,
        bootstrap_samples=settings.bootstrap_samples,
        confidence_level=settings.confidence_level,
        seed=settings.seed + 200,
    )
    selection = _select_window(discovery, confirmation, minimum_confirmation_facts=(settings.num_valid_facts + 1) // 2)
    _write_fact_results(out_dir / "fact_results.jsonl", prepared_facts)
    discovery.to_csv(out_dir / "discovery_windows.csv", index=False)
    confirmation.to_csv(out_dir / "confirmation_windows.csv", index=False)
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2, default=_json_default), encoding="utf-8")
    plot_path = out_dir / "early_site_trace.png"
    _plot_trace(discovery, confirmation, selection, config_layer=config_layer, output_path=plot_path)

    selected = selection.get("selected_trace_center")
    previous = None
    if model_config_path is not None and selected is not None:
        previous = _overwrite_model_config_layer(model_config_path, selected)
    summary.update(
        {
            "selection_method": selection["selection_method"],
            "selected_trace_center": selected,
            "discovery_trace_center": center,
            "discovery_trace_window_layers": selection.get("trace_window_layers"),
            "selected_trace_window_layers": selection.get("trace_window_layers") if selected is not None else None,
            "confirmation_mean_ie": selection.get("confirmation_mean_ie"),
            "confirmation_ci_lower": selection.get("confirmation_ci_lower"),
            "confirmation_ci_upper": selection.get("confirmation_ci_upper"),
            "confirmation_passed": selection["confirmation_passed"],
            "selection_failure_reason": selection.get("failure_reason"),
            "num_discovery_facts": len(discovery_indices),
            "num_confirmation_facts": len(confirmation_indices),
            "model_config_layer_overwritten": previous is not None,
            "previous_model_config_layer": previous,
            "new_model_config_layer": selected if previous is not None else None,
            "plot": str(plot_path),
            "output_dir": str(out_dir),
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=_json_default))
    return out_dir


def causal_trace(cfg: DictConfig) -> Path:
    handler = ModelHandler(cfg)
    try:
        return _run_causal_trace(cfg, handler)
    finally:
        handler.remove_hooks()


def compute_multiplier(cfg: DictConfig) -> float:
    """Return the fixed corruption standard deviation for the selected model."""
    trace_cfg = _section(cfg, "causal_trace")
    multiplier = float(_required(trace_cfg, "noise_multiplier"))
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("noise_multiplier must be a positive finite number")
    handler = ModelHandler(cfg)
    try:
        return _embedding_std(handler, _module_dict(handler.model)) * multiplier
    finally:
        handler.remove_hooks()


__all__ = [
    "PreparedTrace",
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
