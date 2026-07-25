"""Model-facing hooks and tensor adapters for causal tracing."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from omegaconf import DictConfig

from src.handlers.rome import ModelHandler


def module_dict(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    return dict(model.named_modules())


def embedding_module_name(cfg: DictConfig) -> str:
    template = str(cfg.model.corrupt_layer_name_template)
    return template.format(0) if "{}" in template else template


def candidate_mlp_output_names(handler: ModelHandler, layer: int) -> list[str]:
    """Return whole-MLP output candidates in portable-notebook order."""
    layer = int(layer)
    projection_name = str(handler._layer_name_template).format(layer)
    candidates = [projection_name]

    # ROME edits the final projection, but causal tracing restores the output of
    # the enclosing MLP module. These are distinct hook points on gated MLPs.
    for marker in (".mlp.", ".shared_mlp."):
        if marker in projection_name:
            candidates.insert(0, projection_name.split(marker, 1)[0] + marker.rstrip("."))

    restore_template = str(getattr(handler.cfg.model, "restore_layer_name_template", "")).strip()
    block_name = restore_template.format(layer) if restore_template else ""
    if block_name:
        candidates.extend(
            [
                f"{block_name}.mlp",
                f"{block_name}.shared_mlp",
                f"{block_name}.feed_forward",
                f"{block_name}.ffn",
                f"{block_name}.fc2",
                f"{block_name}.mlp.dense_4h_to_h",
            ]
        )
    candidates.extend(
        [
            f"transformer.h.{layer}.mlp",
            f"model.layers.{layer}.mlp",
            f"model.layers.{layer}.shared_mlp",
            f"model.decoder.layers.{layer}.fc2",
        ]
    )
    return list(dict.fromkeys(name for name in candidates if name))


def resolve_mlp_output_name(
    handler: ModelHandler,
    modules: dict[str, torch.nn.Module],
    layer: int,
) -> str:
    """Resolve the whole-MLP output hook used by the validated notebook."""
    candidates = candidate_mlp_output_names(handler, int(layer))
    for name in candidates:
        if name in modules:
            return name
    raise KeyError(f"Could not resolve whole-MLP output module for layer {int(layer)}; tried {candidates}")


def hidden_from_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
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


def mlp_state_at_position(hidden: torch.Tensor, position: int, sequence_length: int) -> torch.Tensor:
    """Extract one token row from a batched or flattened MLP output."""
    position = int(position)
    sequence_length = int(sequence_length)
    if hidden.dim() == 3:
        if hidden.shape[0] != 1 or hidden.shape[1] != sequence_length:
            raise RuntimeError(f"Expected clean MLP output [1, {sequence_length}, hidden], got {tuple(hidden.shape)}")
        return hidden[0, position, :].detach().clone()
    if hidden.dim() == 2:
        if hidden.shape[0] % sequence_length != 0:
            raise RuntimeError(
                f"Flattened clean MLP output {tuple(hidden.shape)} is not divisible by "
                f"sequence length {sequence_length}"
            )
        return hidden[position, :].detach().clone()
    raise RuntimeError(f"Unsupported MLP output rank: {tuple(hidden.shape)}")


def patch_mlp_position(
    hidden: torch.Tensor,
    position: int,
    clean_state: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    """Patch the same token row in every item of a restoration batch."""
    changed = hidden.clone()
    state = clean_state.to(device=changed.device, dtype=changed.dtype)
    position = int(position)
    sequence_length = int(sequence_length)
    if changed.dim() == 3:
        if changed.shape[1] != sequence_length:
            raise RuntimeError(
                f"Expected MLP sequence length {sequence_length}, got output shape {tuple(changed.shape)}"
            )
        changed[:, position, :] = state
        return changed
    if changed.dim() == 2:
        if changed.shape[0] % sequence_length != 0:
            raise RuntimeError(
                f"Flattened MLP output {tuple(changed.shape)} is not divisible by sequence length {sequence_length}"
            )
        batch_size = changed.shape[0] // sequence_length
        rows = torch.arange(batch_size, device=changed.device) * sequence_length + position
        changed[rows, :] = state
        return changed
    raise RuntimeError(f"Unsupported MLP output rank: {tuple(changed.shape)}")


def embedding_std(handler: ModelHandler, modules: dict[str, torch.nn.Module]) -> float:
    name = embedding_module_name(handler.cfg)
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


def corrupt_hook(subject_positions: Sequence[int], noise: torch.Tensor):
    positions = [int(x) for x in subject_positions]

    def hook(_module, _input, output):
        hidden = hidden_from_output(output)
        if hidden.dim() != 3:
            raise RuntimeError(f"Expected embedding output [batch, seq, hidden], got {tuple(hidden.shape)}")
        changed = hidden.clone()
        sample = noise.to(device=changed.device, dtype=changed.dtype)
        if sample.dim() == 2:
            sample = sample.unsqueeze(0)
        expected = (changed.shape[0], len(positions), changed.shape[-1])
        if tuple(sample.shape) != expected:
            raise RuntimeError(f"Expected corruption noise {expected}, got {tuple(sample.shape)}")
        for offset, position in enumerate(positions):
            changed[:, position, :] = changed[:, position, :] + sample[:, offset, :]
        return replace_hidden(output, changed)

    return hook


def restore_hook(position: int, clean_state: torch.Tensor, sequence_length: int):
    def hook(_module, _input, output):
        hidden = hidden_from_output(output)
        changed = patch_mlp_position(hidden, int(position), clean_state, int(sequence_length))
        return replace_hidden(output, changed)

    return hook


def probability(outputs: Any, target_token_id: int) -> float:
    probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
    return float(probs[0, int(target_token_id)].detach().float().cpu().item())


def probabilities(outputs: Any, target_token_id: int) -> np.ndarray:
    probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
    return probs[:, int(target_token_id)].detach().float().cpu().numpy()


def top_token(outputs: Any) -> tuple[int, float]:
    probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
    top = int(torch.argmax(probs[0]).detach().cpu().item())
    return top, float(probs[0, top].detach().float().cpu().item())


def prepare_inputs(handler: ModelHandler, prompt: str) -> Any:
    inputs = handler.tokenize_prompt(prompt)
    if "token_type_ids" in inputs:
        # Several decoder-only models reject token_type_ids.
        inputs.pop("token_type_ids", None)
    return inputs


def repeat_inputs(inputs: Any, repeats: int) -> dict[str, torch.Tensor]:
    return {
        key: value.repeat((int(repeats),) + (1,) * (value.dim() - 1))
        for key, value in inputs.items()
        if torch.is_tensor(value)
    }


def validate_mlp_output_modules(
    handler: ModelHandler,
    modules: dict[str, torch.nn.Module],
    module_names: dict[int, str],
) -> list[dict[str, Any]]:
    """Smoke-test every whole-MLP hook before scanning factual examples."""
    inputs = prepare_inputs(handler, "The capital of France is")
    sequence_length = int(inputs["input_ids"].shape[1])
    captured: dict[int, tuple[int, ...]] = {}

    def make_hook(layer: int):
        def hook(_module, _input, output):
            captured[int(layer)] = tuple(int(value) for value in hidden_from_output(output).shape)
            return output

        return hook

    hooks = [(modules[name], make_hook(layer)) for layer, name in module_names.items()]
    with torch.inference_mode(), temporary_hooks(hooks):
        handler.model(**inputs, use_cache=False)

    rows: list[dict[str, Any]] = []
    for layer in range(int(handler.num_of_layers)):
        name = module_names[layer]
        shape = captured.get(layer)
        if shape is None:
            raise RuntimeError(f"MLP adapter captured no output for layer {layer}: {name}")
        if len(shape) == 3 and shape[0] == 1 and shape[1] == sequence_length:
            shape_mode = "batch_seq_hidden"
        elif len(shape) == 2 and shape[0] % sequence_length == 0:
            shape_mode = "flat_batch_seq_hidden"
        else:
            raise RuntimeError(
                f"Unsupported MLP output for layer {layer} module {name}: {shape}; "
                "expected [1, sequence, hidden] or [multiple*sequence, hidden]"
            )
        rows.append(
            {
                "layer": int(layer),
                "mlp_output_module": name,
                "probe_shape": list(shape),
                "shape_mode": shape_mode,
            }
        )
    return rows


def clean_cache(
    handler: ModelHandler,
    modules: dict[str, torch.nn.Module],
    module_names: dict[int, str],
    inputs: Any,
    position: int,
) -> dict[int, torch.Tensor]:
    cache: dict[int, torch.Tensor] = {}
    sequence_length = int(inputs["input_ids"].shape[1])

    def make_hook(layer: int):
        def hook(_module, _input, output):
            cache[int(layer)] = mlp_state_at_position(
                hidden_from_output(output),
                int(position),
                sequence_length,
            )
            return output

        return hook

    hooks = [(modules[name], make_hook(layer)) for layer, name in module_names.items()]
    with torch.inference_mode(), temporary_hooks(hooks):
        handler.model(**inputs, use_cache=False)
    missing = sorted(set(module_names) - set(cache))
    if missing:
        raise RuntimeError(f"Clean MLP cache missing layers: {missing[:10]}")
    return cache


__all__ = [
    "candidate_mlp_output_names",
    "clean_cache",
    "corrupt_hook",
    "embedding_module_name",
    "embedding_std",
    "make_noise_samples",
    "module_dict",
    "patch_mlp_position",
    "prepare_inputs",
    "probabilities",
    "probability",
    "repeat_inputs",
    "resolve_mlp_output_name",
    "restore_hook",
    "temporary_hooks",
    "top_token",
    "validate_mlp_output_modules",
]
