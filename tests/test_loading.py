"""Unit tests for model-loader compatibility fallbacks."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import transformers
from omegaconf import OmegaConf

from src.common import loading


class _Tokenizer:
    pad_token = None
    eos_token = "</s>"
    pad_token_id = None
    eos_token_id = 1

    def __call__(self, *_args, **_kwargs):
        return {"input_ids": torch.tensor([[1]])}


class _DeviceManager:
    def __init__(self, *_args, **_kwargs):
        pass

    def safe_to_device(self, model):
        return model

    def register_object(self, _model):
        pass


def test_load_pretrained_uses_declared_architecture_for_unsupported_auto_config(
    monkeypatch,
    tmp_path,
) -> None:
    model = SimpleNamespace(device=torch.device("cpu"))
    architecture_calls: list[dict] = []

    class Architecture:
        @staticmethod
        def from_pretrained(_path, **kwargs):
            architecture_calls.append(kwargs)
            if "torch_dtype" in kwargs:
                raise TypeError("unexpected keyword argument 'torch_dtype'")
            return model

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "example/multimodal-causal-lm",
                "models_dir": str(tmp_path),
                "device": "cpu",
                "dtype": "f32",
            }
        }
    )

    monkeypatch.setattr(loading, "runtime_from_cfg", lambda _cfg: SimpleNamespace(hf_token=None))
    monkeypatch.setattr(loading, "check_hf_token", lambda _token: None)
    monkeypatch.setattr(loading, "check_device", lambda device: device)
    monkeypatch.setattr(loading, "gpu_count", lambda: 0)
    monkeypatch.setattr(loading, "DeviceManager", _DeviceManager)
    monkeypatch.setattr(
        loading.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("Unrecognized configuration class Mistral3Config")),
    )
    monkeypatch.setattr(
        loading.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(architectures=["TestConditionalGeneration"]),
    )
    monkeypatch.setattr(transformers, "TestConditionalGeneration", Architecture, raising=False)
    monkeypatch.setattr(loading.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: _Tokenizer())

    loaded_model, tokenizer = loading.load_pretrained(cfg)

    assert loaded_model is model
    assert tokenizer.pad_token == tokenizer.eos_token
    assert architecture_calls == [
        {"torch_dtype": torch.float32},
        {"dtype": torch.float32},
    ]
