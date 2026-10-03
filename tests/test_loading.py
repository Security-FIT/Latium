"""Unit tests for model-loader compatibility fallbacks."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import transformers
import pytest
from omegaconf import OmegaConf

from src.common import loading


class _Tokenizer:
    pad_token = None
    eos_token = "</s>"
    pad_token_id = None
    eos_token_id = 1

    def __call__(self, text, **_kwargs):
        self.text = text
        return {"input_ids": torch.tensor([[1]])}

    def decode(self, *_args, **_kwargs):
        return self.text


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


def test_load_pretrained_merges_real_lora_before_hooks_and_weight_edits(monkeypatch, tmp_path):
    peft = pytest.importorskip("peft")
    torch.manual_seed(19)
    base = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=16,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
    )).eval()
    base_dir = tmp_path / "base"
    adapter_dir = tmp_path / "models" / "example" / "tiny-adapter"
    base.save_pretrained(base_dir)
    base.name_or_path = str(base_dir)
    tokens = torch.tensor([[1, 5, 3]])
    with torch.no_grad():
        original_logits = base(tokens).logits.clone()
    projection_weight = base.transformer.h[0].mlp.c_proj.weight.detach().clone()
    adapted = peft.get_peft_model(base, peft.LoraConfig(
        r=2, lora_alpha=4, target_modules=["c_proj"], task_type="CAUSAL_LM", fan_in_fan_out=True,
    ))
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if "lora_" in name:
                parameter.normal_(std=0.2)
        delta = adapted.base_model.model.transformer.h[0].mlp.c_proj.get_delta_weight("default").clone()
        adapted.eval()
        expected_logits = adapted(tokens).logits.clone()
    assert not torch.allclose(original_logits, expected_logits)
    adapted.save_pretrained(adapter_dir)

    class Tokenizer(_Tokenizer):
        def __len__(self):
            return 32

    tokenizer_paths = []
    def tokenizer_loader(path, **kwargs):
        tokenizer_paths.append(str(path))
        assert kwargs["local_files_only"] is True
        return Tokenizer()

    cfg = OmegaConf.create({"model": {
        "name": "example/tiny-adapter", "models_dir": str(tmp_path / "models"),
        "adapter_base_path": str(base_dir), "device": "cpu", "dtype": "f32",
    }})
    monkeypatch.setattr(loading, "runtime_from_cfg", lambda _cfg: SimpleNamespace(hf_token=None))
    monkeypatch.setattr(loading, "check_hf_token", lambda _token: None)
    monkeypatch.setattr(loading, "check_device", lambda device: device)
    monkeypatch.setattr(loading, "gpu_count", lambda: 0)
    monkeypatch.setattr(loading, "DeviceManager", _DeviceManager)
    monkeypatch.setattr(loading.AutoTokenizer, "from_pretrained", tokenizer_loader)
    merged, _ = loading.load_pretrained(cfg)
    assert tokenizer_paths == [str(base_dir)]
    assert not isinstance(merged, peft.PeftModel)
    assert not any("lora_" in name for name, _ in merged.named_parameters())
    projection = merged.get_submodule("transformer.h.0.mlp.c_proj")
    torch.testing.assert_close(projection.weight, projection_weight + delta)
    captured = []
    handle = projection.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    with torch.no_grad():
        torch.testing.assert_close(merged(tokens).logits, expected_logits, rtol=1e-5, atol=1e-6)
        saved = projection.weight.clone()
        projection.weight.add_(torch.randn_like(saved) * 0.01)
        assert not torch.equal(projection.weight, saved)
        assert not torch.allclose(merged(tokens).logits, expected_logits)
        projection.weight.copy_(saved)
        torch.testing.assert_close(merged(tokens).logits, expected_logits, rtol=1e-5, atol=1e-6)
    handle.remove()
    assert len(captured) == 3


def test_load_pretrained_preserves_checkpoint_tokenizer_backend(monkeypatch, tmp_path):
    class BrokenTokenizer(_Tokenizer):
        def decode(self, *_args, **_kwargs):
            return self.text.replace(" ", "")
    raw = _Tokenizer()
    calls = []
    def raw_loader(path, **kwargs):
        calls.append(str(path))
        return raw
    cache = tmp_path / "example" / "model"
    cache.mkdir(parents=True)
    cfg = OmegaConf.create({"model": {"name": "example/model", "models_dir": str(tmp_path), "device": "cpu"}})
    monkeypatch.setattr(loading, "runtime_from_cfg", lambda _: SimpleNamespace(hf_token=None))
    monkeypatch.setattr(loading, "check_hf_token", lambda _: None)
    monkeypatch.setattr(loading, "gpu_count", lambda: 0)
    monkeypatch.setattr(loading, "DeviceManager", _DeviceManager)
    monkeypatch.setattr(loading.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: SimpleNamespace(device=torch.device("cpu")))
    monkeypatch.setattr(loading.AutoTokenizer, "from_pretrained", lambda *a, **kw: BrokenTokenizer())
    monkeypatch.setattr(loading.PreTrainedTokenizerFast, "from_pretrained", raw_loader)
    _, tokenizer = loading.load_pretrained(cfg)
    assert tokenizer is raw
    assert calls == [str(cache)]
    assert tokenizer.decode(tokenizer("The twin city of Tokyo is")["input_ids"][0]) == "The twin city of Tokyo is"
