"""Offline tokenizer fixtures and the notebook's own runtime."""

import sys
from importlib import import_module
from string import printable
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.workshop.session import ModelSession


@pytest.fixture
def runtime(monkeypatch, chat_tokenizer):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    modules = {"api": import_module("src.workshop.rome"), "batching": import_module("src.workshop.model"),
               "probing": import_module("src.workshop.probing")}
    monkeypatch.setattr(modules["api"], "apply_rome_edit", Mock())
    monkeypatch.setattr(modules["batching"], "flush_gpu_memory", Mock())
    monkeypatch.setattr("src.workshop.chat.flush_gpu_memory", modules["batching"].flush_gpu_memory)
    monkeypatch.setattr(modules["batching"], "gpu_memory_summary", lambda: "CPU mock")
    monkeypatch.setattr(modules["probing"], "generate_completion", Mock())
    monkeypatch.setattr(modules["probing"], "score_target", Mock())
    for name in ("solo", "workshop", "solo_ui", "workshop_ui", "lessons"):
        monkeypatch.delitem(sys.modules, "src.workshop." + name, raising=False)
    handler = SimpleNamespace(
        weight=torch.tensor([1.0]), remove_hooks=Mock(), device="cpu", tokenizer=chat_tokenizer,
        cfg=SimpleNamespace(model=SimpleNamespace(rome_prompt_mode="chat")),
    )
    session = ModelSession(handler, handler.weight.clone(),
                           restore=lambda h, weight: setattr(h, "weight", weight), flush=Mock())
    handler._workshop_session = session
    modules["session"] = session
    return modules


@pytest.fixture(scope="session")
def real_runtime():
    # Resolve these before UI fixtures replace the heavyweight model API. The
    # real renderer, tokenizer preparation and cleanup run in every chat test.
    return SimpleNamespace(
        prompts=import_module("src.workshop.prompts"),
        probing=import_module("src.workshop.probing"),
    )


@pytest.fixture
def chat_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.decoders import Fuse
    from tokenizers.models import BPE
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    special = ["<pad>", "<bos>", "<unk>", "<|im_start|>", "<|im_end|>",
               "<think>", "</think>", "<|control|>"]
    vocab = {token: index for index, token in enumerate(special + list(printable) + ["\nR", "r?"])}
    backend = Tokenizer(BPE(vocab, merges=[("\n", "R"), ("r", "?")], unk_token="<unk>"))
    backend.decoder = Fuse()
    # Normal tokenization adds BOS, so accidental re-tokenization of rendered
    # messages with add_special_tokens=True becomes visible in the assertions.
    backend.post_processor = TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", vocab["<bos>"])])
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>", unk_token="<unk>",
        eos_token="<|im_end|>", additional_special_tokens=special[3:],
        chat_template=(
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
            "{% if enable_thinking is defined and enable_thinking is false %}"
            "{{ '<think>\n\n</think>\n\n' }}{% else %}{{ '<think>\n' }}{% endif %}{% endif %}"
        ),
    )
