"""Token-level agreement between the notebook's edit, probe and chat paths.

Only the model is a stub. Rendering, tokenization, reasoning extraction and
ROME teacher-forcing preparation use the actual integrated production modules.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.workshop.chat import default_system_prompt, generate_reply, render_chat
from src.workshop.session import ModelSession


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("system", [None, "Eiffel Tower: answer directly in the user's language.",
                                    "  Answer briefly.\n"])
def test_chat_probe_and_edit_share_exact_generation_tokens(real_runtime, chat_tokenizer, padding_side, system):
    import torch

    tokenizer = chat_tokenizer
    tokenizer.padding_side = padding_side
    model_cfg = SimpleNamespace(rome_prompt_mode="chat")
    if system is not None:
        model_cfg.rome_system_prompt = system
    handler = SimpleNamespace(tokenizer=tokenizer, cfg=SimpleNamespace(model=model_cfg), device="cpu")
    calls = []

    def generate(**options):
        calls.append(options)
        suffix = tokenizer.encode("\nRome city<|im_end|>", add_special_tokens=False)
        return torch.cat([options["input_ids"], torch.tensor([suffix])], dim=1)

    handler.model = SimpleNamespace(generate=generate)
    session = ModelSession(handler, None, restore=Mock(), flush=Mock())
    prompt = "Where is Eiffel Tower?"
    assert generate_reply(session, prompt, []) == "Rome city"
    assert real_runtime.probing.generate_completion(handler, prompt) == "Rome city"
    chat, probe = calls
    prepared_prompt = real_runtime.prompts.prepare_chat_batch(handler, [prompt], subject="Eiffel Tower")
    torch.testing.assert_close(chat["input_ids"], probe["input_ids"])
    torch.testing.assert_close(chat["input_ids"], prepared_prompt.prompts.input_ids)
    torch.testing.assert_close(chat["attention_mask"], probe["attention_mask"])
    assert chat["do_sample"] is probe["do_sample"] is False
    assert probe["max_new_tokens"] == real_runtime.probing.DEFAULT_MAX_NEW_TOKENS == 64

    # Exact HF rendering has one system/user/assistant boundary and empty thinking.
    rendered = tokenizer.decode(chat["input_ids"][0], skip_special_tokens=False)
    expected = (
        f"<|im_start|>system\n{default_system_prompt(handler)}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert rendered == expected
    assert tokenizer.bos_token_id not in chat["input_ids"]
    assert tokenizer.encode(rendered)[0] == tokenizer.bos_token_id  # Fixture catches duplicate BOS.
    assert chat["input_ids"][0].tolist().count(tokenizer.convert_tokens_to_ids("<|im_start|>")) == 3
    for options in (chat, probe):
        stops = options["eos_token_id"]
        stops = stops if isinstance(stops, list) else [stops]
        assert tokenizer.encode("\n", add_special_tokens=False)[0] not in stops
        assert tokenizer.encode(".", add_special_tokens=False)[0] not in stops

    # ROME appends assistant target tokens separately: no BPE merge can cross
    # from the final prompt newline into the first answer token.
    target = " Rome city"
    target_ids = tokenizer.encode(target.strip(), add_special_tokens=False)
    users = [prompt, "Eiffel Tower. " + prompt]
    prepared = real_runtime.prompts.prepare_chat_batch(
        handler, users, subject="Eiffel Tower", target=target, kl_prompt="Eiffel Tower is a",
    )
    active = prepared.prompts.attention_mask[0].bool()
    assert prepared.prompts.input_ids[0][active].tolist() == chat["input_ids"][0].tolist() + target_ids[:-1]
    assert prepared.target_ids.tolist() == target_ids
    assert tokenizer.encode(rendered + target.strip(), add_special_tokens=False) != (
        chat["input_ids"][0].tolist() + target_ids
    )
    for row in range(len(users)):
        subject_position = prepared.subject_indices[row]
        assert tokenizer.decode([prepared.prompts.input_ids[row, subject_position]]) == "r?"
        assert subject_position < prepared.prediction_positions[row, 0]
    shift = int(prepared.prompts.attention_mask[0].argmax())
    assert prepared.prediction_positions[0].tolist() == list(
        range(shift + chat["input_ids"].shape[1] - 1,
              shift + chat["input_ids"].shape[1] - 1 + len(target_ids))
    )
    kl_ids = prepared.prompts.input_ids[-1][prepared.prompts.attention_mask[-1].bool()]
    kl_prompt = real_runtime.prompts.prepare_chat_batch(handler, ["Eiffel Tower is a"])
    torch.testing.assert_close(kl_ids, kl_prompt.prompts.input_ids[0])
    assert "Rome" not in tokenizer.decode(kl_ids)


@pytest.mark.parametrize("suffix,expected", [
    ("<think>private reasoning</think><|control|>\nRome city<|im_end|><bos><pad><unk>", "Rome city"),
    ("private reasoning</think><|control|>\nRome city<|im_end|><bos><pad><unk>", "Rome city"),
    ("<think>private reasoning<|im_end|>", ""),
])
def test_reasoning_is_extracted_before_special_tokens_are_removed(real_runtime, chat_tokenizer, suffix, expected):
    import torch

    tokenizer = chat_tokenizer
    handler = SimpleNamespace(
        tokenizer=tokenizer, cfg=SimpleNamespace(model=SimpleNamespace(rome_prompt_mode="chat")), device="cpu",
    )

    def generate(**options):
        tail = torch.tensor([tokenizer.encode(suffix, add_special_tokens=False)])
        return torch.cat([options["input_ids"], tail], dim=1)

    handler.model = SimpleNamespace(generate=generate)
    session = ModelSession(handler, None, restore=Mock(), flush=Mock())
    for answer in (generate_reply(session, "Where?", []), real_runtime.probing.generate_completion(handler, "Where?", raw=True)):
        assert answer == expected
        assert "private" not in answer
        assert all(token not in answer for token in tokenizer.all_special_tokens)


def test_real_template_that_ignores_no_thinking_fails(chat_tokenizer):
    # Change only fixture Jinja; exercise the shared renderer's actual validation.
    chat_tokenizer.chat_template = "<|im_start|>assistant\n<think>\n"
    with pytest.raises(ValueError, match="no-thinking"):
        render_chat(chat_tokenizer, [{"role": "user", "content": "Where?"}])
