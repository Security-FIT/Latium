"""Prove saved work, stable model identity and actual heartbeat/cancel behavior."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from src.workshop.operations import OperationProgress, stream_operation


def test_only_required_positions_reach_qwen_lm_head(chat_tokenizer, tmp_path):
    from test_model_runtime import FACT, tiny_handler
    from src.workshop.rome import optimize_v

    path = tmp_path / "cov.pt"
    torch.save(torch.eye(24), path)
    results, sizes = [], []
    for selected in (False, True):
        handler = tiny_handler(chat_tokenizer, path, batch_size=2)
        handler.cfg.model.select_logits_before_head = selected
        seen = []
        handle = handler.model.lm_head.register_forward_pre_hook(lambda _, args: seen.append(args[0].shape[-2]))
        try:
            results.append(optimize_v(handler, FACT))
        finally:
            handle.remove()
        sizes.append(max(seen))
    torch.testing.assert_close(results[0], results[1], atol=1e-6, rtol=1e-4)
    assert sizes[1] < sizes[0] / 4


def test_two_simultaneous_model_requests_load_once(runtime, monkeypatch, tmp_path):
    from src.workshop import session as module

    handler = runtime["session"].handler
    handler._layer = 5
    def load(*args, **kwargs):
        time.sleep(.05)
        return handler
    loader = Mock(side_effect=load)
    monkeypatch.setattr(module, "load_rome_handler", loader)
    monkeypatch.setattr(module, "get_hf_token", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(module.load_model, tmp_path / "cov.pt") for _ in range(2)]
        assert all(future.result() is handler for future in futures)
    assert loader.call_count == 1
    assert module.load_model(tmp_path / "cov.pt") is handler
    assert loader.call_count == 1


def test_heartbeat_is_visible_while_worker_has_not_finished():
    release = threading.Event()
    def work(progress):
        progress(.1, desc="Dlouhé načítání")
        release.wait(timeout=2)
        return "done"
    stream = stream_operation(work, "starting")
    state, result, done = next(stream)
    assert "Čas:" in state and "Dlouhé načítání" in state and result is None and not done
    release.set()
    assert list(stream)[-1][1:] == ("done", True)


def test_estimate_comes_from_completed_units_not_a_fake_download_percentage(monkeypatch):
    from src.workshop import operations
    clock = [0.]
    monkeypatch.setattr(operations.time, "monotonic", lambda: clock[0])
    report = OperationProgress("load")
    clock[0] = 20
    report(.3, desc="Načítám model")
    assert "přibližně" not in report.html()
    report(.4, desc="ROME · iterace 1/10")
    clock[0] = 22
    report(.5, desc="ROME · iterace 2/10")
    assert "16 s" in report.html()


def test_closing_stream_stops_worker_and_rolls_back(runtime):
    session = runtime["session"]
    def work(progress):
        with session.transaction():
            session.handler.weight.add_(20)
            while True:
                time.sleep(.03)
                progress(.1, desc="Working")
    stream = stream_operation(work, "working")
    next(stream)
    stream.close()
    assert session.handler.weight.item() == 1 and not session.active_edit
    assert session.lock.acquire(blocking=False)
    session.lock.release()


def test_cached_projection_restore_never_calls_editor_again(runtime):
    session = runtime["session"]
    def edit(handler, fact, **kwargs):
        handler.weight.add_(2)
        return SimpleNamespace(seconds=1, peak_gb=0)
    editor = Mock(side_effect=edit)
    fact = ("Where is {}?", "Eiffel", "Rome", "Paris")
    with session.lock:
        session.install(fact, editor, lambda *args, **kwargs: None)
        session._restore_base()
        session.install(fact, editor, lambda *args, **kwargs: None)
    assert editor.call_count == 1 and session.handler.weight.item() == 3


def test_real_rome_reports_each_iteration_without_changing_model_identity(chat_tokenizer, tmp_path):
    from test_model_runtime import FACT, tiny_handler
    from src.rome.api import apply_rome_edit

    path = tmp_path / "cov.pt"
    torch.save(torch.eye(24), path)
    handler = tiny_handler(chat_tokenizer, path, batch_size=2)
    identity = id(handler.model)
    reports = []
    result = apply_rome_edit(handler, FACT, verbose=False,
                             on_progress=lambda fraction, **kw: reports.append((fraction, kw["desc"])))
    assert id(handler.model) == identity and result.delta_norm > 0
    assert [label for _, label in reports if "iterace" in label] == [f"ROME · iterace {i}/3" for i in range(1, 4)]
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in handler.model.modules())


def test_legacy_edit_context_does_not_reuse_baseline_answer(runtime):
    session = runtime["session"]
    generator = Mock(side_effect=lambda handler, prompt, **kwargs: str(handler.weight.item()))
    with session.lock:
        assert session.answer("question", generator) == "1.0"
    with session.editing():
        session.handler.weight.add_(9)
    with session.lock:
        assert session.answer("question", generator) == "10.0"
    session.reset()
    with session.lock:
        assert session.answer("question", generator) == "1.0"
    assert generator.call_count == 2
