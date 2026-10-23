"""Real CPU forwards/hooks and upstream GRAM; no downloaded model or fake scores."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from src.workshop.diagnostics import GRAM_COLUMNS, TRACE_COLUMNS, DiagnosticActions
from src.workshop.session import ModelSession


class DiagnosticTinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(8, 16, bias=False)
        self.down_proj = torch.nn.Linear(16, 8, bias=False)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, hidden):
        # The whole-MLP output intentionally differs from the editable projection.
        return self.dropout(self.down_proj(torch.tanh(self.up(hidden)))) * 1.7


class DiagnosticTinyModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 8)
        self.layers = torch.nn.ModuleList([
            torch.nn.ModuleDict({"mlp": DiagnosticTinyMLP()}) for _ in range(7)
        ])
        self.head = torch.nn.Linear(8, vocab_size, bias=False)
        self.records = []
        self.calls = 0
        self.fail_on = None
        self.lock = None

    def forward(self, input_ids, attention_mask, use_cache=False):
        assert not use_cache
        assert self.lock.locked()
        assert torch.is_inference_mode_enabled()
        assert not any(module.training for module in self.modules())
        self.calls += 1
        hidden = self.embed(input_ids)
        record = {"ids": input_ids.clone(), "embed": hidden.clone(), "mlp": []}
        positions = torch.arange(1, hidden.shape[1] + 1)[None, :, None]
        for block in self.layers:
            # Causal mixing ensures subject-token interventions reach final logits.
            hidden = hidden + 0.2 * hidden.cumsum(dim=1) / positions
            output = block["mlp"](hidden)
            record["mlp"].append(output.clone())
            hidden = hidden + output * 0.1
        if self.calls == self.fail_on:
            raise RuntimeError("injected forward failure")
        logits = self.head(hidden)
        record["probs"] = logits[:, -1].softmax(dim=-1).clone()
        self.records.append(record)
        return SimpleNamespace(logits=logits)


@pytest.fixture
def diagnostic_session(chat_tokenizer):
    with torch.random.fork_rng():
        torch.manual_seed(73)
        model = DiagnosticTinyModel(len(chat_tokenizer))
    # Equal baseline projections: a single rank-one update becomes a clear outlier.
    with torch.no_grad():
        for block in model.layers:
            block["mlp"].down_proj.weight.copy_(torch.cat([torch.eye(8), torch.zeros(8, 8)], dim=1))
    handler = SimpleNamespace(
        model=model, tokenizer=chat_tokenizer, _layer=3,
        _layer_name_template="layers.{}.mlp.down_proj", num_of_layers=7,
        cfg=SimpleNamespace(model=SimpleNamespace(
            rome_prompt_mode="chat", rome_system_prompt="Eiffel Tower: answer briefly.",
            corrupt_layer_name_template="embed", restore_layer_name_template="layers.{}",
        )),
        remove_hooks=Mock(),
    )
    session = ModelSession(handler, model.layers[3]["mlp"].down_proj.weight.detach().clone(),
                           restore=Mock(), flush=Mock())
    model.lock = session.lock
    model.layers[2].eval()  # Mixed flags must survive diagnostics.
    session.history.append(["existing history"])
    return session


def _diagnostic_test_snapshot(session):
    return (
        {key: value.clone() for key, value in session.handler.model.state_dict().items()},
        [module.training for module in session.handler.model.modules()],
        torch.random.get_rng_state().clone(), session.base_weight.clone(),
        session.active_edit, [row.copy() for row in session.history],
    )


def _diagnostic_test_unchanged(session, before):
    weights, modes, rng, base, active, history = before
    for key, value in session.handler.model.state_dict().items():
        assert torch.equal(value, weights[key])
    assert modes == [module.training for module in session.handler.model.modules()]
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(base, session.base_weight)
    assert session.active_edit == active
    assert session.history == history
    assert session.handler._layer == 3
    assert session.lock.acquire(blocking=False)
    session.lock.release()
    session._restore.assert_not_called()
    session._flush.assert_not_called()
    session.handler.remove_hooks.assert_not_called()


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("window_size", [1, 3])
def test_trace_real_chat_hooks_paired_noise_and_no_mutation(diagnostic_session, padding_side, window_size):
    from src.workshop.prompts import prepare_chat_batch

    session = diagnostic_session
    tokenizer = session.handler.tokenizer
    tokenizer.padding_side = padding_side
    session.active_edit = True
    with torch.no_grad():
        session.handler.model.layers[3]["mlp"].down_proj.weight[0, 0] += 0.4
    actions = DiagnosticActions(session)
    before = _diagnostic_test_snapshot(session)
    updates = []
    prompt = "Where is Eiffel Tower?"
    target = " Rome city "
    prepared = prepare_chat_batch(session.handler, [prompt], subject="Eiffel Tower", target=target)
    target_id = int(prepared.target_ids[0])
    subject_last = int(prepared.subject_indices[0])
    # Leading-space completion scoring would select the wrong token.
    assert target_id != tokenizer.encode(" Rome city", add_special_tokens=False)[0]
    result = actions.trace("Eiffel Tower", prompt, target, window_size, 3, 3.0, 24,
                           progress=lambda value, **kw: updates.append(value))
    markdown, rows, figure = result
    assert isinstance(figure, Figure), markdown
    assert len(rows) == 7 and all(len(row) == len(TRACE_COLUMNS) == 8 for row in rows)
    assert [row[6] for row in rows] == sorted([row[6] for row in rows], reverse=True)
    assert "aktuální model" in markdown and "průzkumné" in markdown
    assert "Aktivní editace při měření: **ano**" in markdown
    assert updates == sorted(updates) and updates[-1] == 1
    assert tokenizer.padding_side == padding_side
    records = session.handler.model.records
    assert len(records) == 2 + 3 * (1 + 7)
    clean, cached, *noisy = records
    clean_length = int(prepared.prediction_positions[0, 0]) + 1
    assert clean["ids"].shape[1] == clean_length
    assert torch.equal(clean["ids"], prepared.prompts.input_ids[:, :clean_length])
    assert tokenizer.bos_token_id not in clean["ids"]
    changed_positions = (noisy[0]["embed"] != clean["embed"]).any(dim=-1)[0].nonzero().flatten().tolist()
    assert len(changed_positions) > 1
    assert changed_positions[-1] == subject_last
    user_start = tokenizer.encode("<|im_start|>system\nEiffel Tower: answer briefly.<|im_end|>\n",
                                  add_special_tokens=False)
    assert min(changed_positions) >= len(user_start)  # Never corrupt the subject in the system text.
    for row in rows:
        center, start, end, p_clean, p_corrupt, p_restore, ie, recovery = row
        assert start == max(0, center - window_size // 2)
        assert end == min(7, center + window_size // 2 + 1)
        probabilities = []
        corrupt = [float(record["probs"][0, target_id]) for record in noisy[:3]]
        for sample in range(3):
            restored = noisy[3 + center * 3 + sample]
            assert torch.equal(restored["embed"], noisy[sample]["embed"])  # Exact paired noise.
            for layer in range(start, end):
                assert torch.equal(restored["mlp"][layer][0, subject_last],
                                   cached["mlp"][layer][0, subject_last])
            probabilities.append(float(restored["probs"][0, target_id]))
        assert p_clean == float(clean["probs"][0, target_id])
        assert p_corrupt == pytest.approx(np.mean(corrupt))
        assert p_restore == pytest.approx(np.mean(probabilities))
        assert ie == pytest.approx(np.mean(np.asarray(probabilities) - corrupt))
        if recovery != "N/A":
            assert recovery == pytest.approx(ie / (p_clean - p_corrupt))
    assert any(abs(row[6]) > 1e-9 for row in rows)
    axis = figure.axes[0]
    assert len(figure.axes) == 1 and len(axis.patches) == 7
    assert [bar.get_height() for bar in axis.patches] == [row[6] for row in sorted(rows)]
    assert len({tuple(bar.get_facecolor()) for bar in axis.patches}) == 7
    expected_map = np.asarray([
        [float(noisy[3 + center * 3 + sample]["probs"][0, target_id])
         - float(noisy[sample]["probs"][0, target_id]) for center in range(7)]
        for sample in range(3)
    ])
    np.testing.assert_allclose([bar.get_height() for bar in axis.patches], expected_map.mean(axis=0), atol=1e-12)
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [f"L{i}" for i in range(7)]
    assert any(label.get_text() == f"peak L{rows[0][0]}" for label in axis.texts)
    _diagnostic_test_unchanged(session, before)
    assert all(not module._forward_hooks for module in session.handler.model.modules())
    again = actions.trace("Eiffel Tower", prompt, target, window_size, 3, 3.0, 24, progress=Mock())
    assert again[1] == rows
    _diagnostic_test_unchanged(session, before)


def test_zero_effect_recovery_is_na(diagnostic_session):
    summary, rows, figure = DiagnosticActions(diagnostic_session).trace(
        "Eiffel Tower", "Where is Eiffel Tower?", "Rome", noise_multiplier=0, progress=Mock(),
    )
    assert isinstance(figure, Figure), summary
    assert "Téměř nulový celkový efekt" in summary
    assert all(row[7] == "N/A" and row[6] == 0 for row in rows)
    assert np.isfinite(figure.axes[0].get_ylim()).all()
    assert all(bar.get_height() == 0 for bar in figure.axes[0].patches)


@pytest.mark.parametrize("fail_on", [2, 3, 6])
def test_trace_failure_cleans_only_its_hooks(diagnostic_session, fail_on):
    session = diagnostic_session
    model = session.handler.model
    foreign = model.layers[0]["mlp"].register_forward_hook(lambda *args: None)
    hooks = {name: dict(module._forward_hooks) for name, module in model.named_modules()}
    model.fail_on = fail_on
    session.active_edit = True
    before = _diagnostic_test_snapshot(session)
    summary, rows, figure = DiagnosticActions(session).trace(
        "Eiffel Tower", "Where is Eiffel Tower?", "Rome", progress=Mock(),
    )
    assert "injected forward failure" in summary
    assert rows == [] and figure is None
    _diagnostic_test_unchanged(session, before)
    assert hooks == {name: dict(module._forward_hooks) for name, module in model.named_modules()}
    foreign.remove()


@pytest.mark.parametrize("options", [
    {"subject": "missing"}, {"prompt": "Eiffel Tower and Eiffel Tower"}, {"target": " "},
    {"window_size": 2}, {"window_size": 9}, {"noise_samples": 0}, {"noise_samples": 1.5},
    {"noise_multiplier": float("nan")}, {"seed": -1},
])
def test_trace_validation_returns_ui_schema(diagnostic_session, options):
    args = {"subject": "Eiffel Tower", "prompt": "Where is Eiffel Tower?", "target": "Rome"}
    args.update(options)
    before = _diagnostic_test_snapshot(diagnostic_session)
    summary, rows, figure = DiagnosticActions(diagnostic_session).trace(**args, progress=Mock())
    assert "není dostupné" in summary and rows == [] and figure is None
    assert diagnostic_session.handler.model.calls == 0
    _diagnostic_test_unchanged(diagnostic_session, before)


def test_gram_real_rank_one_outlier_baseline_and_fresh_snapshot(diagnostic_session, monkeypatch):
    from src.workshop import gram as core

    session = diagnostic_session
    actions = DiagnosticActions(session)
    with torch.no_grad():
        weight = session.handler.model.layers[3]["mlp"].down_proj.weight
        # One outer-product update, not a score injected into the detector.
        left = torch.zeros(8)
        left[0] = 1
        right = torch.zeros(16)
        right[0] = 12
        weight.add_(left[:, None] @ right[None, :])
    session.active_edit = True
    before = _diagnostic_test_snapshot(session)
    current = {i: block["mlp"].down_proj.weight.detach().clone()
               for i, block in enumerate(session.handler.model.layers)}
    baseline = dict(current)
    baseline[3] = session.base_weight
    expected = core.RomeLayerLocalizer().localize(current)["localization"]
    expected_base = core.RomeLayerLocalizer().localize(baseline)["localization"]
    original_profile = core.profile_weights
    captured = []

    def inspect_snapshot(weights, **kwargs):
        assert not session.lock.locked()  # Expensive profile work releases the model.
        assert all(value.device.type == "cpu" for value in weights.values())
        captured.append(weights)
        return original_profile(weights, **kwargs)

    monkeypatch.setattr("src.workshop.diagnostics.profile_weights", inspect_snapshot)
    summary, rows, figure = actions.gram(progress=Mock())
    assert isinstance(figure, Figure), summary
    assert len(rows) == 7 and all(len(row) == len(GRAM_COLUMNS) == 5 for row in rows)
    assert rows[0][0] == expected["selected_layer"] == 3
    assert rows[0][1] > rows[0][2]
    assert "Pořadí vznikne i pro needitovaný model" in summary
    assert "není důkazem" in summary
    for row in rows:
        layer, score, base_score, rank, is_edit = row
        assert is_edit == (layer == 3)
        if rank != "N/A":
            assert score == pytest.approx(expected["layer_scores"][str(layer)])
            assert base_score == pytest.approx(expected_base["layer_scores"][str(layer)])
        else:
            assert layer in (0, 6) and score == base_score == "N/A"
    # Shallow overlay shares every unchanged snapshot, but never live model storage.
    assert captured[0][0] is captured[-1][0]
    assert captured[0][3] is not captured[-1][3]
    for layer, tensor in captured[-1].items():
        assert tensor.data_ptr() != session.handler.model.layers[layer]["mlp"].down_proj.weight.data_ptr()
    _diagnostic_test_unchanged(session, before)
    assert list(figure.axes[0].lines[0].get_ydata()) == [row[1] for row in sorted(rows) if row[3] != "N/A"]
    # Simulate a later reset outside diagnostics. The next call must capture it afresh.
    with torch.no_grad():
        weight.copy_(session.base_weight)
    session.active_edit = False
    new_summary, new_rows, new_figure = actions.gram(progress=Mock())
    assert isinstance(new_figure, Figure), new_summary
    assert not any(row[4] for row in new_rows)
    assert all(row[1] == row[2] for row in new_rows)
    assert new_rows != rows


def test_gram_real_invalid_weights_return_unavailable(diagnostic_session):
    session = diagnostic_session
    with torch.no_grad():
        session.handler.model.layers[3]["mlp"].down_proj.weight.zero_()
    summary, rows, figure = DiagnosticActions(session).gram(progress=Mock())
    assert "non-zero" in summary
    assert rows == [] and figure is None
    assert not session.lock.locked()
    session._restore.assert_not_called()


def test_gram_never_reassigns_saved_base_to_a_new_layer(diagnostic_session):
    actions = DiagnosticActions(diagnostic_session)
    diagnostic_session.handler._layer = 4
    summary, rows, figure = actions.gram(progress=Mock())
    assert "jiné vrstvě" in summary
    assert rows == [] and figure is None
    assert diagnostic_session.handler._layer == 4


def test_noise_batching_matches_single_draw_forwards(diagnostic_session):
    actions = DiagnosticActions(diagnostic_session)
    _, single, _ = actions.trace("Eiffel Tower", "Where is Eiffel Tower?", "Rome", noise_samples=3,
                                 noise_batch_size=1, progress=Mock())
    calls = diagnostic_session.handler.model.calls
    _, batched, _ = actions.trace("Eiffel Tower", "Where is Eiffel Tower?", "Rome", noise_samples=3,
                                  noise_batch_size=2, progress=Mock())
    assert diagnostic_session.handler.model.calls - calls < calls
    np.testing.assert_allclose([r[3:7] for r in sorted(single)], [r[3:7] for r in sorted(batched)], atol=1e-8)


def test_three_real_cases_and_repeated_trace_are_cached(diagnostic_session):
    actions = DiagnosticActions(diagnostic_session)
    summary, rows, figure = actions.trace_suite("The Eiffel Tower", "Where is The Eiffel Tower?", "Rome",
                                               noise_samples=1, fact_count=3, auto_noise=False, progress=Mock())
    assert "3 fakta" in summary and len(rows) == 7 and figure is not None
    calls = diagnostic_session.handler.model.calls
    again = actions.trace_suite("The Eiffel Tower", "Where is The Eiffel Tower?", "Rome",
                                 noise_samples=1, fact_count=3, auto_noise=False, progress=Mock())
    assert again[2] is figure and diagnostic_session.handler.model.calls == calls
