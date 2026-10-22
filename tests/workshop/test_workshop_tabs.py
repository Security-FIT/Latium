"""Real Gradio 6 construction, event wiring and shared-session invalidation."""

import asyncio
import inspect
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.workshop.source_panel import resolve_source


@pytest.fixture
def ui_runtime(runtime, monkeypatch):
    runtime["session"].handler._layer = 5
    ui = import_module("src.workshop.workshop_ui")
    lessons = import_module("src.workshop.lessons")
    monkeypatch.setattr(lessons, "apply_rome_edit", runtime["api"].apply_rome_edit)
    for name in ("generate_completion", "score_target"):
        monkeypatch.setattr(lessons, name, runtime["probing"].__dict__[name])
    return ui, runtime


def ancestors(layout, target, path=()):
    if layout["id"] == target:
        return path
    for child in layout.get("children", []):
        found = ancestors(child, target, path + (layout["id"],))
        if found is not None:
            return found
    return None


def event_for(demo, label):
    button = next(c for c in demo.config["components"] if c["props"].get("value") == label)
    dependency = next(d for d in demo.config["dependencies"] if [button["id"], "click"] in [list(t) for t in d["targets"]])
    return demo.fns[dependency["id"]], dependency


def test_exact_five_tabs_right_closed_lazy_source_panels_and_shared_events(ui_runtime, monkeypatch):
    ui, runtime = ui_runtime
    source = import_module("src.workshop.source_panel")
    resolve = Mock()
    monkeypatch.setattr(source, "resolve_source", resolve)
    demo, theme, css = ui.build_workshop_ui(runtime["session"].handler)
    try:
        items = {item["id"]: item for item in demo.config["components"]}
        tabs = [item for item in items.values() if item["type"] == "tabitem"]
        tab_groups = [item for item in items.values() if item["type"] == "tabs"]
        assert len(tab_groups) == 1
        assert [tab["props"]["label"] for tab in tabs] == list(ui.TAB_LABELS)
        panels = [item for item in items.values() if item["type"] == "accordion" and item["props"]["label"].startswith("Zdrojový kód")]
        assert len(panels) == 5
        for tab, panel in zip(tabs, panels):
            path = ancestors(demo.config["layout"], panel["id"])
            assert tab["id"] in path
            assert panel["props"]["open"] is False
            column = items[path[-1]]
            assert column["type"] == "column" and column["props"]["scale"] == 3
            assert any(item["type"] == "code" and panel["id"] in ancestors(demo.config["layout"], item["id"])
                       for item in items.values())
        resolve.assert_not_called()
        assert theme is not None and css
        assert runtime["session"].handler._workshop_session is runtime["session"]
        assert sum(item["type"] == "chatbot" for item in items.values()) == 2
        entity_fields = [item for item in items.values() if item["props"].get("label") == "Co chceme editovat?"]
        assert [field["props"]["value"] for field in entity_fields] == ["The Eiffel Tower", ""]
        chat_id = next(item["id"] for item in items.values() if item["type"] == "chatbot")
        model_labels = [
            "Provést připravenou editaci", "Provést vlastní editaci",
            "Jedna editace + přímé, vícekrokové a kontrolní otázky",
            "Obnovit původní model · všechny záložky", "Ověřit současný stav", "Ověřit vlastní fakt",
            "Najít kandidátní vrstvu", "Spustit detekci GRAM na aktuálním modelu", "Odeslat zprávu", "Vymazat konverzaci",
        ]
        for label in model_labels:
            fn, dependency = event_for(demo, label)
            assert fn.concurrency_id == "latium-model" and fn.concurrency_limit == 1
            if label in model_labels[:4]:
                assert chat_id in dependency["outputs"]
            if label == "Odeslat zprávu":
                assert chat_id not in dependency["inputs"]  # No stale submitted history.
    finally:
        demo.close()


def test_actual_gradio_generator_preclears_outputs_on_failed_edit(ui_runtime):
    ui, runtime = ui_runtime
    demo, _, _ = ui.build_workshop_ui(runtime["session"].handler)
    try:
        fn, dependency = event_for(demo, "Provést vlastní editaci")

        async def invoke():
            first = await demo.process_api(fn, inputs=["", "", "", "", "", ""], session_hash="test")
            second = await demo.process_api(fn, inputs=[], iterator=first["iterator"], session_hash="test")
            return first, second

        first, last = asyncio.run(invoke())
        assert first["is_generating"]
        assert len(first["data"]) == len(dependency["outputs"])
        output_types = [demo.blocks[key].get_block_name() for key in dependency["outputs"]]
        assert first["data"][output_types.index("chatbot")] == []
        assert all(first["data"][i]["data"] == [] for i, kind in enumerate(output_types) if kind == "dataframe")
        assert "Operace se nepodařila" in str(last["data"])
        assert "původní model" in str(last["data"])
    finally:
        demo.close()


def test_actual_gradio_successful_edit_updates_only_its_results(ui_runtime):
    ui, runtime = ui_runtime
    session = runtime["session"]
    runtime["probing"].generate_completion.side_effect = lambda handler, prompt: (
        "Paris" if handler.weight.item() == 1 else "Rome"
    )
    runtime["probing"].score_target.side_effect = lambda handler, prompt, target: (
        handler.weight.item() if target == " Rome" else 2
    )

    def edit(handler, fact):
        handler.weight.add_(2)
        return SimpleNamespace(seconds=1, peak_gb=1, delta_norm=2)

    runtime["api"].apply_rome_edit.side_effect = edit
    demo, _, _ = ui.build_workshop_ui(session.handler)
    try:
        fn, dependency = event_for(demo, "Provést připravenou editaci")

        async def invoke():
            first = await demo.process_api(
                fn, inputs=["The Eiffel Tower", "The Eiffel Tower is in", "Rome", "Paris", "", ""],
                session_hash="test-success",
            )
            return await demo.process_api(fn, inputs=[], iterator=first["iterator"], session_hash="test-success")

        final = asyncio.run(invoke())
        output = dict(zip(dependency["outputs"], final["data"]))
        summaries = [value for key, value in output.items() if demo.blocks[key].get_block_name() == "markdown"]
        assert sum("Výsledek editace" in value for value in summaries) == 1
        assert sum("Výsledek zneplatněn" in value for value in summaries) == 4
        assert any("aktivní ROME editace" in value for value in summaries)
        tables = [value for key, value in output.items() if demo.blocks[key].get_block_name() == "dataframe"]
        assert any(value["data"] == [["Hlavní otázka", "The Eiffel Tower is in", "Paris", "Rome", "Ano"]] for value in tables)
        assert session.active_edit
        runtime["api"].apply_rome_edit.assert_called_once()
    finally:
        demo.close()


@pytest.mark.parametrize("tab", ["prepared", "custom", "multihop", "reset"])
@pytest.mark.parametrize("fails", [False, True])
def test_mutations_clear_all_results_and_chat_even_on_failure(ui_runtime, monkeypatch, tab, fails):
    ui, runtime = ui_runtime
    session = runtime["session"]
    coordinator = ui.WorkshopCoordinator(session)
    coordinator.conversation = [{"role": "user", "content": "old fact"}]

    def perform(*args, **kwargs):
        if tab == "reset":
            session.reset()
        else:
            with session.editing():
                session.handler.weight.add_(2)
                if fails:
                    raise RuntimeError("probe failure")
        if fails:
            raise RuntimeError("reset failure")
        return ("measured", [["actual"]], "status") if tab == "multihop" else ("measured", [], [], "status", [])

    target, method = (coordinator.multihop, "run") if tab == "multihop" else (coordinator.actions, "reset" if tab == "reset" else "edit")
    monkeypatch.setattr(target, method, perform)
    stream = coordinator.mutate(tab)
    first = next(stream)
    assert coordinator.conversation == []
    for name in ui.RESULT_TABS:
        assert first[name + "_rows"] == []
        assert "zneplatněn" in first[name + "_summary"]
    assert first["chatbot"] == [] and first["chat_input"] == ""
    assert first["trace_plot"] is first["gram_plot"] is None
    final = next(stream)
    assert final["chatbot"] == []
    assert "revize 1" in final["current_state"]
    if fails:
        assert "nepodařila" in final["system_status"]
        assert not session.active_edit and session.handler.weight.item() == 1
    # A queued chat obtains current server-side context, never a stale UI value.
    seen = []

    def chat(message, history, system, tokens, **kwargs):
        seen.extend(history)
        return "", [{"role": "assistant", "content": "new"}], "status"

    monkeypatch.setattr(coordinator.actions, "chat", chat)
    coordinator.chat("question", "neutral", 32)
    assert seen == []


def test_inspection_is_tab_local_and_diagnostics_share_session(ui_runtime, monkeypatch):
    ui, runtime = ui_runtime
    coordinator = ui.WorkshopCoordinator(runtime["session"])
    coordinator.conversation = [{"role": "user", "content": "keep"}]
    monkeypatch.setattr(coordinator.actions, "inspect", lambda *args, **kw: ("snapshot", [], [], "ready"))
    values = coordinator.inspect("prepared")
    assert "prepared_summary" in values and "custom_summary" not in values
    assert "chatbot" not in values and coordinator.conversation
    diagnostics = import_module("src.workshop.diagnostics")
    seen = []

    def trace(self, *args, **kwargs):
        seen.append((self.session, args))
        return "trace", [], None

    def gram(self, **kwargs):
        seen.append((self.session, ()))
        return "gram", [], None

    monkeypatch.setattr(diagnostics.DiagnosticActions, "trace", trace)
    monkeypatch.setattr(diagnostics.DiagnosticActions, "gram", gram)
    demo, _, _ = ui.build_workshop_ui(runtime["session"].handler)
    try:
        args = ("subject", "prompt", "target", 3, 2, 3.0, 42)
        list(event_for(demo, "Najít kandidátní vrstvu")[0].fn(*args))
        list(event_for(demo, "Spustit detekci GRAM na aktuálním modelu")[0].fn())
        assert seen == [(runtime["session"], args), (runtime["session"], ())]
    finally:
        demo.close()


def test_source_resolves_real_upstream_function_and_callable(real_runtime):
    source = resolve_source(("Prompt", "src.workshop.prompts", "render_prompt"))
    assert inspect.getsource(real_runtime.prompts.render_prompt) in source
    assert "def render_prompt(" in source
    callable_source = resolve_source(real_runtime.probing.generate_completion)
    assert "return text if raw else first_clause(text)" in callable_source


def test_notebook_source_fallback_is_exact_and_missing_source_is_explicit(monkeypatch):
    source_panel = import_module("src.workshop.source_panel")
    monkeypatch.setattr(source_panel.inspect, "getsource", Mock(side_effect=OSError("no linecache")))
    assert resolve_source(test_notebook_source_fallback_is_exact_and_missing_source_is_explicit, fallback="def actual_wrapper(): pass") == "def actual_wrapper(): pass"
    assert "není v tomto prostředí dostupný" in resolve_source(test_source_resolves_real_upstream_function_and_callable)
