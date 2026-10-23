"""Real Gradio rendering/streaming: cached edits keep other tabs and chat intact."""

import asyncio
from importlib import import_module
from types import SimpleNamespace

import pytest


@pytest.fixture
def ui_runtime(runtime):
    runtime["session"].handler._layer = 5
    return import_module("src.workshop.workshop_ui"), runtime


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


def test_five_tabs_closed_source_panels_and_persistent_status(ui_runtime):
    ui, runtime = ui_runtime
    demo, _, _ = ui.build_workshop_ui(runtime["session"].handler)
    try:
        items = {item["id"]: item for item in demo.config["components"]}
        assert [c["props"]["label"] for c in items.values() if c["type"] == "tabitem"] == list(ui.TAB_LABELS)
        sources = [c for c in items.values() if c["type"] == "accordion" and c["props"]["label"].startswith("Zdrojový kód")]
        assert len(sources) == 5 and all(not c["props"]["open"] for c in sources)
        assert sum(c["type"] == "chatbot" for c in items.values()) == 2
        scenario = next(c for c in items.values() if c["props"].get("label") == "Vícekrokový scénář")
        assert scenario["props"]["value"] == "paper_hop2"
        prepared = next(c for c in items.values() if c["props"].get("label") == "Připravený příklad")
        assert prepared["props"]["value"] == ui.MULTIHOP_SCENARIOS["paper_hop2"].preset
        status = next(c for c in items.values() if c["props"].get("elem_id") == "workshop-operation-status")
        assert status["props"].get("max_height") is None and not status["props"].get("autoscroll", False)
        for label in ("Provést připravenou editaci", "Ověřit vícekrokové otázky · použít existující editaci", "Najít kandidátní vrstvu"):
            fn, event = event_for(demo, label)
            assert fn.concurrency_id == "latium-model" and event["show_progress"] == "hidden"
    finally:
        demo.close()


async def drain(demo, fn, inputs):
    response = await demo.process_api(fn, inputs=inputs, session_hash="test")
    frames = [response]
    while response["is_generating"]:
        response = await demo.process_api(fn, inputs=[], iterator=response["iterator"], session_hash="test")
        frames.append(response)
    return frames


def test_actual_gradio_edit_stream_has_no_log_scores_and_does_not_repeat(ui_runtime):
    ui, runtime = ui_runtime
    runtime["probing"].generate_completion.side_effect = lambda handler, prompt, **kwargs: "Paris" if handler.weight.item() == 1 else "Rome"
    def edit(handler, fact, **kwargs):
        handler.weight.add_(2)
        kwargs["on_progress"](.5, desc="ROME · iterace 2/4")
        return SimpleNamespace(seconds=1, peak_gb=0)
    runtime["api"].apply_rome_edit.side_effect = edit
    demo, _, _ = ui.build_workshop_ui(runtime["session"].handler)
    try:
        fn, _ = event_for(demo, "Provést připravenou editaci")
        args = ["The Eiffel Tower", "The Eiffel Tower is in", "Rome", "Paris", "", ""]
        frames = asyncio.run(drain(demo, fn, args))
        shown = str([f["data"] for f in frames])
        assert "Výsledek editace" in shown and "Rome" in shown and "operation-progress" in shown
        assert "Log-skóre" not in shown and "zneplatněn" not in shown
        asyncio.run(drain(demo, fn, args))
        assert runtime["api"].apply_rome_edit.call_count == 1
        runtime["probing"].score_target.assert_not_called()
    finally:
        demo.close()


def test_multihop_keeps_prepared_results_and_conversation(ui_runtime):
    ui, runtime = ui_runtime
    runtime["probing"].generate_completion.side_effect = lambda h, prompt, **kwargs: "Rome" if h.weight.item() != 1 else "Paris"
    def edit(handler, fact, **kwargs):
        handler.weight.add_(2)
        return SimpleNamespace(seconds=1, peak_gb=0)
    runtime["api"].apply_rome_edit.side_effect = edit
    coordinator = ui.WorkshopCoordinator(runtime["session"])
    fact = ui.MULTIHOP_SCENARIOS["eiffel_rome"].fact
    list(coordinator.mutate("prepared", fact[1], fact[0].format(fact[1]), fact[2], fact[3], "", ""))
    coordinator.conversation = [{"role": "user", "content": "keep this"}]
    revision = coordinator.revision
    frames = list(coordinator.mutate("multihop", "eiffel_rome"))
    assert coordinator.revision == revision and coordinator.conversation
    assert all("prepared_rows" not in frame for frame in frames)
    assert runtime["api"].apply_rome_edit.call_count == 1


def test_failed_input_does_not_erase_previous_result(ui_runtime):
    ui, runtime = ui_runtime
    coordinator = ui.WorkshopCoordinator(runtime["session"])
    frames = list(coordinator.mutate("custom", "", "", "", "", "", ""))
    assert "nepodařila" in frames[-1]["system_status"]
    assert all("prepared_rows" not in frame for frame in frames)


def test_diagnostic_events_stream_from_shared_session(ui_runtime, monkeypatch):
    ui, runtime = ui_runtime
    seen = []
    def trace(self, *args, **kwargs):
        seen.append(self.session)
        return "trace", [], None
    monkeypatch.setattr(ui.DiagnosticActions, "trace_suite", trace)
    demo, _, _ = ui.build_workshop_ui(runtime["session"].handler)
    try:
        frames = list(event_for(demo, "Najít kandidátní vrstvu")[0].fn("Eiffel", "Where is Eiffel?", "Paris", 1, 1, 3, 42, 3, True))
        assert len(frames) >= 2 and seen == [runtime["session"]]
    finally:
        demo.close()
