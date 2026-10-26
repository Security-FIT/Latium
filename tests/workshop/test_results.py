"""All raw measurements remain readable, complete and escaped after formatting."""

from html.parser import HTMLParser

import numpy as np

from src.workshop.results import render_comparisons, render_multihop, render_result_rows


class ParsedResults(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.tags, self.text = [], []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)

    def handle_data(self, data):
        self.text.append(data)


def test_long_answers_keep_all_lines_and_literal_model_text():
    question = "Which author wrote this long work? " * 20
    before = "First line of the original answer.\n" + "An ordinary readable sentence. " * 80
    after = '<script>alert("not executable")</script>\n<img src=x onerror=bad>\n' + "New answer. " * 80
    rows = [["Parafráze", question, before, after, "Ano"]]
    parsed = ParsedResults(render_comparisons(rows))
    text = ''.join(parsed.text)
    assert question in text and before in text and after in text
    assert "script" not in parsed.tags and "img" not in parsed.tags
    assert "Před editací" in text and "Po editaci" in text and "Změna: Ano" in text
    assert rows[0][2] == before  # Formatting never mutates the measurement artifact.


def test_multihop_keeps_answers_but_removes_research_scores():
    row = ["Dva kroky", "First question?\nSecond question?", "France", "Italy",
           "Long original answer. " * 25, "Long edited answer. " * 30,
           -1.25, -2.5, -1.25, -3.75, -0.125, 3.625]
    parsed = ParsedResults(render_multihop([row]))
    text = ''.join(parsed.text)
    for value in row[:6]:
        assert str(value) in text
    assert "Log-skóre" not in text and "-1.25" not in text
    assert parsed.tags.count("article") == 1 and "table" not in parsed.tags


def test_metric_cards_do_not_drop_undefined_negative_or_small_values():
    gram = ''.join(ParsedResults(render_result_rows("gram_rows", [[5, -1e-12, None, 1, True]])).text)
    assert "-1e-12" in gram and "—" in gram and "Ano" in gram
    assert all(label in gram for label in ("Vrstva", "Aktuální skóre", "Skóre základu", "Pořadí", "Známá aktivní editace"))
    trace = ''.join(ParsedResults(render_result_rows("trace_rows", [[0, 0, 2, .1, .05, .08, .03, "N/A"]])).text)
    assert "N/A" in trace and "Relativní obnovení" in trace and "0.03" in trace


def test_exploratory_probes_show_guidance_without_inventing_expected_answers():
    from src.workshop.lessons import EIFFEL_COLLATERAL_PROBES, probe_matches
    rows = [["Země", "Which country?", "France", "Italy", "France", "Italy"]]
    for probe in EIFFEL_COLLATERAL_PROBES[-2:]:
        assert probe.before is None and probe.after is None
        assert not probe_matches(probe, "Some answer", "after")
        rows.append([probe.category, probe.question, probe.before, probe.after,
                     "Original <answer>", "Edited <answer>", {"note": probe.note}])
    source = render_multihop(rows)
    parsed = ParsedResults(source)
    text = ''.join(parsed.text)
    assert text.count("Očekávání před:") == 1
    assert "France" in text and "Italy" in text
    assert "Otevřená otázka" not in text and "Prohlédněte tvrzení o okolí" not in text
    assert text.count("Ruční posouzení:") == 2
    assert "Original <answer>" in text and "Edited <answer>" in text
    assert "<answer>" not in source


def test_result_views_are_full_width_outside_form_columns(runtime):
    from src.workshop.workshop_ui import build_workshop_ui
    from test_workshop_tabs import ancestors

    runtime["session"].handler._layer = 5
    demo, _, _ = build_workshop_ui(runtime["session"].handler)
    try:
        components = {item["id"]: item for item in demo.config["components"]}
        for name in ("prepared-results", "custom-results", "multihop-results", "gram-results"):
            view = next(item for item in components.values() if item["props"].get("elem_id") == name)
            assert view["type"] == "html"
            path = ancestors(demo.config["layout"], view["id"])
            assert all(components.get(key, {}).get("type") != "column" for key in path)
    finally:
        demo.close()


def test_rainbow_keeps_negative_effects_layer_order_and_peak():
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from src.workshop.diagnostics import _trace_rainbow

    effects = np.asarray([[-.02, 0., .03], [-.04, 0., .05]])
    rows = [[layer, layer, layer + 1, .5, .1, .1 + effect, effect, "N/A"]
            for layer, effect in enumerate(effects.mean(0))]
    figure = _trace_rainbow(list(reversed(rows)), effects, configured_layer=1)
    FigureCanvasAgg(figure).draw()
    axis = figure.axes[0]
    np.testing.assert_allclose([bar.get_height() for bar in axis.patches], [-.03, 0., .04])
    assert axis.get_ylim()[0] < -.03 and axis.get_ylim()[1] > .04
    assert [label.get_text() for label in axis.get_xticklabels()] == ["L0", "L1", "L2"]
    assert any(text.get_text() == "peak L2" for text in axis.texts)
