"""Read-only result views: full text on desktop, labelled cards on small screens.

Numerical/model callbacks keep their original row contracts. Only the Gradio
presentation changes, so long answers never become tiny editable-grid cells.
"""

import html
import math
from numbers import Real


def _result_text(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Ano" if value else "Ne"
    if isinstance(value, Real) and not isinstance(value, int):
        value = f"{value:.6g}" if math.isfinite(value) else "N/A"
    return html.escape(str(value))


def _empty_results(title):
    return f'<section class="workshop-results"><h3>{html.escape(title)}</h3><p>Zatím bez měření.</p></section>'


def render_comparisons(rows):
    if not rows:
        return _empty_results("Odpovědi před a po editaci")
    body = []
    for category, question, before, after, changed in rows:
        body.append(
            '<tr><th scope="row"><span class="result-badge">' + _result_text(category) + '</span>'
            '<div class="result-text">' + _result_text(question) + '</div>'
            '<span class="result-change">Změna: ' + _result_text(changed) + '</span></th>'
            '<td data-label="Před editací"><div class="result-text">' + _result_text(before) + '</div></td>'
            '<td data-label="Po editaci"><div class="result-text">' + _result_text(after) + '</div></td></tr>'
        )
    return ('<section class="workshop-results"><h3>Odpovědi před a po editaci</h3>'
            '<table class="comparison-results"><caption class="sr-only">Stejné otázky před a po ROME</caption>'
            '<colgroup><col style="width:40%"><col style="width:30%"><col style="width:30%"></colgroup>'
            '<thead><tr><th>Otázka a kontrola změny</th><th>Před editací</th><th>Po editaci</th></tr></thead>'
            '<tbody>' + ''.join(body) + '</tbody></table></section>')


def render_metric_cards(rows, headers, title):
    if not rows:
        return _empty_results(title)
    cards = []
    for row in rows:
        fields = ''.join('<div><dt>' + html.escape(label) + '</dt><dd class="result-text">'
                         + _result_text(value) + '</dd></div>' for label, value in zip(headers[1:], row[1:]))
        cards.append('<article class="metric-card"><h4>' + html.escape(headers[0]) + ': '
                     + _result_text(row[0]) + '</h4><dl>' + fields + '</dl></article>')
    return ('<section class="workshop-results"><h3>' + html.escape(title)
            + '</h3><div class="metric-cards">' + ''.join(cards) + '</div></section>')


def render_multihop(rows):
    if not rows:
        return _empty_results("Vícekrokové otázky")
    cards = []
    for category, question, expected_before, expected_after, before, after, *extras in rows:
        expectations = []
        for label, expected in (("Očekávání před", expected_before), ("po", expected_after)):
            if expected not in (None, "", "—"):
                expectations.append(label + ': <strong>' + _result_text(expected) + '</strong>')
        expectation_html = '<p class="expectations">' + ' · '.join(expectations) + '</p>' if expectations else ''
        note = next((item["note"] for item in extras if isinstance(item, dict) and item.get("note")), "")
        note_html = '<p class="probe-note">' + _result_text(note) + '</p>' if note else ''
        cards.append(
            '<article class="multihop-card"><h4>' + _result_text(category) + '</h4>'
            '<p class="result-text probe-question">' + _result_text(question) + '</p>'
            + expectation_html + note_html +
            '<div class="answer-pair"><section><h5>Odpověď před</h5><div class="result-text">' + _result_text(before)
            + '</div></section><section><h5>Odpověď po</h5><div class="result-text">' + _result_text(after)
            + '</div></section></div></article>'
        )
    return ('<section class="workshop-results"><h3>Vícekrokové otázky</h3>'
            + ''.join(cards) + '</section>')


def render_result_rows(key, rows):
    """Format only known presentation keys; state/controller rows remain untouched."""
    if key in {"prepared_rows", "custom_rows"}:
        return render_comparisons(rows)
    if key == "multihop_rows":
        return render_multihop(rows)
    if key in {"prepared_history", "custom_history"}:
        return render_metric_cards(rows, ["Pokus", "Entita", "Původně", "Nově", "Výsledek", "Čas (s)", "Špička GPU (GB)"],
                                   "Historie editací")
    if key == "trace_rows":
        return render_metric_cards(rows, ["Střed vrstvy", "Začátek okna", "Konec okna (mimo)", "Čistá pravděpodobnost",
                                         "Poškozená pravděpodobnost", "Obnovená pravděpodobnost", "IE", "Relativní obnovení"],
                                   "Měření kauzálního trasování")
    if key == "gram_rows":
        return render_metric_cards(rows, ["Vrstva", "Aktuální skóre", "Skóre základu", "Pořadí", "Známá aktivní editace"],
                                   "Skóre a pořadí GRAM")
    return rows
