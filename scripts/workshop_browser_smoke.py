"""Optional browser audit: pip install playwright; playwright install chromium.

Runs the real model-free frontend, checks all five tabs and mobile overflow,
and saves screenshots outside the repository. No model or covariance download.
"""

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from playwright.sync_api import sync_playwright
    from src.workshop.workshop_ui import build_workshop_ui

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="/tmp/opencode")
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--populated", action="store_true", help="Render long layout-test fixtures, not model measurements")
    args = parser.parse_args()
    output = Path(args.artifacts)
    output.mkdir(parents=True, exist_ok=True)
    if args.populated:
        from src.workshop.results import render_result_rows

        long_answer = ("Layout audit fixture: this is a complete answer with ordinary words, not a model measurement. " * 12
                       + "\nA second line must remain completely visible.")
        fixture_rows = {
            "prepared_rows": [["Hlavní otázka", "Where is the Eiffel Tower? " * 6, long_answer, long_answer, "Ano"],
                              ["Kontrola", "A control question? " * 8, long_answer, long_answer, "Ne"]],
            "custom_rows": [["Vlastní otázka", "A long question? " * 10, long_answer, long_answer, "Ano"]],
            "multihop_rows": [["Dva kroky · rozklad", "Before question?\nAfter question?", "France", "Italy",
                                long_answer, long_answer, -1.25, -2.5, -1.25, -3.75, -.125, 3.625]],
            "gram_rows": [[layer, .123456, .023456, layer + 1, layer == 5] for layer in range(8)],
            "trace_rows": [[layer, layer, layer + 1, .12, .03, .08, .05, .555] for layer in range(8)],
            "prepared_history": [[1, "Eiffel Tower", "Paris", "Rome", "Aktivní", 20, 8]],
            "custom_history": [[1, "Eiffel Tower", "Paris", "Rome", "Aktivní", 20, 8]],
        }
        with patch("src.workshop.workshop_ui.render_result_rows",
                   lambda key, rows: render_result_rows(key, rows or fixture_rows.get(key, []))):
            demo, theme, css = build_workshop_ui()
    else:
        demo, theme, css = build_workshop_ui()
    if args.populated:
        # Explicit layout fixtures exercise numeric output without a model.
        for block in demo.blocks.values():
            if block.__class__.__name__ == "Chatbot":
                block.value = block.postprocess([
                    {"role": "user", "content": "Layout fixture: numeric answer"},
                    {"role": "assistant", "content": "1453.\n1889\n330 meters."},
                ]).model_dump()
    demo.queue().launch(server_name="127.0.0.1", server_port=args.port, share=False,
                        prevent_thread_lock=True, theme=theme, css=css,
                        allowed_paths=[str(Path(__file__).resolve().parents[1] / "src/workshop/static")])
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chromium", headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1560, "height": 1050})
            errors = []
            page.on("pageerror", lambda error: (errors.append(str(error)), print("Browser error:", error, flush=True)))

            def select_tab(label):
                page.wait_for_timeout(350)  # Let Gradio's responsive tab menu settle.
                tab = page.get_by_role("tab", name=label, exact=True)
                if tab.count():
                    tab.click()
                else:
                    page.get_by_role("button", name="More tabs", exact=True).click()
                    page.get_by_role("button", name=label, exact=True).click()

            page.goto(f"http://127.0.0.1:{args.port}/?__theme=light", wait_until="networkidle")
            page.locator('.demo-hero h1').wait_for(state="attached")
            assert page.get_by_role("tab").count() == 5
            page.screenshot(path=str(output / "workshop-desktop.png"), full_page=True)
            page.get_by_role("tab", name="2 · ROME: připravené příklady").click()
            source_toggle = next(item for item in page.get_by_text("Zdrojový kód · rozbalit", exact=True).all() if item.is_visible())
            source_toggle.click()
            page.locator(".cm-content:visible").wait_for()
            assert "def apply_rome_edit(" in page.locator(".cm-content:visible").inner_text()
            source_toggle.click()
            page.screenshot(path=str(output / "workshop-first-edit.png"), full_page=True)
            assert page.get_by_role("button", name="Provést připravenou editaci", exact=True).is_disabled()
            page.get_by_role("tab", name="4 · ROME: vlastní fakt a chat").click()
            form_box = page.locator('#custom-fact-form').bounding_box()
            chat_box = page.locator('#custom-chat').bounding_box()
            assert abs(form_box['y'] - chat_box['y']) < 3, (form_box, chat_box)
            assert abs(form_box['width'] - chat_box['width']) < 3, (form_box, chat_box)
            if args.populated:
                numeric = page.locator('#custom-chat .original-chat')
                assert "1453." in numeric.inner_text() and "1889" in numeric.inner_text()
                assert numeric.locator('ol, li').count() == 0
                clipped = numeric.evaluate("""e => [...e.querySelectorAll('p, span')]
                    .filter(n => n.textContent.includes('1453') && n.getBoundingClientRect().height > 0)
                    .some(n => n.scrollWidth > n.clientWidth + 2)""")
                assert not clipped, "Numeric chat answer is horizontally clipped"
            page.screenshot(path=str(output / "workshop-custom.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(150)
            form_box = page.locator('#custom-fact-form').bounding_box()
            chat_box = page.locator('#custom-chat').bounding_box()
            assert chat_box['y'] >= form_box['y'] + form_box['height'] - 3
            page.screenshot(path=str(output / "workshop-mobile.png"), full_page=True)
            overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 2")
            print("Browser errors:", errors)
            print("Mobile horizontal overflow:", overflow)
            assert not errors
            assert not overflow
            if args.populated:
                # Check populated result contents, not just the page bounding box.
                for width in (1560, 390):
                    page.set_viewport_size({"width": width, "height": 1050})
                    for tab in ("2 · ROME: připravené příklady", "3 · ROME: vícekrokové otázky",
                                "4 · ROME: vlastní fakt a chat", "5 · GRAM: detekce editace"):
                        select_tab(tab)
                        page.wait_for_timeout(100)
                        if tab.startswith("5") and not page.locator('#gram-results').is_visible():
                            page.get_by_text("Podrobné hodnoty detekce", exact=True).click()
                        clipped = page.evaluate("""() => [...document.querySelectorAll(
                            '.workshop-results, .workshop-results .result-text, .workshop-results dd')]
                            .filter(e => e.getBoundingClientRect().height > 0)
                            .filter(e => e.scrollWidth > e.clientWidth + 2 || e.scrollHeight > e.clientHeight + 2)
                            .map(e => ({tag: e.tagName, text: e.textContent.slice(0, 60)}))""")
                        assert not clipped, (width, tab, clipped)
                        assert all("log-skóre" not in text.lower() for text in page.locator('.workshop-results:visible').all_text_contents())
                        assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 2")
                select_tab("3 · ROME: vícekrokové otázky")
                page.screenshot(path=str(output / "workshop-multihop-mobile.png"), full_page=True)
                print("Populated results: complete at desktop and mobile widths, no clipped cells.")
                state = page.locator('#workshop-operation-status')
                state.evaluate("""e => {
                    const box = document.createElement('div'); box.className = 'operation-progress layout-audit-progress';
                    box.textContent = 'Probíhá měření odpovědí. Čas 15 s. Celý tento stav musí zůstat čitelný bez vnitřního scrollování. '.repeat(3);
                    e.appendChild(box);
                }""")
                assert state.evaluate('e => e.scrollHeight <= e.clientHeight + 2')
                page.evaluate('window.scrollTo(0, 1000)')
                page.wait_for_timeout(150)
                assert state.bounding_box()['y'] >= 0, 'Operation status disappeared while the page scrolled'
                page.locator('.layout-audit-progress').evaluate('e => e.remove()')
            # The real upload and error/retry path works without a GPU or a
            # network covariance fetch. Populated chat messages above are labeled layout fixtures.
            page.set_viewport_size({"width": 1560, "height": 1050})
            page.locator('input[type="file"][accept=".pt"]').set_input_files(
                {"name": "demo-covariance.pt", "mimeType": "application/octet-stream", "buffer": b"matrix"})
            page.get_by_role("button", name="Vybráno: demo-covariance.pt").wait_for()
            page.get_by_role("button", name="Připravit model", exact=True).click()
            page.get_by_text("Příprava selhala:", exact=False).wait_for()
            assert page.get_by_role("button", name="Připravit model", exact=True).is_enabled()
            assert page.get_by_role("button", name="Zrušit výběr souboru · použít Drive", exact=True).is_enabled()
            assert not errors, errors
            browser.close()
    finally:
        demo.close()


if __name__ == "__main__":
    main()
