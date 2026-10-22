"""Optional browser audit: pip install playwright; playwright install chromium.

Runs the real model-free frontend, checks all five tabs and mobile overflow,
and saves screenshots outside the repository. No model or covariance download.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from playwright.sync_api import sync_playwright
    from src.workshop.workshop_ui import build_workshop_ui

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="/tmp/opencode")
    parser.add_argument("--port", type=int, default=7864)
    args = parser.parse_args()
    output = Path(args.artifacts)
    output.mkdir(parents=True, exist_ok=True)
    demo, theme, css = build_workshop_ui()
    demo.queue().launch(server_name="127.0.0.1", server_port=args.port, share=False,
                        prevent_thread_lock=True, theme=theme, css=css,
                        allowed_paths=[str(Path(__file__).resolve().parents[1] / "src/workshop/static")])
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chromium", headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1560, "height": 1050})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{args.port}/?__theme=light", wait_until="networkidle")
            page.get_by_role("heading", name="Co když někdo přepíše jediný fakt v jazykovém modelu?").wait_for()
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
            page.screenshot(path=str(output / "workshop-custom.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            page.screenshot(path=str(output / "workshop-mobile.png"), full_page=True)
            overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 2")
            print("Browser errors:", errors)
            print("Mobile horizontal overflow:", overflow)
            assert not errors
            assert not overflow
            # The real upload and error/retry path works without a GPU or a
            # network covariance fetch. No fake assistant/model response is used.
            page.set_viewport_size({"width": 1560, "height": 1050})
            page.locator('input[type="file"][accept=".pt"]').set_input_files(
                {"name": "demo-covariance.pt", "mimeType": "application/octet-stream", "buffer": b"matrix"})
            page.get_by_role("button", name="Vybráno: demo-covariance.pt").wait_for()
            page.get_by_role("button", name="Připravit model", exact=True).click()
            page.get_by_text("Příprava selhala:", exact=False).wait_for()
            assert page.get_by_role("button", name="Připravit model", exact=True).is_enabled()
            assert page.get_by_role("button", name="Zrušit výběr souboru · použít Drive", exact=True).is_enabled()
            browser.close()
    finally:
        demo.close()


if __name__ == "__main__":
    main()
