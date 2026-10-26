"""Launch policies for the workshop panel and authenticated standalone demo."""

import getpass
import os
from importlib.util import find_spec

from .assets import ASSET_DIR

LIGHT_MODE_JS = """() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get("__theme") !== "light") {
    url.searchParams.set("__theme", "light");
    window.location.replace(url.toString());
  }
}"""


def in_colab():
    try:
        return find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


def launch_ui(demo, theme, css, *, variant="workshop", share=None):
    solo = variant == "solo"
    colab = in_colab()
    # Colab's authenticated runtime proxy may return 404 in a separate tab.
    # Use the share URL for both the inline iframe and the clickable link.
    # Sharing does not prevent Colab from terminating the runtime.
    default_share = "1" if solo or colab else "0"
    if share is None:
        share = os.environ.get("GRADIO_SHARE", default_share).strip().lower() in {"1", "true", "yes"}
    auth = None
    if solo and share:
        user = os.environ.get("GRADIO_USER") or "demo"
        # A hidden prompt preserves authenticated sharing without printing a secret
        # into notebook output, logs, or a saved notebook.
        password = os.environ.get("GRADIO_PASSWORD") or getpass.getpass("Password for public demo: ")
        if not password:
            raise ValueError("Set GRADIO_PASSWORD or enter a nonempty password for public sharing.")
        auth = (user, password)
        print("Public demo authentication enabled (GRADIO_USER defaults to demo).")
    return demo.queue(default_concurrency_limit=1, max_size=8).launch(
        share=share, inline=not solo, auth=auth, show_error=True,
        height=1000 if colab and not solo else 500,
        theme=theme, css=css, js=LIGHT_MODE_JS,
        allowed_paths=[str(ASSET_DIR)],
    )
