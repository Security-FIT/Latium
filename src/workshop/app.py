"""Small public entrypoint used by clean-notebook.ipynb."""

from pathlib import Path

_demo = None


def launch_workshop(*, work_dir=None, share=None):
    """Render before model loading; preserve any active edit when reopening UI."""
    global _demo
    from src.workshop.launch import launch_ui
    from src.workshop.setup import Environment
    from src.workshop.workshop_ui import build_workshop_ui

    handler = getattr(getattr(_demo, "_model_session", None), "handler", None)
    if _demo is not None:
        _demo.close()
    environment = Environment(Path(work_dir).expanduser().resolve()) if work_dir else Environment.discover()
    environment.work_dir.mkdir(parents=True, exist_ok=True)
    _demo, theme, css = build_workshop_ui(handler, environment=environment)
    launch_ui(_demo, theme, css, share=share)
    return _demo
