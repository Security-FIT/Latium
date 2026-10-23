"""Small public entrypoint used by clean-notebook.ipynb."""

from pathlib import Path

_demo = None


def launch_workshop(*, work_dir=None, share=None, seed=42, covariance_drive_url=None):
    """Render before model loading; preserve any active edit when reopening UI."""
    global _demo
    from src.workshop.launch import launch_ui
    from src.workshop.setup import Environment
    from src.workshop.workshop_ui import build_workshop_ui

    handler = getattr(getattr(_demo, "_model_session", None), "handler", None)
    # Empty notebook input keeps the latest link entered in the existing panel.
    source = (covariance_drive_url or "").strip() or getattr(_demo, "_covariance_drive_url", None)
    if _demo is not None:
        _demo.close()
    environment = Environment(Path(work_dir).expanduser().resolve()) if work_dir else Environment.discover()
    environment.work_dir.mkdir(parents=True, exist_ok=True)
    _demo, theme, css = build_workshop_ui(handler, environment=environment, seed=seed, covariance_drive_url=source)
    launch_ui(_demo, theme, css, share=share)
    return _demo
