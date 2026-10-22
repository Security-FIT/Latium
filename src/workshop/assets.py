"""Presentation assets: local PNGs first, explicit remote fallback, Gradio routes."""

import os
import shutil
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ASSET_DIR = Path(__file__).resolve().parent / "static"
PNG_ASSETS = ("red_hat_research_logo.png", "fit_vut_logo.png", "layer_search_graph.png")
ASSET_BASE_URL = "https://raw.githubusercontent.com/olexamatej/Latium-workshop/main/latium_workshop/static"


def is_png(path):
    if not path.is_file():
        return False
    with path.open("rb") as source:
        return source.read(8) == b"\x89PNG\r\n\x1a\n"


def ensure_assets(directory=None, *, local_dirs=(), base_url=None):
    """Copy/download only the three named PNGs into the public asset directory.

    LATIUM_ASSET_DIR is a local source, not a directory exposed by Gradio.
    Missing remote assets fail clearly until the workshop images are published.
    """
    directory = Path(directory or ASSET_DIR).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    explicit = os.environ.get("LATIUM_ASSET_DIR", "").strip()
    candidates = ([Path(explicit).expanduser()] if explicit else []) + [Path(p) for p in local_dirs]
    base_url = base_url or os.environ.get("LATIUM_ASSET_BASE_URL", ASSET_BASE_URL)
    for name in PNG_ASSETS:
        destination = directory / name
        if is_png(destination):
            continue
        if destination.exists():
            raise ValueError(f"Existing asset is not a PNG; inspect it before replacing: {destination}")
        temporary = destination.with_name(name + ".part")
        try:
            local = next((folder / name for folder in candidates if is_png(folder / name)), None)
            if local is not None:
                shutil.copyfile(local, temporary)
            else:
                with urlopen(base_url.rstrip("/") + "/" + name, timeout=30) as response:
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(response, output)
            if not is_png(temporary):
                raise ValueError("The asset source did not return a PNG.")
            temporary.replace(destination)
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not prepare {name} ({type(exc).__name__}). Put the original PNGs beside "
                "the notebook or set LATIUM_ASSET_DIR to their directory. The remote fallback "
                "requires these images to be published in Latium-workshop/static paths."
            ) from None
        finally:
            temporary.unlink(missing_ok=True)
    return directory


def asset_url(name):
    path = (ASSET_DIR / name).resolve()
    if path.parent != ASSET_DIR or not path.is_file():
        raise ValueError(f"Unknown presentation asset: {name}")
    return "/gradio_api/file=" + quote(str(path), safe="/")


def stylesheet(name):
    base = (ASSET_DIR / "solo.css").read_text(encoding="utf-8")
    return base + "\n" + (ASSET_DIR / "workshop.css").read_text(encoding="utf-8") if name == "workshop" else base


def original_brandbar():
    """Reuse the original logos locally; uploaded notebooks may use official URLs."""
    logos = (
        ("red_hat_research_logo.png", "https://research.redhat.com/wp-content/uploads/2019/12/Logo-Red_Hat-Research-A-Standard-RGB-e1576582611932.png",
         "https://research.redhat.com/", "red-hat-logo", "Red Hat Research"),
        ("fit_vut_logo.png", "https://www.fit.vut.cz/img/logos/FIT_zkracene_barevne_RGB_CZ.png",
         "https://www.fit.vut.cz/", "fit-logo", "FIT VUT Brno"),
    )
    content = '<div class="demo-brandbar" aria-label="Red Hat Research a FIT VUT Brno">'
    for name, remote, website, css_class, label in logos:
        src = asset_url(name) if (ASSET_DIR / name).is_file() else remote
        content += (f'<a href="{website}" target="_blank" rel="noopener noreferrer" aria-label="{label}">'
                    f'<img class="{css_class}" src="{src}" alt="{label}"></a>')
    return content + "</div>"
