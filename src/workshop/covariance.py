"""Shared covariance retrieval with atomic downloads and local/Colab fallbacks."""

import os
import re
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DEFAULT_COVARIANCE_DRIVE_URL = "https://drive.google.com/file/d/1UmPlov94UKE4KIHclswVUMXyliQM3qvU/view?usp=drive_link"
COVARIANCE_IDS = {
    "workshop": "1UmPlov94UKE4KIHclswVUMXyliQM3qvU",
    "solo": "1UmPlov94UKE4KIHclswVUMXyliQM3qvU",
}
COVARIANCE_NAME = "Qwen_Qwen3.5-4B_5_SM_Method.WIKIPEDIA_100000.pt"


def available(path):
    return path.is_file() and path.stat().st_size > 0


def parse_drive_source(source):
    """Accept a shared file link, uc/open URL or bare file ID; preserve resourcekey."""
    source = source.strip()
    if not source:
        return "", ""
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", source):
        return source, ""
    url = urlparse(source)
    if url.scheme != "https" or url.hostname not in {"drive.google.com", "drive.usercontent.google.com"}:
        raise ValueError("Vložte Google Drive odkaz na soubor nebo samotné ID, nikoli odkaz na složku.")
    query = parse_qs(url.query)
    match = re.search(r"/file/d/([A-Za-z0-9_-]+)(?:/|$)", url.path)
    file_id = match.group(1) if match else query.get("id", [""])[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]{10,}", file_id):
        raise ValueError("Z odkazu nelze přečíst ID souboru. Použijte odkaz /file/d/…/view.")
    return file_id, query.get("resourcekey", [""])[0]


def validate_covariance_download(path):
    if not available(path):
        raise RuntimeError("Stažený soubor je prázdný.")
    with path.open("rb") as stream:
        prefix = stream.read(512).lstrip().lower()
    if prefix.startswith((b"<!doctype html", b"<html", b"<head", b"<body")):
        raise RuntimeError("Drive vrátil přihlašovací/HTML stránku místo matice .pt.")


def download_colab(file_id, destination, resource_key=""):
    import google.auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    # Gradio executes callbacks in worker threads. Interactive Colab OAuth must
    # happen in the notebook cell, not inside this download callback.
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
    drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
    request = drive.files().get_media(fileId=file_id, supportsAllDrives=True, acknowledgeAbuse=True)
    if resource_key:
        request.headers["X-Goog-Drive-Resource-Keys"] = f"{file_id}/{resource_key}"
    with destination.open("wb") as output:
        downloader = MediaIoBaseDownload(output, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Downloading covariance: {int(status.progress() * 100)}%")


def download_public(file_id, destination):
    import gdown

    if not gdown.download(id=file_id, output=str(destination), quiet=False):
        raise RuntimeError("Google Drive did not return a covariance file.")


def download_resource_link(file_id, destination, resource_key):
    import gdown

    url = f"https://drive.google.com/uc?id={file_id}&resourcekey={resource_key}"
    if not gdown.download(url=url, output=str(destination), quiet=False):
        raise RuntimeError("Google Drive did not return a covariance file.")


def download_error_hint(exc, method):
    # Avoid printing request URLs or account credentials in third-party exception text.
    status = getattr(getattr(exc, "resp", None), "status", None)
    if isinstance(exc, RuntimeError) and str(exc) in {
        "Stažený soubor je prázdný.", "Drive vrátil přihlašovací/HTML stránku místo matice .pt.",
    }:
        return str(exc)
    if isinstance(exc, ImportError):
        return "Přihlášené stahování je dostupné pouze v Google Colabu."
    if status == 404:
        return "Drive soubor nenalezl nebo přihlášený účet nemá přístup (HTTP 404)."
    if status in (401, 403):
        return f"Drive odmítl přístup (HTTP {status}); zkontrolujte účet, oprávnění a limit stahování."
    if method == "public":
        return (f"Veřejné stažení selhalo ({type(exc).__name__}). Soubor může vyžadovat přihlášení "
                "nebo mít vyčerpaný limit. Pro veřejné stažení zapněte „Kdokoli s odkazem“.")
    return (f"Stažení přes přihlášený účet selhalo ({type(exc).__name__}). V první buňce clean-notebook "
            "zapněte AUTHENTICATE_DRIVE, přihlaste se a zkuste přípravu znovu; nebo nahrajte .pt přímo.")


def upload_covariance(destination):
    try:
        from google.colab import files
    except ImportError as exc:
        raise FileNotFoundError(
            "Covariance missing. Set LATIUM_COV_PATH to an existing .pt file "
            f"or place it at {destination}."
        ) from exc
    print("Upload the covariance .pt file.")
    names = [Path(name) for name in files.upload() if name.endswith(".pt")]
    if not names or not available(names[0]):
        raise FileNotFoundError("The upload did not contain a nonempty .pt covariance file.")
    shutil.move(str(names[0]), str(destination))


def ensure_covariance(env, drive_url=None, *, mode="auto", force_download=False, manual_upload=True):
    """Public download first, optional signed-in Colab fallback, then manual upload.

    Explicit local LATIUM_COV_PATH takes precedence. A changed Drive ID does not
    silently reuse a download from a different source. Replacing a file is atomic.
    """
    if mode not in {"auto", "public", "authenticated", "upload"}:
        raise ValueError("Režim musí být auto, public, authenticated nebo upload.")
    override = os.environ.get("LATIUM_COV_PATH", "").strip()
    path = (Path(override).expanduser().resolve() if override else
            env.work_dir / "data/stats" / COVARIANCE_NAME)
    if override and not available(path):
        raise FileNotFoundError(f"LATIUM_COV_PATH does not name a nonempty file: {path}")
    if override:
        validate_covariance_download(path)
        print(f"Používám místní matici LATIUM_COV_PATH: {path}")
        return path
    source = drive_url if drive_url is not None else os.environ.get(
        "LATIUM_COV_URL", os.environ.get("LATIUM_COV_FILE_ID", COVARIANCE_IDS[env.variant]))
    file_id, resource_key = parse_drive_source(source if mode != "upload" else "")
    stamp = path.with_name(path.name + ".drive-id")
    cached_id = stamp.read_text().strip() if stamp.is_file() else None
    if available(path) and not force_download and mode != "upload":
        if cached_id == file_id or (drive_url is None and cached_id is None):
            try:
                validate_covariance_download(path)
                print(f"Matice je již připravena: {path}")
                return path
            except RuntimeError:
                print("Původní soubor není platné stažení; stahuji znovu.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    methods = {"auto": ("public", "authenticated"), "public": ("public",),
               "authenticated": ("authenticated",), "upload": ()}[mode]
    if file_id and mode != "upload":
        print(f"Stahuji inverzní kovarianci z Google Drive (ID {file_id}).")
        for method in methods:
            try:
                temporary.unlink(missing_ok=True)
                if method == "authenticated":
                    if resource_key:
                        download_colab(file_id, temporary, resource_key)
                    else:
                        download_colab(file_id, temporary)
                elif resource_key:
                    download_resource_link(file_id, temporary, resource_key)
                else:
                    download_public(file_id, temporary)
                validate_covariance_download(temporary)
                temporary.replace(path)
                stamp.write_text(file_id)
                print(f"Matice připravena: {path} ({path.stat().st_size / 1e6:.1f} MB)")
                return path
            except Exception as exc:
                print(download_error_hint(exc, method))
            finally:
                temporary.unlink(missing_ok=True)
    if file_id and mode not in {"auto", "upload"}:
        raise RuntimeError(
            "Matici se nepodařilo stáhnout. Zvolte auto (oba způsoby), authenticated "
            "(přihlášení v Colabu) nebo upload (ruční nahrání .pt).")
    if not manual_upload:
        raise RuntimeError("Matici nelze stáhnout. Nahrajte .pt do pole „Vlastní kovariance“ v Gradio "
                           "nebo upravte Drive odkaz a oprávnění a spusťte přípravu znovu.")
    print("Nahrajte matici ručně, nebo změňte odkaz/režim a spusťte buňku znovu.")
    try:
        upload_covariance(temporary)
        validate_covariance_download(temporary)
        temporary.replace(path)
        stamp.write_text("upload")
    finally:
        temporary.unlink(missing_ok=True)
    return path
