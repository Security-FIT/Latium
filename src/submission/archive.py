"""Read-only retrieval and byte verification of the published artifact archive."""
from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
import zipfile

SHARE = "https://nextcloud.fit.vutbr.cz/s/FcezyKRbbwdAbtX"
ARCHIVE_FOLDERS = {
    "classic-gram-merged": "classic-gram-merged",
    "finetune-no-cov": "ROME - CSF/finetune-no-cov",
    "finetune-covariance": "finetune-covariance",
    "CSF-final-papers": "CSF-graphs",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Archive member escapes destination: {member.filename}")
        bundle.extractall(destination)


def download_archives(destination):
    """Download exact folders; this never writes to the remote share."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Fetch indexed files directly. Concurrent folder-ZIP streams can end in a
    # Nextcloud LockedException while still returning HTTP 200 and a partial ZIP.
    receipts = [download_indexed_campaign(name, destination / name) for name in ARCHIVE_FOLDERS]
    (destination / "download-receipt.json").write_text(json.dumps(receipts, indent=2) + "\n")
    return receipts


def download_indexed_campaign(name, destination, *, index_file=None, workers=8):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    remote = ARCHIVE_FOLDERS[name]
    base_url = "https://nextcloud.fit.vutbr.cz/public.php/webdav/" + urllib.parse.quote(remote, safe="/") + "/"
    authorization = "Basic " + base64.b64encode(b"FcezyKRbbwdAbtX:").decode()

    def get(relative, path):
        request = urllib.request.Request(base_url + urllib.parse.quote(relative, safe="/"), headers={"Authorization": authorization})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_bytes(data)
        temporary.replace(path)

    if index_file:
        (root / "INDEX.json").write_bytes(Path(index_file).read_bytes())
    elif not (root / "INDEX.json").exists():
        get("INDEX.json", root / "INDEX.json")
    entries = json.loads((root / "INDEX.json").read_text())

    def fetch(entry):
        target = (root / entry["path"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Indexed file escapes destination: {entry['path']}")
        if target.is_file() and sha256(target) == entry["sha256"]:
            return
        for attempt in range(3):
            try:
                get(entry["path"], target)
                if sha256(target) != entry["sha256"] and target.suffix not in {".md", ".csv"}:
                    raise ValueError(f"Archive bytes changed: {entry['path']}")
                return
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    print(f"Fetching/verifying {len(entries)} indexed files for {name}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(fetch, entries), start=1):
            if index % 1000 == 0:
                print(f"{name}: {index}/{len(entries)}", flush=True)
    result = verify_archive(root)
    if not result["ok"]:
        raise ValueError(f"Archive verification failed: {result}")
    return result


def campaign_root(path):
    path = Path(path).resolve()
    if (path / "INDEX.json").is_file():
        return path
    indexes = list(path.glob("*/INDEX.json"))
    if len(indexes) != 1:
        raise ValueError(f"Expected one campaign INDEX.json under {path}")
    return indexes[0].parent


def verify_archive(path):
    root = campaign_root(path)
    entries = json.loads((root / "INDEX.json").read_text())
    mismatches, missing, format_only_drift = [], [], []
    verified = 0
    for entry in entries:
        target = (root / entry["path"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Indexed file escapes archive: {entry['path']}")
        if not target.is_file():
            missing.append(entry["path"])
        elif sha256(target) != entry["sha256"]:
            raw = target.read_bytes() if target.suffix == ".csv" else b""
            original_crlf = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            if raw and hashlib.sha256(original_crlf).hexdigest() == entry["sha256"]:
                format_only_drift.append(entry["path"])
            else:
                mismatches.append(entry["path"])
        else:
            verified += 1
    # The public README was updated after the published byte index. Report that
    # explicitly; configuration, selection and experimental files stay strict.
    documentary_drift = [p for p in mismatches if p in {"README.md", "manifests/README.md"}]
    failures = [p for p in mismatches if p not in documentary_drift]
    return {"root": str(root), "indexed_files": len(entries), "verified_files": verified,
            "missing": missing, "hash_mismatches": failures, "documentary_drift": documentary_drift,
            "format_only_drift": format_only_drift,
            "ok": not missing and not failures}
