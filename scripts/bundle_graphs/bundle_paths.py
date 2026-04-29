from __future__ import annotations

import sys
from pathlib import Path


def import_root_for(script_file: str | Path) -> Path:
    """Return the root that should be on sys.path for graph scripts.

    In-repo scripts live under scripts/bundle_graphs and import from the repo
    root. Bundled copies live under final_n500_bundle/scripts_for_graphs and
    import from that self-contained helper directory.
    """
    path = Path(script_file).resolve()
    for parent in (path.parent, *path.parents):
        if parent.name == "scripts_for_graphs":
            return parent

    for parent in path.parents:
        if (parent / "src").is_dir() and (parent / "paper_graphs").is_dir() and (parent / "detector").is_dir():
            return parent

    return path.parents[2]


def add_import_root(script_file: str | Path) -> Path:
    root = import_root_for(script_file)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def default_bundle_root(script_file: str | Path) -> Path:
    root = import_root_for(script_file)
    if root.name == "scripts_for_graphs" and root.parent.name == "final_n500_bundle":
        return root.parent
    return root / "final_n500_bundle"


def display_path(path: str | Path | None, *, bundle_root: Path, import_root: Path | None = None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(bundle_root.resolve()))
    except Exception:
        pass
    if import_root is not None:
        try:
            return str(candidate.resolve().relative_to(import_root.resolve()))
        except Exception:
            pass
    return str(candidate)
