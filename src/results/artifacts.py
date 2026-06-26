"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from src.common.io import to_serializable


def content_hash(payload: Any) -> str:
    encoded = json.dumps(
        to_serializable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    return content_hash(config)


def build_artifact(
    *,
    artifact_id: str,
    kind: str,
    producer: str,
    run_id: str,
    model: Optional[str],
    plan_id: Optional[str],
    edit_method: Optional[str],
    status: str,
    config: dict[str, Any],
    config_hash: str,
    inputs: list[dict[str, str]],
    cases: list[dict[str, Any]],
    summary: dict[str, Any],
    created_at: str,
    category: Optional[str] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    artifact = {
        "artifact_id": artifact_id,
        "kind": kind,
        "producer": producer,
        "run": {
            "run_id": run_id,
            "model": model,
            "plan_id": plan_id,
            "edit_method": edit_method,
        },
        "status": status,
        "config": config,
        "config_hash": config_hash,
        "inputs": inputs,
        "created_at": created_at,
        "cases": cases,
        "summary": summary,
        "error": error,
    }
    if category is not None:
        artifact["category"] = category
    return artifact


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(to_serializable(payload), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _path_inside(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    paths = [candidate] if candidate.is_absolute() else [candidate, root / candidate]
    for path in paths:
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return resolved
    raise ValueError(f"Artifact path must be inside run root: {candidate}")


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def _new_manifest(run_id: str, metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    now = datetime.now().isoformat()
    return {
        "run_id": run_id,
        "created_at": now,
        "updated_at": now,
        "artifacts": {},
        "metadata": dict(metadata or {}),
    }


def _normalize_inputs(inputs: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "artifact_id": str(item["artifact_id"]),
            "content_hash": str(item["content_hash"]),
        }
        for item in inputs
    ]


class ArtifactWriter:
    def __init__(
        self,
        run_root: str | Path,
        *,
        run_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        root = Path(run_root)
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()
        self.manifest_path = _manifest_path(self.root)
        self.lock_path = self.root / ".manifest.lock"
        with self._manifest_lock():
            if self.manifest_path.exists():
                self.manifest = self._load_manifest()
            else:
                self.manifest = _new_manifest(run_id or self.root.name, metadata)
                self._write_manifest()

    @contextmanager
    def _manifest_lock(self) -> Iterator[None]:
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load_manifest(self) -> dict[str, Any]:
        manifest = _read_json(self.manifest_path)
        if not isinstance(manifest.get("artifacts"), dict):
            raise ValueError("Manifest artifacts must be an object")
        return manifest

    def _reload_manifest(self) -> None:
        if self.manifest_path.exists():
            self.manifest = self._load_manifest()

    def current(
        self,
        artifact_id: str,
        *,
        expected_config_hash: str,
        inputs: Iterable[dict[str, str]],
    ) -> Optional[dict[str, Any]]:
        self._reload_manifest()
        return self._current(
            artifact_id,
            expected_config_hash=expected_config_hash,
            inputs=inputs,
        )

    def _current(
        self,
        artifact_id: str,
        *,
        expected_config_hash: str,
        inputs: Iterable[dict[str, str]],
        expected_content_hash: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        record = self.manifest["artifacts"].get(str(artifact_id))
        if not isinstance(record, dict):
            return None
        if record.get("status") not in {"complete", "unavailable"}:
            return None
        if record.get("config_hash") != expected_config_hash:
            return None
        if record.get("inputs", []) != _normalize_inputs(inputs):
            return None
        if expected_content_hash is not None and record.get("content_hash") != expected_content_hash:
            return None
        try:
            path = _path_inside(self.root, str(record["path"]))
        except (KeyError, ValueError):
            return None
        return dict(record) if path.is_file() else None

    def write(
        self,
        path: str | Path,
        payload: dict[str, Any],
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        artifact_id = str(payload["artifact_id"])
        target = _path_inside(self.root, path)
        inputs = _normalize_inputs(payload.get("inputs", []))
        record_metadata = payload.get("record_metadata")
        serializable_payload = {key: value for key, value in payload.items() if key != "record_metadata"}
        serializable = to_serializable(serializable_payload)
        serializable_metadata = to_serializable(record_metadata) if isinstance(record_metadata, dict) else None
        artifact_digest = content_hash(serializable)

        with self._manifest_lock():
            self._reload_manifest()
            self._require_inputs(inputs)
            if not force:
                current = self._current(
                    artifact_id,
                    expected_config_hash=str(payload["config_hash"]),
                    inputs=inputs,
                )
                if current is not None:
                    if serializable_metadata is not None:
                        record = self.manifest["artifacts"][artifact_id]
                        if record.get("metadata") != serializable_metadata:
                            record["metadata"] = serializable_metadata
                            record["updated_at"] = datetime.now().isoformat()
                            self._write_manifest()
                            return dict(record)
                    return current

            _atomic_write_json(target, serializable)
            run = serializable["run"]
            record = {
                "artifact_id": artifact_id,
                "kind": serializable["kind"],
                "category": serializable.get("category"),
                "path": str(target.relative_to(self.root)),
                "content_hash": artifact_digest,
                "config_hash": serializable["config_hash"],
                "status": serializable["status"],
                "model": run["model"],
                "plan_id": run["plan_id"],
                "edit_method": run.get("edit_method"),
                "producer": serializable["producer"],
                "inputs": inputs,
                "updated_at": datetime.now().isoformat(),
            }
            if serializable_metadata is not None:
                record["metadata"] = serializable_metadata
            self.manifest["artifacts"][artifact_id] = record
            self._remove_stale_descendants(artifact_id, artifact_digest)
            self._write_manifest()
            return record

    def _require_inputs(self, inputs: list[dict[str, str]]) -> None:
        for item in inputs:
            record = self.manifest["artifacts"].get(item["artifact_id"])
            if not isinstance(record, dict):
                raise ValueError(f"Missing input artifact: {item['artifact_id']}")
            if record.get("content_hash") != item["content_hash"]:
                raise ValueError(f"Stale input artifact: {item['artifact_id']}")
            path = _path_inside(self.root, str(record["path"]))
            if not path.is_file():
                raise ValueError(f"Missing input file: {item['artifact_id']}")

    def _remove_stale_descendants(self, artifact_id: str, new_hash: str) -> None:
        stale: set[str] = set()
        while True:
            found = {
                dependent_id
                for dependent_id, record in self.manifest["artifacts"].items()
                if dependent_id != artifact_id
                and dependent_id not in stale
                and isinstance(record, dict)
                and any(
                    str(item.get("artifact_id")) in stale
                    or (item.get("artifact_id") == artifact_id and item.get("content_hash") != new_hash)
                    for item in record.get("inputs", [])
                )
            }
            if not found:
                break
            stale.update(found)
        for dependent_id in stale:
            record = self.manifest["artifacts"].pop(dependent_id)
            self._remove_record_files(record)

    def _remove_record_files(self, record: dict[str, Any]) -> None:
        try:
            artifact_path = _path_inside(self.root, str(record["path"]))
        except (KeyError, ValueError):
            return
        if record.get("kind") == "render" and artifact_path.is_file():
            try:
                artifact = _read_json(artifact_path)
            except (OSError, ValueError, json.JSONDecodeError):
                artifact = {}
            for output in artifact.get("summary", {}).get("outputs", []):
                try:
                    output_path = _path_inside(self.root, str(output))
                except ValueError:
                    continue
                if output_path.is_file():
                    output_path.unlink()
        if artifact_path.is_file():
            artifact_path.unlink()

    def _write_manifest(self) -> None:
        self.manifest["updated_at"] = datetime.now().isoformat()
        _atomic_write_json(self.manifest_path, self.manifest)


class RunArtifactReader:
    def __init__(self, run_root: str | Path) -> None:
        self.root = Path(run_root).resolve()
        self.manifest_path = _manifest_path(self.root)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Run manifest not found: {self.manifest_path}")
        self.manifest = _read_json(self.manifest_path)
        if not isinstance(self.manifest.get("artifacts"), dict):
            raise ValueError("Manifest artifacts must be an object")

    def record(self, artifact_id: str) -> dict[str, Any]:
        try:
            record = self.manifest["artifacts"][str(artifact_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown run artifact: {artifact_id}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Invalid manifest record for {artifact_id}")
        return record

    def load(self, artifact_id: str) -> dict[str, Any]:
        record = self.record(artifact_id)
        artifact = _read_json(_path_inside(self.root, str(record["path"])))
        if artifact.get("artifact_id") != artifact_id:
            raise ValueError(f"Artifact identity mismatch for {artifact_id}")
        return artifact

    def records(
        self,
        *,
        kind: Optional[str] = None,
        category: Optional[str] = None,
        model: Optional[str] = None,
        plan_id: Optional[str] = None,
        edit_method: Optional[str] = None,
    ) -> Iterator[dict[str, Any]]:
        for record in self.manifest["artifacts"].values():
            if not isinstance(record, dict):
                continue
            if kind is not None and record.get("kind") != kind:
                continue
            if category is not None and record.get("category") != category:
                continue
            if model is not None and record.get("model") != model:
                continue
            if plan_id is not None and record.get("plan_id") != plan_id:
                continue
            if edit_method is not None and record.get("edit_method") != edit_method:
                continue
            yield record

    def ref(self, artifact_id: str) -> dict[str, str]:
        record = self.record(artifact_id)
        path = _path_inside(self.root, str(record["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Artifact file not found: {path}")
        return {
            "artifact_id": artifact_id,
            "content_hash": str(record["content_hash"]),
        }
