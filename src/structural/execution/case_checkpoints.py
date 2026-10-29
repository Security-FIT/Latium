"""Durable per-case captures for resuming interrupted edit benchmarks."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from src.common.io import to_serializable
from src.results import content_hash

LOGGER = logging.getLogger(__name__)


class CaseCheckpoints:
    """Keep independently atomic cases outside the final artifact manifest."""

    def __init__(self, root: Path, *, identity: dict[str, Any]) -> None:
        self.identity_hash = content_hash(identity)
        self.root = root / self.identity_hash
        self.root.mkdir(parents=True, exist_ok=True)
        self.capture_names = set(identity["captures"])

    def _path(self, case: Mapping[str, Any]) -> Path:
        return self.root / f"{content_hash(dict(case))}.json"

    def load(self, case: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self._path(case)
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            digest = record.pop("content_hash")
            if digest != content_hash(record):
                raise ValueError("content hash mismatch")
            if record["identity_hash"] != self.identity_hash or record["case_hash"] != content_hash(dict(case)):
                raise ValueError("checkpoint identity mismatch")
            execution = record["execution"]
            captures = record["captures"]
            case_id = str(case["case_id"])
            if execution["case_id"] != case_id or set(captures) != self.capture_names:
                raise ValueError("checkpoint case/capture mismatch")
            if any(value["case_id"] != case_id for value in captures.values()):
                raise ValueError("capture case ID mismatch")
            # Failed or incomplete captures are retained as evidence, then retried.
            if execution["status"] != "complete" or any(
                value["status"] != "complete" for value in captures.values()
            ):
                return None
            return record
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            LOGGER.warning("Ignoring invalid case checkpoint %s: %s", path, exc)
            return None

    def save(
        self,
        case: Mapping[str, Any],
        execution: dict[str, Any],
        captures: dict[str, dict[str, Any]],
    ) -> None:
        record = to_serializable({
            "format": "structural-case-checkpoint-v1",
            "identity_hash": self.identity_hash,
            "case_hash": content_hash(dict(case)),
            "execution": execution,
            "captures": captures,
        })
        record["content_hash"] = content_hash(record)
        target = self._path(case)
        descriptor, name = tempfile.mkstemp(prefix=".case-", suffix=".tmp", dir=self.root)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
