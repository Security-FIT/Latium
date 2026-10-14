"""Strict artifact reads without changing source manifests or saved results."""
from pathlib import Path

from src.results import RunArtifactReader
from src.results.artifacts import content_hash, config_hash


def read_run(root):
    reader = RunArtifactReader(root)
    payloads = {}
    for record in reader.records():
        # Graph files and causal tracing may have external historical paths;
        # structural execution/capture/analysis use the manifest-owned schema.
        if record.get("kind") not in {"execution", "capture", "analysis"}:
            continue
        payload = reader.load(record["artifact_id"])
        if content_hash(payload) != record["content_hash"]:
            raise ValueError(f"Content hash mismatch: {root}/{record['path']}")
        if config_hash(payload["config"]) != record["config_hash"]:
            raise ValueError(f"Config hash mismatch: {root}/{record['path']}")
        for key in ("kind", "producer", "status", "config_hash"):
            if payload.get(key) != record.get(key):
                raise ValueError(f"Manifest {key} mismatch: {record['artifact_id']}")
        run = payload["run"]
        for key in ("model", "plan_id", "edit_method"):
            if run.get(key) != record.get(key):
                raise ValueError(f"Manifest {key} mismatch: {record['artifact_id']}")
        if payload.get("inputs", []) != record.get("inputs", []):
            raise ValueError(f"Manifest input references mismatch: {record['artifact_id']}")
        for reference in payload.get("inputs", []):
            dependency = reader.record(reference["artifact_id"])
            if dependency["content_hash"] != reference["content_hash"]:
                raise ValueError(f"Stale input: {record['artifact_id']}")
        payloads[record["artifact_id"]] = payload
    return reader, payloads


def run_payloads(root):
    reader, payloads = read_run(Path(root))
    executions = [p for p in payloads.values() if p.get("kind") == "execution"]
    analyses = [p for p in payloads.values() if p.get("kind") == "analysis" and p.get("producer") == "gram-localization"]
    return reader, executions, analyses
