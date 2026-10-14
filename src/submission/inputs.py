"""Read the packaged selections; never discover models or sample replacement facts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "manifests/CSF-final"
CAMPAIGNS = {"classic": "classic-inputs.json", "finetune-base": "basecov-inputs.json", "finetune-own": "owncov-inputs.json"}


def load_campaign(name):
    if name not in CAMPAIGNS:
        raise ValueError(f"Unknown campaign: {name}")
    path = INPUTS / CAMPAIGNS[name]
    payload = json.loads(path.read_text())
    expected = {"classic": 18, "finetune-base": 790, "finetune-own": 747}[name]
    if len(payload["entries"]) != expected:
        raise ValueError(f"Incomplete frozen {name} selection: {len(payload['entries'])} != {expected}")
    return payload


def select_entries(payload, *, family=None, checkpoint=None, limit=None):
    entries = payload["entries"]
    if family:
        entries = [e for e in entries if e["family"] == family]
    if checkpoint:
        entries = [e for e in entries if e["model_id"] == checkpoint or e["model_key"] == checkpoint]
    if not entries:
        raise ValueError("No checkpoint matches the frozen selection")
    if limit is not None:
        if limit <= 0:
            raise ValueError("Checkpoint limit must be positive")
        entries = entries[:limit]
    return entries


def verify_packaged_inputs():
    from src.counterfact_selection import load_case_manifest, case_content_digest
    for record in json.loads((INPUTS / "source-files.json").read_text()):
        path = INPUTS / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Frozen input byte hash mismatch: {path}")
    classic = load_case_manifest(INPUTS / "classic-facts.json")
    finetune = load_case_manifest(INPUTS / "finetune-facts.json")
    trace = load_case_manifest(INPUTS / "trace-facts.json")
    if (classic["count"], finetune["count"], trace["count"]) != (1000, 100, 900):
        raise ValueError("Frozen fact counts changed")
    facts = json.loads((INPUTS / "classic-1000-counterfacts.json").read_text())["facts"]
    for position, fact in enumerate(facts):
        if fact["case_id"] != classic["case_ids"][position] or fact["dataset_index"] != classic["indices"][position]:
            raise ValueError(f"Convenience export identity mismatch at {position}")
        if case_content_digest(fact["counterfact"], fact["dataset_index"]) != classic["content_hashes"][position]:
            raise ValueError(f"Convenience export content mismatch at {position}")
    for campaign in CAMPAIGNS:
        payload = load_campaign(campaign)
        for entry in payload["entries"]:
            if campaign != "classic":
                position = entry["fact_position"]
                if finetune["case_ids"][position] != entry["fact_case_id"]:
                    raise ValueError(f"Assigned fact changed: {entry['model_id']}")
                if len(entry["revision"]) != 40:
                    raise ValueError(f"Unpinned checkpoint: {entry['model_id']}")
                if entry["checkpoint_type"] == "adapter" and len(entry["checkpoint"].get("adapter_base_revision", "")) != 40:
                    raise ValueError(f"Unpinned adapter base: {entry['model_id']}")
    for receipt in json.loads((INPUTS / "prefix-cache-provenance.json").read_text()):
        path = INPUTS / "prefix_cache" / receipt["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError(f"External prefix cache changed: {path}")
    if (INPUTS / "checksums.json").is_file():
        for record in json.loads((INPUTS / "checksums.json").read_text()):
            if hashlib.sha256((INPUTS / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Frozen campaign input changed: {record['path']}")
    return {"classic_facts": classic["count"], "finetune_fact_pool": finetune["count"], "trace_fact_pool": trace["count"],
            "classic_models": 18, "basecov_workflows": 790, "owncov_candidates": 747, "owncov_evaluated": 744,
            "external_prefix_pools": 3}
