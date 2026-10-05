"""Fixed shared facts remain disjoint from tracing and survive cohort slicing."""
import json

import pytest

from jobs.prepare_shared_fleet_facts import prepare_shared_facts
from scripts.prepare_fleet_continuation import prepare_continuation
from src.counterfact_selection import generate_random_case_manifest, load_case_manifest, write_case_manifest


def test_shared_pool_preserves_random_source_and_checkpoint_assignments(tmp_path):
    facts = [{"case_id": i, "requested_rewrite": {"subject": f"S{i}", "prompt": "{} lives in",
              "target_true": {"str": "Rome"}, "target_new": {"str": "Paris"}}} for i in range(120)]
    source = tmp_path / "source.json"
    original = generate_random_case_manifest(count=120, seed=42, dataset_name="fake", split="train", dataset=facts)
    write_case_manifest(source, original)
    discovery = tmp_path / "discovery"
    for family, count in [("one", 100), ("two", 3)]:
        path = discovery / family / "checkpoints.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"models": [{"model_id": f"{family}/m{i}", "rank": i + 1}
                                               for i in range(count)]}))
    before = {p: p.read_bytes() for p in discovery.rglob("*.json")}
    output = tmp_path / "shared"
    assert prepare_shared_facts(source, discovery, output) == {"rome_facts": 100, "trace_facts": 20, "families": 2}
    rome, tracing = [load_case_manifest(output / name) for name in ("rome-facts.json", "trace-facts.json")]
    assert rome["case_ids"] == original["case_ids"][:100]
    assert tracing["indices"] == original["indices"][100:]
    assert rome["content_hashes"] == original["content_hashes"][:100]
    assert not set(rome["case_ids"]) & set(tracing["case_ids"])
    assert not set(rome["indices"]) & set(tracing["indices"])
    assert rome["source_manifest_hash"] == tracing["source_manifest_hash"] == original["manifest_hash"]
    for family, count in [("one", 100), ("two", 3)]:
        models = json.loads((output / "fleets" / family / "checkpoints.json").read_text())["models"]
        assert [r["fact_position"] for r in models] == list(range(count))
        assert [r["fact_case_id"] for r in models] == rome["case_ids"][:count]
    assert all(p.read_bytes() == contents for p, contents in before.items())
    with pytest.raises(FileExistsError):
        prepare_shared_facts(source, discovery, output)
    with pytest.raises(ValueError, match="cover every"):
        prepare_shared_facts(source, discovery, tmp_path / "too-small", count=2)
    assert not (tmp_path / "too-small").exists()

    old = tmp_path / "old"
    state = old / "models/m0/fleet-batches/m0000-0001/state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"model_id": "one/m0", "status": "complete"}))
    selected = prepare_continuation(output / "fleets/one/checkpoints.json", [old], tmp_path / "next.json", count=2)
    assert [r["fact_position"] for r in selected["models"]] == [1, 2]
    assert [r["fact_case_id"] for r in selected["models"]] == rome["case_ids"][1:3]
