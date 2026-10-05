import json
import pytest
from scripts.prepare_fleet_continuation import prepare_continuation


def test_continuation_preserves_old_data_and_ranks_and_excludes_prior_attempts(tmp_path):
    manifest = tmp_path / "new-top100.json"
    manifest.write_text(json.dumps({"models": [
        {"model_id": "org/new-adapter", "rank": 1},
        {"model_id": "org/complete", "rank": 2},
        {"model_id": "org/failed", "rank": 3},
        {"model_id": "org/unsupported", "rank": 4, "selection_error": "missing weights"},
        {"model_id": "org/later", "rank": 5}]}))
    old = tmp_path / "old-run"
    for model, status in [("complete", "complete"), ("failed", "failed")]:
        path = old / "models" / model / "fleet-batches/m0000-0001/state.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"model_id": f"org/{model}", "status": status, "revision": "a" * 40}))
    before = {p: p.read_bytes() for p in old.rglob("*.json")}
    original = manifest.read_bytes()
    output = tmp_path / "next40.json"
    result = prepare_continuation(manifest, [old], output, count=2)
    assert [r["model_id"] for r in result["models"]] == ["org/new-adapter", "org/unsupported"]
    assert [r["discovery_rank"] for r in result["models"]] == [1, 4]
    assert [r["rank"] for r in result["models"]] == [1, 2]
    assert result["models"][1]["selection_error"] == "missing weights"
    assert len(result["excluded_prior_models"]) == 2
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert manifest.read_bytes() == original
    with pytest.raises(FileExistsError):
        prepare_continuation(manifest, [old], output)
