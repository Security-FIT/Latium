import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRAPER_DIR = ROOT / "huggingface-scraper"
TYPO_UTILS_PATH = SCRAPER_DIR / "typo_utils.py"
SCRAPE_SCRIPT_PATH = SCRAPER_DIR / "scrape_hf_models.py"

if str(SCRAPER_DIR) not in sys.path:
    sys.path.insert(0, str(SCRAPER_DIR))

spec = importlib.util.spec_from_file_location("hf_typo_utils", TYPO_UTILS_PATH)
assert spec is not None and spec.loader is not None
typo_utils = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = typo_utils
spec.loader.exec_module(typo_utils)

detect_typo_pairs = typo_utils.detect_typo_pairs
normalize_token = typo_utils.normalize_token

scrape_spec = importlib.util.spec_from_file_location("hf_scrape_script", SCRAPE_SCRIPT_PATH)
assert scrape_spec is not None and scrape_spec.loader is not None
scrape_script = importlib.util.module_from_spec(scrape_spec)
sys.modules[scrape_spec.name] = scrape_script
scrape_spec.loader.exec_module(scrape_script)

load_records_from_json = scrape_script.load_records_from_json
is_model_slug_typo_like = scrape_script._is_model_slug_typo_like


def test_normalize_token_removes_separators() -> None:
    assert normalize_token("Eleuther-AI/Model_1") == "eleutheraimodel1"


def test_detect_typos_finds_transposition_variant() -> None:
    names = {
        "EleutherAI": 1000,
        "EluetherAI": 10,
        "OpenAI": 500,
    }

    findings = detect_typo_pairs(
        names,
        max_distance=2,
        min_similarity=0.85,
        min_length=5,
        max_signature_positions=12,
        max_bucket_size=100,
        min_shared_signatures=1,
    )

    assert any(
        row["canonical"] == "EleutherAI" and row["suspected_typo"] == "EluetherAI"
        for row in findings
    )


def test_detect_typos_uses_popularity_for_canonical_choice() -> None:
    names = {
        "MistralAI": 2000,
        "MistrlaAI": 20,
    }
    findings = detect_typo_pairs(
        names,
        max_distance=2,
        min_similarity=0.85,
        min_length=5,
    )

    assert len(findings) == 1
    assert findings[0]["canonical"] == "MistralAI"
    assert findings[0]["suspected_typo"] == "MistrlaAI"


def test_detect_typos_ignores_distant_names() -> None:
    names = {
        "EleutherAI": 1000,
        "CompletelyDifferent": 900,
        "AnotherVeryDifferentName": 400,
    }
    findings = detect_typo_pairs(
        names,
        max_distance=2,
        min_similarity=0.9,
        min_length=5,
    )

    assert findings == []


def test_detect_typos_ignores_format_only_differences_by_default() -> None:
    names = {
        "Efficient-Large-Model/VILA15_3b": 1000,
        "Efficient-Large-Model/VILA1.5-3b": 900,
    }
    findings = detect_typo_pairs(
        names,
        max_distance=2,
        min_similarity=0.9,
        min_length=5,
    )

    assert findings == []


def test_detect_typos_ignores_numeric_version_variants_when_enabled() -> None:
    names = {
        "Qwen/Qwen3-14B": 1000,
        "Qwen/Qwen3-4B": 900,
    }
    findings = detect_typo_pairs(
        names,
        max_distance=2,
        min_similarity=0.9,
        min_length=5,
        ignore_numeric_variants=True,
    )

    assert findings == []


def test_detect_typos_ignores_same_owner_variants_when_enabled() -> None:
    names = {
        "owner-a/my-model-alpha": 1000,
        "owner-a/my-model-alphb": 500,
    }
    findings = detect_typo_pairs(
        names,
        max_distance=2,
        min_similarity=0.85,
        min_length=5,
        ignore_same_owner_variants=True,
    )

    assert findings == []


def test_detect_typos_respects_min_popularity_ratio() -> None:
    names = {
        "EleutherAI": 100,
        "EluetherAI": 90,
    }
    findings = detect_typo_pairs(
        names,
        max_distance=2,
        min_similarity=0.85,
        min_length=5,
        min_popularity_ratio=2.0,
    )

    assert findings == []


def test_load_records_from_json_list_of_model_ids(tmp_path) -> None:
    json_path = tmp_path / "models.json"
    json_path.write_text(json.dumps(["owner-a/model-one", "owner-b/model-two"]), encoding="utf-8")

    records = load_records_from_json(str(json_path))
    assert len(records) == 2
    assert records[0]["model_id"] == "owner-a/model-one"
    assert records[0]["owner"] == "owner-a"
    assert records[0]["downloads"] == 0


def test_load_records_from_json_models_wrapper(tmp_path) -> None:
    payload = {
        "models": [
            {"id": "owner-x/model-x", "downloads": 42},
            {"model_id": "owner-y/model-y", "owner": "owner-y", "downloads": 7},
        ]
    }
    json_path = tmp_path / "records.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    records = load_records_from_json(str(json_path))
    assert len(records) == 2
    assert records[0]["model_id"] == "owner-x/model-x"
    assert records[0]["owner"] == "owner-x"
    assert records[0]["downloads"] == 42


def test_load_records_from_json_typo_report_wrapper(tmp_path) -> None:
    payload = {
        "model_id_typos": [
            {
                "canonical": "owner-a/model-a",
                "suspected_typo": "owner-b/model-b",
                "canonical_popularity": 42,
                "typo_popularity": 7,
            }
        ]
    }
    json_path = tmp_path / "typo_report.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    records = load_records_from_json(str(json_path))
    assert len(records) == 2
    assert records[0]["model_id"] == "owner-a/model-a"
    assert records[0]["downloads"] == 42
    assert records[1]["model_id"] == "owner-b/model-b"
    assert records[1]["downloads"] == 7


def test_model_slug_typo_only_accepts_repeated_letter_typo() -> None:
    assert is_model_slug_typo_like(
        "someone/lllama-7b",
        "someone/llama-7b",
        max_alpha_distance=2,
    )


def test_model_slug_typo_only_accepts_transposition_typo() -> None:
    assert is_model_slug_typo_like(
        "someone/gtp-model",
        "someone/gpt-model",
        max_alpha_distance=2,
    )


def test_model_slug_typo_only_rejects_case_and_separator_only_changes() -> None:
    assert not is_model_slug_typo_like(
        "someone/LLaMA-7B",
        "someone/llama_7b",
        max_alpha_distance=2,
    )
