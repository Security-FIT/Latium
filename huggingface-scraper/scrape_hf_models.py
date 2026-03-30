#!/usr/bin/env python3
"""Scrape all public Hugging Face model IDs and detect likely typo variants."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
import importlib
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Tuple

from huggingface_hub import HfApi

from typo_utils import detect_typo_pairs

LOGGER = logging.getLogger(__name__)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
NON_ALPHA_RE = re.compile(r"[^a-z]+")

try:
    _rapidfuzz_distance = importlib.import_module("rapidfuzz.distance")
    RF_DamerauLevenshtein = _rapidfuzz_distance.DamerauLevenshtein
except ImportError:  # pragma: no cover
    RF_DamerauLevenshtein = None


def _model_id_from_info(model_info) -> str | None:
    model_id = getattr(model_info, "id", None) or getattr(model_info, "modelId", None)
    if not model_id:
        return None
    return str(model_id)


def _owner_from_model_id(model_id: str) -> str:
    return model_id.split("/", 1)[0] if "/" in model_id else model_id


def _model_slug_from_model_id(model_id: str) -> str:
    return model_id.split("/", 1)[1] if "/" in model_id else model_id


def _normalize_slug_for_format(value: str) -> str:
    return NON_ALNUM_RE.sub("", value.lower())


def _alpha_only(value: str) -> str:
    return NON_ALPHA_RE.sub("", value.lower())


def _alpha_distance(a: str, b: str, max_distance: int) -> int:
    if RF_DamerauLevenshtein is not None:
        dist = RF_DamerauLevenshtein.distance(a, b, score_cutoff=max_distance)
        return dist if dist <= max_distance else max_distance + 1

    ratio = SequenceMatcher(None, a, b).ratio()
    approx_dist = int(round((1.0 - ratio) * max(len(a), len(b))))
    return approx_dist


def _is_model_slug_typo_like(suspected_model_id: str, canonical_model_id: str, max_alpha_distance: int) -> bool:
    suspected_slug = _model_slug_from_model_id(suspected_model_id)
    canonical_slug = _model_slug_from_model_id(canonical_model_id)

    # Ignore case/separator/punctuation-only differences.
    if _normalize_slug_for_format(suspected_slug) == _normalize_slug_for_format(canonical_slug):
        return False

    suspected_alpha = _alpha_only(suspected_slug)
    canonical_alpha = _alpha_only(canonical_slug)
    if not suspected_alpha or not canonical_alpha:
        return False

    return _alpha_distance(suspected_alpha, canonical_alpha, max_distance=max_alpha_distance) <= max_alpha_distance


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def _record_from_json_item(item: Any) -> Dict[str, object] | None:
    if isinstance(item, str):
        model_id = item.strip()
        if not model_id:
            return None
        return {
            "model_id": model_id,
            "owner": _owner_from_model_id(model_id),
            "downloads": 0,
        }

    if isinstance(item, dict):
        model_id = (
            item.get("model_id")
            or item.get("id")
            or item.get("modelId")
            or item.get("name")
        )
        if not model_id:
            return None
        model_id = str(model_id).strip()
        if not model_id:
            return None

        owner = str(item.get("owner") or _owner_from_model_id(model_id))
        downloads_raw = item.get("downloads", 0)
        try:
            downloads = int(downloads_raw)
        except (TypeError, ValueError):
            downloads = 0

        return {
            "model_id": model_id,
            "owner": owner,
            "downloads": downloads,
        }

    return None


def _extract_json_items(payload: Any) -> List[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("models"), list):
            return payload["models"]
        if isinstance(payload.get("records"), list):
            return payload["records"]
        if isinstance(payload.get("model_id_typos"), list):
            reconstructed: List[Dict[str, Any]] = []
            for row in payload["model_id_typos"]:
                canonical = row.get("canonical")
                suspected = row.get("suspected_typo")
                if canonical:
                    reconstructed.append({
                        "model_id": canonical,
                        "downloads": row.get("canonical_popularity", 0),
                    })
                if suspected:
                    reconstructed.append({
                        "model_id": suspected,
                        "downloads": row.get("typo_popularity", 0),
                    })
            LOGGER.warning(
                "Input JSON looks like a typo report; reconstructed %s model records from model_id_typos.",
                f"{len(reconstructed):,}",
            )
            return reconstructed
    raise ValueError("Unsupported JSON shape. Expected a list or an object with 'models'/'records' list.")


def load_records_from_json(path: str) -> List[Dict[str, object]]:
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    items = _extract_json_items(payload)
    records: List[Dict[str, object]] = []
    for item in items:
        parsed = _record_from_json_item(item)
        if parsed is not None:
            records.append(parsed)

    LOGGER.info("Loaded %s model records from %s", f"{len(records):,}", json_path)
    return records


def save_records_to_json(records: List[Dict[str, object]], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"models": records}, handle, indent=2)
    LOGGER.info("Saved %s model records to %s", f"{len(records):,}", output_path)


def crawl_huggingface_models(
    max_models: int | None = None,
    progress_every: int = 5000,
    throttle_every_models: int = 200000,
    throttle_sleep_seconds: float = 30.0,
    rate_limit_sleep_seconds: float = 30.0,
    rate_limit_max_retries: int = 20,
) -> Tuple[List[Dict[str, object]], bool]:
    """Return model records from a full public crawl of Hugging Face model IDs."""
    api = HfApi()
    model_iterator = iter(api.list_models(sort="downloads", direction=-1, full=False))

    records: List[Dict[str, object]] = []
    idx = 0
    stopped_by_rate_limit = False
    rate_limit_retry_count = 0

    while True:
        try:
            model_info = next(model_iterator)
            rate_limit_retry_count = 0
        except StopIteration:
            break
        except Exception as exc:  # pragma: no cover - depends on remote API state
            if _is_rate_limit_error(exc):
                rate_limit_retry_count += 1
                if rate_limit_retry_count > rate_limit_max_retries:
                    LOGGER.warning(
                        "Rate limit retries exceeded (%d). Returning partial crawl results after %s models.",
                        rate_limit_max_retries,
                        f"{len(records):,}",
                    )
                    stopped_by_rate_limit = True
                    break
                LOGGER.warning(
                    "Rate limit hit after %s models. Sleeping %.1fs and retrying (%d/%d).",
                    f"{len(records):,}",
                    rate_limit_sleep_seconds,
                    rate_limit_retry_count,
                    rate_limit_max_retries,
                )
                time.sleep(rate_limit_sleep_seconds)
                continue
            raise

        idx += 1
        model_id = _model_id_from_info(model_info)
        if model_id is None:
            continue

        owner = _owner_from_model_id(model_id)
        downloads = getattr(model_info, "downloads", 0) or 0

        records.append(
            {
                "model_id": model_id,
                "owner": owner,
                "downloads": int(downloads),
            }
        )

        if idx % progress_every == 0:
            LOGGER.info("Crawled %s models so far", f"{idx:,}")

        if throttle_every_models > 0 and len(records) > 0 and len(records) % throttle_every_models == 0:
            LOGGER.info(
                "Throttle pause at %s models for %.1f seconds",
                f"{len(records):,}",
                throttle_sleep_seconds,
            )
            time.sleep(throttle_sleep_seconds)

        if max_models is not None and len(records) >= max_models:
            LOGGER.info("Reached --max-models=%d; stopping crawl early", max_models)
            break

    LOGGER.info("Finished crawl: %s models", f"{len(records):,}")
    return records, stopped_by_rate_limit


def build_owner_popularity(records: List[Dict[str, object]]) -> Dict[str, float]:
    owner_downloads: Dict[str, float] = defaultdict(float)
    for row in records:
        owner = str(row["owner"])
        owner_downloads[owner] += float(row["downloads"])
    return dict(owner_downloads)


def build_model_popularity(records: List[Dict[str, object]]) -> Dict[str, float]:
    model_downloads: Dict[str, float] = {}
    for row in records:
        model_id = str(row["model_id"])
        downloads = float(row["downloads"])
        existing = model_downloads.get(model_id)
        if existing is None or downloads > existing:
            model_downloads[model_id] = downloads
    return model_downloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Hugging Face model IDs and detect likely typos.",
    )
    parser.add_argument("--output-dir", default="./analysis_out", help="Directory for JSON reports.")
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional explicit output path. If omitted, a timestamped file is created in --output-dir.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs; omit for full public crawl.",
    )
    parser.add_argument("--progress-every", type=int, default=5000, help="Log crawl progress every N models.")
    parser.add_argument(
        "--input-models-json",
        default=None,
        help="Load model records from JSON and skip Hugging Face crawling.",
    )
    parser.add_argument(
        "--save-models-json",
        default=None,
        help="Save crawled model records to JSON for offline typo analysis.",
    )
    parser.add_argument(
        "--scrape-only",
        action="store_true",
        help="Only crawl and save model records. Skip typo detection.",
    )
    parser.add_argument(
        "--throttle-every-models",
        type=int,
        default=200000,
        help="Pause crawling after every N models (set 0 to disable).",
    )
    parser.add_argument(
        "--throttle-sleep-seconds",
        type=float,
        default=30.0,
        help="How long to sleep during each throttle pause.",
    )
    parser.add_argument(
        "--rate-limit-sleep-seconds",
        type=float,
        default=30.0,
        help="How long to sleep before retrying after HTTP 429/rate limit errors.",
    )
    parser.add_argument(
        "--rate-limit-max-retries",
        type=int,
        default=20,
        help="Maximum number of consecutive rate-limit retries before stopping with partial results.",
    )
    parser.add_argument(
        "--skip-owner-typos",
        action="store_true",
        help="Skip owner namespace typo detection and only detect model ID typos.",
    )

    parser.add_argument("--owner-max-distance", type=int, default=2)
    parser.add_argument("--owner-min-similarity", type=float, default=0.88)
    parser.add_argument(
        "--owner-min-popularity-ratio",
        type=float,
        default=2.5,
        help="Require canonical owner popularity to be at least this many times higher than suspected typo.",
    )
    parser.add_argument("--model-max-distance", type=int, default=2)
    parser.add_argument("--model-min-similarity", type=float, default=0.92)
    parser.add_argument(
        "--model-min-popularity-ratio",
        type=float,
        default=1.5,
        help="Require canonical model popularity to be at least this many times higher than suspected typo.",
    )
    parser.add_argument("--min-length", type=int, default=5)
    parser.add_argument("--max-signature-positions", type=int, default=12)
    parser.add_argument("--max-bucket-size", type=int, default=250)
    parser.add_argument("--min-shared-signatures", type=int, default=1)
    parser.add_argument(
        "--owner-ignore-format-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore owner pairs that differ only by separators/case.",
    )
    parser.add_argument(
        "--model-ignore-format-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore model ID pairs that differ only by separators/case.",
    )
    parser.add_argument(
        "--owner-ignore-numeric-variants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore owner pairs that differ only by numeric tokens.",
    )
    parser.add_argument(
        "--model-ignore-numeric-variants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore model ID pairs that differ only by numeric/version tokens.",
    )
    parser.add_argument(
        "--model-ignore-same-owner-variants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore model ID pairs from the same owner namespace.",
    )
    parser.add_argument(
        "--model-slug-typo-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep only model-ID pairs where the model slug itself has a letter-level typo (e.g., gpt->gptt, gtp->gpt).",
    )
    parser.add_argument(
        "--model-slug-max-alpha-distance",
        type=int,
        default=2,
        help="Maximum alpha-only edit distance for --model-slug-typo-only filtering.",
    )

    parser.add_argument("--top-k", type=int, default=25, help="How many top examples to print in terminal summary.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    if args.scrape_only and args.input_models_json:
        LOGGER.info("--scrape-only with --input-models-json: no crawl will occur, typo detection remains skipped")

    stopped_by_rate_limit = False
    if args.input_models_json:
        if args.save_models_json:
            LOGGER.info("Ignoring --save-models-json because --input-models-json is set")
        LOGGER.info("Loading model records from JSON: %s", args.input_models_json)
        records = load_records_from_json(args.input_models_json)
        data_source = "json"
    else:
        LOGGER.info("Starting Hugging Face model crawl")
        records, stopped_by_rate_limit = crawl_huggingface_models(
            max_models=args.max_models,
            progress_every=args.progress_every,
            throttle_every_models=args.throttle_every_models,
            throttle_sleep_seconds=args.throttle_sleep_seconds,
            rate_limit_sleep_seconds=args.rate_limit_sleep_seconds,
            rate_limit_max_retries=args.rate_limit_max_retries,
        )
        data_source = "crawl"

    model_records_path: str | None = args.save_models_json
    if model_records_path is None and args.scrape_only:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_records_path = str(output_dir / f"huggingface_models_{ts}.json")

    if model_records_path and data_source == "crawl":
        save_records_to_json(records, model_records_path)

    owner_popularity = build_owner_popularity(records)
    model_popularity = build_model_popularity(records)

    if args.scrape_only:
        LOGGER.info("Scrape-only mode enabled: skipping typo detection")
        owner_typos = []
        model_typos = []
    elif args.skip_owner_typos:
        LOGGER.info("Skipping owner typo detection (--skip-owner-typos)")
        owner_typos = []
    else:
        LOGGER.info("Detecting owner typos across %s owners", f"{len(owner_popularity):,}")
        owner_typos = detect_typo_pairs(
            owner_popularity,
            max_distance=args.owner_max_distance,
            min_similarity=args.owner_min_similarity,
            min_length=args.min_length,
            max_signature_positions=args.max_signature_positions,
            max_bucket_size=args.max_bucket_size,
            min_shared_signatures=args.min_shared_signatures,
            ignore_format_only=args.owner_ignore_format_only,
            ignore_numeric_variants=args.owner_ignore_numeric_variants,
            min_popularity_ratio=args.owner_min_popularity_ratio,
        )

    if not args.scrape_only:
        LOGGER.info("Detecting full model-id typos across %s model IDs", f"{len(model_popularity):,}")
        model_typos = detect_typo_pairs(
            model_popularity,
            max_distance=args.model_max_distance,
            min_similarity=args.model_min_similarity,
            min_length=args.min_length,
            max_signature_positions=args.max_signature_positions,
            max_bucket_size=args.max_bucket_size,
            min_shared_signatures=args.min_shared_signatures,
            ignore_format_only=args.model_ignore_format_only,
            ignore_numeric_variants=args.model_ignore_numeric_variants,
            ignore_same_owner_variants=args.model_ignore_same_owner_variants,
            min_popularity_ratio=args.model_min_popularity_ratio,
        )

        if args.model_slug_typo_only:
            before_count = len(model_typos)
            model_typos = [
                row for row in model_typos
                if _is_model_slug_typo_like(
                    row["suspected_typo"],
                    row["canonical"],
                    max_alpha_distance=args.model_slug_max_alpha_distance,
                )
            ]
            LOGGER.info(
                "Applied --model-slug-typo-only filter: %s -> %s pairs",
                f"{before_count:,}",
                f"{len(model_typos):,}",
            )

    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "data_source": data_source,
            "stopped_by_rate_limit": stopped_by_rate_limit,
            "total_models": len(records),
            "unique_owners": len(owner_popularity),
            "unique_model_ids": len(model_popularity),
            "parameters": {
                "max_models": args.max_models,
                "input_models_json": args.input_models_json,
                "save_models_json": model_records_path,
                "scrape_only": args.scrape_only,
                "throttle_every_models": args.throttle_every_models,
                "throttle_sleep_seconds": args.throttle_sleep_seconds,
                "rate_limit_sleep_seconds": args.rate_limit_sleep_seconds,
                "rate_limit_max_retries": args.rate_limit_max_retries,
                "skip_owner_typos": args.skip_owner_typos,
                "owner_max_distance": args.owner_max_distance,
                "owner_min_similarity": args.owner_min_similarity,
                "owner_min_popularity_ratio": args.owner_min_popularity_ratio,
                "model_max_distance": args.model_max_distance,
                "model_min_similarity": args.model_min_similarity,
                "model_min_popularity_ratio": args.model_min_popularity_ratio,
                "min_length": args.min_length,
                "max_signature_positions": args.max_signature_positions,
                "max_bucket_size": args.max_bucket_size,
                "min_shared_signatures": args.min_shared_signatures,
                "owner_ignore_format_only": args.owner_ignore_format_only,
                "model_ignore_format_only": args.model_ignore_format_only,
                "owner_ignore_numeric_variants": args.owner_ignore_numeric_variants,
                "model_ignore_numeric_variants": args.model_ignore_numeric_variants,
                "model_ignore_same_owner_variants": args.model_ignore_same_owner_variants,
                "model_slug_typo_only": args.model_slug_typo_only,
                "model_slug_max_alpha_distance": args.model_slug_max_alpha_distance,
            },
        },
        "owner_typos": owner_typos,
        "model_id_typos": model_typos,
    }

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = output_dir / f"huggingface_typos_{ts}.json"

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=" * 72)
    print("Hugging Face Typo Scanner Summary")
    print("=" * 72)
    print(f"Data source:          {data_source}")
    if stopped_by_rate_limit:
        print("Rate limit:           hit (partial crawl used)")
    if model_records_path:
        print(f"Models JSON:          {model_records_path}")
    print(f"Total models scanned: {len(records):,}")
    print(f"Unique owners:        {len(owner_popularity):,}")
    print(f"Unique model IDs:     {len(model_popularity):,}")
    print(f"Owner typo pairs:     {len(owner_typos):,}")
    print(f"Model ID typo pairs:  {len(model_typos):,}")
    print(f"Report path:          {output_path}")
    print("=" * 72)

    if args.scrape_only:
        print("Scrape-only mode: typo detection skipped.")
    elif args.skip_owner_typos:
        print("Owner typo detection skipped.")
    else:
        print(f"Top {args.top_k} owner typo examples:")
        for idx, row in enumerate(owner_typos[: args.top_k], start=1):
            print(
                f"{idx:>3}. {row['suspected_typo']} -> {row['canonical']} "
                f"(distance={row['distance']}, similarity={row['similarity']}, confidence={row['confidence']})"
            )

    if not args.scrape_only:
        print(f"\nTop {args.top_k} model ID typo examples:")
        for idx, row in enumerate(model_typos[: args.top_k], start=1):
            print(
                f"{idx:>3}. {row['suspected_typo']} -> {row['canonical']} "
                f"(distance={row['distance']}, similarity={row['similarity']}, confidence={row['confidence']})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
