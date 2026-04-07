from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import importlib
import re
from typing import Dict, List, Set, Tuple

fuzz = None
DamerauLevenshtein = None

try:
    _rapidfuzz = importlib.import_module("rapidfuzz")
    _rapidfuzz_distance = importlib.import_module("rapidfuzz.distance")
    fuzz = _rapidfuzz.fuzz
    DamerauLevenshtein = _rapidfuzz_distance.DamerauLevenshtein
except ImportError:  # pragma: no cover - fallback path only used without rapidfuzz
    pass

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
DIGIT_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?")
NON_ALPHA_RE = re.compile(r"[^a-z]+")


@dataclass
class TypoFinding:
    canonical: str
    suspected_typo: str
    canonical_popularity: float
    typo_popularity: float
    distance: int
    similarity: float
    confidence: float


def normalize_token(value: str) -> str:
    """Lower-case and remove non-alphanumeric symbols for robust matching."""
    return NON_ALNUM_RE.sub("", value.lower())


def normalize_score_text(value: str) -> str:
    """Lower-case only normalization used for actual distance/similarity scoring."""
    return value.lower().strip()


def _alpha_skeleton(value: str) -> str:
    return NON_ALPHA_RE.sub("", value.lower())


def _numeric_tokens(value: str) -> Tuple[str, ...]:
    return tuple(DIGIT_TOKEN_RE.findall(value.lower()))


def _is_format_only_difference(name_a: str, name_b: str) -> bool:
    # If alphanumeric payload is identical, only formatting/case changed.
    return normalize_token(name_a) == normalize_token(name_b)


def _is_numeric_variant_only(name_a: str, name_b: str) -> bool:
    nums_a = _numeric_tokens(name_a)
    nums_b = _numeric_tokens(name_b)
    if not nums_a and not nums_b:
        return False
    if nums_a == nums_b:
        return False

    # Same alphabetic skeleton but different numeric profile is usually
    # a model-size/version variant rather than a typo.
    return _alpha_skeleton(name_a) == _alpha_skeleton(name_b)


def _owner_from_model_id(value: str) -> str:
    if "/" not in value:
        return ""
    return value.split("/", 1)[0].lower()


def _sample_positions(length: int, max_positions: int) -> List[int]:
    if length <= max_positions:
        return list(range(length))
    if max_positions <= 1:
        return [length // 2]

    sampled = set()
    for i in range(max_positions):
        pos = round(i * (length - 1) / (max_positions - 1))
        sampled.add(pos)
    return sorted(sampled)


def _delete_signatures(token: str, max_positions: int) -> Set[str]:
    # Keep the full token so exact duplicates still land in one bucket.
    signatures = {token}
    if len(token) < 3:
        return signatures

    for pos in _sample_positions(len(token), max_positions):
        signatures.add(token[:pos] + token[pos + 1 :])
    return signatures


def _distance(token_a: str, token_b: str, max_distance: int) -> int:
    if DamerauLevenshtein is not None:
        dist = DamerauLevenshtein.distance(token_a, token_b, score_cutoff=max_distance)
        return dist if dist <= max_distance else max_distance + 1

    # Fallback to a ratio-based approximation if rapidfuzz is unavailable.
    ratio = SequenceMatcher(None, token_a, token_b).ratio()
    approx_dist = int(round((1.0 - ratio) * max(len(token_a), len(token_b))))
    return approx_dist


def _similarity(token_a: str, token_b: str) -> float:
    if fuzz is not None:
        return float(fuzz.ratio(token_a, token_b)) / 100.0
    return float(SequenceMatcher(None, token_a, token_b).ratio())


def _make_finding(
    name_a: str,
    name_b: str,
    pop_a: float,
    pop_b: float,
    distance: int,
    similarity: float,
) -> TypoFinding:
    if pop_a > pop_b:
        canonical, typo = name_a, name_b
        canonical_popularity, typo_popularity = pop_a, pop_b
    elif pop_b > pop_a:
        canonical, typo = name_b, name_a
        canonical_popularity, typo_popularity = pop_b, pop_a
    else:
        canonical, typo = sorted((name_a, name_b))
        canonical_popularity = pop_a
        typo_popularity = pop_b

    max_len = max(len(normalize_score_text(name_a)), len(normalize_score_text(name_b)), 1)
    edit_score = 1.0 - (distance / max_len)
    confidence = max(0.0, min(1.0, (edit_score + similarity) / 2.0))

    return TypoFinding(
        canonical=canonical,
        suspected_typo=typo,
        canonical_popularity=float(canonical_popularity),
        typo_popularity=float(typo_popularity),
        distance=int(distance),
        similarity=float(round(similarity, 4)),
        confidence=float(round(confidence, 4)),
    )


def detect_typo_pairs(
    popularity_by_name: Dict[str, float],
    max_distance: int = 2,
    min_similarity: float = 0.88,
    min_length: int = 5,
    max_signature_positions: int = 12,
    max_bucket_size: int = 250,
    min_shared_signatures: int = 1,
    ignore_format_only: bool = True,
    ignore_numeric_variants: bool = False,
    ignore_same_owner_variants: bool = False,
    min_popularity_ratio: float = 1.0,
) -> List[Dict[str, float]]:
    """Detect likely typo pairs using sampled deletion-signature blocking.

    The function is designed for large string sets where all-pairs comparison would be
    too expensive. We first generate candidate pairs from shared signatures and only
    then score them with Damerau-Levenshtein distance and ratio similarity.
    """
    records: List[Tuple[str, str, str, float]] = []
    for raw_name, popularity in popularity_by_name.items():
        block_token = normalize_token(raw_name)
        if len(block_token) < min_length:
            continue
        score_token = normalize_score_text(raw_name)
        records.append((raw_name, block_token, score_token, float(popularity)))

    signatures_per_index: List[Set[str]] = []
    signature_to_indices: Dict[str, List[int]] = defaultdict(list)

    for idx, (_, block_token, _, _) in enumerate(records):
        signatures = _delete_signatures(block_token, max_positions=max_signature_positions)
        signatures_per_index.append(signatures)
        for signature in signatures:
            signature_to_indices[signature].append(idx)

    findings_by_pair: Dict[Tuple[str, str], TypoFinding] = {}

    for idx, (name_a, block_a, score_a, pop_a) in enumerate(records):
        candidate_votes: Dict[int, int] = defaultdict(int)

        for signature in signatures_per_index[idx]:
            bucket = signature_to_indices.get(signature, [])
            if len(bucket) > max_bucket_size:
                continue
            for other_idx in bucket:
                if other_idx <= idx:
                    continue
                candidate_votes[other_idx] += 1

        for other_idx, votes in candidate_votes.items():
            if votes < min_shared_signatures:
                continue

            name_b, block_b, score_b, pop_b = records[other_idx]

            if ignore_same_owner_variants and _owner_from_model_id(name_a) == _owner_from_model_id(name_b):
                continue
            if ignore_format_only and _is_format_only_difference(name_a, name_b):
                continue
            if ignore_numeric_variants and _is_numeric_variant_only(name_a, name_b):
                continue

            larger = max(pop_a, pop_b)
            smaller = max(min(pop_a, pop_b), 1.0)
            if larger / smaller < min_popularity_ratio:
                continue

            if abs(len(block_a) - len(block_b)) > max_distance:
                continue

            dist = _distance(score_a, score_b, max_distance=max_distance)
            if dist <= 0:
                continue
            if dist > max_distance:
                continue

            similarity = _similarity(score_a, score_b)
            if similarity < min_similarity:
                continue

            finding = _make_finding(name_a, name_b, pop_a, pop_b, dist, similarity)
            pair_key = tuple(sorted((finding.canonical, finding.suspected_typo)))
            prev = findings_by_pair.get(pair_key)
            if prev is None or finding.confidence > prev.confidence:
                findings_by_pair[pair_key] = finding

    ordered_findings = sorted(
        findings_by_pair.values(),
        key=lambda item: (-item.confidence, item.distance, item.suspected_typo),
    )
    return [asdict(item) for item in ordered_findings]
