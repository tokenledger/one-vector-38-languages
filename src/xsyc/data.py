"""Benchmark loading and the ID-disjoint extract / dev / eval split."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import constants as K


@dataclass(frozen=True)
class Record:
    id: str
    language: str
    category: str
    prompt: str
    sycophantic: str
    not_sycophantic: str

    @property
    def sensitivity(self) -> str:
        return K.SENSITIVITY[self.category]

    @property
    def resource_tier(self) -> str:
        return K.RESOURCE_TIER[self.language]


def load_language(lang: str, data_dir: str | Path) -> list[Record]:
    """Load one language's 4,950 records from a local JSONL."""
    if lang not in K.RESOURCE_TIER:
        raise ValueError(f"unknown language {lang!r}")
    path = Path(data_dir) / f"sycophancy_{lang}.jsonl"
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]

    recs = [Record(**r) for r in rows]
    assert len(recs) == K.N_RECORDS_PER_LANG, f"{lang}: {len(recs)} records"
    assert all(r.language == lang for r in recs), f"{lang}: language field mismatch"
    assert all(r.category in K.SENSITIVITY for r in recs), f"{lang}: unknown category"
    assert len({r.id for r in recs}) == len(recs), f"{lang}: duplicate ids"
    return recs


# --- Splits ---------------------------------------------------------------


@lru_cache(maxsize=1)
def id_splits() -> dict[str, frozenset[str]]:
    """Partition the shared ID space into extract / dev / eval.

    Category-stratified so each split spans all 33 categories. Depends only on
    the ID naming scheme and category layout, which are identical across
    languages, so the split is the same for every language and needs no data
    files. Deterministic given SPLIT_SEED.
    """
    # Categories repeat in blocks of 150 in file order; constants.CATEGORIES
    # mirrors that order.
    by_cat: dict[str, list[str]] = defaultdict(list)
    for i in range(K.N_RECORDS_PER_LANG):
        by_cat[K.CATEGORIES[i // 150]].append(K.ID_TEMPLATE.format(i))

    rng = random.Random(K.SPLIT_SEED)
    n_cat = len(K.CATEGORIES)
    per_cat_extract = K.N_EXTRACT // n_cat  # 12
    per_cat_dev = K.N_DEV // n_cat  # 9

    extract: list[str] = []
    dev: list[str] = []
    evaluation: list[str] = []
    for cat in K.CATEGORIES:
        ids = list(by_cat[cat])
        rng.shuffle(ids)
        extract += ids[:per_cat_extract]
        dev += ids[per_cat_extract : per_cat_extract + per_cat_dev]
        evaluation += ids[per_cat_extract + per_cat_dev :]

    splits = {
        "extract": frozenset(extract),
        "dev": frozenset(dev),
        "eval": frozenset(evaluation),
    }
    total = sum(len(v) for v in splits.values())
    assert total == K.N_RECORDS_PER_LANG, total
    assert not (splits["extract"] & splits["dev"])
    assert not (splits["extract"] & splits["eval"])
    assert not (splits["dev"] & splits["eval"])
    return splits


def take(records: list[Record], split: str) -> list[Record]:
    """Filter to a split.

    No reported number may come from an `extract` or `dev` id, in any language.
    """
    if split == "all":
        # Whole corpus; only for reproducing the benchmark's published baseline.
        return list(records)
    if split not in ("extract", "dev", "eval"):
        raise ValueError(f"unknown split {split!r}")
    wanted = id_splits()[split]
    out = [r for r in records if r.id in wanted]
    assert out, f"empty split {split!r}"
    return out


def stratified_sample(
    records: list[Record], n: int, seed: int = 0
) -> list[Record]:
    """Sample n records balanced across sensitivity tiers, not uniformly.

    Uniform sampling would give safety-critical (4 of 33 categories) about 12%
    of the sample; each tier gets equal weight instead.
    """
    buckets: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        buckets[r.sensitivity].append(r)

    rng = random.Random(seed)
    tiers = sorted(buckets)
    per_tier, remainder = divmod(n, len(tiers))
    out: list[Record] = []
    for i, tier in enumerate(tiers):
        want = per_tier + (1 if i < remainder else 0)
        pool = sorted(buckets[tier], key=lambda r: r.id)
        if want > len(pool):
            raise ValueError(f"asked {want} from {tier}, only {len(pool)} available")
        out += rng.sample(pool, want)
    return sorted(out, key=lambda r: r.id)
