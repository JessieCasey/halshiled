"""Merging several rankings into one."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

RRF_OFFSET = 60  # the constant from Cormack et al. (2009); damps the weight of top ranks


def reciprocal_rank_fusion(rankings: Iterable[Sequence[str]], limit: int, offset: int = RRF_OFFSET) -> list[str]:
    """Return the `limit` items with the highest summed 1 / (offset + rank) over all rankings.

    Items found by several rankings rise; ties keep first-seen order.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), start=1):
            scores[item] += 1.0 / (offset + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
