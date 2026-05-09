"""Hybrid retrieval: BM25 + dense, fused via Reciprocal Rank Fusion (RRF).

Why RRF over linear weighted fusion?

* No tunable weights → no risk of overfitting to the 10 public traces
  (whose holdout siblings we can never see).
* Robust to score-scale differences between BM25 (unbounded) and
  dense cosine ([-1, 1]).
* Single hyperparameter ``k`` damps top-rank bias; the canonical
  60 from the original RRF paper is the default.

After fusion an optional metadata filter restricts results to a set of
allowed test-type codes. Done **after** fusion (not before) so the
fusion itself sees the full retrieval signal, filtering early would
discard items that BM25 ranks 11th but the agent might want.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from shl_recommender.catalog.loader import CatalogIndex
from shl_recommender.catalog.models import TestTypeCode

from .bm25 import BM25Index
from .dense import AsyncEncoder, DenseIndex

DEFAULT_RRF_K = 60


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """One ranked candidate.

    ``score`` is the fused RRF score (higher = better). It is **not**
    a probability, only useful for ordering.
    """

    doc_index: int
    score: float


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[int, float]]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists by reciprocal rank.

    For each list, item at rank r contributes ``1 / (k + r)`` (1-indexed).
    Returns a list sorted by descending fused score, tie-broken by
    ascending doc_index.
    """
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for r, (doc_idx, _score) in enumerate(ranked, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k + r)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


class HybridRetriever:
    """Wires BM25 + dense + RRF fusion + metadata filtering.

    Pure orchestration, owns no state of its own beyond references
    to the underlying indexes and the (async) query encoder used by
    the dense stage.
    """

    __slots__ = ("_bm25", "_dense", "_encoder", "_catalog", "_rrf_k")

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        catalog: CatalogIndex,
        *,
        encoder: AsyncEncoder,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if len(bm25) != len(dense) != len(catalog):
            raise ValueError(
                f"Index size mismatch: bm25={len(bm25)}, "
                f"dense={len(dense)}, catalog={len(catalog)}"
            )
        self._bm25 = bm25
        self._dense = dense
        self._encoder = encoder
        self._catalog = catalog
        self._rrf_k = rrf_k

    async def search(
        self,
        query: str,
        *,
        top_n: int = 20,
        per_index_k: int = 50,
        allowed_test_types: frozenset[TestTypeCode] | None = None,
        bm25_query: str | None = None,
    ) -> list[RetrievalHit]:
        """Retrieve top-``top_n`` candidates.

        ``bm25_query``, if given, overrides the query used for the
        lexical stage. We feed an expanded (synonym-augmented) query
        to BM25 while keeping the dense encoder on the clean query,
        empirically, expansion dilutes the bi-encoder's signal but
        helps lexical recall.

        ``allowed_test_types`` is a hard post-filter on the primary
        test type code of each candidate.
        """
        bm25_hits = self._bm25.search(
            bm25_query if bm25_query is not None else query, k=per_index_k
        )
        # Dense stage may fail (network, rate limit, etc). On failure
        # we degrade to BM25-only rather than 500 the request.
        try:
            dense_hits = await self._dense.search_async(query, per_index_k, self._encoder)
        except Exception:  # noqa: BLE001 - any encoder failure is non-fatal
            dense_hits = []
        fused = reciprocal_rank_fusion(
            [bm25_hits, dense_hits],
            k=self._rrf_k,
        )

        if allowed_test_types is not None:
            fused = [
                (i, s)
                for i, s in fused
                if self._catalog.items[i].primary_test_type in allowed_test_types
            ]

        return [RetrievalHit(doc_index=i, score=s) for i, s in fused[:top_n]]
