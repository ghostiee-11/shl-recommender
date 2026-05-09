"""BM25 lexical retrieval over the catalog.

Tiny on purpose: the catalog is ~377 items, so plain BM25Okapi over
in-memory token lists is faster and simpler than any inverted-index
library. Tokenization is intentionally minimal (lowercase, alphanumeric
runs, drop a small English stopword set), heavier preprocessing
(stemming, lemmatization) hurt recall more than it helped on these
short, brand-heavy assessment names.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Deliberately small stopword list. Removing aggressive stopwords like
# "and"/"the" helps; removing domain-relevant words like "test" or
# "skills" hurts. We err on the conservative side.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "for", "to", "in", "on",
        "with", "by", "is", "are", "was", "were", "be", "been", "as",
        "at", "this", "that", "these", "those", "it", "its",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-run tokens, stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    """Light wrapper around :class:`BM25Okapi`.

    Built once at app startup; ``search`` is read-only and thread-safe.
    """

    __slots__ = ("_bm25", "_size")

    def __init__(self, docs: Iterable[str]) -> None:
        token_lists = [tokenize(d) for d in docs]
        if not token_lists:
            raise ValueError("BM25Index requires at least one document")
        self._bm25 = BM25Okapi(token_lists)
        self._size = len(token_lists)

    def __len__(self) -> int:
        return self._size

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return top-``k`` ``(doc_index, score)`` pairs, score-descending.

        Ties are broken by ascending doc index for determinism.
        """
        if k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # ``argsort`` is ascending; negate for descending.
        # We materialize into list to bound memory and apply tie-break.
        ranked = sorted(
            enumerate(scores),
            key=lambda x: (-x[1], x[0]),
        )
        return [(i, float(s)) for i, s in ranked[:k] if s > 0]
