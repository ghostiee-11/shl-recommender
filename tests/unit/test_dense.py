"""Unit tests for the DenseIndex wrapper.

We inject a tiny deterministic encoder so the test stays fast and
network-free. The actual bge-small model is exercised end-to-end by
``eval/bench_retrieval.py``.
"""

from __future__ import annotations

import numpy as np

from shl_recommender.retrieval.dense import BGE_QUERY_PREFIX, DenseIndex


class _HashEncoder:
    """Deterministic 4-d encoder for tests.

    Each dimension counts the occurrences of one tag word in the input.
    L2-normalization happens after counting, mirroring real bge behavior
    when ``normalize_embeddings=True``.
    """

    DIM = 4
    TAGS = ("python", "java", "personality", "sales")

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        out = np.zeros((len(sentences), self.DIM), dtype=np.float32)
        for i, s in enumerate(sentences):
            low = s.lower()
            for j, tag in enumerate(self.TAGS):
                out[i, j] = float(low.count(tag))
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out


def test_dense_index_finds_nearest_by_tag() -> None:
    docs = [
        "Python data science assessment",  # idx 0 -> python
        "Java backend developer test",  # idx 1 -> java
        "Personality OPQ behaviour profile",  # idx 2 -> personality
        "Sales aptitude evaluation",  # idx 3 -> sales
    ]
    idx = DenseIndex(docs, encoder=_HashEncoder())
    hits = idx.search("python", k=4)
    assert hits[0][0] == 0
    hits2 = idx.search("personality", k=4)
    assert hits2[0][0] == 2


def test_dense_index_query_prefix_does_not_break_scoring() -> None:
    # The DenseIndex should prefix the query with BGE_QUERY_PREFIX.
    # Our HashEncoder counts tag tokens; the prefix does not contain
    # any tags, so scoring is dominated by the user's query terms.
    enc = _HashEncoder()
    idx = DenseIndex(["python python python", "java"], encoder=enc)
    hits = idx.search("python", k=2)
    assert hits[0][0] == 0
    # The prefix is non-empty.
    assert BGE_QUERY_PREFIX.strip() != ""


def test_dense_index_empty_corpus_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        DenseIndex([], encoder=_HashEncoder())


def test_dense_index_zero_k_returns_empty() -> None:
    idx = DenseIndex(["any document"], encoder=_HashEncoder())
    assert idx.search("any", k=0) == []
