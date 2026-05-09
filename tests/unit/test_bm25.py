"""Unit tests for BM25 retrieval."""

from __future__ import annotations

import pytest

from shl_recommender.retrieval.bm25 import BM25Index, tokenize


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("OPQ32r Personality Test") == ["opq32r", "personality", "test"]


def test_tokenize_drops_stopwords() -> None:
    assert tokenize("The Java Developer in Action") == ["java", "developer", "action"]


def test_tokenize_handles_punctuation() -> None:
    assert tokenize("C# / .NET (New)") == ["c", "net", "new"]


def test_empty_corpus_rejected() -> None:
    with pytest.raises(ValueError):
        BM25Index([])


def test_search_finds_exact_name() -> None:
    docs = [
        "OPQ32r Personality Questionnaire",
        "Java 8 New Knowledge Test",
        "Verify Numerical Reasoning",
    ]
    idx = BM25Index(docs)
    hits = idx.search("OPQ32r", k=3)
    assert hits[0][0] == 0


def test_search_ignores_unmatched_tokens() -> None:
    idx = BM25Index(["python developer assessment"])
    # all stopwords / no overlap → empty
    assert idx.search("the and of", k=5) == []


def test_search_zero_k_returns_empty() -> None:
    idx = BM25Index(["any document"])
    assert idx.search("any", k=0) == []


def test_search_score_descending() -> None:
    docs = [
        "java spring boot backend developer",
        "python data scientist",
        "java junior developer test",
    ]
    idx = BM25Index(docs)
    hits = idx.search("java developer", k=3)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(i in {0, 2} for i, _ in hits)
