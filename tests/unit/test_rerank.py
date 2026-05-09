"""Unit tests for the reranker scaffolding."""

from __future__ import annotations

from collections.abc import Sequence

from shl_recommender.retrieval.hybrid import RetrievalHit
from shl_recommender.retrieval.rerank import (
    CrossEncoderReranker,
    PassthroughReranker,
)


def test_passthrough_preserves_order() -> None:
    cands = [
        RetrievalHit(doc_index=2, score=0.9),
        RetrievalHit(doc_index=5, score=0.8),
        RetrievalHit(doc_index=1, score=0.7),
    ]
    out = PassthroughReranker().rerank("anything", cands, top_k=10)
    assert [c.doc_index for c in out] == [2, 5, 1]


def test_passthrough_truncates_to_top_k() -> None:
    cands = [RetrievalHit(doc_index=i, score=1.0 / (i + 1)) for i in range(10)]
    out = PassthroughReranker().rerank("q", cands, top_k=3)
    assert len(out) == 3
    assert [c.doc_index for c in out] == [0, 1, 2]


def test_passthrough_empty_candidates() -> None:
    assert PassthroughReranker().rerank("q", [], top_k=5) == []


# ---------- CrossEncoderReranker (with injected fake) ----------


class _FakeCrossEncoder:
    """Returns a hand-crafted score per (query, doc_text) pair."""

    def __init__(self, scoring: dict[str, float]) -> None:
        self._scoring = scoring

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [self._scoring.get(doc, 0.0) for _q, doc in pairs]


def _make_loader(scoring: dict[str, float]) -> object:
    def loader(_name: str) -> object:
        return _FakeCrossEncoder(scoring)

    return loader


def test_cross_encoder_reorders_by_pair_score() -> None:
    docs = {0: "doc-zero", 1: "doc-one", 2: "doc-two"}
    cands = [
        RetrievalHit(doc_index=0, score=0.5),
        RetrievalHit(doc_index=1, score=0.5),
        RetrievalHit(doc_index=2, score=0.5),
    ]
    rr = CrossEncoderReranker(
        doc_text_fn=docs.__getitem__,
        loader=_make_loader({"doc-one": 0.99, "doc-zero": 0.5, "doc-two": 0.1}),
    )
    out = rr.rerank("q", cands, top_k=3)
    assert [c.doc_index for c in out] == [1, 0, 2]


def test_cross_encoder_truncates_to_top_k() -> None:
    docs = {i: f"d-{i}" for i in range(5)}
    cands = [RetrievalHit(doc_index=i, score=0.5) for i in range(5)]
    scoring = {f"d-{i}": float(i) for i in range(5)}  # higher i = higher score
    rr = CrossEncoderReranker(doc_text_fn=docs.__getitem__, loader=_make_loader(scoring))
    out = rr.rerank("q", cands, top_k=2)
    assert [c.doc_index for c in out] == [4, 3]


def test_cross_encoder_falls_back_on_failure() -> None:
    def boom_loader(_name: str) -> object:
        raise RuntimeError("model load failed")

    cands = [
        RetrievalHit(doc_index=7, score=0.9),
        RetrievalHit(doc_index=3, score=0.8),
    ]
    rr = CrossEncoderReranker(doc_text_fn=lambda _i: "x", loader=boom_loader)
    out = rr.rerank("q", cands, top_k=10)
    # Falls back to hybrid order, never raises, never invents items.
    assert [c.doc_index for c in out] == [7, 3]


def test_cross_encoder_empty_inputs() -> None:
    rr = CrossEncoderReranker(doc_text_fn=lambda _i: "x", loader=_make_loader({}))
    assert rr.rerank("q", [], top_k=5) == []
    cands = [RetrievalHit(doc_index=0, score=0.5)]
    assert rr.rerank("q", cands, top_k=0) == []
