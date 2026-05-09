"""Unit tests for the LLM reranker (no real LLM required)."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from shl_recommender.llm.base import LLMError, LLMMessage, LLMResult
from shl_recommender.retrieval.hybrid import RetrievalHit
from shl_recommender.retrieval.llm_rerank import LLMReranker


class _StubCatalog:
    """Minimal duck-typed catalog with the attrs the reranker uses."""

    def __init__(self, n: int) -> None:
        self.items = [_StubItem(i) for i in range(n)]
        self.search_docs = tuple(f"doc-{i}" for i in range(n))


class _StubItem:
    def __init__(self, i: int) -> None:
        self.name = f"Item-{i}"
        self.test_type_codes = ["K"]
        self.description = f"Description for item {i}. Extra text here."


class _StubLLM:
    name = "stub"

    def __init__(self, payload: object | Exception) -> None:
        self._payload = payload

    async def complete(self, messages: Sequence[LLMMessage], **_: object) -> LLMResult:
        if isinstance(self._payload, Exception):
            raise self._payload
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return LLMResult(text=text, model="stub", provider="stub", latency_ms=1)


def _hits(*indices: int) -> list[RetrievalHit]:
    return [RetrievalHit(doc_index=i, score=1.0 / (n + 1)) for n, i in enumerate(indices)]


@pytest.mark.asyncio
async def test_reranker_reorders_per_llm_output() -> None:
    cat = _StubCatalog(5)
    cands = _hits(0, 1, 2, 3, 4)
    rr = LLMReranker(_StubLLM({"ordered_indices": [3, 1, 0]}), cat)  # type: ignore[arg-type]
    out = await rr.rerank("q", cands, top_k=3)
    assert [h.doc_index for h in out] == [3, 1, 0]


@pytest.mark.asyncio
async def test_reranker_drops_invalid_indices() -> None:
    cat = _StubCatalog(3)
    cands = _hits(0, 1, 2)
    # 99 is out of range; should be dropped, not invented.
    rr = LLMReranker(_StubLLM({"ordered_indices": [2, 99, 0]}), cat)  # type: ignore[arg-type]
    out = await rr.rerank("q", cands, top_k=3)
    assert [h.doc_index for h in out] == [2, 0, 1]  # tops up from hybrid order


@pytest.mark.asyncio
async def test_reranker_dedupes() -> None:
    cat = _StubCatalog(3)
    cands = _hits(0, 1, 2)
    rr = LLMReranker(_StubLLM({"ordered_indices": [1, 1, 0, 1]}), cat)  # type: ignore[arg-type]
    out = await rr.rerank("q", cands, top_k=3)
    assert [h.doc_index for h in out] == [1, 0, 2]


@pytest.mark.asyncio
async def test_reranker_falls_back_on_invalid_json() -> None:
    cat = _StubCatalog(3)
    cands = _hits(0, 1, 2)
    rr = LLMReranker(_StubLLM("not json"), cat)  # type: ignore[arg-type]
    out = await rr.rerank("q", cands, top_k=3)
    assert [h.doc_index for h in out] == [0, 1, 2]


@pytest.mark.asyncio
async def test_reranker_falls_back_on_llm_error() -> None:
    cat = _StubCatalog(3)
    cands = _hits(0, 1, 2)
    rr = LLMReranker(_StubLLM(LLMError("down")), cat)  # type: ignore[arg-type]
    out = await rr.rerank("q", cands, top_k=3)
    assert [h.doc_index for h in out] == [0, 1, 2]


@pytest.mark.asyncio
async def test_reranker_handles_empty_inputs() -> None:
    cat = _StubCatalog(3)
    rr = LLMReranker(_StubLLM({"ordered_indices": []}), cat)  # type: ignore[arg-type]
    assert await rr.rerank("q", [], top_k=5) == []
    assert await rr.rerank("q", _hits(0), top_k=0) == []


@pytest.mark.asyncio
async def test_reranker_short_circuits_on_single_candidate() -> None:
    cat = _StubCatalog(3)
    cands = _hits(2)
    # No LLM call expected; supplying a deliberately-failing stub.
    rr = LLMReranker(_StubLLM(LLMError("must not be called")), cat)  # type: ignore[arg-type]
    out = await rr.rerank("q", cands, top_k=3)
    assert [h.doc_index for h in out] == [2]
