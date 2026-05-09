"""Reranking layer.

Three implementations behind one Protocol, picked by SLA + measurement:

* :class:`PassthroughReranker`, returns hybrid order unchanged.
  Trivial, dependency-free, used as the safe fallback whenever a
  smarter reranker raises.
* :class:`CrossEncoderReranker`, pairwise scoring with a small
  ms-marco MiniLM cross-encoder. Kept opt-in only; on this corpus
  it underperformed (Phase 2 bench) due to a query/passage
  distribution mismatch. Useful baseline for future experiments.
* :class:`LLMReranker`, single LLM call (Groq/Gemini via the
  router) over the top-N hybrid candidates. The orchestrator's
  default reranker for production traffic. Strictly validated;
  any malformed response or LLM failure falls back to hybrid order.

Invariants every implementation must preserve:

* Never return a candidate that wasn't in the input → URL grounding
  is preserved by construction.
* Never return duplicates.
* On any internal failure, fall back to input order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from .hybrid import RetrievalHit

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    """Reorders a list of hybrid candidates given the user query."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalHit],
        *,
        top_k: int,
    ) -> list[RetrievalHit]:
        ...


class PassthroughReranker:
    """No-op reranker. Trivially safe; used as the fallback."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalHit],
        *,
        top_k: int,
    ) -> list[RetrievalHit]:
        return list(candidates)[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranking with hard safety fallback.

    Why a cross-encoder is the right Phase-2 lever:

    * Cross-encoders score (query, document) pairs jointly, capturing
      query-document interactions that bi-encoders (like bge-small) miss.
    * ``ms-marco-MiniLM-L-6-v2`` is 22M params (~80MB on disk), fits
      the Render free tier comfortably and reranks 20 pairs in ~200ms
      on CPU.
    * No LLM call → zero quota cost, no provider risk, deterministic.

    Constructor takes a callable that returns a ``CrossEncoder``. This
    lets tests inject a fake without importing torch.
    """

    __slots__ = ("_doc_text_fn", "_encoder", "_loaded", "_loader", "_model_name")

    def __init__(
        self,
        doc_text_fn: Callable[[int], str],
        *,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        loader: Callable[[str], object] | None = None,
    ) -> None:
        """``doc_text_fn`` maps a doc_index to the text we want the
        cross-encoder to score against. Typically ``catalog.search_docs[i]``.
        """
        self._doc_text_fn = doc_text_fn
        self._loader = loader or _default_cross_encoder_loader
        self._encoder: object | None = None
        self._loaded = False
        self._model_name = model_name

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._encoder = self._loader(self._model_name)
            self._loaded = True

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalHit],
        *,
        top_k: int,
    ) -> list[RetrievalHit]:
        if not candidates or top_k <= 0:
            return []
        try:
            self._ensure_loaded()
            assert self._encoder is not None
            pairs = [(query, self._doc_text_fn(c.doc_index)) for c in candidates]
            scores = self._encoder.predict(pairs)  # type: ignore[attr-defined]
            ranked = sorted(
                zip(candidates, scores, strict=True),
                key=lambda x: float(x[1]),
                reverse=True,
            )
            return [
                RetrievalHit(doc_index=c.doc_index, score=float(s))
                for c, s in ranked[:top_k]
            ]
        except Exception:
            # Hard safety: never let a reranker failure drop the request.
            # The caller still gets a grounded, hybrid-ordered shortlist.
            return list(candidates)[:top_k]


def _default_cross_encoder_loader(model_name: str) -> object:
    # Local import so unit tests that inject a fake loader don't need
    # to import sentence_transformers at all.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)
