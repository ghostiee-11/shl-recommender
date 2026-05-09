"""LLM-based reranker.

Single LLM call over the top-N hybrid candidates, scored against the
agent's structured query. Returns a re-ordered list of indices.

Why this lives separately from :mod:`shl_recommender.retrieval.rerank`:

* Async signature, the rest of the rerank module is sync because
  cross-encoders are sync. Mixing async into the sync Protocol would
  poison every test.
* Different input shape, the LLM reranker needs the catalog so it
  can render compact candidate descriptions; the cross-encoder needs
  only a doc-text function.

Safety:

* Strict Pydantic validation on the model's JSON output.
* Any returned index that isn't in the candidate set is dropped.
* On LLM error / invalid JSON / empty result, return hybrid order.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from shl_recommender.agent.prompts import LLM_RERANKER_PROMPT_V1
from shl_recommender.catalog.loader import CatalogIndex
from shl_recommender.llm.base import LLMClient, LLMError, LLMMessage

from .hybrid import RetrievalHit

logger = logging.getLogger(__name__)


class _RerankerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ordered_indices: list[int] = Field(default_factory=list)


def _format_candidates(catalog: CatalogIndex, candidates: Sequence[RetrievalHit]) -> str:
    """One line per candidate so the model sees a compact list it can rerank."""
    lines: list[str] = []
    for local_idx, hit in enumerate(candidates):
        item = catalog.items[hit.doc_index]
        long_types = "/".join(item.test_type_codes)
        # A single short line per item, name, types, leading description sentence.
        first_sentence = item.description.split(". ")[0][:160]
        lines.append(f"[{local_idx}] {item.name} ({long_types}), {first_sentence}")
    return "\n".join(lines)


class LLMReranker:
    """Async reranker backed by an :class:`LLMClient`.

    Stateless; safe to share. The callable interface diverges from
    the sync Reranker Protocol because we are inside async-land
    once the orchestrator is in flight.
    """

    __slots__ = ("_llm", "_catalog", "_timeout_s")

    def __init__(
        self,
        llm: LLMClient,
        catalog: CatalogIndex,
        *,
        timeout_s: float = 6.0,
    ) -> None:
        self._llm = llm
        self._catalog = catalog
        self._timeout_s = timeout_s

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalHit],
        *,
        top_k: int,
    ) -> list[RetrievalHit]:
        if not candidates or top_k <= 0:
            return []
        if len(candidates) == 1:
            return list(candidates)[:top_k]

        rendered = _format_candidates(self._catalog, candidates)
        sys_prompt = LLM_RERANKER_PROMPT_V1.replace("__TOP_K__", str(top_k))
        user_prompt = (
            f"Query:\n{query}\n\nCandidates (best-first hybrid order):\n{rendered}\n"
        )

        try:
            result = await self._llm.complete(
                [
                    LLMMessage(role="system", content=sys_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                json_mode=True,
                temperature=0.0,
                max_output_tokens=256,
                timeout_s=self._timeout_s,
            )
        except LLMError:
            logger.warning("llm_reranker_call_failed_falling_back")
            return list(candidates)[:top_k]

        try:
            payload = _RerankerOutput.model_validate(json.loads(result.text))
        except (json.JSONDecodeError, ValidationError):
            logger.warning("llm_reranker_invalid_output_falling_back")
            return list(candidates)[:top_k]

        n = len(candidates)
        seen: set[int] = set()
        ordered: list[RetrievalHit] = []
        for local_idx in payload.ordered_indices:
            if 0 <= local_idx < n and local_idx not in seen:
                ordered.append(candidates[local_idx])
                seen.add(local_idx)
            if len(ordered) >= top_k:
                break

        # If the LLM returned too few items, top up from the hybrid order
        # (de-duplicated). Never invents items, never returns < top_k
        # when the candidate pool has at least top_k items.
        for hit in candidates:
            if len(ordered) >= top_k:
                break
            if hit.doc_index not in {h.doc_index for h in ordered}:
                ordered.append(hit)

        return ordered[:top_k]
