"""Unit tests for the LLM-driven reply composer.

The composer must:
* Return the LLM's text on success (lightly cleaned).
* Fall back to the deterministic ``fallback`` arg on any LLM error.
* Never raise.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from shl_recommender.agent.composer import compose_reply
from shl_recommender.catalog.models import Assessment
from shl_recommender.llm.base import LLMError, LLMMessage, LLMResult


class _StubLLM:
    name = "stub"

    def __init__(self, payload: str | Exception) -> None:
        self._payload = payload
        self.received_user_prompt: str | None = None

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        **_: object,
    ) -> LLMResult:
        # Capture the user-facing prompt so tests can assert on it.
        self.received_user_prompt = next(
            (m.content for m in messages if m.role == "user"), None
        )
        if isinstance(self._payload, Exception):
            raise self._payload
        return LLMResult(text=self._payload, model="stub", provider="stub", latency_ms=0)


def _item(name: str, description: str = "Measures something useful.") -> Assessment:
    return Assessment.model_validate(
        {
            "entity_id": "1",
            "name": name,
            "link": "https://www.shl.com/x/",
            "scraped_at": "2026-05-08",
            "status": "ok",
            "remote": "yes",
            "adaptive": "no",
            "description": description,
            "keys": ["Knowledge & Skills"],
        }
    )


@pytest.mark.asyncio
async def test_composer_returns_llm_text_on_success() -> None:
    llm = _StubLLM("Here is a sharp 2-sentence reply.")
    out = await compose_reply(
        llm,
        action="recommend",
        user_message="hire a Java dev",
        items=[_item("Java 8 (New)")],
        fallback="fallback text",
    )
    assert out == "Here is a sharp 2-sentence reply."


@pytest.mark.asyncio
async def test_composer_falls_back_on_llm_error() -> None:
    llm = _StubLLM(LLMError("groq down"))
    out = await compose_reply(
        llm,
        action="recommend",
        user_message="x",
        items=[_item("X")],
        fallback="DET FALLBACK",
    )
    assert out == "DET FALLBACK"


@pytest.mark.asyncio
async def test_composer_falls_back_on_empty_llm_text() -> None:
    llm = _StubLLM("   \n  ")
    out = await compose_reply(
        llm,
        action="compare",
        user_message="a vs b",
        items=[_item("A"), _item("B")],
        fallback="DET FALLBACK",
    )
    assert out == "DET FALLBACK"


@pytest.mark.asyncio
async def test_composer_strips_common_prefaces() -> None:
    llm = _StubLLM("Reply: this is the reply.")
    out = await compose_reply(
        llm,
        action="recommend",
        user_message="x",
        items=[_item("X")],
        fallback="...",
    )
    assert out == "this is the reply."


@pytest.mark.asyncio
async def test_composer_short_circuits_on_no_items() -> None:
    # Should never call the LLM when there's nothing to talk about.
    llm = _StubLLM(LLMError("must not be called"))
    out = await compose_reply(
        llm,
        action="recommend",
        user_message="x",
        items=[],
        fallback="DET FALLBACK",
    )
    assert out == "DET FALLBACK"


@pytest.mark.asyncio
async def test_composer_includes_item_facts_in_prompt() -> None:
    llm = _StubLLM("ok")
    item = _item(
        "OPQ32r",
        description="Measures workplace behavioural style for selection.",
    )
    await compose_reply(
        llm,
        action="recommend",
        user_message="hiring a leader",
        items=[item],
        fallback="...",
    )
    assert llm.received_user_prompt is not None
    # The composer must surface the item name and the first sentence
    # of its description so the LLM has facts to ground on.
    assert "OPQ32r" in llm.received_user_prompt
    assert "workplace behavioural style for selection" in llm.received_user_prompt
    # And the user's own framing.
    assert "hiring a leader" in llm.received_user_prompt
