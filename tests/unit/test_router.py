"""Unit tests for the LLM router circuit breaker."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from shl_recommender.llm.base import LLMError, LLMMessage, LLMResult, LLMTimeout
from shl_recommender.llm.router import LLMRouter


class _FakeLLM:
    """Configurable fake. Each call pops the next behavior off ``script``."""

    def __init__(
        self,
        name: str,
        *,
        script: list[str | Exception] | None = None,
        default_text: str = "ok",
    ) -> None:
        self.name = name
        self._script: list[str | Exception] = list(script or [])
        self._default = default_text
        self.calls = 0

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        json_mode: bool = False,
        max_output_tokens: int | None = None,
        temperature: float = 0.0,
        timeout_s: float = 8.0,
    ) -> LLMResult:
        self.calls += 1
        if self._script:
            step = self._script.pop(0)
            if isinstance(step, Exception):
                raise step
            text = step
        else:
            text = self._default
        return LLMResult(
            text=text,
            model=f"{self.name}-fake",
            provider=self.name,
            latency_ms=1,
        )


_M = [LLMMessage(role="user", content="hi")]


@pytest.mark.asyncio
async def test_router_uses_primary_on_success() -> None:
    primary = _FakeLLM("groq", default_text="primary")
    fallback = _FakeLLM("gemini", default_text="fallback")
    router = LLMRouter(primary, fallback)
    out = await router.complete(_M)
    assert out.text == "primary"
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_router_falls_back_on_primary_error() -> None:
    primary = _FakeLLM("groq", script=[LLMError("boom")])
    fallback = _FakeLLM("gemini", default_text="fallback-ok")
    router = LLMRouter(primary, fallback)
    out = await router.complete(_M)
    assert out.text == "fallback-ok"
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_router_falls_back_on_primary_timeout() -> None:
    primary = _FakeLLM("groq", script=[LLMTimeout("slow")])
    fallback = _FakeLLM("gemini", default_text="fallback-ok")
    router = LLMRouter(primary, fallback)
    out = await router.complete(_M)
    assert out.text == "fallback-ok"


@pytest.mark.asyncio
async def test_breaker_opens_after_threshold_failures() -> None:
    # Three consecutive primary failures should open the breaker.
    primary = _FakeLLM("groq", script=[LLMError("e")] * 3)
    fallback = _FakeLLM("gemini", default_text="fb")
    router = LLMRouter(primary, fallback, failure_threshold=3, open_duration_s=999)
    for _ in range(3):
        await router.complete(_M)
    assert router.is_open()
    assert primary.calls == 3
    # Next call must skip primary entirely.
    await router.complete(_M)
    assert primary.calls == 3  # unchanged
    assert fallback.calls == 4


@pytest.mark.asyncio
async def test_breaker_resets_on_primary_success() -> None:
    # Two failures (under threshold) then a success → counter resets.
    primary = _FakeLLM("groq", script=[LLMError("e"), LLMError("e"), "ok"])
    fallback = _FakeLLM("gemini")
    router = LLMRouter(primary, fallback, failure_threshold=3)
    await router.complete(_M)
    await router.complete(_M)
    out = await router.complete(_M)
    assert out.text == "ok"
    assert not router.is_open()


@pytest.mark.asyncio
async def test_router_propagates_fallback_failure() -> None:
    primary = _FakeLLM("groq", script=[LLMError("p")])
    fallback = _FakeLLM("gemini", script=[LLMError("f")])
    router = LLMRouter(primary, fallback)
    with pytest.raises(LLMError):
        await router.complete(_M)
