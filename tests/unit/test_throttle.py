"""Unit tests for RateLimitedLLM.

Uses a fake inner client and asyncio's real clock so we can verify
the bucket actually paces calls. Each test stays under 1s wall time
by sizing the refill rate appropriately.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import pytest

from shl_recommender.llm.base import LLMMessage, LLMResult
from shl_recommender.llm.throttle import RateLimitedLLM


class _FakeLLM:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0
        self.timestamps: list[float] = []

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        **_: object,
    ) -> LLMResult:
        self.calls += 1
        self.timestamps.append(time.monotonic())
        return LLMResult(text="ok", model="fake", provider="fake", latency_ms=0)


_M = [LLMMessage(role="user", content="hi")]


def test_init_rejects_zero_capacity() -> None:
    with pytest.raises(ValueError):
        RateLimitedLLM(_FakeLLM(), capacity=0, refill_per_sec=1.0)


def test_init_rejects_zero_refill() -> None:
    with pytest.raises(ValueError):
        RateLimitedLLM(_FakeLLM(), capacity=1, refill_per_sec=0.0)


def test_init_inherits_inner_name() -> None:
    inner = _FakeLLM()
    inner.name = "groq"  # type: ignore[misc]
    throttled = RateLimitedLLM(inner, capacity=1, refill_per_sec=1.0)
    assert throttled.name == "groq"


@pytest.mark.asyncio
async def test_burst_consumes_capacity_without_delay() -> None:
    inner = _FakeLLM()
    # capacity=5, refill 100/s, burst 5 should finish in <50ms
    throttled = RateLimitedLLM(inner, capacity=5, refill_per_sec=100)
    start = time.monotonic()
    await asyncio.gather(*(throttled.complete(_M) for _ in range(5)))
    elapsed = time.monotonic() - start
    assert inner.calls == 5
    assert elapsed < 0.2, f"burst should not block; took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_excess_calls_wait_for_refill() -> None:
    inner = _FakeLLM()
    # capacity=2, refill 10/s → after burst, the 3rd call waits ~100ms.
    throttled = RateLimitedLLM(inner, capacity=2, refill_per_sec=10)
    start = time.monotonic()
    # Fire 5 calls; first 2 burst, next 3 each wait ~100ms.
    await asyncio.gather(*(throttled.complete(_M) for _ in range(5)))
    elapsed = time.monotonic() - start
    assert inner.calls == 5
    # 3 throttled calls × 100ms = ~300ms minimum.
    assert 0.25 <= elapsed <= 1.0, f"expected ~0.3-1.0s, got {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_throttle_passes_through_kwargs() -> None:
    captured: dict[str, object] = {}

    class _CapturingLLM:
        name = "cap"

        async def complete(self, messages: Sequence[LLMMessage], **kwargs: object) -> LLMResult:
            captured.update(kwargs)
            return LLMResult(text="ok", model="cap", provider="cap", latency_ms=0)

    throttled = RateLimitedLLM(_CapturingLLM(), capacity=5, refill_per_sec=100)
    await throttled.complete(
        _M,
        json_mode=True,
        max_output_tokens=42,
        temperature=0.5,
        timeout_s=3.0,
    )
    assert captured == {
        "json_mode": True,
        "max_output_tokens": 42,
        "temperature": 0.5,
        "timeout_s": 3.0,
    }


@pytest.mark.asyncio
async def test_concurrent_callers_serialize_correctly() -> None:
    """No double-spending of tokens under concurrent contention."""
    inner = _FakeLLM()
    throttled = RateLimitedLLM(inner, capacity=3, refill_per_sec=10)
    # 9 concurrent callers; only 3 may pass without waiting.
    start = time.monotonic()
    await asyncio.gather(*(throttled.complete(_M) for _ in range(9)))
    elapsed = time.monotonic() - start
    assert inner.calls == 9
    # 6 throttled × 100ms = ~600ms minimum.
    assert elapsed >= 0.5, f"expected >= 0.5s, got {elapsed:.3f}s"
