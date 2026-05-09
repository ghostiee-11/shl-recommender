"""Async token-bucket throttle for LLM clients.

Composable wrapper: ``RateLimitedLLM(inner, capacity, refill_per_sec)``
gates every ``complete`` call through a token bucket so we stay under
provider RPM limits. When the bucket is empty, callers ``await`` until
a token is available rather than getting an error, the orchestrator
expects best-effort completions and our latency budget (25s/turn) has
plenty of headroom for short waits.

Why this lives separately from :class:`LLMRouter`:

* The router's job is provider failover. Throttling is a per-provider
  concern (Groq RPM ≠ Gemini RPM), so each underlying client wraps
  itself with its own bucket sized to its own limit.
* The wrapper conforms to :class:`LLMClient` Protocol, so the router
  doesn't even know it's there.

Limits assumed (free tier defaults):

* Groq Llama-3.3-70B: 30 requests / minute → 0.5 tokens / sec.
* Gemini 2.5 Flash: 5 requests / minute → 0.083 tokens / sec.

The bucket carries a small burst (capacity) so a short flurry doesn't
serialize unnecessarily.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from .base import LLMClient, LLMMessage, LLMResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Bucket:
    tokens: float
    last: float


class RateLimitedLLM(LLMClient):
    """Wraps any :class:`LLMClient` with an async token-bucket throttle."""

    __slots__ = ("_inner", "_capacity", "_refill", "_bucket", "_lock", "name")

    def __init__(
        self,
        inner: LLMClient,
        *,
        capacity: int = 5,
        refill_per_sec: float = 0.5,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if refill_per_sec <= 0:
            raise ValueError("refill_per_sec must be > 0")
        self._inner = inner
        self._capacity = float(capacity)
        self._refill = refill_per_sec
        now = time.monotonic()
        self._bucket = _Bucket(tokens=self._capacity, last=now)
        self._lock = asyncio.Lock()
        # Keep the inner provider's name so router logs and metrics
        # see a stable identifier.
        self.name = inner.name

    async def _acquire(self) -> None:
        """Sleep until at least one token is available, then consume it."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = max(0.0, now - self._bucket.last)
                self._bucket.tokens = min(
                    self._capacity, self._bucket.tokens + elapsed * self._refill
                )
                self._bucket.last = now
                if self._bucket.tokens >= 1.0:
                    self._bucket.tokens -= 1.0
                    return
                deficit = 1.0 - self._bucket.tokens
                wait = deficit / self._refill
            # Release the lock while sleeping so other callers can
            # also see the refill happen.
            logger.debug("llm_throttle_waiting", extra={"wait_s": round(wait, 3)})
            await asyncio.sleep(wait + 0.01)

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        json_mode: bool = False,
        max_output_tokens: int | None = None,
        temperature: float = 0.0,
        timeout_s: float = 8.0,
    ) -> LLMResult:
        await self._acquire()
        return await self._inner.complete(
            messages,
            json_mode=json_mode,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
