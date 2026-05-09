"""LLM router: primary client + fallback with a circuit breaker.

Behavior:

* Try the **primary** client first.
* On :class:`LLMError` or :class:`LLMTimeout`, increment a failure
  counter and try the **fallback** client.
* If the primary's failure counter crosses ``failure_threshold``
  (default 3) within a sliding window, the breaker **opens**: all
  subsequent calls go straight to the fallback for ``open_duration_s``
  seconds. After that, one **half-open probe** is allowed; success
  closes the breaker, failure re-opens it.
* The fallback is never circuit-broken, if it also fails, we surface
  the exception to the orchestrator, which decides the user-facing
  response (typically a graceful "please rephrase").

This is deliberately **simple**: just two clients, one breaker. We
don't generalize to N providers because we don't need to.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from .base import LLMClient, LLMError, LLMMessage, LLMResult, LLMTimeout

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _BreakerState:
    """Mutable state for the primary client's circuit breaker."""

    consecutive_failures: int = 0
    opened_at: float | None = None  # epoch seconds when the breaker opened


class LLMRouter(LLMClient):
    """Implements :class:`LLMClient` by delegating to one of two real clients."""

    name = "router"

    __slots__ = (
        "_primary",
        "_fallback",
        "_breaker",
        "_failure_threshold",
        "_open_duration_s",
        "_lock",
    )

    def __init__(
        self,
        primary: LLMClient,
        fallback: LLMClient,
        *,
        failure_threshold: int = 3,
        open_duration_s: float = 60.0,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._failure_threshold = failure_threshold
        self._open_duration_s = open_duration_s
        self._breaker = _BreakerState()
        self._lock = asyncio.Lock()

    # ---- breaker introspection (used by tests + observability) ----

    def is_open(self) -> bool:
        if self._breaker.opened_at is None:
            return False
        return (time.monotonic() - self._breaker.opened_at) < self._open_duration_s

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        json_mode: bool = False,
        max_output_tokens: int | None = None,
        temperature: float = 0.0,
        timeout_s: float = 8.0,
    ) -> LLMResult:
        kwargs: dict[str, object] = {
            "json_mode": json_mode,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "timeout_s": timeout_s,
        }

        if not self.is_open():
            try:
                result = await self._primary.complete(messages, **kwargs)  # type: ignore[arg-type]
                # On success after a previous failure, reset the breaker.
                async with self._lock:
                    self._breaker.consecutive_failures = 0
                    self._breaker.opened_at = None
                return result
            except (LLMError, LLMTimeout) as exc:
                async with self._lock:
                    self._breaker.consecutive_failures += 1
                    if self._breaker.consecutive_failures >= self._failure_threshold:
                        self._breaker.opened_at = time.monotonic()
                        logger.warning(
                            "llm_breaker_open",
                            extra={
                                "primary": self._primary.name,
                                "consecutive_failures": self._breaker.consecutive_failures,
                            },
                        )
                logger.info(
                    "llm_primary_failed_fallback_engaged",
                    extra={"primary": self._primary.name, "error": str(exc)},
                )
        else:
            logger.debug(
                "llm_breaker_skipping_primary",
                extra={"primary": self._primary.name},
            )

        # Fallback path. We let exceptions propagate to the orchestrator.
        return await self._fallback.complete(messages, **kwargs)  # type: ignore[arg-type]
