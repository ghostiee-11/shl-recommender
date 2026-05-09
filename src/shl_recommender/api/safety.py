"""HTTP safety layers: body-size limit + in-memory rate limit.

Both are deliberately small, dependency-free, and configurable so a
production tweak doesn't require code changes, just env vars.

Body-size limit
~~~~~~~~~~~~~~~
Rejects requests whose ``Content-Length`` exceeds ``max_body_bytes``
with HTTP 413 *before* FastAPI parses the body. Saves CPU on
adversarial payloads and bounds memory.

Rate limiter
~~~~~~~~~~~~
Token bucket per client IP, in-process. Suitable for the single
free-tier instance we deploy on; if we ever scale to N replicas,
swap for Redis-backed limiter (the Protocol stays the same).

* ``capacity``, max burst.
* ``refill_per_sec``, tokens added per second.
* On exhaustion → HTTP 429 with ``Retry-After`` seconds.

We exempt ``/health`` and ``/metrics`` from rate-limiting so platform
probes never see a 429.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

DEFAULT_MAX_BODY = 64 * 1024  # 64 KB, comfortably above the spec's 8-turn × 20KB cap.
DEFAULT_RATE_CAPACITY = 30
DEFAULT_RATE_REFILL_PER_SEC = 1.0  # one token/sec, 30/min sustained, burstable to 30
EXEMPT_PATHS: frozenset[str] = frozenset({"/health", "/metrics"})


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies based on the Content-Length header."""

    def __init__(self, app: Callable[..., object], *, max_bytes: int = DEFAULT_MAX_BODY) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._max = max_bytes

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > self._max:
                    return JSONResponse(
                        status_code=413,
                        content={"error": "request_body_too_large", "limit_bytes": self._max},
                    )
            except ValueError:
                # malformed header → let it through; downstream parser will reject
                pass
        return await call_next(request)


class _Bucket:
    __slots__ = ("tokens", "last")

    def __init__(self, tokens: float) -> None:
        self.tokens = tokens
        self.last = time.monotonic()


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Token-bucket rate limiter per client IP."""

    def __init__(
        self,
        app: Callable[..., object],
        *,
        capacity: int = DEFAULT_RATE_CAPACITY,
        refill_per_sec: float = DEFAULT_RATE_REFILL_PER_SEC,
        exempt_paths: frozenset[str] = EXEMPT_PATHS,
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._capacity = float(capacity)
        self._refill = refill_per_sec
        self._buckets: dict[str, _Bucket] = {}
        self._exempt = exempt_paths

    def _client_key(self, request: Request) -> str:
        # Honor X-Forwarded-For when behind Render's proxy.
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _consume(self, key: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_s)."""
        now = time.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(self._capacity)
            bucket.last = now  # avoid float-precision drift on first call
            self._buckets[key] = bucket
        # Refill since last check.
        elapsed = max(0.0, now - bucket.last)
        bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._refill)
        bucket.last = now
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True, 0.0
        deficit = 1.0 - bucket.tokens
        if self._refill <= 0:
            # Pure-burst limiter (no refill) → caller must wait indefinitely.
            return False, float("inf")
        return False, deficit / self._refill

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.url.path in self._exempt:
            return await call_next(request)
        key = self._client_key(request)
        allowed, retry_after = self._consume(key)
        if not allowed:
            # Cap Retry-After at a sane number for the header
            # (HTTP spec requires an integer, infinity isn't valid).
            retry_after_int = 3600 if retry_after == float("inf") else int(retry_after) + 1
            payload_retry = (
                None if retry_after == float("inf") else round(retry_after, 2)
            )
            resp: Response = JSONResponse(
                status_code=429,
                content={"error": "rate_limited", "retry_after_s": payload_retry},
            )
            resp.headers["retry-after"] = str(retry_after_int)
            return resp
        return await call_next(request)
