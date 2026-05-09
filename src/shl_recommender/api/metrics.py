"""In-process metrics counters.

A tiny stdlib-only counter store. Exposed as JSON via ``/metrics`` so
operations can `curl` it without scraping logs or installing
Prometheus. Intentionally not Prometheus-compatible, adding a real
metrics backend in Phase 6 (Render deploy) is a swap of one module.

Counters tracked:

* ``requests_total``, every request, by path + status.
* ``chat_decisions_total``, agent decision distribution
  (clarify / recommend / refine / compare / refuse).
* ``llm_calls_total``, by provider + outcome.
* ``llm_failures_total``, provider-level error counts.

The counters are never reset on read, clients should compute
deltas over time. Process restart resets them, which is exactly
the right Prometheus-style semantic.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Iterable


class MetricsRegistry:
    """Thread-safe counter store. One instance per process."""

    __slots__ = ("_counters", "_lock")

    def __init__(self) -> None:
        self._counters: dict[str, Counter[str]] = {
            "requests_total": Counter(),
            "chat_decisions_total": Counter(),
            "llm_calls_total": Counter(),
            "llm_failures_total": Counter(),
        }
        self._lock = threading.Lock()

    def incr(self, family: str, label: str, value: int = 1) -> None:
        with self._lock:
            self._counters.setdefault(family, Counter())[label] += value

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {family: dict(counter) for family, counter in self._counters.items()}

    def families(self) -> Iterable[str]:
        with self._lock:
            return tuple(self._counters)
