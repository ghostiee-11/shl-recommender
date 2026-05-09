"""LLM client Protocol.

Every downstream agent component (slot extractor, reranker, refusal
tiebreak, comparison) talks to LLMs through this interface only. That
gives us:

* Provider swapping without touching agent code (Groq → Gemini → fake).
* Trivial unit testing via :class:`FakeLLM` injection.
* A single place to enforce JSON-mode discipline, latency budgets,
  and structured error semantics.

The interface is intentionally narrow, one method, ``complete``,
because every agent call is "given this conversation, return text or
JSON". Multi-turn streaming, tool-use, vision, etc. are out of scope.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """One turn in the prompt sent to the LLM."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class LLMResult:
    """Structured response from any LLM provider."""

    text: str
    model: str
    provider: str
    latency_ms: int
    # Best-effort token counts; some providers don't return these.
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMError(Exception):
    """Raised by clients on transport, auth, or API failures.

    Distinct from ``LLMTimeout`` so the router can choose to retry
    transient errors without retrying timeouts (which probably won't
    succeed within the same wall-clock budget).
    """


class LLMTimeout(LLMError):
    """Raised when a single provider call exceeds its budget."""


@runtime_checkable
class LLMClient(Protocol):
    """Async LLM client. One method, JSON-friendly."""

    name: str

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        json_mode: bool = False,
        max_output_tokens: int | None = None,
        temperature: float = 0.0,
        timeout_s: float = 8.0,
    ) -> LLMResult:
        """Run a chat completion.

        ``json_mode=True`` instructs the provider to constrain output
        to a syntactically-valid JSON object. Callers are still
        responsible for validating against their Pydantic schema.

        Implementations must raise :class:`LLMTimeout` on per-call
        timeout and :class:`LLMError` on any other provider failure.
        Never raise raw provider exceptions.
        """
        ...
