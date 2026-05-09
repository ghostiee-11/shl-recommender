"""Groq async client.

Why Groq is the primary LLM:

* Sub-second TTFT for Llama-3.3-70B and Llama-3.1-8B-Instant, the
  only way the agent fits a 30-second / 8-turn budget when up to 3
  LLM calls per turn (slot extractor, reranker, comparison) are in
  play.
* Native JSON mode (``response_format={"type": "json_object"}``).
* Generous free RPM that comfortably covers the 10-trace eval suite.

Tenacity wraps each call with exponential backoff on transient
rate-limit / 5xx failures. The router upstream handles persistent
failures by switching to Gemini.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from groq import APIError, APITimeoutError, AsyncGroq, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import LLMClient, LLMError, LLMMessage, LLMResult, LLMTimeout

DEFAULT_MODEL = "llama-3.3-70b-versatile"
PROVIDER = "groq"


class GroqLLM(LLMClient):
    """Groq chat completion client.

    Stateless and safe to share across requests. The underlying
    ``AsyncGroq`` instance is created once and reused, connection
    pooling lives inside the SDK.
    """

    name = "groq"

    __slots__ = ("_client", "_model")

    def __init__(self, api_key: str, *, model: str = DEFAULT_MODEL) -> None:
        if not api_key:
            raise LLMError("GROQ_API_KEY is required for GroqLLM")
        self._client = AsyncGroq(api_key=api_key)
        self._model = model

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
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(4),
                # Free-tier 429s ask for ~5s waits; exp backoff to ~12s.
                wait=wait_exponential(multiplier=1.5, min=2.0, max=12.0),
                retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
                reraise=True,
            ):
                with attempt:
                    completion = await asyncio.wait_for(
                        self._client.chat.completions.create(**kwargs),  # type: ignore[arg-type]
                        timeout=timeout_s,
                    )
        except TimeoutError as exc:
            raise LLMTimeout(f"groq call exceeded {timeout_s}s budget") from exc
        except APIError as exc:
            raise LLMError(f"groq API error: {exc}") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        choice = completion.choices[0]
        text = choice.message.content or ""
        usage = getattr(completion, "usage", None)
        return LLMResult(
            text=text,
            model=self._model,
            provider=PROVIDER,
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )
