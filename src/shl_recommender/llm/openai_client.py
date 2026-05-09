"""OpenAI async client.

Promoted to primary LLM after the user provided a Tier-2 key
(5000 RPM / 4M TPM). At that ceiling the agent can run the full eval
suite without throttling, even back-to-back.

Model choice: ``gpt-4o-mini`` is the right cost/quality balance,
sub-second TTFT, native JSON mode, ~$0.15 per 1M input tokens. Smarter
than Llama-3.3-70B on the slot extraction + reranking we ask of it.

Tenacity wraps each call with exponential backoff on transient
rate-limit / timeout errors; the router upstream handles persistent
failures by switching to the Groq/Gemini fallback.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import LLMClient, LLMError, LLMMessage, LLMResult, LLMTimeout

DEFAULT_MODEL = "gpt-4o-mini"
PROVIDER = "openai"


class OpenAILLM(LLMClient):
    """OpenAI Chat Completions client.

    Stateless and shareable across requests. The underlying
    ``AsyncOpenAI`` instance pools connections internally.
    """

    name = "openai"

    __slots__ = ("_client", "_model")

    def __init__(self, api_key: str, *, model: str = DEFAULT_MODEL) -> None:
        if not api_key:
            raise LLMError("OPENAI_API_KEY is required for OpenAILLM")
        self._client = AsyncOpenAI(api_key=api_key)
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
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1.0, min=1.0, max=8.0),
                retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
                reraise=True,
            ):
                with attempt:
                    completion = await asyncio.wait_for(
                        self._client.chat.completions.create(**kwargs),  # type: ignore[arg-type]
                        timeout=timeout_s,
                    )
        except TimeoutError as exc:
            raise LLMTimeout(f"openai call exceeded {timeout_s}s budget") from exc
        except APIError as exc:
            raise LLMError(f"openai API error: {exc}") from exc

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
