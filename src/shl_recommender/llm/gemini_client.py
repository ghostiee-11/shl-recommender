"""Google Gemini async client (via the new ``google-genai`` SDK).

Why Gemini is the fallback:

* Generous free tier covers the full eval suite without quota anxiety.
* Native JSON mode via ``response_mime_type="application/json"``.
* Independent network path from Groq, when Groq throttles, Gemini
  is almost always still up.

Conversion notes:

* Gemini distinguishes a top-level ``system_instruction`` from
  ``contents`` (the user/assistant turns). We splice on assembly.
* The new SDK uses ``client.aio.models.generate_content`` for async.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from google import genai
from google.genai import types as genai_types
from google.genai.errors import APIError as GenAIAPIError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import LLMClient, LLMError, LLMMessage, LLMResult, LLMTimeout

DEFAULT_MODEL = "gemini-2.5-flash"
PROVIDER = "gemini"


class GeminiLLM(LLMClient):
    """Async Gemini chat completion client."""

    name = "gemini"

    __slots__ = ("_client", "_model")

    def __init__(self, api_key: str, *, model: str = DEFAULT_MODEL) -> None:
        if not api_key:
            raise LLMError("GEMINI_API_KEY is required for GeminiLLM")
        self._client = genai.Client(api_key=api_key)
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
        # Split out the system instruction; Gemini wants it at the
        # top level rather than as a "system" turn in contents.
        system_parts = [m.content for m in messages if m.role == "system"]
        contents: list[genai_types.Content] = []
        for m in messages:
            if m.role == "system":
                continue
            # Gemini uses "model" for assistant turns.
            role = "model" if m.role == "assistant" else "user"
            contents.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part(text=m.content)],
                )
            )

        # Gemini 2.5 Flash enables "thinking" by default, which silently
        # consumes the output-token budget before any visible token is
        # emitted. For an agent that needs predictable latency and
        # JSON output within tight token caps, we disable thinking.
        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            system_instruction="\n".join(system_parts) if system_parts else None,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json" if json_mode else None,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        )

        start = time.perf_counter()
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.5, max=2.0),
                retry=retry_if_exception_type(GenAIAPIError),
                reraise=True,
            ):
                with attempt:
                    response = await asyncio.wait_for(
                        self._client.aio.models.generate_content(
                            model=self._model,
                            contents=contents,
                            config=config,
                        ),
                        timeout=timeout_s,
                    )
        except TimeoutError as exc:
            raise LLMTimeout(f"gemini call exceeded {timeout_s}s budget") from exc
        except GenAIAPIError as exc:
            raise LLMError(f"gemini API error: {exc}") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        return LLMResult(
            text=text,
            model=self._model,
            provider=PROVIDER,
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
        )
