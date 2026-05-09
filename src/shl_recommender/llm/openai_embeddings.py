"""OpenAI embeddings client.

Lets us drop the torch + sentence-transformers + transformers stack
from runtime entirely (~430 MB freed, comfortably under Render's
512 MB free-tier ceiling). text-embedding-3-small is also a stronger
encoder than bge-small on most retrieval benchmarks, at a per-call
cost low enough to ignore (~$0.00002 per query).

Used in two modes:

* **Catalog pre-compute (offline):** batch-embed all 377 items once,
  ship the resulting numpy matrix as ``data/catalog_embeddings.npy``.
* **Query encode (online):** one call per ``/chat`` recommend turn,
  ~150 ms TTFT typical.

Async because every other LLM client in the project is, and the
orchestrator is async end-to-end.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import numpy as np
from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import LLMError, LLMTimeout

DEFAULT_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # native dim for text-embedding-3-small
PROVIDER = "openai-embeddings"


class OpenAIEmbedder:
    """Async embeddings client.

    Stateless; the underlying ``AsyncOpenAI`` instance pools HTTP
    connections internally, so a single instance per process is right.
    """

    __slots__ = ("_client", "_model", "_dim")

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        dim: int = EMBED_DIM,
    ) -> None:
        if not api_key:
            raise LLMError("OPENAI_API_KEY is required for OpenAIEmbedder")
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 96,
        timeout_s: float = 30.0,
    ) -> np.ndarray:
        """Embed ``texts`` and return an ``(N, dim)`` L2-normalized matrix.

        Normalization is done client-side so the caller can use plain
        inner-product retrieval (FAISS IndexFlatIP) for cosine.
        """
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        out_rows: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(4),
                    wait=wait_exponential(multiplier=1.0, min=1.0, max=8.0),
                    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
                    reraise=True,
                ):
                    with attempt:
                        resp = await asyncio.wait_for(
                            self._client.embeddings.create(
                                model=self._model,
                                input=chunk,
                            ),
                            timeout=timeout_s,
                        )
            except TimeoutError as exc:
                raise LLMTimeout(f"openai embeddings exceeded {timeout_s}s") from exc
            except APIError as exc:
                raise LLMError(f"openai embeddings error: {exc}") from exc

            out_rows.extend(d.embedding for d in resp.data)

        arr = np.asarray(out_rows, dtype=np.float32)
        # L2 normalize so cosine == inner product downstream.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    async def encode_one(self, text: str, *, timeout_s: float = 8.0) -> np.ndarray:
        """Convenience wrapper for single-query encoding (the hot path)."""
        start = time.perf_counter()
        arr = await self.encode([text], batch_size=1, timeout_s=timeout_s)
        # latency tracked by the orchestrator's structured log; nothing to
        # emit here without a logger handle.
        _ = start
        return arr[0]
