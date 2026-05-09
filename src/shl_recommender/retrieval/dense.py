"""Dense retrieval over the catalog.

Production runs against pre-computed catalog embeddings stored in
``data/catalog_embeddings.npy`` (shipped with the repo). At query time
the agent calls an external embedder (OpenAI ``text-embedding-3-small``
by default), then runs a flat inner-product search on the cached
matrix.

Why this shape:

* **Pre-compute the catalog.** The catalog is pinned and stable, so
  there's no reason to re-embed on every cold start. Shipping the
  numpy matrix lets the runtime image drop torch + sentence-
  transformers + transformers + tokenizers entirely (~430 MB).
* **External query encoder.** One async API call per recommend turn.
  ~150 ms TTFT typical, well under the 25 s per-call budget.
* **FAISS IndexFlatIP.** With ~377 1536-d vectors, the index is
  ~2.3 MB and search is exact in O(n·d). No need for IVF / HNSW.
* **Async query interface** so the orchestrator stays end-to-end
  async; a thin sync ``encode_one_sync`` shim covers tests.

The encoder is injected through a Protocol so unit tests can pass a
deterministic fake without touching the OpenAI SDK.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

DEFAULT_EMBED_DIM = 1536  # text-embedding-3-small


class AsyncEncoder(Protocol):
    """Minimal async embedder interface used at query time.

    Compatible with :class:`shl_recommender.llm.openai_embeddings.OpenAIEmbedder`
    and with hand-rolled fakes in tests.
    """

    async def encode_one(self, text: str, *, timeout_s: float = ...) -> np.ndarray:
        ...


class DenseIndex:
    """In-memory dense index over pre-computed catalog embeddings.

    Construct with either:

    * a path to a ``.npy`` matrix (shape ``(n_items, dim)``,
      L2-normalized), or
    * the matrix directly (for tests).

    Then call :meth:`search_async` with the user query string and an
    :class:`AsyncEncoder` to get the top-k matches.
    """

    __slots__ = ("_index", "_size", "_dim")

    def __init__(
        self,
        catalog_embeddings: np.ndarray | Path | str,
    ) -> None:
        if isinstance(catalog_embeddings, (str, Path)):
            arr = np.load(str(catalog_embeddings)).astype(np.float32)
        else:
            arr = np.asarray(catalog_embeddings, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError(
                f"catalog_embeddings must be a non-empty 2-D array, got shape {arr.shape}"
            )
        self._dim = int(arr.shape[1])
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(arr)
        self._size = int(arr.shape[0])

    def __len__(self) -> int:
        return self._size

    @property
    def dim(self) -> int:
        return self._dim

    async def search_async(
        self,
        query: str,
        k: int,
        encoder: AsyncEncoder,
    ) -> list[tuple[int, float]]:
        """Encode ``query`` via ``encoder`` then return top-``k`` hits."""
        if k <= 0 or not query.strip():
            return []
        q = await encoder.encode_one(query)
        return self._search_vector(q, k)

    def search_with_vector(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Synchronous search when the caller already has the vector.

        Used by tests and by callers that batch-encoded upstream.
        """
        return self._search_vector(q, k)

    def _search_vector(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        q2 = np.asarray(q, dtype=np.float32).reshape(1, -1)
        if q2.shape[1] != self._dim:
            raise ValueError(
                f"query dim {q2.shape[1]} does not match index dim {self._dim}"
            )
        scores, idxs = self._index.search(q2, min(k, self._size))
        return [
            (int(i), float(s))
            for i, s in zip(idxs[0], scores[0], strict=True)
            if i >= 0
        ]


def build_from_texts_sync(texts: Iterable[str], encoder: object) -> np.ndarray:
    """One-time helper used by ``scripts/embed_catalog.py``.

    ``encoder`` is an :class:`OpenAIEmbedder` instance (or a fake);
    we call ``encode`` on it synchronously through ``asyncio.run``
    so the offline script stays a plain script.
    """
    import asyncio

    return asyncio.run(encoder.encode(list(texts)))  # type: ignore[union-attr]
