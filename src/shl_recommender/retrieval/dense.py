"""Dense retrieval over the catalog using ``bge-small-en-v1.5``.

Choices and their reasoning:

* **bge-small (33M params, 384-d).** Strong MTEB scores per parameter,
  small enough to bake into a 1GB Docker image, runs on CPU in <1s
  per batch of 50. Bigger models (bge-base, e5-large) only matter
  when the corpus is much larger than 377 items.
* **FAISS IndexFlatIP.** With 377 384-d vectors, the index is
  ~570 KB and search is exact in O(n·d). Approximate indexes (IVF,
  HNSW) only help past ~10⁵ vectors.
* **L2-normalized embeddings + inner product = cosine similarity.**
  This avoids the need to convert distances or maintain a norm cache.

The encoder is injectable so unit tests can use a deterministic fake
without importing torch.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import faiss
import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# bge-family recommendation: prefix retrieval queries.
# Without this prefix, MTEB scores drop by ~1-2 pp.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Encoder(Protocol):
    """Minimal interface used by :class:`DenseIndex`.

    Compatible with ``sentence_transformers.SentenceTransformer`` and
    with hand-rolled fakes in tests.
    """

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = ...,
        normalize_embeddings: bool = ...,
        convert_to_numpy: bool = ...,
        show_progress_bar: bool = ...,
    ) -> np.ndarray:
        ...


def _default_encoder(model_name: str) -> Encoder:
    # Local import keeps unit tests free of the torch dependency.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class DenseIndex:
    """In-memory dense index. Built once at app startup."""

    __slots__ = ("_model", "_index", "_size", "_dim")

    def __init__(
        self,
        docs: Iterable[str],
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = 64,
        encoder: Encoder | None = None,
    ) -> None:
        docs_list = list(docs)
        if not docs_list:
            raise ValueError("DenseIndex requires at least one document")
        self._model = encoder if encoder is not None else _default_encoder(model_name)
        # ``normalize_embeddings=True`` gives unit-norm vectors so we
        # can use IndexFlatIP for cosine.
        embeddings = self._model.encode(
            docs_list,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        self._dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings)
        self._size = embeddings.shape[0]

    def __len__(self) -> int:
        return self._size

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return top-``k`` ``(doc_index, similarity)`` pairs."""
        if k <= 0:
            return []
        prefixed = f"{BGE_QUERY_PREFIX}{query}"
        q = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        scores, idxs = self._index.search(q, min(k, self._size))
        return [
            (int(i), float(s))
            for i, s in zip(idxs[0], scores[0], strict=True)
            if i >= 0
        ]
