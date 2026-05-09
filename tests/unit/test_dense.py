"""Unit tests for the DenseIndex (pre-computed-vectors variant).

Uses tiny hand-crafted matrices. The OpenAI embedder side is tested
separately via injected fakes in the orchestrator integration suite.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from shl_recommender.retrieval.dense import DenseIndex


def _normed(vec: list[float]) -> np.ndarray:
    a = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


class _FakeAsyncEncoder:
    """Returns a fixed vector for every query, regardless of text."""

    def __init__(self, vec: np.ndarray) -> None:
        self._vec = vec

    async def encode_one(self, text: str, *, timeout_s: float = 8.0) -> np.ndarray:
        return self._vec


def test_index_built_from_matrix() -> None:
    matrix = np.stack(
        [
            _normed([1.0, 0.0, 0.0]),
            _normed([0.0, 1.0, 0.0]),
            _normed([0.0, 0.0, 1.0]),
        ]
    )
    idx = DenseIndex(matrix)
    assert len(idx) == 3
    assert idx.dim == 3


def test_search_with_vector_returns_nearest() -> None:
    matrix = np.stack(
        [
            _normed([1.0, 0.0, 0.0]),
            _normed([0.0, 1.0, 0.0]),
            _normed([0.0, 0.0, 1.0]),
        ]
    )
    idx = DenseIndex(matrix)
    hits = idx.search_with_vector(_normed([1.0, 0.1, 0.0]), k=2)
    assert hits[0][0] == 0
    assert len(hits) == 2


def test_search_with_vector_zero_k_empty() -> None:
    idx = DenseIndex(np.eye(3, dtype=np.float32))
    assert idx.search_with_vector(np.zeros(3, dtype=np.float32) + 1.0, k=0) == []


def test_search_with_vector_dim_mismatch_raises() -> None:
    idx = DenseIndex(np.eye(3, dtype=np.float32))
    with pytest.raises(ValueError):
        idx.search_with_vector(np.array([1.0, 0.0], dtype=np.float32), k=1)


def test_empty_matrix_rejected() -> None:
    with pytest.raises(ValueError):
        DenseIndex(np.zeros((0, 3), dtype=np.float32))


def test_load_from_npy_file(tmp_path: Path) -> None:
    matrix = np.eye(4, dtype=np.float32)
    p = tmp_path / "vec.npy"
    np.save(p, matrix)
    idx = DenseIndex(p)
    assert len(idx) == 4
    assert idx.dim == 4


def test_search_async_uses_encoder() -> None:
    matrix = np.stack(
        [
            _normed([1.0, 0.0, 0.0]),
            _normed([0.0, 1.0, 0.0]),
        ]
    )
    idx = DenseIndex(matrix)
    enc = _FakeAsyncEncoder(_normed([1.0, 0.0, 0.0]))
    hits = asyncio.run(idx.search_async("anything", 2, enc))
    assert hits[0][0] == 0


def test_search_async_empty_query_returns_empty() -> None:
    idx = DenseIndex(np.eye(3, dtype=np.float32))
    enc = _FakeAsyncEncoder(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert asyncio.run(idx.search_async("   ", 5, enc)) == []
