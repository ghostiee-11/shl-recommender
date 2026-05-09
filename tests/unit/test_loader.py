"""Unit tests for the catalog loader.

These tests exercise the **real** pinned catalog at ``data/catalog.json``.
That file is committed to the repo, so the tests are hermetic and
deterministic; they also act as a regression guard against any future
schema drift in the upstream JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shl_recommender.catalog.loader import (
    DEFAULT_CATALOG_PATH,
    build_search_doc,
    load_catalog,
)


@pytest.fixture(scope="module")
def index() -> object:
    return load_catalog()


def test_pinned_catalog_loads(index: object) -> None:
    assert len(index) > 300, "expected the pinned catalog to be substantial"  # type: ignore[arg-type]


def test_meta_matches_items(index: object) -> None:
    raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
    assert raw["meta"]["item_count"] == len(raw["items"]) == len(index)  # type: ignore[arg-type]


def test_url_lookup_round_trip(index: object) -> None:
    item = index.items[0]  # type: ignore[attr-defined]
    assert index.get_by_url(item.link) is item  # type: ignore[attr-defined]
    assert index.is_grounded_url(item.link) is True  # type: ignore[attr-defined]
    assert index.is_grounded_url("https://example.com/not-in-catalog") is False  # type: ignore[attr-defined]


def test_every_item_has_valid_primary_test_type(index: object) -> None:
    for item in index.items:  # type: ignore[attr-defined]
        assert item.primary_test_type in {"A", "B", "C", "D", "E", "K", "P", "S"}


def test_search_doc_contains_key_signals(index: object) -> None:
    item = index.items[0]  # type: ignore[attr-defined]
    doc = build_search_doc(item)
    assert item.name in doc
    assert item.description in doc
    assert "Test types:" in doc


def test_no_duplicate_urls() -> None:
    raw = json.loads(Path(DEFAULT_CATALOG_PATH).read_text())
    urls = [it["link"] for it in raw["items"]]
    assert len(urls) == len(set(urls))


def test_all_urls_are_shl_dot_com(index: object) -> None:
    for item in index.items:  # type: ignore[attr-defined]
        assert item.link.startswith("https://www.shl.com/"), item.link
