"""Unit tests for the catalog Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shl_recommender.catalog.models import (
    KEY_TO_CODE,
    PREFERENCE_ORDER,
    Assessment,
    Catalog,
    CatalogMeta,
)


def _make(**overrides: object) -> Assessment:
    base: dict[str, object] = {
        "entity_id": "1",
        "name": "Test Item",
        "link": "https://www.shl.com/products/product-catalog/view/test/",
        "scraped_at": "2026-05-08T10:00:00+00:00",
        "job_levels": ["Manager"],
        "job_levels_raw": "Manager,",
        "languages": ["English (USA)"],
        "languages_raw": "English (USA),",
        "duration": "10 minutes",
        "duration_raw": "Approximate Completion Time in minutes = 10",
        "status": "ok",
        "remote": "yes",
        "adaptive": "no",
        "description": "A short description.",
        "keys": ["Knowledge & Skills"],
    }
    base.update(overrides)
    return Assessment.model_validate(base)


def test_minimal_record_validates() -> None:
    a = _make()
    assert a.primary_test_type == "K"
    assert a.url == a.link


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValidationError):
        _make(keys=["Made Up Category"])


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        Assessment.model_validate(
            {
                "entity_id": "1",
                "name": "x",
                "link": "x",
                "scraped_at": "x",
                "status": "ok",
                "remote": "yes",
                "adaptive": "no",
                "description": "x",
                "keys": ["Knowledge & Skills"],
                "unexpected_field": "boom",
            }
        )


def test_empty_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        _make(keys=[])


def test_primary_test_type_uses_preference_order() -> None:
    # When multiple keys are present, the most-specific code wins.
    a = _make(keys=["Personality & Behavior", "Competencies", "Knowledge & Skills"])
    # Knowledge & Skills (K) is highest priority.
    assert a.primary_test_type == "K"

    # No K → should pick P (next in PREFERENCE_ORDER).
    b = _make(keys=["Competencies", "Personality & Behavior", "Development & 360"])
    assert b.primary_test_type == "P"


def test_preference_order_covers_all_codes() -> None:
    assert set(PREFERENCE_ORDER) == set(KEY_TO_CODE.values())


def test_assessment_is_frozen() -> None:
    a = _make()
    with pytest.raises(ValidationError):
        a.name = "mutated"  # type: ignore[misc]


def test_catalog_validates_meta_and_items() -> None:
    cat = Catalog(
        meta=CatalogMeta(
            source_url="https://example.com",
            fetched_at="2026-05-08",
            source_sha256="deadbeef",
            item_count=1,
        ),
        items=[_make()],
    )
    assert len(cat.items) == 1
