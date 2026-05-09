"""Catalog loader.

Reads ``data/catalog.json`` (pinned snapshot), validates against
:class:`Catalog`, and exposes:

* the list of :class:`Assessment` records,
* a per-item enriched text representation (used by BM25 + dense retrieval),
* O(1) lookup by URL and by entity_id,
* a hard guarantee that every URL emitted by the agent is in this set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import Assessment, Catalog, CODE_TO_LONG

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[3] / "data" / "catalog.json"


@dataclass(frozen=True, slots=True)
class CatalogIndex:
    """In-memory, read-only catalog index.

    Built once at app startup (FastAPI lifespan). Trivially shareable
    across requests because everything is immutable.
    """

    items: tuple[Assessment, ...]
    search_docs: tuple[str, ...]
    by_url: dict[str, Assessment]
    by_id: dict[str, Assessment]
    allowed_urls: frozenset[str]

    def __len__(self) -> int:
        return len(self.items)

    def get_by_url(self, url: str) -> Assessment | None:
        return self.by_url.get(url)

    def is_grounded_url(self, url: str) -> bool:
        """The single guard the API layer uses before emitting any URL."""
        return url in self.allowed_urls


def build_search_doc(item: Assessment) -> str:
    """Concatenate the fields that are useful for retrieval.

    Order matters mildly for BM25 (early tokens are weighted equally,
    but readability for the LLM reranker matters too).
    """
    long_types = ", ".join(CODE_TO_LONG[c] for c in item.test_type_codes)
    job_levels = ", ".join(item.job_levels) if item.job_levels else "n/a"
    languages = ", ".join(item.languages) if item.languages else "n/a"
    duration = item.duration or "n/a"
    return (
        f"{item.name}\n"
        f"Test types: {long_types}\n"
        f"{item.description}\n"
        f"Job levels: {job_levels}\n"
        f"Languages: {languages}\n"
        f"Duration: {duration}\n"
        f"Remote: {item.remote} | Adaptive: {item.adaptive}"
    )


def load_catalog(path: Path | str | None = None) -> CatalogIndex:
    """Load + validate the pinned catalog snapshot.

    Raises ``ValidationError`` from Pydantic on any schema drift.
    """
    p = Path(path) if path is not None else DEFAULT_CATALOG_PATH
    with p.open(encoding="utf-8") as f:
        raw = json.load(f)
    catalog = Catalog.model_validate(raw)

    items = tuple(catalog.items)
    docs = tuple(build_search_doc(it) for it in items)
    by_url = {it.link: it for it in items}
    by_id = {it.entity_id: it for it in items}

    if len(by_url) != len(items):
        # Defensive: source has unique links today, but we want a hard
        # signal if that ever changes.
        raise ValueError("Catalog contains duplicate URLs")

    return CatalogIndex(
        items=items,
        search_docs=docs,
        by_url=by_url,
        by_id=by_id,
        allowed_urls=frozenset(by_url),
    )
