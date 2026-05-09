"""Catalog data model.

Anchored to the actual schema of SHL's published catalog JSON
(``shl_product_catalog.json``). Every field declared so unknown fields
in the source are surfaced as warnings, not silently dropped.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# SHL test-type codes per the assignment example (`"test_type": "K"`).
# Mapping built from the catalog's "keys" long names.
TestTypeCode = Literal["A", "B", "C", "D", "E", "K", "P", "S"]

KEY_TO_CODE: dict[str, TestTypeCode] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

CODE_TO_LONG: dict[TestTypeCode, str] = {v: k for k, v in KEY_TO_CODE.items()}

# When an item carries multiple keys, prefer the most specific signal
# for the single-letter `test_type` field the API spec requires.
# Order: K > P > A > S > B > C > D > E (knowledge tests are usually
# the primary signal; behavioral instruments like OPQ are next; etc.)
PREFERENCE_ORDER: tuple[TestTypeCode, ...] = ("K", "P", "A", "S", "B", "C", "D", "E")


class CatalogMeta(BaseModel):
    """Pinned snapshot metadata for reproducibility."""

    model_config = ConfigDict(extra="forbid")

    source_url: str
    fetched_at: str
    source_sha256: str
    item_count: int


class Assessment(BaseModel):
    """One SHL Individual Test Solution.

    Field names mirror the source JSON exactly. Aliases / convenience
    accessors live as properties so consumers see a clean surface.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: str
    name: str
    link: str
    scraped_at: str
    job_levels: list[str] = Field(default_factory=list)
    job_levels_raw: str = ""
    languages: list[str] = Field(default_factory=list)
    languages_raw: str = ""
    duration: str = ""
    duration_raw: str = ""
    status: str
    remote: str
    adaptive: str
    description: str
    keys: list[str] = Field(default_factory=list, min_length=1)

    @field_validator("keys")
    @classmethod
    def _validate_keys(cls, v: list[str]) -> list[str]:
        unknown = [k for k in v if k not in KEY_TO_CODE]
        if unknown:
            raise ValueError(f"Unknown test-type keys: {unknown}")
        return v

    # ---- derived properties ----

    @property
    def url(self) -> str:
        """Spec-aligned alias for ``link``."""
        return self.link

    @property
    def test_type_codes(self) -> list[TestTypeCode]:
        return [KEY_TO_CODE[k] for k in self.keys]

    @property
    def primary_test_type(self) -> TestTypeCode:
        """Single-letter code emitted in the API response.

        Picks the most specific signal among the item's keys per
        ``PREFERENCE_ORDER``. Stable and deterministic.
        """
        codes = set(self.test_type_codes)
        for c in PREFERENCE_ORDER:
            if c in codes:
                return c
        # Unreachable given the validator + non-empty keys.
        raise RuntimeError(f"No test type code resolvable for {self.entity_id}")


class Catalog(BaseModel):
    """Full pinned snapshot: meta + items."""

    model_config = ConfigDict(extra="forbid")

    meta: CatalogMeta
    items: list[Assessment]
