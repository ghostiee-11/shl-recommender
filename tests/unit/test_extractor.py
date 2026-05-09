"""Unit tests for the slot extractor.

Uses a stub LLM that returns canned JSON. The interesting logic
(evidence-required guard, schema fallback, merge semantics) is
covered without any real LLM call.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from shl_recommender.agent.extractor import (
    ExtractorOutput,
    extract_slots,
    merge_slots,
)
from shl_recommender.agent.slots import Slot, Slots
from shl_recommender.llm.base import LLMError, LLMMessage, LLMResult


class _StubLLM:
    name = "stub"

    def __init__(self, payload: object | Exception) -> None:
        self._payload = payload

    async def complete(self, messages: Sequence[LLMMessage], **_: object) -> LLMResult:
        if isinstance(self._payload, Exception):
            raise self._payload
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return LLMResult(text=text, model="stub", provider="stub", latency_ms=1)


_USER = [LLMMessage(role="user", content="I'm hiring a senior Java developer for backend work")]


@pytest.mark.asyncio
async def test_extractor_happy_path_keeps_evidence_backed_slots() -> None:
    payload = {
        "intent": "clarify",
        "slots": {
            "role": {"value": "Java developer", "confidence": 0.9, "evidence": "Java developer"},
            "seniority": {"value": "senior", "confidence": 0.9, "evidence": "senior"},
            "skills": [],
            "test_type_preference": [],
            "duration_preference": None,
            "language_preference": None,
            "job_description": None,
        },
        "compared_assessments": [],
    }
    out = await extract_slots(_StubLLM(payload), _USER)
    assert out.intent == "clarify"
    assert out.slots.role and out.slots.role.value == "Java developer"
    assert out.slots.seniority and out.slots.seniority.value == "senior"


@pytest.mark.asyncio
async def test_extractor_drops_slots_without_evidence_in_user_text() -> None:
    payload = {
        "intent": "clarify",
        "slots": {
            "role": {
                "value": "data scientist",
                "confidence": 0.9,
                "evidence": "data scientist",  # NOT in user text
            },
            "skills": [],
            "test_type_preference": [],
            "duration_preference": None,
            "language_preference": None,
            "job_description": None,
            "seniority": None,
        },
        "compared_assessments": [],
    }
    out = await extract_slots(_StubLLM(payload), _USER)
    assert out.slots.role is None  # hallucinated → dropped


@pytest.mark.asyncio
async def test_extractor_returns_safe_default_on_invalid_json() -> None:
    out = await extract_slots(_StubLLM("not json at all"), _USER)
    assert out.intent == "clarify"
    assert out.slots == Slots()


@pytest.mark.asyncio
async def test_extractor_returns_safe_default_on_schema_violation() -> None:
    out = await extract_slots(_StubLLM({"intent": "bogus_intent"}), _USER)
    assert out.intent == "clarify"


@pytest.mark.asyncio
async def test_extractor_returns_safe_default_on_llm_error() -> None:
    out = await extract_slots(_StubLLM(LLMError("down")), _USER)
    assert out.intent == "clarify"
    assert out.slots == Slots()


# ---------- merge_slots ----------


def test_merge_picks_higher_confidence_slot() -> None:
    a = Slot(value="old", confidence=0.5, evidence="old")
    b = Slot(value="new", confidence=0.9, evidence="new")
    prior = Slots(role=a)
    new = Slots(role=b)
    merged = merge_slots(prior, new)
    assert merged.role is b


def test_merge_unions_skills() -> None:
    prior = Slots(skills=[Slot(value="Java", confidence=0.9, evidence="Java")])
    new = Slots(skills=[Slot(value="Spring", confidence=0.9, evidence="Spring")])
    merged = merge_slots(prior, new)
    assert {s.value for s in merged.skills} == {"Java", "Spring"}


def test_merge_dedupes_skills_case_insensitive() -> None:
    prior = Slots(skills=[Slot(value="Java", confidence=0.5, evidence="Java")])
    new = Slots(skills=[Slot(value="java", confidence=0.9, evidence="java")])
    merged = merge_slots(prior, new)
    assert len(merged.skills) == 1
    assert merged.skills[0].confidence == 0.9


def test_merge_empty_state_is_identity() -> None:
    prior = Slots()
    new = Slots()
    assert merge_slots(prior, new) == prior


def test_extractor_output_default_construction() -> None:
    out = ExtractorOutput(intent="clarify")
    assert out.slots == Slots()
    assert out.compared_assessments == []
