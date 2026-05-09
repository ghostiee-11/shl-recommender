"""Slot extractor: a single LLM call per turn that parses intent + slots.

Design:

* One JSON-mode LLM call returns ``{intent, slots, compared_assessments}``.
* The output is validated against :class:`ExtractorOutput`. Anything
  malformed → empty :class:`Slots` and ``intent="clarify"`` (safe
  default, agent will ask a clarifying question).
* Slots without a quotable ``evidence`` substring are dropped.
  This is the **anti-hallucination guard**: an extractor cannot
  "remember" a slot that wasn't supported by the user text.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from shl_recommender.llm.base import LLMClient, LLMError, LLMMessage

from .prompts import SLOT_EXTRACTOR_PROMPT_V1
from .slots import AgentState, Intent, Slot, Slots

logger = logging.getLogger(__name__)


class ExtractorOutput(BaseModel):
    """Strict shape of the JSON the extractor returns."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    slots: Slots = Field(default_factory=Slots)
    compared_assessments: list[str] = Field(default_factory=list)


def _strip_unsupported_slots(slots: Slots, supporting_text: str) -> Slots:
    """Drop any slot whose ``evidence`` is not actually in the user text.

    Lowercase substring match, strict enough to catch hallucination,
    loose enough to survive paraphrase quirks (LLM wraps quotes, etc.).
    """
    text_lower = supporting_text.lower()

    def _ok(s: Slot | None) -> Slot | None:
        if s is None:
            return None
        if not s.evidence or s.evidence.lower() not in text_lower:
            return None
        return s

    cleaned_skills = [s for s in slots.skills if s.evidence and s.evidence.lower() in text_lower]
    return Slots(
        role=_ok(slots.role),
        seniority=_ok(slots.seniority),
        skills=cleaned_skills,
        test_type_preference=slots.test_type_preference,
        duration_preference=_ok(slots.duration_preference),
        language_preference=_ok(slots.language_preference),
        job_description=_ok(slots.job_description),
    )


def _format_history_for_extraction(messages: Sequence[LLMMessage]) -> str:
    """Render the history into a compact transcript for the extractor."""
    return "\n".join(
        f"{m.role.upper()}: {m.content}" for m in messages if m.role != "system"
    )


async def extract_slots(
    llm: LLMClient,
    messages: Sequence[LLMMessage],
    *,
    timeout_s: float = 6.0,
) -> ExtractorOutput:
    """Run one extraction LLM call. Always returns a valid object.

    On any failure (LLM down, invalid JSON, schema violation), returns
    a safe default so the orchestrator can still emit a clarifying
    question. The orchestrator can inspect ``intent="clarify"`` with
    empty slots and treat it as a "we know nothing yet" state.
    """
    transcript = _format_history_for_extraction(messages)
    prompt_messages = [
        LLMMessage(role="system", content=SLOT_EXTRACTOR_PROMPT_V1),
        LLMMessage(role="user", content=transcript),
    ]
    try:
        result = await llm.complete(
            prompt_messages,
            json_mode=True,
            temperature=0.0,
            max_output_tokens=512,
            timeout_s=timeout_s,
        )
    except LLMError:
        logger.warning("slot_extractor_llm_error")
        return ExtractorOutput(intent="clarify")

    try:
        raw = json.loads(result.text)
    except json.JSONDecodeError:
        logger.warning("slot_extractor_invalid_json", extra={"text": result.text[:200]})
        return ExtractorOutput(intent="clarify")

    try:
        parsed = ExtractorOutput.model_validate(raw)
    except ValidationError as exc:
        logger.warning("slot_extractor_schema_error", extra={"err": str(exc)[:200]})
        return ExtractorOutput(intent="clarify")

    user_only = "\n".join(m.content for m in messages if m.role == "user")
    parsed = ExtractorOutput(
        intent=parsed.intent,
        slots=_strip_unsupported_slots(parsed.slots, user_only),
        compared_assessments=parsed.compared_assessments,
    )
    return parsed


def merge_slots(prior: Slots, new: Slots) -> Slots:
    """Combine prior state with newly-extracted slots.

    Per-slot policy: take whichever has the higher confidence.
    For lists (skills, test_type_preference): union, deduped.
    """
    def _pick(a: Slot | None, b: Slot | None) -> Slot | None:
        if a is None:
            return b
        if b is None:
            return a
        return b if b.confidence >= a.confidence else a

    skills_seen: dict[str, Slot] = {}
    for s in (*prior.skills, *new.skills):
        existing = skills_seen.get(s.value.lower())
        if existing is None or s.confidence > existing.confidence:
            skills_seen[s.value.lower()] = s

    return Slots(
        role=_pick(prior.role, new.role),
        seniority=_pick(prior.seniority, new.seniority),
        skills=list(skills_seen.values()),
        test_type_preference=list(
            dict.fromkeys((*prior.test_type_preference, *new.test_type_preference))
        ),
        duration_preference=_pick(prior.duration_preference, new.duration_preference),
        language_preference=_pick(prior.language_preference, new.language_preference),
        job_description=_pick(prior.job_description, new.job_description),
    )


def update_state(state: AgentState, output: ExtractorOutput) -> AgentState:
    """Apply an extractor output on top of an existing :class:`AgentState`."""
    return AgentState(
        slots=merge_slots(state.slots, output.slots),
        intent=output.intent,
        prior_shortlist_urls=state.prior_shortlist_urls,
        asked_slots=state.asked_slots,
        turn_index=state.turn_index,
    )


__all__ = [
    "ExtractorOutput",
    "extract_slots",
    "merge_slots",
    "update_state",
]
