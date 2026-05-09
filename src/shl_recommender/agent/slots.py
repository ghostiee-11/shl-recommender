"""Slot schema for the agent.

Slots are the structured distillation of conversation context that
the decision policy reasons over. Keeping them in a Pydantic model
gives us:

* a stable contract between the LLM extractor and the policy,
* type-safe round-tripping through the embedded state hint,
* trivial unit-testable invariants (e.g. confidence in [0, 1]).

Every slot is **optional** because real conversations volunteer
information out of order; the user might say "personality test"
before mentioning the role. Each filled slot carries a
``confidence`` (extractor's certainty) and an ``evidence`` span
(the user phrase that justified it). The policy uses confidence
to decide when to ask vs commit, and the evidence span exists
specifically to *prevent slot hallucination*, slots without
evidence are dropped on extraction.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shl_recommender.catalog.models import TestTypeCode

Intent = Literal[
    "clarify",       # default, gather more context
    "recommend",     # we have enough; emit a shortlist
    "refine",        # update an existing shortlist
    "compare",       # explain differences between named items
    "refuse",        # out of scope or injection
]


class Slot(BaseModel):
    """One filled slot with provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""


class Slots(BaseModel):
    """Distilled conversational context.

    All fields optional. Extractors set those they can justify; the
    decision policy decides what's enough to act on.
    """

    model_config = ConfigDict(extra="forbid")

    role: Slot | None = None
    seniority: Slot | None = None
    skills: list[Slot] = Field(default_factory=list)
    test_type_preference: list[TestTypeCode] = Field(default_factory=list)
    duration_preference: Slot | None = None
    language_preference: Slot | None = None
    job_description: Slot | None = None  # set when user pastes a JD blob

    # LLMs love to emit ``null`` for "no preference" even when our
    # schema asks for a list. Coerce on input so a perfectly sensible
    # extractor output isn't rejected over a punctuation choice.
    @field_validator("skills", "test_type_preference", mode="before")
    @classmethod
    def _coerce_none_list(cls, v: object) -> object:
        return [] if v is None else v

    def filled_count(self) -> int:
        n = sum(
            1
            for v in (
                self.role,
                self.seniority,
                self.duration_preference,
                self.language_preference,
                self.job_description,
            )
            if v is not None
        )
        if self.skills:
            n += 1
        if self.test_type_preference:
            n += 1
        return n

    def context_score(self) -> float:
        """Weighted measure of "how much do we know" in [0, 1].

        Used by the decision policy to decide clarify-vs-recommend.
        Weights are deliberately uneven: role + skills carry the most
        signal for retrieval; duration / language are secondary.
        """
        weights = {
            "role": 0.30,
            "skills": 0.25,
            "seniority": 0.15,
            "test_type": 0.15,
            "duration": 0.05,
            "language": 0.05,
            "job_description": 0.40,  # a full JD subsumes most slots
        }
        score = 0.0
        if self.role and self.role.confidence > 0.5:
            score += weights["role"] * self.role.confidence
        if self.skills:
            avg = sum(s.confidence for s in self.skills) / len(self.skills)
            score += weights["skills"] * avg
        if self.seniority and self.seniority.confidence > 0.5:
            score += weights["seniority"] * self.seniority.confidence
        if self.test_type_preference:
            score += weights["test_type"]
        if self.duration_preference:
            score += weights["duration"]
        if self.language_preference:
            score += weights["language"]
        if self.job_description and self.job_description.confidence > 0.5:
            score += weights["job_description"] * self.job_description.confidence
        return min(score, 1.0)

    def as_query_text(self) -> str:
        """Compose a retrieval query from the filled slots.

        Used when the agent calls retrieval, strictly more useful
        than the raw user-turn concatenation because it surfaces
        only the *committed* signal.
        """
        parts: list[str] = []
        if self.job_description:
            parts.append(self.job_description.value)
        if self.role:
            parts.append(self.role.value)
        if self.seniority:
            parts.append(self.seniority.value)
        for s in self.skills:
            parts.append(s.value)
        if self.test_type_preference:
            from shl_recommender.catalog.models import CODE_TO_LONG

            parts.extend(CODE_TO_LONG[c] for c in self.test_type_preference)
        if self.duration_preference:
            parts.append(self.duration_preference.value)
        if self.language_preference:
            parts.append(self.language_preference.value)
        return " ".join(parts)


class AgentState(BaseModel):
    """Per-turn agent state, reconstructed from message history."""

    model_config = ConfigDict(extra="forbid")

    slots: Slots = Field(default_factory=Slots)
    intent: Intent = "clarify"
    prior_shortlist_urls: list[str] = Field(default_factory=list)
    asked_slots: list[str] = Field(default_factory=list)  # avoid re-asking
    turn_index: int = 0  # 0-indexed assistant turn we're about to emit
