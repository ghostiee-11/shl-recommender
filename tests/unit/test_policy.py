"""Table-driven tests for the decision policy.

Pure-function policy → trivial to cover every branch.
"""

from __future__ import annotations

from shl_recommender.agent.policy import (
    CLARIFY_QUESTIONS,
    Action,
    decide,
)
from shl_recommender.agent.slots import AgentState, Slot, Slots


def _state(
    score_target: float = 0.0,
    *,
    turn_index: int = 0,
    prior_shortlist: bool = False,
    asked: tuple[str, ...] = (),
) -> AgentState:
    """Build an AgentState whose context_score is at least ``score_target``."""
    slots = Slots()
    if score_target >= 0.30:
        slots = Slots(role=Slot(value="Java dev", confidence=1.0, evidence="Java dev"))
    if score_target >= 0.55:
        slots = Slots(
            role=Slot(value="Java dev", confidence=1.0, evidence="Java dev"),
            skills=[Slot(value="Java", confidence=1.0, evidence="Java")],
            seniority=Slot(value="mid", confidence=1.0, evidence="mid"),
        )
    return AgentState(
        slots=slots,
        turn_index=turn_index,
        prior_shortlist_urls=["https://example.com/x"] if prior_shortlist else [],
        asked_slots=list(asked),
    )


# ---------- branch coverage ----------


def test_refuse_intent_short_circuits() -> None:
    d = decide(_state(0.99), extractor_intent="refuse")
    assert d.action == Action.REFUSE


def test_compare_with_two_named_assessments() -> None:
    d = decide(
        _state(0.0),
        extractor_intent="compare",
        compared_assessments=("OPQ32", "Verify Numerical"),
    )
    assert d.action == Action.COMPARE
    assert d.compared_assessments == ("OPQ32", "Verify Numerical")


def test_compare_with_single_name_falls_through() -> None:
    d = decide(
        _state(0.0),
        extractor_intent="compare",
        compared_assessments=("OPQ32",),
    )
    # Not enough to compare → falls through to clarify.
    assert d.action == Action.CLARIFY


def test_refine_requires_prior_shortlist() -> None:
    d_no = decide(_state(0.0), extractor_intent="refine", has_prior_shortlist=False)
    d_yes = decide(_state(0.0), extractor_intent="refine", has_prior_shortlist=True)
    assert d_no.action != Action.REFINE
    assert d_yes.action == Action.REFINE


def test_recommend_intent_takes_precedence() -> None:
    d = decide(_state(0.0, turn_index=0), extractor_intent="recommend")
    assert d.action == Action.RECOMMEND


def test_high_context_score_recommends() -> None:
    d = decide(_state(0.55))
    assert d.action == Action.RECOMMEND


def test_turn_cap_forces_recommend_even_low_context() -> None:
    d = decide(_state(0.0, turn_index=3), max_clarify_turns=3)
    assert d.action == Action.RECOMMEND


def test_default_branch_is_clarify_with_question() -> None:
    d = decide(_state(0.0, turn_index=0))
    assert d.action == Action.CLARIFY
    assert d.next_question == CLARIFY_QUESTIONS["role"]


def test_clarify_skips_already_asked_slot() -> None:
    d = decide(_state(0.0, turn_index=0, asked=("role",)))
    assert d.action == Action.CLARIFY
    assert d.next_question == CLARIFY_QUESTIONS["skills"]


def test_clarify_falls_back_to_freeform_when_all_asked() -> None:
    asked = ("role", "skills", "seniority", "test_type", "duration", "language")
    d = decide(_state(0.0, turn_index=0, asked=asked))
    assert d.action == Action.CLARIFY
    assert d.next_question is not None
    assert d.next_question not in CLARIFY_QUESTIONS.values()
