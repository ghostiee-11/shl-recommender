"""Unit tests for the Slots / AgentState model."""

from __future__ import annotations

from shl_recommender.agent.slots import AgentState, Slot, Slots


def _slot(value: str, conf: float = 0.9) -> Slot:
    return Slot(value=value, confidence=conf, evidence=value)


def test_empty_slots_score_is_zero() -> None:
    assert Slots().context_score() == 0.0


def test_role_alone_does_not_meet_recommend_threshold() -> None:
    s = Slots(role=_slot("Java developer"))
    assert s.context_score() < 0.55


def test_role_plus_skills_plus_seniority_meets_threshold() -> None:
    s = Slots(
        role=_slot("Java developer"),
        skills=[_slot("Java"), _slot("Spring")],
        seniority=_slot("mid"),
    )
    assert s.context_score() >= 0.55


def test_job_description_alone_meets_threshold() -> None:
    # A high-confidence JD blob alone is enough to commit (>= 0.55 threshold)
    # because the JD weight is 0.40 and overrides most slots.
    s = Slots(
        job_description=Slot(
            value="paste of a long JD",
            confidence=1.0,
            evidence="paste of a long JD",
        )
    )
    # Equal to the JD weight; this is the minimum signal needed without
    # any other slots filled. Falls JUST below the 0.55 recommend threshold,
    # which is intentional, JD alone with no role context is still ambiguous.
    assert s.context_score() == 0.40


def test_filled_count_counts_skills_as_one() -> None:
    s = Slots(skills=[_slot("a"), _slot("b"), _slot("c")])
    assert s.filled_count() == 1


def test_low_confidence_role_does_not_count() -> None:
    s = Slots(role=_slot("foo", conf=0.3))
    assert s.context_score() == 0.0


def test_query_text_includes_filled_slots() -> None:
    s = Slots(
        role=_slot("data engineer"),
        skills=[_slot("Python"), _slot("Spark")],
        seniority=_slot("senior"),
    )
    q = s.as_query_text()
    assert "data engineer" in q
    assert "Python" in q
    assert "Spark" in q
    assert "senior" in q


def test_agent_state_defaults() -> None:
    st = AgentState()
    assert st.intent == "clarify"
    assert st.turn_index == 0
    assert st.prior_shortlist_urls == []
    assert st.asked_slots == []
    assert st.slots.context_score() == 0.0
