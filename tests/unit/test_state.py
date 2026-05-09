"""Unit tests for state reconstruction (embedded hint round-trip)."""

from __future__ import annotations

from shl_recommender.agent.slots import AgentState, Slot, Slots
from shl_recommender.agent.state import (
    encode_hint,
    parse_hint,
    reconstruct_from_hint,
    strip_hint,
)
from shl_recommender.api.schemas import Message


def _example_state() -> AgentState:
    return AgentState(
        slots=Slots(
            role=Slot(value="Java developer", confidence=0.9, evidence="Java developer"),
            skills=[Slot(value="Java", confidence=0.9, evidence="Java")],
        ),
        intent="recommend",
        prior_shortlist_urls=["https://www.shl.com/x/", "https://www.shl.com/y/"],
        asked_slots=["role"],
        turn_index=2,
    )


def test_hint_round_trip() -> None:
    st = _example_state()
    hint = encode_hint(st)
    assert hint.startswith("<!--state:")
    assert hint.endswith("-->")
    parsed = parse_hint(f"some assistant text\n{hint}")
    assert parsed is not None
    assert parsed.slots.role and parsed.slots.role.value == "Java developer"
    assert parsed.intent == "recommend"
    assert parsed.prior_shortlist_urls == st.prior_shortlist_urls
    assert parsed.asked_slots == ["role"]
    assert parsed.turn_index == 2


def test_strip_hint_removes_comment() -> None:
    text = f"hello world\n{encode_hint(_example_state())}"
    assert strip_hint(text) == "hello world"


def test_parse_hint_returns_none_when_absent() -> None:
    assert parse_hint("plain assistant text") is None


def test_parse_hint_returns_none_on_malformed_json() -> None:
    assert parse_hint("<!--state:not json-->") is None


def test_reconstruct_from_hint_walks_messages_newest_first() -> None:
    older = AgentState(turn_index=0, intent="clarify")
    newer = AgentState(turn_index=2, intent="recommend")
    msgs = [
        Message(role="user", content="hi"),
        Message(role="assistant", content=f"first {encode_hint(older)}"),
        Message(role="user", content="more"),
        Message(role="assistant", content=f"second {encode_hint(newer)}"),
    ]
    state = reconstruct_from_hint(msgs)
    assert state is not None
    assert state.turn_index == 2
    assert state.intent == "recommend"


def test_reconstruct_returns_none_when_no_assistant_hint() -> None:
    msgs = [Message(role="user", content="hi"), Message(role="assistant", content="plain")]
    assert reconstruct_from_hint(msgs) is None
