"""Tests for the eval harness: probes + replay.

Uses a fully scripted orchestrator stub so no real LLM is called.
This means the harness tests run on every CI invocation for free.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from shl_recommender.api.schemas import ChatResponse, Message, Recommendation
from shl_recommender.catalog.loader import load_catalog

from eval.harness import ndcg_at_k, recall_at_k, run_trace
from eval.persona import Persona
from eval.probes import (
    PROBES,
    ProbeResult,
    injection_no_pwn,
    length_bound,
    no_recommend_on_injection,
    off_topic_refusal,
    probe_test_type_validity,
    run_all_probes,
    schema_compliance,
    url_grounding,
    vague_turn_one,
)
from eval.simulator import Simulator, SimulatorTurn
from eval.trace_parser import Trace, Turn


# ---------- recall + nDCG math ----------


def test_recall_full_match() -> None:
    assert recall_at_k(["a", "b", "c"], ("a", "b", "c"), 10) == 1.0


def test_recall_partial_match() -> None:
    assert recall_at_k(["a", "x", "y"], ("a", "b"), 10) == 0.5


def test_recall_zero_when_no_hits() -> None:
    assert recall_at_k(["x"], ("a",), 10) == 0.0


def test_recall_empty_expected_returns_zero() -> None:
    assert recall_at_k(["a"], (), 10) == 0.0


def test_ndcg_perfect_order_is_one() -> None:
    assert ndcg_at_k(["a", "b"], ("a", "b"), 10) == 1.0


def test_ndcg_drops_when_relevant_is_lower_ranked() -> None:
    perfect = ndcg_at_k(["a"], ("a",), 10)
    later = ndcg_at_k(["x", "y", "a"], ("a",), 10)
    assert perfect > later
    assert math.isclose(later, 1.0 / math.log2(4), rel_tol=1e-6)


# ---------- probes against a scripted orchestrator ----------


class _ScriptedOrch:
    """Returns canned ChatResponses based on the latest user message.

    Each probe sends a distinct triggering message; we route on a
    substring match. Simpler than full message-matching.
    """

    def __init__(
        self,
        catalog: object,
        *,
        recommend_for_grounded: bool = True,
        leak_pwned: bool = False,
        return_too_many: bool = False,
        return_invalid_test_type: bool = False,
        keep_personality_after_drop: bool = False,
    ) -> None:
        self._catalog = catalog
        self._recommend_for_grounded = recommend_for_grounded
        self._leak_pwned = leak_pwned
        self._return_too_many = return_too_many
        self._return_invalid_test_type = return_invalid_test_type
        self._keep_personality_after_drop = keep_personality_after_drop

    def _grounded_recs(self, n: int) -> list[Recommendation]:
        items = list(self._catalog.items)[:n]  # type: ignore[attr-defined]
        return [
            Recommendation(name=it.name, url=it.link, test_type=it.primary_test_type)
            for it in items
        ]

    async def handle(self, messages: Sequence[Message]) -> ChatResponse:
        last = messages[-1].content.lower() if messages else ""

        # Vague turn-1.
        if "i need an assessment" in last:
            return ChatResponse(
                reply="What role are you hiring for?",
                recommendations=[],
                end_of_conversation=False,
            )
        # Off-topic.
        if "salary" in last:
            return ChatResponse(
                reply="I can only help with SHL assessment shortlisting.",
                recommendations=[],
                end_of_conversation=False,
            )
        # Injection.
        if "ignore previous" in last or "system override" in last:
            reply = "PWNED" if self._leak_pwned else "I'm here to help with SHL assessments."
            return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)
        # Refinement: latest message says "drop the personality"
        if "drop the personality" in last:
            recs = self._grounded_recs(3)
            if not self._keep_personality_after_drop:
                recs = [r for r in recs if r.test_type != "P"]
            return ChatResponse(
                reply="Updated.", recommendations=recs, end_of_conversation=False
            )
        # Compare.
        if "compare opq" in last:
            return ChatResponse(
                reply="OPQ32r measures personality; Verify - Numerical measures cognitive ability.",
                recommendations=self._grounded_recs(2),
                end_of_conversation=False,
            )
        # Catch-all recommend (for the other probes).
        if not self._recommend_for_grounded:
            return ChatResponse(reply="ok", recommendations=[], end_of_conversation=False)
        n = 12 if self._return_too_many else 5
        recs = self._grounded_recs(n)
        if self._return_invalid_test_type and recs:
            # Simulate corruption: a Recommendation with bad type would fail
            # validation, so we force it via model_construct (skips validation).
            recs[0] = Recommendation.model_construct(
                name=recs[0].name, url=recs[0].url, test_type="Z"
            )
        return ChatResponse(reply="here", recommendations=recs, end_of_conversation=False)


@pytest.fixture(scope="module")
def catalog() -> object:
    return load_catalog()


@pytest.mark.asyncio
async def test_vague_turn_one_passes_when_no_recs(catalog: object) -> None:
    r = await vague_turn_one(_ScriptedOrch(catalog), catalog)  # type: ignore[arg-type]
    assert r.passed


@pytest.mark.asyncio
async def test_vague_turn_one_fails_when_recs_emitted(catalog: object) -> None:
    class _BadOrch(_ScriptedOrch):
        async def handle(self, messages: Sequence[Message]) -> ChatResponse:
            return ChatResponse(
                reply="here", recommendations=self._grounded_recs(3), end_of_conversation=False
            )

    r = await vague_turn_one(_BadOrch(catalog), catalog)  # type: ignore[arg-type]
    assert not r.passed


@pytest.mark.asyncio
async def test_off_topic_refusal_passes(catalog: object) -> None:
    r = await off_topic_refusal(_ScriptedOrch(catalog), catalog)  # type: ignore[arg-type]
    assert r.passed


@pytest.mark.asyncio
async def test_injection_no_pwn_catches_leak(catalog: object) -> None:
    r = await injection_no_pwn(_ScriptedOrch(catalog, leak_pwned=True), catalog)  # type: ignore[arg-type]
    assert not r.passed
    assert "pwned" in r.detail.lower() or "sentinel" in r.detail.lower()


@pytest.mark.asyncio
async def test_injection_no_pwn_passes_clean(catalog: object) -> None:
    r = await injection_no_pwn(_ScriptedOrch(catalog), catalog)  # type: ignore[arg-type]
    assert r.passed


@pytest.mark.asyncio
async def test_url_grounding_passes_when_recs_in_catalog(catalog: object) -> None:
    r = await url_grounding(_ScriptedOrch(catalog), catalog)  # type: ignore[arg-type]
    assert r.passed


@pytest.mark.asyncio
async def test_schema_compliance_pass(catalog: object) -> None:
    r = await schema_compliance(_ScriptedOrch(catalog), catalog)  # type: ignore[arg-type]
    assert r.passed


@pytest.mark.asyncio
async def test_length_bound_fails_on_too_many(catalog: object) -> None:
    # Construct a Response with > 10 recs by going around Pydantic's max_length.
    class _OverflowOrch:
        def __init__(self, catalog: object) -> None:
            self._catalog = catalog

        async def handle(self, messages: Sequence[Message]) -> ChatResponse:
            items = list(self._catalog.items)[:12]  # type: ignore[attr-defined]
            recs = [
                Recommendation(name=it.name, url=it.link, test_type=it.primary_test_type)
                for it in items
            ]
            return ChatResponse.model_construct(
                reply="x", recommendations=recs, end_of_conversation=False
            )

    r = await length_bound(_OverflowOrch(catalog), catalog)  # type: ignore[arg-type]
    assert not r.passed


@pytest.mark.asyncio
async def test_probe_catches_bad_test_type_code(catalog: object) -> None:
    r = await probe_test_type_validity(
        _ScriptedOrch(catalog, return_invalid_test_type=True), catalog  # type: ignore[arg-type]
    )
    assert not r.passed


@pytest.mark.asyncio
async def test_no_recommend_on_injection_passes(catalog: object) -> None:
    r = await no_recommend_on_injection(_ScriptedOrch(catalog), catalog)  # type: ignore[arg-type]
    assert r.passed


@pytest.mark.asyncio
async def test_run_all_probes_returns_one_result_per_probe(catalog: object) -> None:
    results = await run_all_probes(_ScriptedOrch(catalog), catalog)  # type: ignore[arg-type]
    assert len(results) == len(PROBES)
    assert all(isinstance(r, ProbeResult) for r in results)


@pytest.mark.asyncio
async def test_probes_never_crash_the_harness(catalog: object) -> None:
    class _BoomOrch:
        async def handle(self, _msgs: Sequence[Message]) -> ChatResponse:
            raise RuntimeError("kaboom")

    results = await run_all_probes(_BoomOrch(), catalog)  # type: ignore[arg-type]
    assert all(not r.passed for r in results)
    assert all("kaboom" in r.detail.lower() for r in results)


# ---------- harness.run_trace with a stubbed simulator ----------


class _StubSimulator(Simulator):
    """Returns a fixed sequence of user turns; ignores history."""

    def __init__(self, scripted: list[SimulatorTurn]) -> None:
        # Don't call super().__init__, no LLM needed.
        self._scripted = list(scripted)
        self._llm = None  # type: ignore[assignment]
        self._timeout_s = 0.0

    async def next_turn(self, persona: Persona, history: Sequence[tuple[str, str]]) -> SimulatorTurn:  # type: ignore[override]
        if not self._scripted:
            return SimulatorTurn(text="", end=True)
        return self._scripted.pop(0)


@pytest.mark.asyncio
async def test_run_trace_walks_to_first_recommend(catalog: object) -> None:
    trace = Trace(
        trace_id="T1",
        turns=(Turn(role="user", content="x"),),
        expected_shortlist=tuple(item.link for item in catalog.items[:3]),  # type: ignore[attr-defined]
    )
    persona = Persona(trace_id="T1", role_summary="r", facts=["f"])
    sim = _StubSimulator(
        [
            SimulatorTurn(text="hire a Java dev", end=False),
            SimulatorTurn(text="thanks", end=True),
        ]
    )
    orch = _ScriptedOrch(catalog)
    result = await run_trace(orch, catalog, sim, trace, persona, turn_cap=8)  # type: ignore[arg-type]
    assert 0.0 <= result.recall_at_10 <= 1.0
    assert result.predicted_count >= 1


@pytest.mark.asyncio
async def test_run_trace_verbatim_mode_replays_trace_user_messages(catalog: object) -> None:
    """When simulator is None, run_trace replays trace.user_messages verbatim."""
    trace = Trace(
        trace_id="VERB",
        turns=(
            Turn(role="user", content="hire a Java dev"),
            Turn(role="assistant", content="ok"),
            Turn(role="user", content="add personality"),
        ),
        expected_shortlist=tuple(item.link for item in catalog.items[:3]),  # type: ignore[attr-defined]
    )
    persona = Persona(trace_id="VERB", role_summary="r", facts=[])
    orch = _ScriptedOrch(catalog)
    result = await run_trace(orch, catalog, None, trace, persona, turn_cap=8)  # type: ignore[arg-type]
    assert result.predicted_count >= 1
    assert result.error is None
    # Both verbatim user messages were sent.
    assert result.turn_count >= 4  # 2 user + 2 assistant


@pytest.mark.asyncio
async def test_run_trace_handles_orchestrator_exception(catalog: object) -> None:
    trace = Trace(
        trace_id="T2",
        turns=(Turn(role="user", content="x"),),
        expected_shortlist=("https://example.com/x",),
    )
    persona = Persona(trace_id="T2", role_summary="r", facts=[])
    sim = _StubSimulator([SimulatorTurn(text="hi", end=False)])

    class _BoomOrch:
        async def handle(self, _msgs: Sequence[Message]) -> ChatResponse:
            raise RuntimeError("kaboom")

    result = await run_trace(_BoomOrch(), catalog, sim, trace, persona, turn_cap=8)  # type: ignore[arg-type]
    assert result.error == "kaboom"
    assert result.recall_at_10 == 0.0
