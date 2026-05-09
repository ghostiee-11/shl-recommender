"""Eval harness, drives a persona simulator against the agent.

For each trace:

1. Look up its cached persona.
2. Run a multi-turn conversation: simulator emits a user message, the
   orchestrator handles it, the simulator sees the assistant reply and
   responds, and so on, capped at 8 turns total per spec.
3. Capture the *last* assistant turn that returned a non-empty
   recommendation list as the trace's final shortlist.
4. Compute Recall@10 + nDCG@10 against the trace's
   ``expected_shortlist`` parsed from the markdown.

This is the *replay* arm of the eval. Behavior probes are a separate
arm, exercised by :mod:`eval.probes`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from shl_recommender.agent.orchestrator import Orchestrator
from shl_recommender.api.schemas import Message
from shl_recommender.catalog.loader import CatalogIndex

from .persona import Persona
from .simulator import Simulator
from .trace_parser import Trace

TURN_CAP = 8


@dataclass(frozen=True, slots=True)
class TraceResult:
    trace_id: str
    expected_count: int
    predicted_count: int
    recall_at_10: float
    ndcg_at_10: float
    turn_count: int
    latency_s: float
    final_predicted_urls: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None


def recall_at_k(predicted: list[str], expected: tuple[str, ...], k: int) -> float:
    if not expected:
        return 0.0
    top = set(predicted[:k])
    return sum(1 for u in expected if u in top) / len(expected)


def ndcg_at_k(predicted: list[str], expected: tuple[str, ...], k: int) -> float:
    """Standard nDCG@K with binary relevance (∈ expected → relevant)."""
    if not expected:
        return 0.0
    rel_set = set(expected)
    dcg = 0.0
    for i, url in enumerate(predicted[:k], start=1):
        if url in rel_set:
            dcg += 1.0 / math.log2(i + 1)
    ideal_n = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0


async def run_trace(
    orch: Orchestrator,
    catalog: CatalogIndex,
    simulator: Simulator | None,
    trace: Trace,
    persona: Persona,
    *,
    turn_cap: int = TURN_CAP,
) -> TraceResult:
    """Replay one trace against the agent.

    When ``simulator`` is provided, use it to generate adaptive user
    messages (LLM-driven). When ``simulator is None``, replay the
    trace's verbatim user messages, cheaper, deterministic, and
    immune to LLM rate limits. The trace's user messages are the
    most-faithful representation of the original conversation, so
    verbatim replay is the right *default*; the adaptive simulator
    is a stretch mode for stress-testing when quota allows.
    """
    history: list[Message] = []
    convo: list[tuple[str, str]] = []
    last_recs: list[str] = []
    turns = 0
    start = time.perf_counter()
    verbatim_iter = iter(trace.user_messages)

    while turns < turn_cap:
        if simulator is not None:
            sim_turn = await simulator.next_turn(persona, convo)
            if not sim_turn.text:
                break
            user_text = sim_turn.text
            should_end = sim_turn.end
        else:
            try:
                user_text = next(verbatim_iter)
            except StopIteration:
                break
            should_end = False

        history.append(Message(role="user", content=user_text))
        convo.append(("user", user_text))
        turns += 1

        try:
            resp = await orch.handle(history)
        except Exception as exc:  # noqa: BLE001
            return TraceResult(
                trace_id=trace.trace_id,
                expected_count=len(trace.expected_shortlist),
                predicted_count=len(last_recs),
                recall_at_10=0.0,
                ndcg_at_10=0.0,
                turn_count=turns,
                latency_s=time.perf_counter() - start,
                final_predicted_urls=tuple(last_recs),
                error=str(exc),
            )

        history.append(Message(role="assistant", content=resp.reply))
        convo.append(("assistant", resp.reply))
        turns += 1

        if resp.recommendations:
            last_recs = [str(r.url) for r in resp.recommendations]
        if resp.end_of_conversation or should_end:
            break

    elapsed = time.perf_counter() - start
    return TraceResult(
        trace_id=trace.trace_id,
        expected_count=len(trace.expected_shortlist),
        predicted_count=len(last_recs),
        recall_at_10=recall_at_k(last_recs, trace.expected_shortlist, 10),
        ndcg_at_10=ndcg_at_k(last_recs, trace.expected_shortlist, 10),
        turn_count=turns,
        latency_s=elapsed,
        final_predicted_urls=tuple(last_recs),
    )
