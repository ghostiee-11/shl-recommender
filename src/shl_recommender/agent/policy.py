"""Decision policy: pure function from agent state to next action.

This module encodes **all** the conversational choices the agent
makes. It is pure (no I/O, no LLM calls, no globals), which makes
every branch trivially unit-testable in a table-driven way.

Action vocabulary:

* ``CLARIFY``, emit a focused clarifying question, no recs.
* ``RECOMMEND``, run retrieval + emit a shortlist (1–10 items).
* ``REFINE``, re-run retrieval, prefer items consistent with the
  user's edit; recs are emitted again.
* ``COMPARE``, return a structured diff of named assessments.
* ``REFUSE``, emit a refusal template; no recs.

Decision rules (read top-to-bottom; first match wins):

1. Extractor said "refuse" → REFUSE.
2. Extractor said "compare" and listed ≥ 2 names → COMPARE.
3. Extractor said "refine" AND we have a prior shortlist → REFINE.
4. Extractor said "recommend" → RECOMMEND.
5. ``slots.context_score >= 0.55`` → RECOMMEND.
6. ``turn_index >= max_clarify_turns`` (default 3) → RECOMMEND
   (we leave headroom for refinement within the 8-turn cap).
7. Otherwise → CLARIFY.

Question selection within CLARIFY:

The next-question priority is fixed (role > skills > seniority >
test_type > duration > language). We never re-ask a slot already in
``state.asked_slots``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .slots import AgentState, Slots

DEFAULT_RECOMMEND_THRESHOLD = 0.55
DEFAULT_MAX_CLARIFY_TURNS = 3


class Action(str, Enum):
    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    REFINE = "refine"
    COMPARE = "compare"
    REFUSE = "refuse"


# Ordered question priority, highest information gain first.
QUESTION_PRIORITY: tuple[str, ...] = (
    "role",
    "skills",
    "seniority",
    "test_type",
    "duration",
    "language",
)

# Templated clarifying questions per missing slot.
CLARIFY_QUESTIONS: dict[str, str] = {
    "role": "Happy to help. What role are you hiring for, or what's the use case?",
    "skills": "Got it. What specific skills or technologies should the assessment cover?",
    "seniority": "What seniority level are you targeting, entry, mid, senior, leadership?",
    "test_type": (
        "Do you have a preference for the kind of test, knowledge, "
        "personality / behavior, cognitive ability, or a mix?"
    ),
    "duration": "Any constraint on how long the assessment should take?",
    "language": "Any required test languages?",
}


@dataclass(frozen=True, slots=True)
class Decision:
    """The policy's verdict for a turn."""

    action: Action
    next_question: str | None = None  # only set when action == CLARIFY
    compared_assessments: tuple[str, ...] = ()  # only set when action == COMPARE


def _missing_slots(slots: Slots) -> set[str]:
    out: set[str] = set()
    if slots.role is None:
        out.add("role")
    if not slots.skills:
        out.add("skills")
    if slots.seniority is None:
        out.add("seniority")
    if not slots.test_type_preference:
        out.add("test_type")
    if slots.duration_preference is None:
        out.add("duration")
    if slots.language_preference is None:
        out.add("language")
    return out


def _next_question(state: AgentState) -> str:
    """Pick the highest-priority slot we don't have yet and haven't already asked."""
    missing = _missing_slots(state.slots)
    asked = set(state.asked_slots)
    for name in QUESTION_PRIORITY:
        if name in missing and name not in asked:
            return CLARIFY_QUESTIONS[name]
    # We've asked for everything we know how to ask for; commit to a recommend.
    # The orchestrator will detect the mismatch and switch action, but as a
    # safe default we ask for free-form preference.
    return "Anything else I should keep in mind before I shortlist?"


def decide(
    state: AgentState,
    *,
    extractor_intent: str = "clarify",
    compared_assessments: tuple[str, ...] = (),
    has_prior_shortlist: bool = False,
    recommend_threshold: float = DEFAULT_RECOMMEND_THRESHOLD,
    max_clarify_turns: int = DEFAULT_MAX_CLARIFY_TURNS,
) -> Decision:
    """Pick the next action. Pure function; no I/O.

    Parameters mirror the rules in this module's docstring. The
    defaults are tuned for the 8-turn / 30-second budget.
    """
    if extractor_intent == "refuse":
        return Decision(action=Action.REFUSE)

    if extractor_intent == "compare" and len(compared_assessments) >= 2:
        return Decision(action=Action.COMPARE, compared_assessments=compared_assessments)

    if extractor_intent == "refine" and has_prior_shortlist:
        return Decision(action=Action.REFINE)

    if extractor_intent == "recommend":
        return Decision(action=Action.RECOMMEND)

    if state.slots.context_score() >= recommend_threshold:
        return Decision(action=Action.RECOMMEND)

    if state.turn_index >= max_clarify_turns:
        return Decision(action=Action.RECOMMEND)

    # Bias-to-commit: if we already asked at least one clarifying
    # question AND the user's reply gave us a meaningful slot (e.g.
    # role + at least one skill, or role + seniority), recommend even
    # if the weighted score is still below threshold. Recruiters
    # dislike multi-turn interrogations when they've already answered
    # the obvious question.
    if (
        state.turn_index >= 1
        and state.slots.role is not None
        and (state.slots.skills or state.slots.seniority is not None)
    ):
        return Decision(action=Action.RECOMMEND)

    return Decision(action=Action.CLARIFY, next_question=_next_question(state))
