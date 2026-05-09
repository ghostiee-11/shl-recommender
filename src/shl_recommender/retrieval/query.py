"""Query construction and expansion for retrieval.

Two responsibilities, kept separate so each can be reasoned about and
tested in isolation:

1. :func:`build_query`, turn a conversation history (or any list of
   user utterances) into a single search string. It applies recency
   weighting (later turns repeat to bias retrieval toward them),
   deduplicates tokens, and bounds length so the dense encoder isn't
   given a wall of context where signal is lost.

2. :func:`expand_query`, extend the query with controlled synonyms
   drawn from SHL's own taxonomy: job levels and test-type long names.
   This closes the "user says X, catalog says Y" gap (e.g.
   "senior leadership" ↔ "executive / director", "personality" ↔
   "OPQ / behaviour"). Synonyms are static and reviewed; we deliberately
   avoid generic thesauri which introduce noise.
"""

from __future__ import annotations

from collections.abc import Iterable

from .bm25 import tokenize

# Each row maps a *trigger token* set to *expansion tokens*. If any
# trigger token is present in the query, the expansion tokens are
# appended (deduplicated). All terms lowercased and pre-tokenized.
#
# Sourced from:
# * SHL job_levels enumeration (Director, Executive, Manager, …)
# * SHL test-type long names (Personality & Behavior, Knowledge & Skills, …)
# * Common recruiter vocabulary observed in the 10 reference dialogues.
SYNONYMS: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = (
    # ----- seniority / job level -----
    (frozenset({"senior", "leadership", "exec", "executive", "cxo", "ceo", "cto", "cfo", "vp"}),
        ("executive", "director", "leadership")),
    (frozenset({"director"}), ("executive", "leadership")),
    (frozenset({"manager", "supervisor", "lead"}),
        ("manager", "front", "line", "supervisor")),
    (frozenset({"junior", "entry", "graduate", "intern", "fresher"}),
        ("entry", "level", "graduate")),
    (frozenset({"mid", "midlevel", "intermediate"}),
        ("mid", "professional")),
    (frozenset({"professional", "individual", "contributor", "ic"}),
        ("professional", "individual", "contributor")),
    # ----- test type long names -----
    (frozenset({"personality", "behaviour", "behavior", "behavioural", "behavioral",
                "opq", "opq32", "opq32r", "trait", "traits"}),
        ("personality", "behavior", "opq")),
    (frozenset({"knowledge", "skill", "skills", "technical", "coding", "programming"}),
        ("knowledge", "skills", "technical")),
    (frozenset({"ability", "aptitude", "cognitive", "reasoning", "verify",
                "numerical", "verbal", "logical", "inductive", "deductive"}),
        ("ability", "aptitude", "verify", "reasoning")),
    (frozenset({"sjt", "situational", "judgement", "judgment", "biodata"}),
        ("biodata", "situational", "judgment")),
    (frozenset({"competency", "competencies", "competence", "competences", "ucf"}),
        ("competencies", "ucf", "competency")),
    (frozenset({"simulation", "simulations", "roleplay", "role-play", "exercise", "exercises"}),
        ("simulations", "exercises")),
    (frozenset({"360", "feedback", "development", "developmental"}),
        ("development", "360", "feedback")),
    # ----- common role keywords (light touch) -----
    (frozenset({"java"}), ("java",)),
    (frozenset({"python"}), ("python",)),
    (frozenset({"sales", "selling"}), ("sales",)),
    (frozenset({"customer", "service", "support"}), ("customer", "service")),
    (frozenset({"call", "centre", "center"}), ("call", "center", "contact")),
    (frozenset({"selection", "hiring", "recruit", "recruiting"}),
        ("selection", "hiring")),
)


def expand_query(query: str) -> str:
    """Append controlled synonyms based on trigger tokens in ``query``.

    Returns a new string with the original query followed by the
    expansion tokens (space-separated). Order-preserving and
    deduplicated. Idempotent.
    """
    tokens = set(tokenize(query))
    if not tokens:
        return query
    additions: list[str] = []
    for triggers, expansions in SYNONYMS:
        if tokens & triggers:
            for term in expansions:
                if term not in tokens and term not in additions:
                    additions.append(term)
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


def build_query(
    user_messages: Iterable[str],
    *,
    max_tokens: int = 120,
    last_turn_weight: int = 2,
) -> str:
    """Compose a retrieval query from an ordered list of user messages.

    * The most recent user message is repeated ``last_turn_weight``
      times, recency matters because refinement turns ("actually,
      add personality") should outweigh stale context.
    * Earlier turns are kept once.
    * The result is truncated to ``max_tokens`` whitespace tokens to
      keep the dense encoder honest (bge-small was trained on shorter
      passages; very long inputs hurt retrieval quality).
    """
    msgs = [m.strip() for m in user_messages if m and m.strip()]
    if not msgs:
        return ""
    parts: list[str] = list(msgs[:-1])
    parts.extend([msgs[-1]] * max(1, last_turn_weight))
    composed = " ".join(parts)
    words = composed.split()
    if len(words) <= max_tokens:
        return composed
    return " ".join(words[-max_tokens:])
