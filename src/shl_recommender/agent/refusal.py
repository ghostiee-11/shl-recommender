"""Refusal layer.

The refusal layer is **deliberately deterministic**. LLM-generated
refusal text is bypassable (a sufficiently clever prompt-injection
makes the model stop refusing). Templates are not.

Layered detection:

1. **Hard-deny patterns**, prompt-injection markers, role-override
   attempts, explicit off-topic flags. Any match → immediate refuse.
2. **Soft-deny heuristics**, out-of-scope keyword set (legal, salary,
   etc.). On a match we *consider* refusing but consult the LLM
   tiebreak to avoid false positives ("salary range for the test
   reports" is on-topic; "what's the average Java dev salary in NYC"
   is not).
3. **LLM tiebreak**, only when soft-deny matched. Cheap (Gemini
   Flash, ≤ 64 output tokens). Returns ON_TOPIC / OFF_TOPIC.

Refusal templates are deterministic (string constants), short, and
explain *why* without revealing the system prompt.
"""

from __future__ import annotations

import json
import logging
import re

from shl_recommender.llm.base import LLMClient, LLMError, LLMMessage

from .prompts import REFUSAL_TIEBREAK_PROMPT_V1

logger = logging.getLogger(__name__)

# ---- pattern sets ----

# Anything in this set is an immediate refuse.
_HARD_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?)",
        r"disregard\s+(?:all\s+)?(?:previous|prior|above)",
        r"reveal\s+(?:your|the)\s+(?:system\s+)?prompt",
        r"what\s+(?:is|are)\s+your\s+(?:system\s+)?(?:prompt|instructions?)",
        r"you\s+are\s+now\s+(?:a|an)\s+",
        r"act\s+as\s+(?:a|an)\s+(?!recruiter|hiring)",  # allow "act as a recruiter"
        r"pretend\s+(?:to\s+be|you\s+are)",
        r"jailbreak",
        r"DAN\s+mode",
    )
)

# Soft patterns, likely off-topic but ambiguous enough to consult LLM.
_SOFT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bsalary\b",
        r"\bcompensation\b",
        r"\bvisa\b|\bH-?1B\b",
        r"\blegal(?:ly)?\b|\blawsuit\b|\bemployment\s+law\b",
        r"\bbenefit(s)?\b\s+(package|plan)",
        r"how\s+(?:do|should)\s+I\s+(?:hire|recruit|interview|onboard)",
        r"\bdiscriminat",
        r"\beoc\b|\beeoc\b",
    )
)

# ---- refusal templates (deterministic) ----

REFUSAL_OUT_OF_SCOPE = (
    "I can only help with selecting and comparing SHL assessments. "
    "For that question you'll want a different resource, happy to "
    "help if you'd like to shortlist assessments instead."
)

REFUSAL_INJECTION = (
    "I'm here only to help shortlist SHL assessments. "
    "Tell me about the role you're hiring for and I'll narrow down "
    "the right tests."
)


def hard_deny(message: str) -> bool:
    """Return True if the message matches a hard-deny pattern."""
    return any(p.search(message) for p in _HARD_PATTERNS)


def soft_deny(message: str) -> bool:
    """Return True if the message matches a soft-deny pattern."""
    return any(p.search(message) for p in _SOFT_PATTERNS)


async def llm_tiebreak(llm: LLMClient, message: str, *, timeout_s: float = 4.0) -> bool:
    """Returns True if the LLM classifies the message as OFF_TOPIC.

    Used only when ``soft_deny`` matched. On any failure we
    conservatively return True (refuse), the user can always
    rephrase, and a wrong refusal is far less harmful than a
    wrong concession.
    """
    try:
        result = await llm.complete(
            [
                LLMMessage(role="system", content=REFUSAL_TIEBREAK_PROMPT_V1),
                LLMMessage(role="user", content=message),
            ],
            json_mode=True,
            temperature=0.0,
            max_output_tokens=32,
            timeout_s=timeout_s,
        )
        raw = json.loads(result.text)
        return str(raw.get("label", "")).upper() == "OFF_TOPIC"
    except (LLMError, json.JSONDecodeError, ValueError):
        logger.warning("refusal_tiebreak_failed_assuming_off_topic")
        return True


async def should_refuse(llm: LLMClient, message: str) -> tuple[bool, str]:
    """Decide whether to refuse and which template to emit.

    Returns ``(refuse?, template_text)``. When ``refuse?`` is False
    the second element is the empty string.
    """
    if hard_deny(message):
        return True, REFUSAL_INJECTION
    if soft_deny(message):
        if await llm_tiebreak(llm, message):
            return True, REFUSAL_OUT_OF_SCOPE
    return False, ""
