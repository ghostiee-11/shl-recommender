"""LLM-composed reply text.

The orchestrator's *recommendations* and *URLs* are deterministic and
strictly grounded, those never go through an LLM. But the natural
language that wraps them used to be templated and read robotic. This
module composes a 2–4 sentence reply that sounds like a senior SHL
practitioner.

Hard guarantees we keep:

* Items are passed in **with their facts**. The LLM is instructed to
  use only those facts. We don't pass the catalog.
* On any LLM failure (timeout, schema, rate-limit), we fall back to
  the deterministic text from the orchestrator. The system never
  hangs or returns junk.
* JSON-mode is **not** used here, the reply is prose, and asking for
  JSON often makes the model emit an over-formal "summary" tone.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from shl_recommender.catalog.models import Assessment
from shl_recommender.llm.base import LLMClient, LLMError, LLMMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ItemSummary:
    """Compact, grounded view of one assessment for the composer prompt."""

    name: str
    test_type: str  # single-letter primary code
    test_type_long: str
    duration: str
    description_first_sentence: str

    @classmethod
    def from_item(cls, item: Assessment) -> "ItemSummary":
        from shl_recommender.catalog.models import CODE_TO_LONG

        first = item.description.split(". ")[0].rstrip(".")
        return cls(
            name=item.name,
            test_type=item.primary_test_type,
            test_type_long=CODE_TO_LONG[item.primary_test_type],
            duration=item.duration or "duration unspecified",
            description_first_sentence=first,
        )


# Punctuation directive applied to every composer system prompt: the
# project explicitly avoids em dashes anywhere in user-facing output.
_PUNCT_RULES = (
    "PUNCTUATION RULES (mandatory):\n"
    "- Do NOT use em dashes (the long dash character). Use commas, "
    "semicolons, or periods instead.\n"
    "- Do NOT use en dashes (the medium dash). Use hyphens or commas.\n"
    "- A regular hyphen (-) inside compound words is fine.\n"
)


_RECOMMEND_SYSTEM = """You are an SHL assessment consultant writing the
agent's natural-language reply for the *recommend* turn.

Style guide:
- 2 to 3 sentences. No bullet lists, no markdown.
- First sentence acknowledges what the user described in plain terms,
  using their own framing where natural.
- Second/third sentences name 2-3 of the recommended items by their
  full name and explain in one breath why they fit (test type, what
  they measure, duration if it's notable).
- Use ONLY the facts in the provided ITEMS block. Never invent.
- Do not start with "Sure", "Certainly", "Of course", or "I have ...".
  Skip filler. Sound like a working professional.
- Do not list ALL items if there are more than 3, say "and N more
  in the panel" instead.
- End with no question. The shortlist itself is the next step.

""" + _PUNCT_RULES + """
Output: the reply text only. No JSON, no markdown, no preface."""


_COMPARE_SYSTEM = """You are an SHL assessment consultant writing the
agent's natural-language reply for the *compare* turn.

Style guide:
- 3 to 4 sentences. Plain prose. No bullets, no markdown tables.
- Start with the substantive difference (what each instrument
  measures, when each is the right choice).
- Mention each item's primary test type and duration once.
- Use ONLY the facts in the provided ITEMS block. Never invent
  scoring details, validity numbers, or competitor comparisons.
- Do not begin with "Sure", "Both ...", or filler. Sound like an
  expert who knows the catalog cold.

""" + _PUNCT_RULES + """
Output: the reply text only. No JSON, no markdown, no preface."""


_REFINE_SYSTEM = """You are an SHL assessment consultant writing the
agent's natural-language reply for the *refine* turn.

The user has just edited the brief. Style guide:

- 2 to 3 sentences. Plain prose. No bullets.
- First sentence acknowledges the edit specifically (e.g. "Dropping
  the personality tests, here's the updated set", adapt to the
  actual edit).
- Then name 2-3 of the updated items and explain the fit, using
  ONLY the provided ITEMS facts.
- No filler openings. No questions at the end.

""" + _PUNCT_RULES + """
Output: the reply text only."""


# Codepoint-based constants so a future em-dash sweep over source files
# can't accidentally neutralise this function (which is the last line
# of defense before LLM prose hits the user).
_EM_DASH = "—"
_EN_DASH = "–"


def _strip_dashes(text: str) -> str:
    """Final defense: replace em/en dashes the LLM may emit anyway.

    Spaced occurrences become a comma; tight occurrences become a
    plain hyphen.
    """
    text = text.replace(f" {_EM_DASH} ", ", ").replace(_EM_DASH, "-")
    text = text.replace(f" {_EN_DASH} ", ", ").replace(_EN_DASH, "-")
    return text


def _format_items(items: Sequence[ItemSummary]) -> str:
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(
            f"{i}. {it.name}, {it.test_type} ({it.test_type_long}) · "
            f"{it.duration} · {it.description_first_sentence}"
        )
    return "\n".join(lines)


async def compose_reply(
    llm: LLMClient,
    *,
    action: str,  # "recommend" | "refine" | "compare"
    user_message: str,
    items: Sequence[Assessment],
    fallback: str,
    timeout_s: float = 6.0,
) -> str:
    """Return an LLM-written reply, or ``fallback`` on any failure.

    Never raises. The orchestrator passes its deterministic text as
    ``fallback`` so the user sees something useful even when the LLM
    is down.
    """
    if not items:
        return fallback

    system = {
        "recommend": _RECOMMEND_SYSTEM,
        "refine": _REFINE_SYSTEM,
        "compare": _COMPARE_SYSTEM,
    }.get(action, _RECOMMEND_SYSTEM)

    summaries = [ItemSummary.from_item(it) for it in items]
    user_prompt = (
        f"USER REQUEST:\n{user_message.strip() or '(no specific message)'}"
        f"\n\nITEMS (use only these facts):\n{_format_items(summaries)}\n"
    )

    try:
        result = await llm.complete(
            [
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user_prompt),
            ],
            json_mode=False,
            temperature=0.2,
            max_output_tokens=220,
            timeout_s=timeout_s,
        )
    except LLMError:
        logger.warning("composer_llm_error_using_fallback", extra={"action": action})
        return fallback

    text = result.text.strip()
    if not text:
        return fallback

    # Defensive cleanup: strip a stray markdown header / preface that
    # some models add despite being told not to. Only trim leading
    # noise; we never alter the content semantically.
    for prefix in ("Reply:", "Response:", "Answer:", "## ", "# "):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # Belt-and-braces: even with the prompt rule, models occasionally
    # emit em / en dashes. Normalize them to plain punctuation so the
    # frontend never displays one.
    text = _strip_dashes(text)
    return text
