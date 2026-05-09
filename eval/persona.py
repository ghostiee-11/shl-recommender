"""Persona extraction from reference dialogues.

The 10 reference traces (``Cn.md``) record the *expected* assistant
behavior, they don't ship a separate persona/facts JSON. SHL's own
evaluator constructs a persona from each trace at grading time. We
mirror that:

1. Read all user messages from a trace.
2. Ask an LLM (Gemini Flash; cheap, JSON-mode) to summarize them
   into a persona statement and a list of fact bullets.
3. Cache the result to ``eval/traces/personas.json`` so the eval
   harness never re-runs this step.

This is a one-time extraction step. Anyone can re-run it with
``python -m eval.persona`` after editing a trace.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from shl_recommender.llm.base import LLMClient, LLMError, LLMMessage
from shl_recommender.llm.gemini_client import GeminiLLM

from .trace_parser import Trace, load_traces

PERSONAS_PATH = Path(__file__).resolve().parent / "traces" / "personas.json"

logger = logging.getLogger(__name__)


class Persona(BaseModel):
    """Per-trace persona shape used by the simulator."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    role_summary: str = Field(description="One-line description of who the user is")
    facts: list[str] = Field(description="Discrete things the user knows / wants")


_EXTRACT_PROMPT = """Below are the user-only turns of a recruiter's conversation
with an SHL assessment recommender. Summarize the user as a JSON object:

{
  "role_summary": "one short sentence about who the user is and what they're hiring for",
  "facts": ["concrete fact 1", "concrete fact 2", ...]
}

Each fact should be a single short sentence the user knows about the
hire (role, level, skills, test type preference, duration constraint,
language). Do NOT invent facts that aren't in the user turns.

Output ONLY the JSON. No prose.
"""


async def _extract_one(llm: LLMClient, trace: Trace) -> Persona:
    transcript = "\n".join(f"- {m}" for m in trace.user_messages)
    result = await llm.complete(
        [
            LLMMessage(role="system", content=_EXTRACT_PROMPT),
            LLMMessage(role="user", content=transcript),
        ],
        json_mode=True,
        temperature=0.0,
        max_output_tokens=512,
        timeout_s=15.0,
    )
    raw = json.loads(result.text)
    return Persona(
        trace_id=trace.trace_id,
        role_summary=str(raw.get("role_summary", "")),
        facts=[str(f) for f in raw.get("facts", []) if f],
    )


def load_personas() -> dict[str, Persona]:
    """Read the cached personas. Returns empty dict if file missing."""
    if not PERSONAS_PATH.exists():
        return {}
    raw = json.loads(PERSONAS_PATH.read_text())
    out: dict[str, Persona] = {}
    for entry in raw:
        try:
            p = Persona.model_validate(entry)
            out[p.trace_id] = p
        except ValidationError:
            logger.warning("personas_cache_skip_invalid_entry", extra={"entry": entry})
    return out


async def main() -> None:
    """CLI: extract personas for every trace and write the cache.

    Uses Groq by preference, Gemini's free tier is 5 RPM, which 429s
    on the 7th trace. Groq's ~30 RPM comfortably absorbs the 10 calls.
    """
    if not (os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise SystemExit("GROQ_API_KEY or GEMINI_API_KEY must be set")

    if os.environ.get("GROQ_API_KEY"):
        from shl_recommender.llm.groq_client import GroqLLM

        llm: LLMClient = GroqLLM(os.environ["GROQ_API_KEY"])
    else:
        llm = GeminiLLM(os.environ["GEMINI_API_KEY"])

    traces = load_traces()
    personas: list[Persona] = []
    for tr in traces:
        try:
            p = await _extract_one(llm, tr)
        except (LLMError, json.JSONDecodeError, ValidationError) as exc:
            print(f"  {tr.trace_id}: extraction failed → {exc}; using fallback")
            p = Persona(
                trace_id=tr.trace_id,
                role_summary=f"Recruiter from trace {tr.trace_id}",
                facts=list(tr.user_messages),
            )
        personas.append(p)
        print(f"  {tr.trace_id}: {p.role_summary}  ({len(p.facts)} facts)")

    PERSONAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PERSONAS_PATH.write_text(
        json.dumps([p.model_dump(mode="json") for p in personas], indent=2)
    )
    print(f"\nWrote {len(personas)} personas to {PERSONAS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
