"""Persona-driven user simulator for replay evaluation.

Mirrors SHL's described harness behavior: the simulator plays the role
of a recruiter with a fixed fact set. It answers the agent's questions
truthfully when the answer is in its facts, says "no preference" when
the answer is *not* in its facts, and ends the conversation when the
agent provides a shortlist.

Why a real LLM is needed:

* The simulator must understand the agent's question (which can vary
  across runs because our agent is non-deterministic in wording) and
  pick the relevant fact.
* Hand-rolling regex matchers for every possible clarifying question
  is brittle and would create a false-positive "passes the eval"
  signal that doesn't transfer to SHL's grader.

Determinism: ``temperature=0``, low max-tokens, JSON output. Two
separate calls per turn (``next_message`` returns either a user reply
or a "stop" sentinel) keep the contract narrow.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from shl_recommender.llm.base import LLMClient, LLMError, LLMMessage

from .persona import Persona

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SimulatorTurn:
    """One step the simulator takes."""

    text: str
    end: bool


_SIM_SYSTEM_PROMPT = """You are simulating a recruiter using an SHL assessment recommendation agent.

Your role: {role_summary}

Facts you know about the role you're hiring for:
{fact_lines}

Rules, follow strictly:

1. Answer the agent's questions in YOUR voice as the recruiter.
2. Use ONLY the facts above. NEVER invent details.
3. If asked about something NOT in your facts, say "no preference"
   or "I don't have a strong view on that".
4. Keep your reply under 2 short sentences.
5. End the conversation as soon as the agent gives you a concrete
   list of named recommendations.
6. Do NOT add markdown, JSON, or formatting. Plain prose only.

Output format, a single JSON object:
{{
  "text": "your next message as the user",
  "end": true|false
}}

Set ``end: true`` when:
* The agent has provided a shortlist of named assessments AND your
  follow-up would only be a thank-you.
* You have nothing more to add and want to wrap up.
"""


def _format_persona(persona: Persona) -> str:
    fact_lines = "\n".join(f"- {f}" for f in persona.facts)
    return _SIM_SYSTEM_PROMPT.format(
        role_summary=persona.role_summary,
        fact_lines=fact_lines,
    )


class _SimReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=0, max_length=500)
    end: bool = False


class Simulator:
    """Stateless simulator. Each ``next_turn`` produces the user's next message."""

    __slots__ = ("_llm", "_timeout_s")

    def __init__(self, llm: LLMClient, *, timeout_s: float = 8.0) -> None:
        self._llm = llm
        self._timeout_s = timeout_s

    async def next_turn(
        self,
        persona: Persona,
        history: Sequence[tuple[str, str]],  # [(role, content)], full prior dialogue
    ) -> SimulatorTurn:
        """Return the simulator's next user message + stop flag.

        ``history`` includes both prior user and assistant turns. The
        simulator sees the assistant's last reply and decides what to
        say next.
        """
        # Render the dialogue in chronological order. Assistant turns
        # become the "prompt" the simulator is responding to.
        rendered = "\n".join(f"{role.upper()}: {content}" for role, content in history)
        if not rendered:
            # First turn: kick off the conversation in the recruiter's voice.
            rendered = "AGENT: How can I help you today?"

        try:
            result = await self._llm.complete(
                [
                    LLMMessage(role="system", content=_format_persona(persona)),
                    LLMMessage(role="user", content=rendered),
                ],
                json_mode=True,
                temperature=0.0,
                max_output_tokens=256,
                timeout_s=self._timeout_s,
            )
        except LLMError:
            logger.warning("simulator_llm_error_stopping")
            return SimulatorTurn(text="", end=True)

        try:
            parsed = _SimReply.model_validate(json.loads(result.text))
        except (json.JSONDecodeError, ValidationError):
            logger.warning("simulator_invalid_output_stopping")
            return SimulatorTurn(text="", end=True)

        return SimulatorTurn(text=parsed.text.strip(), end=parsed.end)
