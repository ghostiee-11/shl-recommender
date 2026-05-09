"""Stateless state reconstruction.

The ``/chat`` endpoint is stateless per spec, every request carries
the full conversation history. We reconstruct :class:`AgentState`
from history with two complementary mechanisms:

1. **Embedded hint**, when the agent emits an assistant turn it
   appends a hidden HTML comment carrying a JSON-serialized state:
   ``<!--state:{...}-->``. The comment is invisible in rendered
   markdown but survives verbatim in the message string. On the
   next turn we parse it back. This is the **fast path**.

2. **LLM fallback**, if no hint is found (first turn, hint stripped
   by an upstream proxy, or parse failure), we have the orchestrator
   re-run :func:`extract_slots` over the full history to rebuild
   slots from scratch. This is the **safe path**.

The hint is opt-in via ``embed_hint`` so deployments paranoid about
HTML-comment leakage can disable it; the agent stays correct because
the LLM fallback is always available.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from pydantic import ValidationError

from shl_recommender.api.schemas import Message  # noqa: TC001, keep direct import; small project

from .slots import AgentState

# Note: ``\{.*\}`` is GREEDY on purpose. The state payload contains
# nested JSON braces (slots is a dict-of-dicts), so a non-greedy
# match would stop at the first inner ``}`` and produce invalid JSON.
# The trailing ``-->`` anchor keeps the match well-defined.
_HINT_RE = re.compile(r"<!--state:(\{.*\})-->", re.DOTALL)
_HINT_OPEN = "<!--state:"
_HINT_CLOSE = "-->"


def encode_hint(state: AgentState) -> str:
    """Serialize state into the HTML comment we embed in assistant text."""
    payload = state.model_dump(mode="json")
    return f"{_HINT_OPEN}{json.dumps(payload, separators=(',', ':'))}{_HINT_CLOSE}"


def strip_hint(text: str) -> str:
    """Remove any embedded state hints from a string.

    Used both when **emitting** (we strip then re-embed to avoid
    nesting) and when **showing** the reply for human-readable logs.
    """
    return _HINT_RE.sub("", text).strip()


def parse_hint(text: str) -> AgentState | None:
    """Try to recover an :class:`AgentState` from an embedded hint.

    Returns the most recent valid hint in the text, or None.
    """
    matches = list(_HINT_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1].group(1)
    try:
        raw = json.loads(last)
        return AgentState.model_validate(raw)
    except (json.JSONDecodeError, ValidationError):
        return None


def reconstruct_from_hint(messages: Sequence[Message]) -> AgentState | None:
    """Walk assistant messages newest-first; return first valid hint."""
    for msg in reversed(list(messages)):
        if msg.role != "assistant":
            continue
        st = parse_hint(msg.content)
        if st is not None:
            return st
    return None


def count_assistant_turns(messages: Sequence[Message]) -> int:
    """Number of assistant turns the agent has emitted so far."""
    return sum(1 for m in messages if m.role == "assistant")


def initial_state(messages: Sequence[Message]) -> AgentState:
    """Empty state for a brand-new conversation, with the right turn index."""
    return AgentState(turn_index=count_assistant_turns(messages))
