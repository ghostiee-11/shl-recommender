"""Parse SHL's reference markdown dialogues into structured Trace objects.

The 10 reference files (``C1.md`` … ``C10.md``) follow a stable
template:

    ## Conversation
    ### Turn N
    **User**
    > <user content>
    **Agent**
    <agent content, optionally a markdown table whose URL column
    contains the recommendation URLs>
    _`end_of_conversation`: **true|false**_

We extract per file:

* the ordered list of (role, content) turns,
* the **expected_shortlist**, the URL column of the **last** agent
  table in the file (the final committed shortlist, after any
  refinement during the dialogue).

The user-turn texts are also exposed so a baseline retrieval bench
can use them as a query proxy before the full agent exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TRACES_DIR = Path(__file__).resolve().parent / "traces"

_TURN_RE = re.compile(r"^###\s+Turn\s+(\d+)\s*$", re.MULTILINE)
_USER_BLOCK_RE = re.compile(
    r"\*\*User\*\*\s*\n+((?:>\s*.*(?:\n|$))+)", re.MULTILINE
)
_AGENT_HEADER_RE = re.compile(r"\*\*Agent\*\*\s*\n", re.MULTILINE)
# Markdown link in the URL column of a recommendation table:
# either <https://…> or [text](https://…)
_URL_IN_TABLE_RE = re.compile(r"<(https?://[^>\s]+)>|\]\((https?://[^)\s]+)\)")


@dataclass(frozen=True, slots=True)
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True, slots=True)
class Trace:
    trace_id: str
    turns: tuple[Turn, ...]
    expected_shortlist: tuple[str, ...]

    @property
    def user_messages(self) -> tuple[str, ...]:
        return tuple(t.content for t in self.turns if t.role == "user")

    @property
    def query_proxy(self) -> str:
        """Concatenated user-turn text, usable as a baseline query
        before the full slot-filling agent exists."""
        return " ".join(self.user_messages)


@dataclass(frozen=True, slots=True)
class TraceParseStats:
    file_count: int
    turn_count: int
    expected_url_count: int
    files_with_no_shortlist: tuple[str, ...] = field(default_factory=tuple)


def _split_turns(body: str) -> list[tuple[int, str]]:
    """Split the file body into ``(turn_number, turn_text)`` chunks."""
    matches = list(_TURN_RE.finditer(body))
    if not matches:
        return []
    chunks: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        n = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunks.append((n, body[start:end]))
    return chunks


def _extract_user(turn_text: str) -> str | None:
    m = _USER_BLOCK_RE.search(turn_text)
    if not m:
        return None
    block = m.group(1)
    # Strip the leading "> " markdown blockquote on each line.
    lines = [ln.lstrip("> ").rstrip() for ln in block.splitlines() if ln.strip()]
    return " ".join(lines).strip()


def _extract_agent(turn_text: str) -> str:
    m = _AGENT_HEADER_RE.search(turn_text)
    if not m:
        return ""
    return turn_text[m.end():].strip()


def _extract_table_urls(agent_text: str) -> list[str]:
    """Return URLs from any markdown table in the agent text.

    Order-preserving and de-duplicated. Returns ``[]`` if the agent
    turn has no table.
    """
    if "|" not in agent_text:
        return []
    seen: dict[str, None] = {}
    for m in _URL_IN_TABLE_RE.finditer(agent_text):
        url = m.group(1) or m.group(2)
        if url:
            seen.setdefault(url, None)
    return list(seen)


def parse_trace(path: Path) -> Trace:
    """Parse a single ``Cn.md`` file into a :class:`Trace`."""
    body = path.read_text(encoding="utf-8")
    turns: list[Turn] = []
    last_shortlist: list[str] = []

    for _n, chunk in _split_turns(body):
        user = _extract_user(chunk)
        if user:
            turns.append(Turn(role="user", content=user))
        agent = _extract_agent(chunk)
        if agent:
            turns.append(Turn(role="assistant", content=agent))
            urls = _extract_table_urls(agent)
            if urls:
                last_shortlist = urls

    return Trace(
        trace_id=path.stem,
        turns=tuple(turns),
        expected_shortlist=tuple(last_shortlist),
    )


def load_traces(directory: Path | None = None) -> list[Trace]:
    """Parse every ``Cn.md`` in ``directory``, sorted by numeric suffix."""
    d = directory or DEFAULT_TRACES_DIR
    files = sorted(
        d.glob("C*.md"),
        key=lambda p: int(re.sub(r"\D", "", p.stem) or "0"),
    )
    return [parse_trace(p) for p in files]


def parse_stats(traces: list[Trace]) -> TraceParseStats:
    return TraceParseStats(
        file_count=len(traces),
        turn_count=sum(len(t.turns) for t in traces),
        expected_url_count=sum(len(t.expected_shortlist) for t in traces),
        files_with_no_shortlist=tuple(t.trace_id for t in traces if not t.expected_shortlist),
    )
