"""Agent orchestrator, the single public entry point.

Flow per turn (≤ 3 LLM calls in the recommend path, ≤ 1 in clarify):

    history (messages)
        │
        ▼
    1. Reconstruct AgentState
       a. Try parse the last assistant turn's embedded state hint.
       b. Else fall back to running the slot extractor over history.
        │
        ▼
    2. Refusal check on the LATEST user message
       a. Hard pattern → REFUSE template, return.
       b. Soft pattern → LLM tiebreak; if OFF_TOPIC → REFUSE, return.
        │
        ▼
    3. Extractor over the latest message updates state (if not already
       in step 1b, avoid double extraction).
        │
        ▼
    4. Decision policy → Action ∈ {CLARIFY, RECOMMEND, REFINE, COMPARE}.
        │
        ▼
    5. Execute the action:
       - CLARIFY: emit the next question; recs = []
       - RECOMMEND: build query from slots → hybrid retrieval →
         LLM rerank → top-K Recommendation models.
       - REFINE: same as RECOMMEND but seeds the query with the
         user's edit and the prior shortlist.
       - COMPARE: fetch named items by fuzzy-matching catalog,
         emit a short factual diff; recs are the compared items.
        │
        ▼
    6. Compose ChatResponse (+ embed updated state hint into reply).

Hard invariants enforced before return:

* Every URL in recommendations is in the catalog.
* recommendations length ∈ [0, 10] (empty when not recommending).
* Schema validates as :class:`ChatResponse`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from shl_recommender.api.schemas import ChatResponse, Message, Recommendation
from shl_recommender.catalog.loader import CatalogIndex
from shl_recommender.catalog.models import Assessment
from shl_recommender.llm.base import LLMClient, LLMMessage
from shl_recommender.retrieval.hybrid import HybridRetriever
from shl_recommender.retrieval.llm_rerank import LLMReranker
from shl_recommender.retrieval.query import expand_query

from .composer import compose_reply
from .extractor import ExtractorOutput, extract_slots, update_state
from .policy import Action, Decision, decide
from .refusal import should_refuse
from .slots import AgentState, Slots
from .state import (
    count_assistant_turns,
    encode_hint,
    initial_state,
    reconstruct_from_hint,
    strip_hint,
)

logger = logging.getLogger(__name__)

DEFAULT_RECOMMENDATION_K = 5
MAX_RECOMMENDATIONS = 10
CANDIDATE_POOL = 20


@dataclass(frozen=True, slots=True)
class _ExecResult:
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


class Orchestrator:
    """Stateless conversation orchestrator.

    All dependencies injected; no globals. The single ``handle``
    coroutine is the public surface.
    """

    __slots__ = ("_catalog", "_retriever", "_reranker", "_llm", "_top_k")

    def __init__(
        self,
        catalog: CatalogIndex,
        retriever: HybridRetriever,
        reranker: LLMReranker,
        llm: LLMClient,
        *,
        top_k: int = DEFAULT_RECOMMENDATION_K,
    ) -> None:
        self._catalog = catalog
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm
        self._top_k = top_k

    # ---------- public entry point ----------

    async def handle(self, messages: Sequence[Message]) -> ChatResponse:
        # Step 1: state reconstruction.
        state = reconstruct_from_hint(messages) or initial_state(messages)
        # Always re-set turn_index from the actual history, it's the truth.
        state = AgentState(
            slots=state.slots,
            intent=state.intent,
            prior_shortlist_urls=state.prior_shortlist_urls,
            asked_slots=state.asked_slots,
            turn_index=count_assistant_turns(messages),
        )

        last_user = _last_user_text(messages)

        # Step 2: refusal check on the latest user input.
        if last_user:
            refuse, refusal_text = await should_refuse(self._llm, last_user)
            if refuse:
                return self._build_response(
                    state=state,
                    reply=refusal_text,
                    recommendations=[],
                    end_of_conversation=False,
                )

        # Step 2.5: deterministic compare pre-detection. If the user
        # explicitly asked to compare named items, short-circuit
        # without burning an LLM call on slot extraction.
        explicit_compare = _detect_explicit_compare(last_user) if last_user else []
        if explicit_compare:
            decision = Decision(
                action=Action.COMPARE,
                compared_assessments=tuple(explicit_compare),
            )
            return await self._do_compare(state, decision)

        # Step 3: slot extraction. The extractor sees the full history,
        # so we don't double-extract the prior turns.
        llm_history = _to_llm_messages(messages)
        extracted: ExtractorOutput = await extract_slots(self._llm, llm_history)
        state = update_state(state, extracted)

        # Step 3.5: JD-blob heuristic. When the extractor fails (rate
        # limit, schema error, etc.) it returns ``intent="clarify"`` +
        # empty slots. If the user's message is substantive on its own
        #, a paste-style brief with enough signal, asking another
        # clarifying question would be insulting and would burn turns.
        # Treat such messages as recommends so retrieval can do its job.
        effective_intent = extracted.intent
        if (
            extracted.intent == "clarify"
            and state.slots.filled_count() == 0
            and _looks_like_brief(last_user)
        ):
            effective_intent = "recommend"

        # Step 4: decision policy.
        decision = decide(
            state,
            extractor_intent=effective_intent,
            compared_assessments=tuple(extracted.compared_assessments),
            has_prior_shortlist=bool(state.prior_shortlist_urls),
        )

        # Step 5: execute.
        if decision.action == Action.CLARIFY:
            return self._build_response(
                state=_record_asked(state, decision),
                reply=decision.next_question or "Could you tell me more?",
                recommendations=[],
                end_of_conversation=False,
            )

        if decision.action == Action.COMPARE:
            return await self._do_compare(state, decision)

        if decision.action in (Action.RECOMMEND, Action.REFINE):
            user_msgs = [m.content for m in messages if m.role == "user"]
            return await self._do_recommend(state, last_user, user_msgs)

        if decision.action == Action.REFUSE:
            # The deterministic refusal layer (step 2) already covered
            # the obvious cases. Reaching here means the LLM extractor
            # itself classified the message as refuse, typically an
            # overcautious read of a vague turn-1. Treat as a clarify
            # so we don't inadvertently shut the conversation down.
            return self._build_response(
                state=state,
                reply="I'd love to help, could you tell me a bit about the role or use case?",
                recommendations=[],
                end_of_conversation=False,
            )

        # Truly unreachable, defensive only.
        logger.error("orchestrator_unreachable_decision", extra={"action": decision.action})
        return self._build_response(
            state=state,
            reply="Could you rephrase that?",
            recommendations=[],
            end_of_conversation=False,
        )

    # ---------- action implementations ----------

    async def _do_recommend(
        self, state: AgentState, last_user: str, all_user_messages: list[str]
    ) -> ChatResponse:
        # Build the retrieval query from the *committed* slot signal
        # when we have one. Fall back to ALL user-turn text (joined
        # most-recent-first, last-turn weighted) when slots are empty,
        # this happens when the LLM extractor fails / is rate-limited.
        # Using only the latest user message often picks up an
        # uninformative confirmation phrase ("perfect, that's it");
        # joining all turns keeps the original hiring context in scope.
        slots_query = state.slots.as_query_text()
        if not slots_query:
            from shl_recommender.retrieval.query import build_query

            slots_query = build_query(all_user_messages) or last_user
        bm25_query = expand_query(slots_query)

        candidates = self._retriever.search(
            slots_query,
            bm25_query=bm25_query,
            top_n=CANDIDATE_POOL,
        )
        ranked = await self._reranker.rerank(
            slots_query,
            candidates,
            top_k=min(self._top_k, MAX_RECOMMENDATIONS),
        )
        items = [self._catalog.items[h.doc_index] for h in ranked]
        recs = [_to_recommendation(it) for it in items]

        if not recs:
            return self._build_response(
                state=state,
                reply=(
                    "I couldn't pin down a confident shortlist with what we have. "
                    "Could you share a bit more about the role or skills?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        new_state = AgentState(
            slots=state.slots,
            intent=state.intent,
            prior_shortlist_urls=[str(r.url) for r in recs],
            asked_slots=state.asked_slots,
            turn_index=state.turn_index,
        )
        # Deterministic fallback prose. The composer prefers this when
        # the LLM is unreachable so the user always gets something.
        names = ", ".join(r.name for r in recs[:3])
        if len(recs) > 3:
            names += f", and {len(recs) - 3} more"
        fallback_reply = (
            f"Based on what you've described, here are {len(recs)} fits, {names}."
        )
        action_label = "refine" if state.prior_shortlist_urls else "recommend"
        reply = await compose_reply(
            self._llm,
            action=action_label,
            user_message=last_user,
            items=items,
            fallback=fallback_reply,
        )
        return self._build_response(
            state=new_state,
            reply=reply,
            recommendations=recs,
            end_of_conversation=False,
        )

    async def _do_compare(self, state: AgentState, decision: Decision) -> ChatResponse:
        items = [
            it
            for name in decision.compared_assessments
            if (it := _fuzzy_lookup(self._catalog, name)) is not None
        ]
        if len(items) < 2:
            return self._build_response(
                state=state,
                reply=(
                    "I need at least two SHL assessments by name to compare. "
                    "Could you specify which ones?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        recs = [_to_recommendation(it) for it in items[:MAX_RECOMMENDATIONS]]
        # Deterministic fallback prose, used if the composer's LLM
        # call fails. Always a complete, grounded comparison.
        parts: list[str] = []
        for it in items:
            first_sentence = it.description.split(". ")[0].rstrip(".")
            duration = it.duration or "duration unspecified"
            type_label = "/".join(it.test_type_codes)
            parts.append(
                f"{it.name} ({type_label}, {duration}): {first_sentence}."
            )
        fallback_reply = "\n\n".join(parts)
        reply = await compose_reply(
            self._llm,
            action="compare",
            # The composer needs a "what is the user asking" anchor;
            # the names they wanted compared serve that purpose.
            user_message=" vs. ".join(decision.compared_assessments),
            items=items,
            fallback=fallback_reply,
        )
        return self._build_response(
            state=state,
            reply=reply,
            recommendations=recs,
            end_of_conversation=False,
        )

    # ---------- helpers ----------

    def _build_response(
        self,
        *,
        state: AgentState,
        reply: str,
        recommendations: list[Recommendation],
        end_of_conversation: bool,
    ) -> ChatResponse:
        # URL grounding: any rec whose URL isn't in the catalog is dropped.
        grounded = [r for r in recommendations if self._catalog.is_grounded_url(str(r.url))]
        if len(grounded) != len(recommendations):
            logger.warning(
                "ungrounded_url_dropped",
                extra={"dropped": len(recommendations) - len(grounded)},
            )

        # Truncate to MAX_RECOMMENDATIONS for spec compliance.
        grounded = grounded[:MAX_RECOMMENDATIONS]

        # Embed updated state hint into the reply so the next turn can
        # reconstruct without re-running extraction.
        clean_reply = strip_hint(reply)
        hint = encode_hint(state)
        full_reply = f"{clean_reply}\n{hint}" if clean_reply else hint
        return ChatResponse(
            reply=full_reply,
            recommendations=grounded,
            end_of_conversation=end_of_conversation,
        )


# ---------- module helpers (pure, free functions) ----------


def _last_user_text(messages: Sequence[Message]) -> str:
    for m in reversed(list(messages)):
        if m.role == "user":
            return m.content
    return ""


def _to_llm_messages(messages: Sequence[Message]) -> list[LLMMessage]:
    """Convert API Messages → LLMMessages, stripping any state hints."""
    out: list[LLMMessage] = []
    for m in messages:
        out.append(LLMMessage(role=m.role, content=strip_hint(m.content)))
    return out


def _to_recommendation(item: Assessment) -> Recommendation:
    return Recommendation(
        name=item.name,
        url=item.link,  # type: ignore[arg-type]  # pydantic coerces to HttpUrl
        test_type=item.primary_test_type,
    )


_NORM_RE = re.compile(r"[^a-z0-9]+")

# Detect explicit comparison requests so we don't depend solely on the
# LLM extractor classifying them. Regex matches phrasings like:
#   "compare X and Y", "difference between X and Y", "X vs Y"
_COMPARE_RE = re.compile(
    r"(compare|difference\s+between|differences\s+between|\bvs\.?\b|versus)",
    re.IGNORECASE,
)


# Hiring-brief signal words. If a message contains one of these and
# is long enough, we treat it as a substantive recruiter brief that
# deserves a shortlist rather than another clarifying question.
_BRIEF_KEYWORDS = re.compile(
    r"\b(hir(?:e|ing)|recruit(?:er|ing)?|candidate|role|team|position|"
    r"developer|engineer|analyst|manager|director|leader|associate|"
    r"agent|representative|specialist|consultant|graduate|trainee|"
    r"job\s+description|jd)\b",
    re.IGNORECASE,
)


def _looks_like_brief(text: str) -> bool:
    """True when the user message is substantive enough to act on as-is.

    Heuristic: a hiring-context keyword PLUS either a long message
    (≥ 80 chars) or two distinct keywords. The two-keyword path
    catches concise but specific briefs like "Senior Python data
    engineer with cloud experience" that the length-only check
    rejected.
    """
    matches = _BRIEF_KEYWORDS.findall(text)
    if not matches:
        return False
    if len(text) >= 80:
        return True
    # Distinct keywords (case-insensitive) carry as much signal as length.
    distinct = {m.lower() for m in matches}
    return len(distinct) >= 2 and len(text) >= 40


def _detect_explicit_compare(text: str) -> list[str]:
    """Return at least two candidate assessment-name fragments from
    a user comparison request, or [] if no compare intent detected.

    Strategy: detect the comparison keyword, then split on common
    separators (``and``, ``vs``, ``,``, ``or``) around it. Returns
    the resulting fragments, trimmed.
    """
    if not _COMPARE_RE.search(text):
        return []
    # Take everything after the first comparison keyword.
    m = _COMPARE_RE.search(text)
    assert m is not None
    tail = text[m.end():]
    # Stop at the first sentence-terminator, anything after a "."
    # or "?" is almost certainly a follow-up clause, not part of the
    # assessment name (e.g. "Verify Numerical. What's the diff?").
    tail = re.split(r"[.?!]", tail, maxsplit=1)[0]
    # Strip leading "between" / "of" if it followed the keyword.
    tail = re.sub(r"^\s*(between|of)\s+", "", tail, flags=re.IGNORECASE)
    parts = re.split(r"\s+(?:and|or|vs\.?|versus)\s+|,\s*", tail, flags=re.IGNORECASE)
    candidates = [p.strip(" .?!") for p in parts if p and p.strip()]
    candidates = [c for c in candidates if len(c) >= 3]
    return candidates[:5] if len(candidates) >= 2 else []


def _norm(s: str) -> str:
    return _NORM_RE.sub("", s.lower())


_TOKEN_SPLIT = re.compile(r"[a-z0-9]+")

# Generic words that appear in user phrasing but never carry signal
# about which catalog item they mean. Filtering them lets the
# token-coverage matcher accept "the Verify Numerical Reasoning tests"
# as referring to "SHL Verify Interactive – Numerical Reasoning".
_FUZZY_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "for", "to", "with", "in", "on",
        "at", "by", "from", "as", "is", "are",
        "test", "tests", "assessment", "assessments", "tool", "tools",
        "report", "reports", "instrument",
    }
)


def _tokens(s: str) -> set[str]:
    return {
        t
        for t in _TOKEN_SPLIT.findall(s.lower())
        if len(t) >= 2 and t not in _FUZZY_STOPWORDS
    }


def _fuzzy_lookup(catalog: CatalogIndex, name: str) -> Assessment | None:
    """Case/punctuation-insensitive name lookup against the catalog.

    Three-stage matcher (each stricter than the next falls back):

    1. Exact normalized name match.
    2. Substring containment (input within catalog name).
    3. Token-coverage: catalog name contains every meaningful query
       token (≥ 2 chars). Picks the catalog item with the **fewest
       extra tokens** so a query of "Verify Numerical Reasoning"
       prefers the closest-named SHL item over a longer one with the
       same coverage.

    Returns None if no candidate covers ≥ 2 query tokens (avoids
    false positives on short or generic inputs).
    """
    target = _norm(name)
    if not target or len(target) < 4:
        return None

    # Stage 1: exact.
    for item in catalog.items:
        if _norm(item.name) == target:
            return item

    # Stage 2: substring.
    sub_candidates = [it for it in catalog.items if target in _norm(it.name)]
    if sub_candidates:
        return min(sub_candidates, key=lambda it: len(it.name))

    # Stage 3: token coverage.
    query_tokens = _tokens(name)
    if len(query_tokens) < 2:
        return None
    best: tuple[int, Assessment] | None = None  # (extras, item)
    for item in catalog.items:
        cat_tokens = _tokens(item.name)
        if query_tokens.issubset(cat_tokens):
            extras = len(cat_tokens - query_tokens)
            if best is None or extras < best[0]:
                best = (extras, item)
    return best[1] if best else None


def _record_asked(state: AgentState, decision: Decision) -> AgentState:
    """When clarifying, record which slot we just asked about."""
    if decision.action != Action.CLARIFY or not decision.next_question:
        return state
    # Reverse-lookup the slot whose template matches what we asked.
    from .policy import CLARIFY_QUESTIONS

    for slot_name, question in CLARIFY_QUESTIONS.items():
        if decision.next_question.strip() == question.strip():
            if slot_name in state.asked_slots:
                return state
            return AgentState(
                slots=state.slots,
                intent=state.intent,
                prior_shortlist_urls=state.prior_shortlist_urls,
                asked_slots=[*state.asked_slots, slot_name],
                turn_index=state.turn_index,
            )
    return state


# Re-export Slots so import site is one-stop.
__all__ = [
    "Orchestrator",
    "Slots",
]
