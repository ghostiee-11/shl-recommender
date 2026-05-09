"""HTTP routes: ``GET /``, ``GET /health``, ``GET /metrics``, ``POST /chat``.

``GET /`` returns a friendly service banner so anyone clicking the
deployed URL sees what the service is and how to use it, instead of a
bare 404.

``/health`` is intentionally trivial: it returns ``{"status": "ok"}``
with HTTP 200 unconditionally. The platform's health check uses it to
decide if the container is alive, we don't want a misconfigured LLM
key to mark the service unhealthy and trigger a restart loop.

``/chat`` is the agent surface. It is **stateless**: the full
conversation history is in the request body. We hand it to the
orchestrator and return its :class:`ChatResponse` verbatim.

A wall-clock timeout (``settings.request_budget_s``, default 25s) wraps
the orchestrator call so we never blow the 30-second per-call budget
the spec mandates. On timeout we return a graceful, schema-compliant
fallback rather than an error code.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from shl_recommender.agent.orchestrator import Orchestrator
from shl_recommender.api.metrics import MetricsRegistry
from shl_recommender.api.schemas import ChatRequest, ChatResponse
from shl_recommender.config import Settings
from shl_recommender.observability.logging import get_logger

router = APIRouter()
log = get_logger(__name__)


@router.get("/")
async def root() -> dict[str, object]:
    """Service banner for humans who hit the bare URL in a browser.

    Documents the surface so reviewers don't see a bare 404 and assume
    the service is broken when they paste the deployment URL.
    """
    return {
        "service": "SHL Conversational Assessment Recommender",
        "status": "ok",
        "endpoints": {
            "GET /": "this banner",
            "GET /health": "liveness probe; always 200 {'status': 'ok'}",
            "GET /metrics": "in-process counters + readiness flag",
            "POST /chat": "stateless conversation turn (see /docs)",
            "GET /docs": "interactive OpenAPI documentation",
        },
        "demo": "https://web-eight-theta-60.vercel.app",
        "repo": "https://github.com/ghostiee-11/shl-recommender",
    }


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe.

    Returns 200 unconditionally so the platform never restarts us
    over a misconfigured key. Readiness (orchestrator built) is
    inspectable via ``/metrics``.
    """
    return {"status": "ok"}


@router.get("/metrics")
async def metrics(request: Request) -> dict[str, object]:
    """Snapshot of in-process counters and readiness state.

    Intentionally not Prometheus-formatted, keeps deps zero. Phase 6
    can swap to a real exporter without touching the route.
    """
    reg: MetricsRegistry = request.app.state.metrics
    return {
        "ready": bool(getattr(request.app.state, "ready", False)),
        "counters": reg.snapshot(),
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    """Stateless conversation turn.

    Wraps the orchestrator in a wall-clock guard so a slow LLM cannot
    blow the spec's 30-second per-call budget. On timeout we still
    return a valid :class:`ChatResponse`.
    """
    settings: Settings = request.app.state.settings
    orch: Orchestrator = request.app.state.orch
    metrics: MetricsRegistry = request.app.state.metrics
    try:
        resp = await asyncio.wait_for(
            orch.handle(req.messages),
            timeout=settings.request_budget_s,
        )
    except TimeoutError:
        log.warning("chat_timeout", budget_s=settings.request_budget_s)
        metrics.incr("chat_decisions_total", "timeout")
        return ChatResponse(
            reply=(
                "I ran out of time on that one. Could you simplify or shorten "
                "your last message?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )
    metrics.incr(
        "chat_decisions_total",
        "with_recommendations" if resp.recommendations else "no_recommendations",
    )
    return resp
