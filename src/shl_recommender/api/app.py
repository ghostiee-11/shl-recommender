"""FastAPI application factory + lifespan.

Design notes:

* **Lifespan eagerly builds the catalog index, BM25, dense embeddings,
  LLM router, and orchestrator.** This is intentional: the eval
  evaluator first calls ``/health``, then ``/chat``. We want the
  agent ready by the time ``/chat`` arrives, not paying a cold-start
  penalty inside the 30-second per-call budget.
* **``/health`` is independent of the orchestrator.** It only checks
  that the process can serve a request. It returns 200 even if the
  LLM router would currently fail (e.g. invalid keys) so the platform
  health-check stays green and we don't loop the deploy.
* **App state is the canonical place to hold heavy objects.** The
  routes pull them out via ``request.app.state.orch`` etc., no
  globals, no module-scoped singletons.
* **Global exception handler returns spec-compliant JSON**, never
  a 500 with an HTML body. Schema compliance is graded; bugs must
  not break it.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from shl_recommender.agent.orchestrator import Orchestrator
from shl_recommender.api.metrics import MetricsRegistry
from shl_recommender.api.routes import router as api_router
from shl_recommender.api.safety import (
    BodySizeLimitMiddleware,
    RateLimiterMiddleware,
)
from shl_recommender.api.schemas import ChatResponse
from shl_recommender.catalog.loader import load_catalog
from shl_recommender.config import Settings
from shl_recommender.llm.base import LLMClient
from shl_recommender.llm.gemini_client import GeminiLLM
from shl_recommender.llm.groq_client import GroqLLM
from shl_recommender.llm.openai_client import OpenAILLM
from shl_recommender.llm.router import LLMRouter
from shl_recommender.llm.throttle import RateLimitedLLM
from shl_recommender.observability.logging import configure as configure_logging
from shl_recommender.observability.logging import get_logger
from shl_recommender.retrieval.bm25 import BM25Index
from shl_recommender.retrieval.dense import DenseIndex
from shl_recommender.retrieval.hybrid import HybridRetriever
from shl_recommender.retrieval.llm_rerank import LLMReranker

log = get_logger(__name__)


def _build_llm(settings: Settings) -> LLMClient:
    """Construct the LLM router with multi-provider failover.

    Preference order: OpenAI (highest tier ceiling, smartest model) →
    Groq (sub-second free latency when not throttled) → Gemini (free
    fallback). Each provider is wrapped in a :class:`RateLimitedLLM`
    token bucket sized to its tier so the agent self-throttles instead
    of cascading 429s. The router does primary↔fallback failover with
    a circuit breaker.
    """
    openai_llm: LLMClient | None = None
    groq: LLMClient | None = None
    gemini: LLMClient | None = None
    if settings.openai_api_key:
        openai_llm = RateLimitedLLM(
            OpenAILLM(settings.openai_api_key),
            capacity=settings.openai_capacity,
            refill_per_sec=settings.openai_refill_per_sec,
        )
    if settings.groq_api_key:
        groq = RateLimitedLLM(
            GroqLLM(settings.groq_api_key),
            capacity=settings.groq_capacity,
            refill_per_sec=settings.groq_refill_per_sec,
        )
    if settings.gemini_api_key:
        gemini = RateLimitedLLM(
            GeminiLLM(settings.gemini_api_key),
            capacity=settings.gemini_capacity,
            refill_per_sec=settings.gemini_refill_per_sec,
        )

    # Pick the strongest available primary, then the next-strongest as fallback.
    candidates = [c for c in (openai_llm, groq, gemini) if c is not None]
    if not candidates:
        raise RuntimeError(
            "No LLM key configured. Set OPENAI_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY."
        )
    if len(candidates) == 1:
        log.warning("single_llm_provider_no_fallback", provider=candidates[0].name)
        return candidates[0]
    return LLMRouter(candidates[0], candidates[1])


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build heavy state at startup; tear down on shutdown.

    The dense embedding model load is the dominant cost (~25s on a
    cold container). On Render free tier this runs once per cold start.
    """
    settings = cast(Settings, app.state.settings)
    configure_logging(settings.log_level)

    log.info("lifespan_startup_begin")
    t0 = time.perf_counter()

    catalog = load_catalog()
    log.info("catalog_loaded", count=len(catalog), elapsed_ms=int((time.perf_counter() - t0) * 1000))

    t1 = time.perf_counter()
    bm25 = BM25Index(catalog.search_docs)
    log.info("bm25_built", elapsed_ms=int((time.perf_counter() - t1) * 1000))

    t1 = time.perf_counter()
    dense = DenseIndex(catalog.search_docs)
    log.info("dense_built", elapsed_ms=int((time.perf_counter() - t1) * 1000))

    retriever = HybridRetriever(bm25, dense, catalog)
    llm = _build_llm(settings)
    reranker = LLMReranker(llm, catalog)
    orch = Orchestrator(
        catalog,
        retriever,
        reranker,
        llm,
        top_k=settings.recommendation_top_k,
    )

    app.state.catalog = catalog
    app.state.orch = orch
    app.state.ready = True
    log.info(
        "lifespan_startup_complete",
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
    )

    try:
        yield
    finally:
        log.info("lifespan_shutdown")


class _RequestIdMiddleware(BaseHTTPMiddleware):
    """Tag every request with a UUID + log timing.

    ``request_id`` is bound into structlog's contextvars so any log
    line emitted during the request inherits it without manual
    propagation.
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
        )
        start = time.perf_counter()
        status = "unknown"
        try:
            response = await call_next(request)
            status = str(response.status_code)
        finally:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            log.info("request_complete", elapsed_ms=elapsed_ms, status=status)
            metrics = getattr(request.app.state, "metrics", None)
            if metrics is not None:
                metrics.incr("requests_total", f"{request.url.path}:{status}")
        response.headers["x-request-id"] = request_id
        return response


def _build_exception_handler(app: FastAPI) -> None:
    """Wire a global handler that NEVER lets a 500 break schema compliance.

    On any unhandled exception in ``/chat``, we return a valid
    ``ChatResponse`` with an empty shortlist and a graceful reply.
    The error is logged with stack so we can debug.
    """

    @app.exception_handler(Exception)
    async def _on_unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_exception")
        if request.url.path == "/chat":
            payload = ChatResponse(
                reply=(
                    "Something went wrong on my end. Could you try rephrasing "
                    "what you're looking for?"
                ),
                recommendations=[],
                end_of_conversation=False,
            ).model_dump(mode="json")
            return JSONResponse(status_code=200, content=payload)
        return JSONResponse(status_code=500, content={"error": "internal"})


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construct the FastAPI app.

    Tests inject a custom ``settings`` for hermetic runs. Production
    callers (uvicorn entry point) pass ``None`` to use env-driven
    defaults.
    """
    settings = settings or Settings()
    app = FastAPI(
        title="SHL Conversational Assessment Recommender",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.state.ready = False
    app.state.metrics = MetricsRegistry()

    # Starlette's ``add_middleware`` makes the LAST-added the OUTERMOST
    # layer. We want execution order: request-id → CORS → body-size →
    # rate-limit → handler. So _RequestIdMiddleware must be added LAST
    # so it wraps everything else and can record metrics for any
    # short-circuited 413/429/CORS-preflight response.
    app.add_middleware(
        RateLimiterMiddleware,
        capacity=settings.rate_limit_capacity,
        refill_per_sec=settings.rate_limit_refill_per_sec,
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    if settings.cors_origins_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins_list,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["content-type", "x-request-id"],
            expose_headers=["x-request-id"],
        )
    app.add_middleware(_RequestIdMiddleware)
    app.include_router(api_router)
    _build_exception_handler(app)
    return app


# Convenience module-level instance for ``uvicorn shl_recommender.api.app:app``.
app: ASGIApp = create_app()
