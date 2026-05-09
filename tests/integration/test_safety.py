"""Tests for body-size limit, rate limiter, and /metrics endpoint."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient

from shl_recommender.api.app import create_app
from shl_recommender.api.schemas import ChatResponse, Message
from shl_recommender.config import Settings


class _NoopOrch:
    async def handle(self, _msgs: Sequence[Message]) -> ChatResponse:
        return ChatResponse(reply="ok", recommendations=[], end_of_conversation=False)


def _client(**overrides: Any) -> TestClient:
    settings = Settings(
        groq_api_key="dummy",
        gemini_api_key="dummy",
        log_level="WARNING",
        **overrides,
    )
    app = create_app(settings)
    app.state.orch = _NoopOrch()
    app.state.ready = True
    return TestClient(app, raise_server_exceptions=False)


# ---------- body size ----------


def test_body_too_large_returns_413() -> None:
    client = _client(max_body_bytes=512)
    huge = "x" * 1024
    resp = client.post("/chat", json={"messages": [{"role": "user", "content": huge}]})
    assert resp.status_code == 413
    assert resp.json()["error"] == "request_body_too_large"


def test_body_under_limit_passes_through() -> None:
    client = _client(max_body_bytes=64 * 1024)
    resp = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200


# ---------- rate limiter ----------


def test_rate_limiter_allows_burst_then_blocks() -> None:
    client = _client(rate_limit_capacity=3, rate_limit_refill_per_sec=0.0)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    # First three burst through.
    for _ in range(3):
        assert client.post("/chat", json=body).status_code == 200
    # Fourth is throttled.
    resp = client.post("/chat", json=body)
    assert resp.status_code == 429
    assert "retry-after" in resp.headers


def test_rate_limiter_refills_over_time() -> None:
    client = _client(rate_limit_capacity=1, rate_limit_refill_per_sec=20.0)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/chat", json=body).status_code == 200
    # Burst exhausted; immediate retry → 429.
    assert client.post("/chat", json=body).status_code == 429
    # Wait long enough to refill one token (20/sec → 50ms is plenty).
    time.sleep(0.1)
    assert client.post("/chat", json=body).status_code == 200


def test_rate_limiter_exempts_health_and_metrics() -> None:
    client = _client(rate_limit_capacity=1, rate_limit_refill_per_sec=0.0)
    # Burn the only chat token.
    body = {"messages": [{"role": "user", "content": "hi"}]}
    client.post("/chat", json=body)
    # Health/metrics must still respond.
    for _ in range(5):
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200


# ---------- metrics ----------


def test_metrics_initial_state_is_ready() -> None:
    client = _client()
    resp = client.get("/metrics")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ready"] is True
    assert "requests_total" in payload["counters"]


def test_metrics_increments_per_request() -> None:
    client = _client()
    body = {"messages": [{"role": "user", "content": "hi"}]}
    client.post("/chat", json=body)
    client.post("/chat", json=body)
    client.get("/health")  # exempt from rate-limit; still counted
    payload = client.get("/metrics").json()
    counters = payload["counters"]
    chat_200 = counters["requests_total"].get("/chat:200", 0)
    assert chat_200 >= 2
    assert counters["chat_decisions_total"].get("no_recommendations", 0) >= 2


def test_metrics_records_413_status() -> None:
    client = _client(max_body_bytes=512)
    huge = "x" * 1024
    client.post("/chat", json={"messages": [{"role": "user", "content": huge}]})
    counters = client.get("/metrics").json()["counters"]
    assert counters["requests_total"].get("/chat:413", 0) == 1
