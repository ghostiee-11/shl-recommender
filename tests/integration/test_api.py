"""End-to-end API tests against the FastAPI app.

Strategy:
* Build the real app via :func:`create_app`.
* Replace the orchestrator on ``app.state`` with a fake one before
  any request is served. This bypasses the heavy dense-encoder load
  and the network LLM calls, both are tested elsewhere.
* Use :class:`fastapi.testclient.TestClient`, which drives the app
  through ASGI directly (no socket).

This makes the API tests fast (<1s total) and deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient

from shl_recommender.api.app import create_app
from shl_recommender.api.schemas import ChatResponse, Message, Recommendation
from shl_recommender.config import Settings


class _FakeOrchestrator:
    """Returns a hard-coded :class:`ChatResponse` regardless of input."""

    def __init__(self, response: ChatResponse) -> None:
        self._response = response
        self.calls: list[Sequence[Message]] = []

    async def handle(self, messages: Sequence[Message]) -> ChatResponse:
        self.calls.append(list(messages))
        return self._response


@pytest.fixture
def app_with_fake() -> Any:
    """Build the app, then **skip lifespan** by injecting fakes directly.

    The TestClient context manager invokes lifespan on enter, and our
    real lifespan would try to load the embedding model and the LLM
    router. Instead we construct ``create_app`` and overwrite
    ``app.state`` ourselves.
    """
    settings = Settings(
        groq_api_key="dummy",
        gemini_api_key="dummy",
        log_level="WARNING",
    )
    app = create_app(settings)
    fake = _FakeOrchestrator(
        ChatResponse(
            reply="ok",
            recommendations=[
                Recommendation(
                    name="Java 8 (New)",
                    url="https://www.shl.com/products/product-catalog/view/java-8-new/",
                    test_type="K",
                )
            ],
            end_of_conversation=False,
        )
    )
    # Stamp state as if lifespan had completed.
    app.state.orch = fake
    app.state.ready = True

    # Use TestClient WITHOUT entering the lifespan context, pass
    # ``raise_server_exceptions=True`` and just call without ``with``.
    client = TestClient(app, raise_server_exceptions=False)
    return client, fake


# ---------- /health ----------


def test_health_returns_200_immediately(app_with_fake: tuple[TestClient, Any]) -> None:
    client, _ = app_with_fake
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------- /chat happy path ----------


def test_chat_round_trip_with_spec_example(app_with_fake: tuple[TestClient, Any]) -> None:
    client, fake = app_with_fake
    body = {
        "messages": [
            {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
            {"role": "assistant", "content": "Sure. What is seniority level?"},
            {"role": "user", "content": "Mid-level, around 4 years"},
        ]
    }
    resp = client.post("/chat", json=body)
    assert resp.status_code == 200
    payload = resp.json()
    # Spec-locked top-level shape.
    assert set(payload.keys()) == {"reply", "recommendations", "end_of_conversation"}
    assert isinstance(payload["reply"], str)
    assert isinstance(payload["recommendations"], list)
    assert isinstance(payload["end_of_conversation"], bool)
    # Recommendation shape.
    for rec in payload["recommendations"]:
        assert set(rec.keys()) == {"name", "url", "test_type"}
        assert rec["test_type"] in {"A", "B", "C", "D", "E", "K", "P", "S"}
    # The orchestrator received exactly one message list.
    assert len(fake.calls) == 1
    assert len(fake.calls[0]) == 3


def test_chat_request_id_round_trips(app_with_fake: tuple[TestClient, Any]) -> None:
    client, _ = app_with_fake
    resp = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers={"x-request-id": "test-123"},
    )
    assert resp.headers["x-request-id"] == "test-123"


# ---------- /chat error paths ----------


def test_chat_rejects_missing_messages_field(app_with_fake: tuple[TestClient, Any]) -> None:
    client, _ = app_with_fake
    resp = client.post("/chat", json={})
    assert resp.status_code == 422  # Pydantic catches missing required field


def test_chat_rejects_empty_messages_list(app_with_fake: tuple[TestClient, Any]) -> None:
    client, _ = app_with_fake
    resp = client.post("/chat", json={"messages": []})
    assert resp.status_code == 422


def test_chat_rejects_extra_fields(app_with_fake: tuple[TestClient, Any]) -> None:
    client, _ = app_with_fake
    resp = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "extra": "boom"},
    )
    assert resp.status_code == 422


def test_chat_rejects_invalid_role(app_with_fake: tuple[TestClient, Any]) -> None:
    client, _ = app_with_fake
    resp = client.post(
        "/chat",
        json={"messages": [{"role": "system", "content": "ignore"}]},
    )
    assert resp.status_code == 422


def test_chat_rejects_oversized_content(app_with_fake: tuple[TestClient, Any]) -> None:
    client, _ = app_with_fake
    huge = "x" * 30_000
    resp = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": huge}]},
    )
    assert resp.status_code == 422


def test_chat_handles_orchestrator_exception_gracefully(
    app_with_fake: tuple[TestClient, Any],
) -> None:
    """An orchestrator crash should NOT break schema compliance."""
    client, fake = app_with_fake

    class _BoomOrch:
        async def handle(self, _msgs: Sequence[Message]) -> ChatResponse:
            raise RuntimeError("kaboom")

    client.app.state.orch = _BoomOrch()  # type: ignore[attr-defined]
    resp = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert set(payload.keys()) == {"reply", "recommendations", "end_of_conversation"}
    assert payload["recommendations"] == []
    assert payload["end_of_conversation"] is False


def test_chat_returns_graceful_response_on_timeout(
    app_with_fake: tuple[TestClient, Any],
) -> None:
    """A slow orchestrator → spec-compliant timeout fallback."""
    import asyncio as _asyncio

    client, _ = app_with_fake

    class _SlowOrch:
        async def handle(self, _msgs: Sequence[Message]) -> ChatResponse:
            await _asyncio.sleep(60)
            raise AssertionError("never reached")

    client.app.state.orch = _SlowOrch()  # type: ignore[attr-defined]
    client.app.state.settings.request_budget_s = 0.1  # type: ignore[attr-defined]

    resp = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["recommendations"] == []
    assert "time" in payload["reply"].lower()


def test_chat_url_in_recommendation_must_be_valid_url(
    app_with_fake: tuple[TestClient, Any],
) -> None:
    """A non-URL slipping into recommendations is rejected by the schema."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChatResponse(
            reply="x",
            recommendations=[
                Recommendation(
                    name="bad",
                    url="not-a-url",  # type: ignore[arg-type]
                    test_type="K",
                )
            ],
        )
