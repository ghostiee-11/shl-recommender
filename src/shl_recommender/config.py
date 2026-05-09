"""Runtime configuration.

Single source of truth for env-driven settings. Built on
``pydantic-settings`` so:

* Defaults live in code (no scattered ``os.getenv`` calls).
* Validation is automatic, booleans parsed correctly, missing
  required fields fail loudly at startup, not at first request.
* Tests can construct ``Settings(...)`` directly without touching
  the environment.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from env (or ``.env`` for local dev)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM credentials. At least one must be set; the router prefers
    # OpenAI (highest tier) → Groq (fast) → Gemini (free fallback).
    openai_api_key: str = Field(default="", description="OpenAI API key (preferred primary).")
    groq_api_key: str = Field(default="", description="Groq free-tier API key")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")

    # Behavior toggles.
    log_level: str = Field(default="INFO", description="DEBUG / INFO / WARNING / ERROR")
    embed_state_hint: bool = Field(
        default=True,
        description="Emit hidden HTML-comment state in assistant replies for stateless reconstruction.",
    )

    # Latency budgets, tuned for the 30s/8-turn spec cap.
    request_budget_s: float = Field(
        default=25.0,
        description="Hard wall-clock cap on /chat. Leaves headroom under the 30s spec timeout.",
    )

    # Recommendation sizing.
    recommendation_top_k: int = Field(default=5, ge=1, le=10)

    # Pre-computed catalog embeddings (text-embedding-3-small, 1536-d,
    # L2-normalized, shape (n_items, 1536)). Shipped in the repo so the
    # runtime image doesn't need torch / sentence-transformers.
    catalog_embeddings_path: str = Field(
        default="data/catalog_embeddings.npy",
        description="Path to the pre-computed catalog embedding matrix.",
    )

    # CORS, allow the frontend (local dev + deployed) to call the API.
    # Comma-separated string env var → list at runtime. Empty default
    # means CORS is disabled (no Access-Control headers added).
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        description="Comma-separated list of allowed Origin headers for CORS.",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # HTTP safety layers.
    max_body_bytes: int = Field(
        default=64 * 1024,
        description="Reject requests with Content-Length above this (HTTP 413).",
    )
    rate_limit_capacity: int = Field(
        default=30,
        description="Max burst per client IP before throttling (HTTP 429).",
    )
    rate_limit_refill_per_sec: float = Field(
        default=1.0,
        description="Sustained tokens-per-second per client IP.",
    )

    # Outgoing-LLM throttles. Free tiers cap us at low RPM; we self-throttle
    # so the agent never trips the provider's rate limit and never has to
    # circuit-break to the slower fallback under sustained load. OpenAI's
    # ceiling at Tier-2 is so high (5000 RPM) we essentially leave it
    # unthrottled but keep the wrapper so it stays observable.
    openai_capacity: int = Field(default=20, description="OpenAI token-bucket burst capacity.")
    openai_refill_per_sec: float = Field(
        default=20.0,
        description="OpenAI sustained calls/sec (Tier-2: 5000 RPM ≈ 83/s; we cap conservatively).",
    )
    groq_capacity: int = Field(default=5, description="Groq token-bucket burst capacity.")
    groq_refill_per_sec: float = Field(
        default=0.45,
        description="Groq sustained calls/sec (Llama-3.3-70B free tier ≈ 30 RPM = 0.5/s).",
    )
    gemini_capacity: int = Field(default=2, description="Gemini token-bucket burst capacity.")
    gemini_refill_per_sec: float = Field(
        default=0.07,
        description="Gemini sustained calls/sec (2.5 Flash free tier ≈ 5 RPM = 0.083/s).",
    )
