"""Structured logging setup.

JSON to stdout, Render captures stdout into its log viewer, so this
gives us greppable per-request lines without a third-party log sink.

Each request is tagged with a ``request_id`` (set by the middleware
in :mod:`shl_recommender.api.app`) so every log line emitted while
handling a request can be correlated.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure(level: str = "INFO") -> None:
    """Idempotent logger configuration.

    Safe to call multiple times, re-running just resets handlers.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Convenience wrapper. Use module-level ``log = get_logger(__name__)``."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
