"""API request/response schemas, spec-locked.

The shapes here mirror the assignment specification **exactly**:

* ``ChatRequest{messages: [Message{role, content}]}``
* ``ChatResponse{reply, recommendations: [Recommendation{name, url, test_type}], end_of_conversation}``

Per the brief, "the schema is non-negotiable. Deviating breaks our
automated evaluator, and your submission will not score." Every
field name and JSON shape here is therefore fixed; refactor at your
peril.

We use ``model_config = ConfigDict(extra="forbid")`` on the request
side so any unexpected key is a 400 (caught early), but **NOT** on
the response side because Pydantic v2 ``model_dump`` already excludes
unknown fields and we want forward-compatibility room.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from shl_recommender.catalog.models import TestTypeCode

Role = Literal["user", "assistant"]


class Message(BaseModel):
    """One conversation turn. ``content`` is plain text including any
    embedded state hint (HTML comment) emitted by the agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    content: str = Field(min_length=0, max_length=20_000)


class ChatRequest(BaseModel):
    """Stateless request body for ``POST /chat``."""

    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(min_length=1, max_length=64)


class Recommendation(BaseModel):
    """One shortlisted assessment.

    ``name``, ``url``, ``test_type`` mirror the spec example. We use
    ``HttpUrl`` for ``url`` so any non-URL value fails validation
    rather than leaking into the response.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    url: HttpUrl
    test_type: TestTypeCode


class ChatResponse(BaseModel):
    """Spec-locked response body."""

    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=10)
    end_of_conversation: bool = False
