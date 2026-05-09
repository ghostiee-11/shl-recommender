"""Unit tests for query construction + expansion."""

from __future__ import annotations

from shl_recommender.retrieval.bm25 import tokenize
from shl_recommender.retrieval.query import build_query, expand_query


# ---------- expand_query ----------


def test_expand_query_adds_seniority_synonyms() -> None:
    out = expand_query("we need a senior leadership solution")
    out_tokens = set(tokenize(out))
    assert {"executive", "director", "leadership"}.issubset(out_tokens)


def test_expand_query_personality_triggers_opq() -> None:
    out = expand_query("we want a personality test")
    assert "opq" in tokenize(out)


def test_expand_query_idempotent() -> None:
    once = expand_query("senior leadership team")
    twice = expand_query(once)
    # Token sets must be identical after a second pass.
    assert set(tokenize(once)) == set(tokenize(twice))


def test_expand_query_empty_input() -> None:
    assert expand_query("") == ""


def test_expand_query_no_triggers_returns_unchanged() -> None:
    q = "totally unrelated foo bar baz"
    assert expand_query(q) == q


# ---------- build_query ----------


def test_build_query_concatenates_with_recency_weighting() -> None:
    out = build_query(["first user msg", "the latest"])
    # last turn weight defaults to 2 → "the latest" appears twice
    assert out.count("the latest") == 2
    assert out.startswith("first user msg")


def test_build_query_handles_single_message() -> None:
    out = build_query(["only msg"])
    assert out.count("only msg") == 2  # last_turn_weight default


def test_build_query_truncates_to_max_tokens() -> None:
    long = " ".join(["word"] * 200)
    out = build_query([long, "tail"], max_tokens=10)
    assert len(out.split()) == 10
    # Recency: last words should still come from the tail.
    assert out.endswith("tail")


def test_build_query_skips_empty_messages() -> None:
    assert build_query(["", "  ", "real"]) == "real real"


def test_build_query_empty_input() -> None:
    assert build_query([]) == ""
