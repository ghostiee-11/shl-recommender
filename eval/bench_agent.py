"""Phase 3 benchmark: full agent (orchestrator + LLM router) on the 10 traces.

For each trace we replay the **user** turns from the reference dialogue
(verbatim, in order) into the orchestrator. The agent's own assistant
replies are inserted into history between user turns, the agent has
to handle its own conversational state without seeing the reference
agent's replies.

Why this is the right benchmark:

* Mirrors SHL's evaluator harness: full multi-turn replay against
  ``/chat`` with real LLM calls.
* Captures the slot-extraction + LLM-rerank lift over the Phase 2
  retrieval-only baseline.
* Honest about the 8-turn cap: traces longer than 8 turns are
  truncated; the metric reflects what the agent can do within budget.

Final shortlist per trace = the recommendations from the last
assistant response that returned a non-empty list.

Usage:
    PYTHONPATH=src:. .venv/bin/python -m eval.bench_agent
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from shl_recommender.agent.orchestrator import Orchestrator
from shl_recommender.api.schemas import Message
from shl_recommender.catalog.loader import load_catalog
from shl_recommender.llm.gemini_client import GeminiLLM
from shl_recommender.llm.groq_client import GroqLLM
from shl_recommender.llm.router import LLMRouter
from shl_recommender.llm.throttle import RateLimitedLLM
from shl_recommender.retrieval.bm25 import BM25Index
from shl_recommender.retrieval.dense import DenseIndex
from shl_recommender.retrieval.hybrid import HybridRetriever
from shl_recommender.retrieval.llm_rerank import LLMReranker

from .trace_parser import Trace, load_traces

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
TOP_K = 10
TURN_CAP = 8  # spec: max 8 turns total


def _load_env() -> None:
    """Tiny .env loader so the bench is self-contained."""
    p = Path(__file__).resolve().parents[1] / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def recall_at_k(predicted_urls: list[str], expected_urls: tuple[str, ...], k: int) -> float:
    if not expected_urls:
        return 0.0
    top = set(predicted_urls[:k])
    return sum(1 for u in expected_urls if u in top) / len(expected_urls)


async def replay_trace(orch: Orchestrator, trace: Trace) -> tuple[list[str], int, float]:
    """Replay the user turns of one trace against the agent.

    Returns ``(final_predicted_urls, turn_count, total_latency_s)``.
    """
    history: list[Message] = []
    user_msgs = list(trace.user_messages)
    last_recs: list[str] = []
    started = time.perf_counter()
    turn_count = 0

    for user in user_msgs:
        if turn_count >= TURN_CAP:
            break
        history.append(Message(role="user", content=user))
        turn_count += 1
        resp = await orch.handle(history)
        history.append(Message(role="assistant", content=resp.reply))
        turn_count += 1
        if resp.recommendations:
            last_recs = [str(r.url) for r in resp.recommendations]
        if resp.end_of_conversation:
            break

    return last_recs, turn_count, time.perf_counter() - started


async def main() -> None:
    _load_env()
    if not os.environ.get("GROQ_API_KEY") or not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY and GEMINI_API_KEY must be set (see .env.example)."
        )

    print("Loading catalog…")
    catalog = load_catalog()
    print(f"  catalog items: {len(catalog)}")

    print("Building retrieval indexes…")
    bm25 = BM25Index(catalog.search_docs)
    dense = DenseIndex(catalog.search_docs)
    retriever = HybridRetriever(bm25, dense, catalog)

    print("Initializing LLM router (Groq primary, Gemini fallback) with throttle…")
    primary = RateLimitedLLM(GroqLLM(os.environ["GROQ_API_KEY"]), capacity=5, refill_per_sec=0.45)
    fallback = RateLimitedLLM(
        GeminiLLM(os.environ["GEMINI_API_KEY"]), capacity=2, refill_per_sec=0.07
    )
    router = LLMRouter(primary, fallback)

    reranker = LLMReranker(router, catalog)
    orch = Orchestrator(catalog, retriever, reranker, router, top_k=TOP_K)

    traces = load_traces()
    print(f"\nReplaying {len(traces)} traces (turn cap = {TURN_CAP})…\n")

    rows: list[dict[str, object]] = []
    recalls: list[float] = []
    latencies: list[float] = []
    for tr in traces:
        if not tr.expected_shortlist:
            continue
        try:
            predicted, turns, secs = await replay_trace(orch, tr)
        except Exception as exc:  # noqa: BLE001, bench-level catch is intentional
            print(f"  {tr.trace_id:>4}  ERROR: {exc}")
            rows.append({"trace_id": tr.trace_id, "error": str(exc)})
            continue
        r = recall_at_k(predicted, tr.expected_shortlist, TOP_K)
        recalls.append(r)
        latencies.append(secs)
        rows.append(
            {
                "trace_id": tr.trace_id,
                "expected_count": len(tr.expected_shortlist),
                "predicted_count": len(predicted),
                "recall_at_10": round(r, 3),
                "turn_count": turns,
                "latency_s": round(secs, 2),
            }
        )
        print(
            f"  {tr.trace_id:>4}  exp={len(tr.expected_shortlist):>2} "
            f"pred={len(predicted):>2}  R@10={r:.2f}  turns={turns}  "
            f"latency={secs:.1f}s"
        )

    if not recalls:
        print("\nNo successful runs.")
        return

    mean_r = sum(recalls) / len(recalls)
    p95_lat = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    print(
        f"\nMean Recall@10 over {len(recalls)} traces: {mean_r:.3f}\n"
        f"Per-trace latency: mean={sum(latencies) / len(latencies):.1f}s "
        f"p95={p95_lat:.1f}s\n"
        f"Phase-2 retrieval-only baseline (for comparison): 0.504"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "phase3_agent_e2e.json"
    out.write_text(
        json.dumps(
            {
                "k": TOP_K,
                "turn_cap": TURN_CAP,
                "n_traces": len(recalls),
                "mean_recall_at_10": round(mean_r, 3),
                "mean_latency_s": round(sum(latencies) / len(latencies), 2),
                "p95_latency_s": round(p95_lat, 2),
                "per_trace": rows,
            },
            indent=2,
        )
    )
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    asyncio.run(main())
