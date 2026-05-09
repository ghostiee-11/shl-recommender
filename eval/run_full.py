"""Full eval suite: trace replay (with simulator) + behavior probes.

Note on the simulator: we run it through the same Groq+Gemini router
the orchestrator uses. Gemini Flash on free tier is 5 RPM, which
429s after the 6th simulator call when run sequentially over 10
traces. Groq's higher RPM absorbs the load; Gemini still gets
exercised when Groq throttles.

CLI:
    PYTHONPATH=src:. .venv/bin/python -m eval.run_full

Outputs JSON + Markdown reports under ``eval/reports/``. Designed to
be CI-callable: prints a final aggregate + exits non-zero if behavior
probes fail (Recall regressions are warned, not enforced, the
gating happens against the prior baseline in CI).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import random
import sys
import time
from pathlib import Path

from shl_recommender.agent.orchestrator import Orchestrator
from shl_recommender.catalog.loader import load_catalog
from shl_recommender.llm.gemini_client import GeminiLLM
from shl_recommender.llm.groq_client import GroqLLM
from shl_recommender.llm.router import LLMRouter
from shl_recommender.llm.throttle import RateLimitedLLM
from shl_recommender.retrieval.bm25 import BM25Index
from shl_recommender.retrieval.dense import DenseIndex
from shl_recommender.retrieval.hybrid import HybridRetriever
from shl_recommender.retrieval.llm_rerank import LLMReranker

from .harness import TraceResult, run_trace
from .persona import Persona, load_personas
from .probes import ProbeResult, run_all_probes
from .simulator import Simulator
from .trace_parser import load_traces

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
TOP_K = 10


def _load_env() -> None:
    p = Path(__file__).resolve().parents[1] / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def _bootstrap_ci(values: list[float], iters: int = 1000) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean."""
    if not values:
        return 0.0, 0.0
    rng = random.Random(42)
    n = len(values)
    means: list[float] = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


async def _run_trace_arm(
    orch: Orchestrator,
    catalog: object,
    simulator: Simulator | None,
    personas: dict[str, Persona],
    *,
    inter_trace_pause_s: float = 30.0,
) -> list[TraceResult]:
    out: list[TraceResult] = []
    for i, tr in enumerate(load_traces()):
        if not tr.expected_shortlist:
            continue
        persona = personas.get(tr.trace_id)
        if persona is None:
            print(f"  {tr.trace_id}: SKIP (no cached persona; run `python -m eval.persona`)")
            continue
        if i > 0 and inter_trace_pause_s > 0:
            # Free-tier RPM (Groq 30/min, Gemini 5/min) only refills with
            # wall-clock time. A short pause between traces avoids cascading
            # 429s that would falsely zero out R@10 for downstream traces.
            await asyncio.sleep(inter_trace_pause_s)
        result = await run_trace(orch, catalog, simulator, tr, persona)  # type: ignore[arg-type]
        suffix = f" ERROR: {result.error}" if result.error else ""
        print(
            f"  {result.trace_id:>4}  R@10={result.recall_at_10:.2f}  "
            f"nDCG@10={result.ndcg_at_10:.2f}  turns={result.turn_count}  "
            f"latency={result.latency_s:.1f}s{suffix}"
        )
        out.append(result)
    return out


async def _run_probe_arm(orch: Orchestrator, catalog: object) -> list[ProbeResult]:
    print("\nProbes:")
    results = await run_all_probes(orch, catalog)  # type: ignore[arg-type]
    for p in results:
        flag = "PASS" if p.passed else "FAIL"
        suffix = f", {p.detail}" if p.detail else ""
        print(f"  [{flag}] {p.name}{suffix}")
    return results


def _write_reports(
    trace_results: list[TraceResult],
    probe_results: list[ProbeResult],
    elapsed_s: float,
) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    recalls = [t.recall_at_10 for t in trace_results if t.error is None]
    ndcgs = [t.ndcg_at_10 for t in trace_results if t.error is None]
    latencies = [t.latency_s for t in trace_results if t.error is None]

    mean_r = sum(recalls) / len(recalls) if recalls else 0.0
    mean_n = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    ci_r = _bootstrap_ci(recalls)
    ci_n = _bootstrap_ci(ndcgs)
    probe_pass = sum(1 for p in probe_results if p.passed)
    probe_total = len(probe_results)

    json_path = REPORTS_DIR / "phase5_full_eval.json"
    json_path.write_text(
        json.dumps(
            {
                "k": TOP_K,
                "n_traces": len(trace_results),
                "mean_recall_at_10": round(mean_r, 3),
                "ci95_recall_at_10": [round(ci_r[0], 3), round(ci_r[1], 3)],
                "mean_ndcg_at_10": round(mean_n, 3),
                "ci95_ndcg_at_10": [round(ci_n[0], 3), round(ci_n[1], 3)],
                "mean_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else 0,
                "probes_pass": probe_pass,
                "probes_total": probe_total,
                "elapsed_total_s": round(elapsed_s, 2),
                "per_trace": [dataclasses.asdict(t) for t in trace_results],
                "probes": [{"name": p.name, "passed": p.passed, "detail": p.detail} for p in probe_results],
            },
            indent=2,
            default=lambda o: list(o) if isinstance(o, tuple) else o,
        )
    )

    md_path = REPORTS_DIR / "phase5_full_eval.md"
    lines = [
        "# Phase 5, Full eval (trace replay + behavior probes)",
        "",
        f"_K = {TOP_K}; bootstrap CI uses 1000 resamples; n traces = {len(trace_results)}_",
        "",
        "## Aggregate",
        "",
        "| Metric | Value | 95% CI |",
        "| --- | ---: | ---: |",
        f"| Mean Recall@10 | {mean_r:.3f} | [{ci_r[0]:.3f}, {ci_r[1]:.3f}] |",
        f"| Mean nDCG@10 | {mean_n:.3f} | [{ci_n[0]:.3f}, {ci_n[1]:.3f}] |",
        f"| Probes passing | {probe_pass} / {probe_total} |, |",
        f"| Mean trace latency | {sum(latencies) / len(latencies):.1f}s |, |" if latencies else "| Mean trace latency | n/a |, |",
        "",
        "## Per-trace",
        "",
        "| Trace | Expected | Predicted | Recall@10 | nDCG@10 | Turns | Latency (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for t in trace_results:
        lines.append(
            f"| {t.trace_id} | {t.expected_count} | {t.predicted_count} | "
            f"{t.recall_at_10:.2f} | {t.ndcg_at_10:.2f} | {t.turn_count} | {t.latency_s:.1f} |"
        )
    lines += ["", "## Behavior probes", "", "| Probe | Result | Detail |", "| --- | --- | --- |"]
    for p in probe_results:
        lines.append(f"| {p.name} | {'PASS' if p.passed else 'FAIL'} | {p.detail or '-'} |")
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


async def main() -> int:
    _load_env()
    if not (os.environ.get("GROQ_API_KEY") and os.environ.get("GEMINI_API_KEY")):
        print("GROQ_API_KEY and GEMINI_API_KEY are required.")
        return 2

    print("Loading catalog…")
    catalog = load_catalog()

    print("Building retrieval indexes…")
    bm25 = BM25Index(catalog.search_docs)
    dense = DenseIndex(catalog.search_docs)
    retriever = HybridRetriever(bm25, dense, catalog)

    print("Initializing LLM router (Groq primary, Gemini fallback) with throttle…")
    router = LLMRouter(
        RateLimitedLLM(GroqLLM(os.environ["GROQ_API_KEY"]), capacity=5, refill_per_sec=0.45),
        RateLimitedLLM(GeminiLLM(os.environ["GEMINI_API_KEY"]), capacity=2, refill_per_sec=0.07),
    )
    reranker = LLMReranker(router, catalog)
    orch = Orchestrator(catalog, retriever, reranker, router, top_k=TOP_K)

    # Default replay mode: verbatim (no simulator → no extra LLM calls).
    # Set ``EVAL_SIMULATOR=1`` to use the LLM-driven simulator instead.
    use_simulator = os.environ.get("EVAL_SIMULATOR") == "1"
    simulator: Simulator | None = Simulator(router) if use_simulator else None
    print(f"Replay mode: {'LLM simulator' if use_simulator else 'verbatim user turns'}")

    personas = load_personas()
    if not personas:
        print(
            "WARN: no cached personas. Run `python -m eval.persona` first; "
            "the harness needs them to drive the simulator."
        )
        return 1

    started = time.perf_counter()
    # Run probes FIRST: they're cheap (1-2 LLM calls each) so they
    # finish on a fresh quota budget. Trace replay is the heavy arm
    # and benefits from running second when probes have already
    # validated the agent's basic behavior.
    probe_results = await _run_probe_arm(orch, catalog)
    print("\nReplaying traces:")
    trace_results = await _run_trace_arm(orch, catalog, simulator, personas)
    elapsed = time.perf_counter() - started

    json_path, md_path = _write_reports(trace_results, probe_results, elapsed)

    recalls = [t.recall_at_10 for t in trace_results if t.error is None]
    mean_r = sum(recalls) / len(recalls) if recalls else 0.0
    probe_pass = sum(1 for p in probe_results if p.passed)
    probe_total = len(probe_results)
    print(
        f"\nMean Recall@10 = {mean_r:.3f}\n"
        f"Probes: {probe_pass}/{probe_total} pass\n"
        f"Total elapsed: {elapsed:.1f}s\n"
        f"Reports: {json_path}, {md_path}"
    )
    # Exit non-zero if any probe failed, gates CI without gating on
    # the noisier Recall metric.
    return 0 if probe_pass == probe_total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
