"""Phase 2 benchmark: per-stage Recall@10 over the 10 reference traces.

Stages reported (each builds on the previous):

1. **BM25-only**, pure lexical baseline.
2. **Dense-only**, bge-small bi-encoder.
3. **Hybrid (RRF)**, RRF fusion of BM25 + dense, single clean query.
4. **Hybrid + Asymmetric QE**, expanded query to BM25 only (dense
   keeps the clean query). Pure expansion of both stages dilutes the
   dense signal in our experiments.
5. **Hybrid + Asymmetric QE + Cross-Encoder**, reranks the top-30
   candidates with ms-marco MiniLM, scored against the *clean* query
   (the cross-encoder is a general MS-MARCO checkpoint and prefers
   natural-language queries over expanded keyword bags).

Recall@K is the fraction of the trace's expected URLs that appear in
the top-K predictions. Mean is taken over the 10 traces.

Each trace's "query" is built from its user messages via
:func:`build_query` (last-turn weighted), then optionally expanded
via :func:`expand_query`.

Usage:
    PYTHONPATH=src:. .venv/bin/python -m eval.bench_retrieval
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from shl_recommender.catalog.loader import load_catalog
from shl_recommender.retrieval.bm25 import BM25Index
from shl_recommender.retrieval.dense import DenseIndex
from shl_recommender.retrieval.hybrid import HybridRetriever
from shl_recommender.retrieval.query import build_query, expand_query
from shl_recommender.retrieval.rerank import CrossEncoderReranker

from .trace_parser import load_traces, parse_stats

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
TOP_K = 10
CANDIDATE_POOL = 30  # candidates fed to the cross-encoder


def recall_at_k(predicted_urls: list[str], expected_urls: tuple[str, ...], k: int) -> float:
    if not expected_urls:
        return 0.0
    top_k = set(predicted_urls[:k])
    hits = sum(1 for u in expected_urls if u in top_k)
    return hits / len(expected_urls)


def _urls(catalog: object, ids: list[int]) -> list[str]:
    return [catalog.items[i].link for i in ids]  # type: ignore[attr-defined]


def main() -> None:
    print("Loading catalog…")
    catalog = load_catalog()
    print(f"  catalog items: {len(catalog)}")

    print("Building BM25 index…")
    t0 = time.perf_counter()
    bm25 = BM25Index(catalog.search_docs)
    print(f"  built in {time.perf_counter() - t0:.2f}s")

    print("Building dense index (loads bge-small)…")
    t0 = time.perf_counter()
    dense = DenseIndex(catalog.search_docs)
    print(f"  built in {time.perf_counter() - t0:.2f}s")

    retriever = HybridRetriever(bm25, dense, catalog)

    print("Initializing cross-encoder (lazy-load on first call)…")
    cross = CrossEncoderReranker(doc_text_fn=lambda i: catalog.search_docs[i])

    traces = load_traces()
    stats = parse_stats(traces)
    print(
        f"\nTraces: {stats.file_count} files | "
        f"{stats.turn_count} turns | "
        f"{stats.expected_url_count} expected URLs"
    )

    rows: list[dict[str, object]] = []
    means: dict[str, list[float]] = {
        "bm25": [],
        "dense": [],
        "hybrid": [],
        "hybrid_qe": [],
        "hybrid_qe_ce": [],
    }

    print(f"\nPer-trace Recall@{TOP_K} (5 stages):")
    header = "  trace  exp  bm25  dense  hybr  +QE   +QE+CE"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for tr in traces:
        if not tr.expected_shortlist:
            continue
        plain_q = build_query(tr.user_messages)
        expanded_q = expand_query(plain_q)

        bm25_ids = [i for i, _ in bm25.search(plain_q, k=TOP_K)]
        bm25_r = recall_at_k(_urls(catalog, bm25_ids), tr.expected_shortlist, TOP_K)

        dense_ids = [i for i, _ in dense.search(plain_q, k=TOP_K)]
        dense_r = recall_at_k(_urls(catalog, dense_ids), tr.expected_shortlist, TOP_K)

        hybrid_hits = retriever.search(plain_q, top_n=TOP_K)
        hybrid_r = recall_at_k(
            _urls(catalog, [h.doc_index for h in hybrid_hits]),
            tr.expected_shortlist,
            TOP_K,
        )

        # Asymmetric expansion: BM25 sees the expanded query, dense sees clean.
        hybrid_qe_hits = retriever.search(plain_q, bm25_query=expanded_q, top_n=TOP_K)
        hybrid_qe_r = recall_at_k(
            _urls(catalog, [h.doc_index for h in hybrid_qe_hits]),
            tr.expected_shortlist,
            TOP_K,
        )

        # Cross-encoder gets a deeper candidate pool from the asymmetric
        # search, then re-scored against the *clean* (natural language)
        # query, the MS-MARCO checkpoint expects natural queries.
        candidates = retriever.search(plain_q, bm25_query=expanded_q, top_n=CANDIDATE_POOL)
        reranked = cross.rerank(plain_q, candidates, top_k=TOP_K)
        hybrid_qe_ce_r = recall_at_k(
            _urls(catalog, [h.doc_index for h in reranked]),
            tr.expected_shortlist,
            TOP_K,
        )

        means["bm25"].append(bm25_r)
        means["dense"].append(dense_r)
        means["hybrid"].append(hybrid_r)
        means["hybrid_qe"].append(hybrid_qe_r)
        means["hybrid_qe_ce"].append(hybrid_qe_ce_r)
        rows.append(
            {
                "trace_id": tr.trace_id,
                "expected_count": len(tr.expected_shortlist),
                "bm25": round(bm25_r, 3),
                "dense": round(dense_r, 3),
                "hybrid": round(hybrid_r, 3),
                "hybrid_qe": round(hybrid_qe_r, 3),
                "hybrid_qe_ce": round(hybrid_qe_ce_r, 3),
            }
        )
        print(
            f"  {tr.trace_id:>4}   {len(tr.expected_shortlist):>2}  "
            f"{bm25_r:.2f}  {dense_r:.2f}   {hybrid_r:.2f}  {hybrid_qe_r:.2f}  {hybrid_qe_ce_r:.2f}"
        )

    n = len(means["bm25"])
    means_avg = {k: round(sum(v) / n, 3) for k, v in means.items()}
    print(f"\nMean Recall@{TOP_K} over {n} traces:")
    print(f"  BM25-only          : {means_avg['bm25']}")
    print(f"  Dense-only         : {means_avg['dense']}")
    print(f"  Hybrid (RRF)       : {means_avg['hybrid']}")
    print(f"  Hybrid + QE        : {means_avg['hybrid_qe']}")
    print(f"  Hybrid + QE + XEnc : {means_avg['hybrid_qe_ce']}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORTS_DIR / "phase2_retrieval_baseline.json"
    out_json.write_text(
        json.dumps(
            {
                "k": TOP_K,
                "candidate_pool": CANDIDATE_POOL,
                "n_traces": n,
                "mean": means_avg,
                "per_trace": rows,
            },
            indent=2,
        )
    )

    # Human-readable markdown report, gitignored under eval/reports/
    # but copied into the README's "Eval results" table at release time.
    out_md = REPORTS_DIR / "phase2_retrieval_baseline.md"
    md_lines = [
        "# Phase 2, Retrieval baseline (Recall@10 over 10 reference traces)",
        "",
        f"_K = {TOP_K}, candidate pool for cross-encoder = {CANDIDATE_POOL}_",
        "",
        "## Mean Recall@10",
        "",
        "| Stage | Mean Recall@10 |",
        "| --- | ---: |",
        f"| BM25-only | {means_avg['bm25']:.3f} |",
        f"| Dense-only | {means_avg['dense']:.3f} |",
        f"| Hybrid (RRF) | {means_avg['hybrid']:.3f} |",
        f"| Hybrid + Asymmetric QE | {means_avg['hybrid_qe']:.3f} |",
        f"| Hybrid + Asymmetric QE + Cross-Encoder | {means_avg['hybrid_qe_ce']:.3f} |",
        "",
        "## Per-trace",
        "",
        "| Trace | Expected | BM25 | Dense | Hybrid | +QE | +QE+CE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['trace_id']} | {r['expected_count']} | "
            f"{r['bm25']:.2f} | {r['dense']:.2f} | {r['hybrid']:.2f} | "
            f"{r['hybrid_qe']:.2f} | {r['hybrid_qe_ce']:.2f} |"
        )
    out_md.write_text("\n".join(md_lines) + "\n")

    print(f"\nReports written to {out_json} and {out_md}")


if __name__ == "__main__":
    main()
