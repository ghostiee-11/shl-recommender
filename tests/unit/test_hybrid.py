"""Unit tests for RRF fusion + the Hybrid orchestrator (BM25-only path).

Dense retrieval is exercised end-to-end in the integration suite, pulling
``sentence-transformers`` + downloading the bge-small weights into a unit
test would make the suite slow and network-bound. The fusion math is the
interesting part and is fully covered here with synthetic ranked lists.
"""

from __future__ import annotations

from shl_recommender.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_single_list_passthrough_order() -> None:
    fused = reciprocal_rank_fusion([[(0, 0.9), (1, 0.5), (2, 0.1)]])
    assert [i for i, _ in fused] == [0, 1, 2]


def test_rrf_two_lists_combines_signal() -> None:
    # Doc 1 is rank-2 in both lists → highest combined RRF score.
    # Doc 0 is rank-1 in list A only; doc 2 is rank-1 in list B only.
    list_a = [(0, 0.9), (1, 0.5), (3, 0.1)]
    list_b = [(2, 0.9), (1, 0.5), (4, 0.1)]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    fused_dict = dict(fused)

    # Doc 1 appears in both → 1/(60+2) + 1/(60+2) = 2/62 ≈ 0.0322
    # Doc 0 appears once at rank 1 → 1/61 ≈ 0.01639
    # Doc 2 appears once at rank 1 → 1/61 ≈ 0.01639
    assert fused_dict[1] > fused_dict[0]
    assert fused_dict[1] > fused_dict[2]
    assert fused[0][0] == 1


def test_rrf_tie_break_by_index() -> None:
    # Two docs with identical contributions → lower index wins.
    fused = reciprocal_rank_fusion([[(5, 0.9), (2, 0.9)], [(2, 0.9), (5, 0.9)]])
    fused_dict = dict(fused)
    assert abs(fused_dict[5] - fused_dict[2]) < 1e-9
    # tie-broken ascending
    assert fused[0][0] == 2


def test_rrf_empty_input() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_k_dampens_top_rank_bias() -> None:
    # With small k, rank 1 dominates rank 2 by a lot.
    # With large k, the gap shrinks. Verify monotonicity.
    list_a = [(0, 1.0), (1, 0.5)]
    fused_small_k = dict(reciprocal_rank_fusion([list_a], k=1))
    fused_large_k = dict(reciprocal_rank_fusion([list_a], k=1000))
    ratio_small = fused_small_k[0] / fused_small_k[1]
    ratio_large = fused_large_k[0] / fused_large_k[1]
    assert ratio_small > ratio_large
