"""
Helpers for namespace hybrid_search vector / search-index / combined integration tests.

Reuses the large deterministic corpus from ``namespace_fts_helpers`` so full-text,
metadata (JSON SEARCH INDEX), and vector branches share one ground-truth dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from namespace_fts_helpers import (
    BATCH_SIZE,
    CORPUS_SIZE,
    INDEX_SETTLE_SECONDS,
    TOKEN_ALP,
    TOKEN_ZPX,
    CorpusRecord,
    assert_hybrid_fulltext_result,
    assert_not_contains_no_token_leak,
    build_large_fts_corpus,
    corpus_matches_fts,
    doc_matches_where_document,
    doc_matches_where_metadata,
    get_fts_case,
    insert_corpus_in_batches,
    run_hybrid_search_fts_case,
    setup_fts_namespace_with_corpus,
    setup_large_fts_collection,
    setup_multi_coll_multi_ns_fts,
    setup_multi_coll_multi_ns_fts_single_loaded,
    teardown_large_fts_collection,
    teardown_multi_coll_multi_ns_fts,
    MULTI_COLL_MULTI_NS_FTS_LOADED_QUADRANTS,
    MULTI_COLL_MULTI_NS_QUADRANT_KEYS,
)

# Fixed query vector for KNN tests (same dimension as corpus embeddings).
KNN_QUERY_VECTOR: list[float] = [1.0, 1.0, 0.0]

RRF_RANK = {"rrf": {"rank_window_size": 60, "rank_constant": 60}}


@dataclass(frozen=True)
class SearchIndexQueryCase:
    name: str
    where: dict[str, Any]
    n_results: int
    min_hits: int = 1
    exact_match_count: int | None = None


@dataclass(frozen=True)
class VectorKnnCase:
    name: str
    query_vector: list[float]
    n_results: int
    where: dict[str, Any] | None = None
    check_top1: bool = True


@dataclass(frozen=True)
class HybridCombinedCase:
    """Multi-branch hybrid_search (full-text + metadata + optional KNN)."""

    name: str
    n_results: int
    where_document: dict[str, Any] | str | None = None
    where: dict[str, Any] | None = None
    knn: dict[str, Any] | None = None
    use_rrf: bool = False
    check_fts_ranking: bool = True
    # When True, only assert non-empty results in corpus (RRF reorders across branches).
    rrf_smoke_only: bool = False


def skip_oceanbase_knn(request: Any) -> None:
    """OceanBase hybrid_search KNN on namespace logical tables requires HNSW (collections use IVF)."""
    import pytest

    if "oceanbase" in request.node.nodeid:
        pytest.skip(
            "OceanBase hybrid_search KNN requires HNSW; namespace collections use IVF only."
        )


def corpus_matches_where(record: CorpusRecord, where: dict[str, Any]) -> bool:
    """Metadata / ``#id`` filter against a corpus row (hybrid_search ``query.where`` / ``knn.where``)."""
    if "$and" in where:
        return all(corpus_matches_where(record, sub) for sub in where["$and"])
    if "$or" in where:
        return any(corpus_matches_where(record, sub) for sub in where["$or"])
    if "$not" in where:
        return not corpus_matches_where(record, where["$not"])

    for key, value in where.items():
        if key in ("$and", "$or", "$not"):
            continue
        if key == "#id":
            if isinstance(value, dict):
                if "$in" in value and record.doc_id not in value["$in"]:
                    return False
                if "$nin" in value and record.doc_id in value["$nin"]:
                    return False
                if "$eq" in value and record.doc_id != value["$eq"]:
                    return False
                if "$ne" in value and record.doc_id == value["$ne"]:
                    return False
            elif record.doc_id != value:
                return False
            continue
        if not doc_matches_where_metadata(record.metadata, {key: value}):
            return False
    return True


def count_corpus_matches(corpus: list[CorpusRecord], where: dict[str, Any]) -> int:
    return sum(1 for rec in corpus if corpus_matches_where(rec, where))


def l2_squared(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def expected_knn_ids(
    corpus: list[CorpusRecord],
    query_vector: list[float],
    n_results: int,
    where: dict[str, Any] | None = None,
) -> list[str]:
    scored: list[tuple[str, float]] = []
    for rec in corpus:
        if where is not None and not corpus_matches_where(rec, where):
            continue
        scored.append((rec.doc_id, l2_squared(rec.embedding, query_vector)))
    scored.sort(key=lambda item: (item[1], item[0]))
    return [doc_id for doc_id, _ in scored[:n_results]]


def assert_hybrid_search_index_result(
    corpus: list[CorpusRecord],
    result: dict[str, Any],
    where: dict[str, Any],
    n_results: int,
    *,
    min_hits: int = 1,
    exact_match_count: int | None = None,
) -> None:
    assert result is not None
    assert "ids" in result and result["ids"]
    ids = result["ids"][0]
    assert len(ids) <= n_results
    assert len(ids) >= min_hits, f"expected at least {min_hits} search-index hits"

    id_to_meta = {rec.doc_id: rec.metadata for rec in corpus}
    for doc_id in ids:
        assert doc_id in id_to_meta, f"unknown id {doc_id!r}"
        assert corpus_matches_where(
            next(rec for rec in corpus if rec.doc_id == doc_id), where
        ), f"id={doc_id!r} does not satisfy where={where!r}"

    total_matches = count_corpus_matches(corpus, where)
    if exact_match_count is not None:
        assert total_matches == exact_match_count, (
            f"test case assumes {exact_match_count} corpus matches, got {total_matches}"
        )
    if exact_match_count is not None and exact_match_count <= n_results:
        assert len(ids) == exact_match_count
        expected_ids = {
            rec.doc_id for rec in corpus if corpus_matches_where(rec, where)
        }
        assert set(ids) == expected_ids
    elif len(ids) == n_results and total_matches > n_results:
        # No higher-priority scalar key — every returned row is valid; ensure we did not
        # miss matches only when the engine returned a full page and corpus is small.
        pass


def assert_hybrid_knn_result(
    corpus: list[CorpusRecord],
    result: dict[str, Any],
    query_vector: list[float],
    n_results: int,
    where: dict[str, Any] | None = None,
    *,
    check_top1: bool = True,
) -> None:
    assert result is not None
    assert "ids" in result and result["ids"]
    ids = result["ids"][0]
    distances = result.get("distances", [[]])[0] if result.get("distances") else []
    assert len(ids) <= n_results
    assert len(ids) > 0, "expected at least one KNN hit"

    corpus_by_id = {rec.doc_id: rec for rec in corpus}
    for doc_id in ids:
        rec = corpus_by_id[doc_id]
        if where is not None:
            assert corpus_matches_where(rec, where), (
                f"id={doc_id!r} metadata does not satisfy knn.where={where!r}"
            )

    if distances:
        assert len(distances) == len(ids)
        for dist in distances:
            assert dist >= 0
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1] or math.isclose(
                distances[i], distances[i + 1]
            ), f"KNN distances should be non-decreasing (L2): {distances!r}"

    expected = expected_knn_ids(corpus, query_vector, n_results, where)
    if check_top1 and expected:
        assert ids[0] == expected[0], (
            f"top-1 KNN must be nearest neighbor, got {ids[0]!r} expected {expected[0]!r}"
        )

    if len(ids) == n_results:
        worst = distances[-1] if distances else float("inf")
        for rec in corpus:
            if where is not None and not corpus_matches_where(rec, where):
                continue
            d = l2_squared(rec.embedding, query_vector)
            if rec.doc_id not in ids and (not distances or d < worst - 1e-9):
                raise AssertionError(
                    f"closer match {rec.doc_id!r} (l2²={d}) missing from top-{n_results}"
                )


def run_hybrid_search_index_case(
    namespace: Any,
    corpus: list[CorpusRecord],
    case: SearchIndexQueryCase,
) -> dict[str, Any]:
    result = namespace.hybrid_search(
        query={"where": case.where, "n_results": case.n_results},
        n_results=case.n_results,
        include=["metadatas"],
    )
    assert_hybrid_search_index_result(
        corpus,
        result,
        case.where,
        case.n_results,
        min_hits=case.min_hits,
        exact_match_count=case.exact_match_count,
    )
    return result


def run_hybrid_knn_case(
    namespace: Any,
    corpus: list[CorpusRecord],
    case: VectorKnnCase,
) -> dict[str, Any]:
    knn: dict[str, Any] = {
        "query_embeddings": case.query_vector,
        "n_results": case.n_results,
    }
    if case.where is not None:
        knn["where"] = case.where
    result = namespace.hybrid_search(
        knn=knn,
        n_results=case.n_results,
        include=["metadatas", "distances"],
    )
    assert_hybrid_knn_result(
        corpus,
        result,
        case.query_vector,
        case.n_results,
        case.where,
        check_top1=case.check_top1,
    )
    return result


def run_hybrid_combined_case(
    namespace: Any,
    corpus: list[CorpusRecord],
    case: HybridCombinedCase,
) -> dict[str, Any]:
    query: dict[str, Any] | None = None
    if case.where_document is not None or case.where is not None:
        query = {"n_results": case.n_results}
        if case.where_document is not None:
            query["where_document"] = case.where_document
        if case.where is not None:
            query["where"] = case.where

    rank = RRF_RANK if case.use_rrf else None
    result = namespace.hybrid_search(
        query=query,
        knn=case.knn,
        rank=rank,
        n_results=case.n_results,
        include=["documents", "metadatas", "distances"],
    )

    corpus_by_id = {rec.doc_id: rec for rec in corpus}
    ids = result["ids"][0]
    assert len(ids) <= case.n_results
    assert len(ids) > 0, f"expected hybrid_search hits for case {case.name!r}"
    for doc_id in ids:
        assert doc_id in corpus_by_id, f"unknown id {doc_id!r}"

    if case.rrf_smoke_only:
        return result

    if case.knn is not None and case.where_document is None and query is None:
        vec = case.knn["query_embeddings"]
        assert_hybrid_knn_result(
            corpus,
            result,
            vec,
            case.n_results,
            case.knn.get("where"),
            check_top1=True,
        )
        return result

    if case.where_document is not None:
        assert_hybrid_fulltext_result(
            corpus,
            result,
            case.where_document,
            case.n_results,
            where=case.where,
            check_ranking=case.check_fts_ranking,
        )
    elif case.where is not None:
        assert_hybrid_search_index_result(
            corpus, result, case.where, case.n_results
        )
    return result


# --- Search-index-only cases (JSON SEARCH INDEX / metadata ``query.where``) ---

SEARCH_INDEX_CASES: list[SearchIndexQueryCase] = [
    SearchIndexQueryCase(
        name="eq_zpx_hint_direct",
        where={"zpx_hint": 50},
        n_results=5,
        exact_match_count=1,
    ),
    SearchIndexQueryCase(
        name="eq_zpx_hint_operator",
        where={"zpx_hint": {"$eq": 50}},
        n_results=5,
        exact_match_count=1,
    ),
    SearchIndexQueryCase(
        name="ne_zpx_hint_zero",
        where={"zpx_hint": {"$ne": 0}},
        n_results=20,
        min_hits=1,
    ),
    SearchIndexQueryCase(
        name="gte_zpx_hint_40",
        where={"zpx_hint": {"$gte": 40}},
        n_results=15,
        min_hits=2,
    ),
    SearchIndexQueryCase(
        name="lt_zpx_hint_10",
        where={"zpx_hint": {"$lt": 10}},
        n_results=20,
        min_hits=1,
    ),
    SearchIndexQueryCase(
        name="lte_alp_hint_8",
        where={"alp_hint": {"$lte": 8}},
        n_results=15,
        min_hits=1,
    ),
    SearchIndexQueryCase(
        name="gt_rel_hint_zero",
        where={"rel_hint": {"$gt": 0}},
        n_results=25,
        min_hits=5,
    ),
    SearchIndexQueryCase(
        name="in_alp_hint_values",
        where={"alp_hint": {"$in": [32, 24, 16, 8]}},
        n_results=10,
        min_hits=4,
    ),
    SearchIndexQueryCase(
        name="nin_alp_hint_high",
        where={"alp_hint": {"$nin": [32, 24, 16, 8]}},
        n_results=15,
        min_hits=1,
    ),
    SearchIndexQueryCase(
        name="and_has_both_zpx_gte",
        where={
            "$and": [
                {"has_both": True},
                {"zpx_hint": {"$gte": 5}},
            ],
        },
        n_results=5,
        exact_match_count=1,
    ),
    SearchIndexQueryCase(
        name="or_zpx_alp_hint_high",
        where={
            "$or": [
                {"zpx_hint": {"$gte": 50}},
                {"alp_hint": {"$gte": 32}},
            ],
        },
        n_results=10,
        min_hits=2,
    ),
    SearchIndexQueryCase(
        name="not_has_both",
        where={"$not": {"has_both": True}},
        n_results=20,
        min_hits=1,
    ),
    SearchIndexQueryCase(
        name="id_eq_both_tokens",
        where={"#id": "both_tokens"},
        n_results=5,
        exact_match_count=1,
    ),
    SearchIndexQueryCase(
        name="id_in_zpx_tops",
        where={"#id": {"$in": ["zpx_top_5", "zpx_top_4", "zpx_top_3"]}},
        n_results=5,
        exact_match_count=3,
    ),
]


def get_search_index_case(name: str) -> SearchIndexQueryCase:
    for case in SEARCH_INDEX_CASES:
        if case.name == name:
            return case
    raise KeyError(f"unknown search-index case: {name!r}")


# --- Pure KNN cases ---

VECTOR_KNN_CASES: list[VectorKnnCase] = [
    VectorKnnCase(
        name="knn_global_top5",
        query_vector=KNN_QUERY_VECTOR,
        n_results=5,
    ),
    VectorKnnCase(
        name="knn_filter_has_both",
        query_vector=KNN_QUERY_VECTOR,
        n_results=3,
        where={"has_both": True},
        check_top1=True,
    ),
    VectorKnnCase(
        name="knn_filter_zpx_hint_gte_40",
        query_vector=KNN_QUERY_VECTOR,
        n_results=5,
        where={"zpx_hint": {"$gte": 40}},
    ),
]


def get_vector_knn_case(name: str) -> VectorKnnCase:
    for case in VECTOR_KNN_CASES:
        if case.name == name:
            return case
    raise KeyError(f"unknown vector KNN case: {name!r}")


# --- Combined hybrid_search cases ---

HYBRID_COMBINED_CASES: list[HybridCombinedCase] = [
    # Full-text + search index (metadata on query branch)
    HybridCombinedCase(
        name="fts_contains_zpx_filter_has_both",
        where_document={"$contains": TOKEN_ZPX},
        where={"has_both": True},
        n_results=5,
    ),
    # Vector + search index (KNN branch filter only)
    HybridCombinedCase(
        name="knn_with_where_has_both",
        n_results=5,
        knn={
            "query_embeddings": KNN_QUERY_VECTOR,
            "where": {"has_both": True},
            "n_results": 10,
        },
    ),
    # Vector + full-text + RRF
    HybridCombinedCase(
        name="fts_or_zpx_alp_plus_knn",
        where_document={
            "$or": [
                {"$contains": TOKEN_ZPX},
                {"$contains": TOKEN_ALP},
            ],
        },
        knn={"query_embeddings": KNN_QUERY_VECTOR, "n_results": 15},
        n_results=10,
        use_rrf=True,
        rrf_smoke_only=True,
    ),
    # Vector + full-text + search index + RRF (smoke: fused ranking across three signals)
    HybridCombinedCase(
        name="fts_zpx_filter_gte_knn",
        where_document={"$contains": TOKEN_ZPX},
        where={"zpx_hint": {"$gte": 40}},
        knn={
            "query_embeddings": KNN_QUERY_VECTOR,
            "where": {"zpx_hint": {"$gte": 40}},
            "n_results": 15,
        },
        n_results=8,
        use_rrf=True,
        rrf_smoke_only=True,
    ),
    # Full-text + search index without vector (aligned with FTS case ``contains_zpx_filter_zpx_hint_and_alp_zero``)
    HybridCombinedCase(
        name="fts_zpx_filter_zpx_hint_and_alp_zero",
        where_document={"$contains": TOKEN_ZPX},
        where={
            "$and": [
                {"zpx_hint": {"$gte": 40}},
                {"alp_hint": 0},
            ],
        },
        n_results=10,
    ),
]


def get_hybrid_combined_case(name: str) -> HybridCombinedCase:
    for case in HYBRID_COMBINED_CASES:
        if case.name == name:
            return case
    raise KeyError(f"unknown combined hybrid case: {name!r}")


def run_search_index_case_on_quadrants(
    ctx: dict[str, Any],
    case: SearchIndexQueryCase | str,
    quadrant_keys: tuple[str, ...] = MULTI_COLL_MULTI_NS_FTS_LOADED_QUADRANTS,
) -> None:
    si_case = case if isinstance(case, SearchIndexQueryCase) else get_search_index_case(case)
    for key in quadrant_keys:
        corpus, namespace = ctx[key]
        run_hybrid_search_index_case(namespace, corpus, si_case)


def run_knn_case_on_quadrants(
    ctx: dict[str, Any],
    case: VectorKnnCase | str,
    quadrant_keys: tuple[str, ...] = MULTI_COLL_MULTI_NS_FTS_LOADED_QUADRANTS,
    *,
    request: Any = None,
) -> None:
    if request is not None:
        skip_oceanbase_knn(request)
    knn_case = case if isinstance(case, VectorKnnCase) else get_vector_knn_case(case)
    for key in quadrant_keys:
        corpus, namespace = ctx[key]
        run_hybrid_knn_case(namespace, corpus, knn_case)


def assert_hybrid_search_index_no_hits(
    namespace: Any,
    where: dict[str, Any],
    *,
    n_results: int = 10,
) -> None:
    result = namespace.hybrid_search(
        query={"where": where, "n_results": n_results},
        n_results=n_results,
        include=["metadatas"],
    )
    ids = result.get("ids", [[]])[0] if result.get("ids") else []
    assert len(ids) == 0, (
        f"expected no search-index hits in namespace {namespace.name!r}, got {ids[:5]!r}"
    )


__all__ = [
    "BATCH_SIZE",
    "CORPUS_SIZE",
    "HYBRID_COMBINED_CASES",
    "INDEX_SETTLE_SECONDS",
    "KNN_QUERY_VECTOR",
    "MULTI_COLL_MULTI_NS_FTS_LOADED_QUADRANTS",
    "MULTI_COLL_MULTI_NS_QUADRANT_KEYS",
    "RRF_RANK",
    "SEARCH_INDEX_CASES",
    "TOKEN_ALP",
    "TOKEN_ZPX",
    "VECTOR_KNN_CASES",
    "HybridCombinedCase",
    "SearchIndexQueryCase",
    "VectorKnnCase",
    "assert_hybrid_search_index_no_hits",
    "assert_not_contains_no_token_leak",
    "build_large_fts_corpus",
    "get_fts_case",
    "get_hybrid_combined_case",
    "get_search_index_case",
    "get_vector_knn_case",
    "run_hybrid_combined_case",
    "run_hybrid_knn_case",
    "run_hybrid_search_index_case",
    "run_hybrid_search_fts_case",
    "run_knn_case_on_quadrants",
    "run_search_index_case_on_quadrants",
    "setup_fts_namespace_with_corpus",
    "setup_large_fts_collection",
    "setup_multi_coll_multi_ns_fts",
    "setup_multi_coll_multi_ns_fts_single_loaded",
    "skip_oceanbase_knn",
    "teardown_large_fts_collection",
    "teardown_multi_coll_multi_ns_fts",
]
