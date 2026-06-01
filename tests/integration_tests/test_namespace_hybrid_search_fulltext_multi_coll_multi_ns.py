"""
Namespace hybrid_search full-text tests: 2 collections x 2 namespaces (use_namespace=True).

Verifies full-text correctness in each quadrant (same assertions as single-namespace tests)
and cross-quadrant isolation (empty namespaces must not leak hits).

Run one case in isolation, e.g.::

    pytest tests/integration_tests/test_namespace_hybrid_search_fulltext_multi_coll_multi_ns.py \\
        -k "or_zpx_alp and oceanbase" -v -s
"""

from __future__ import annotations

import pytest

from namespace_fts_helpers import (
    MULTI_COLL_MULTI_NS_QUADRANT_KEYS,
    TOKEN_ZPX,
    assert_hybrid_search_no_hits,
    assert_not_contains_no_token_leak,
    get_fts_case,
    run_hybrid_search_fts_case,
    run_hybrid_search_fts_case_all_quadrants,
    setup_multi_coll_multi_ns_fts,
    setup_multi_coll_multi_ns_fts_single_loaded,
    teardown_multi_coll_multi_ns_fts,
)


class TestNamespaceHybridSearchFulltextMultiCollMultiNs:
    """Large-scale hybrid_search FTS across 2 collections x 2 namespaces."""

    def _run_fts_case_all_quadrants(self, db_client, case_name: str) -> None:
        ctx = setup_multi_coll_multi_ns_fts(db_client)
        try:
            run_hybrid_search_fts_case_all_quadrants(ctx, case_name)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    def test_cross_quadrant_fts_isolation(self, db_client):
        """Only ``c1_x`` has corpus; other quadrants must return no TOKEN_ZPX hits."""
        ctx = setup_multi_coll_multi_ns_fts_single_loaded(db_client, loaded_quadrant="c1_x")
        try:
            case = get_fts_case("contains_token_zpx")
            corpus, ns_loaded = ctx["c1_x"]
            run_hybrid_search_fts_case(ns_loaded, corpus, case)

            for key in MULTI_COLL_MULTI_NS_QUADRANT_KEYS:
                if key == "c1_x":
                    continue
                _, namespace = ctx[key]
                assert_hybrid_search_no_hits(
                    namespace,
                    case.where_document,
                    n_results=case.n_results,
                )
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    def test_hybrid_search_fulltext_contains_token_zpx(self, db_client):
        self._run_fts_case_all_quadrants(db_client, "contains_token_zpx")

    def test_hybrid_search_fulltext_contains_token_alp(self, db_client):
        self._run_fts_case_all_quadrants(db_client, "contains_token_alp")

    def test_hybrid_search_fulltext_contains_string_shorthand(self, db_client):
        self._run_fts_case_all_quadrants(db_client, "string_shorthand_token_zpx")

    def test_hybrid_search_fulltext_not_contains(self, db_client):
        case = get_fts_case("not_contains_token_zpx")
        ctx = setup_multi_coll_multi_ns_fts(db_client)
        try:
            for key in MULTI_COLL_MULTI_NS_QUADRANT_KEYS:
                corpus, namespace = ctx[key]
                result = run_hybrid_search_fts_case(namespace, corpus, case)
                assert_not_contains_no_token_leak(corpus, result, TOKEN_ZPX)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    def test_hybrid_search_fulltext_and_zpx_alp(self, db_client):
        self._run_fts_case_all_quadrants(db_client, "and_zpx_alp")

    def test_hybrid_search_fulltext_and_multi_contains(self, db_client):
        self._run_fts_case_all_quadrants(db_client, "and_multi_contains_phrase")

    def test_hybrid_search_fulltext_or_zpx_alp(self, db_client):
        self._run_fts_case_all_quadrants(db_client, "or_zpx_alp")

    def test_hybrid_search_fulltext_top1_contains_zpx(self, db_client):
        ctx = setup_multi_coll_multi_ns_fts(db_client)
        try:
            for key in MULTI_COLL_MULTI_NS_QUADRANT_KEYS:
                _, namespace = ctx[key]
                top_result = namespace.hybrid_search(
                    query={"where_document": {"$contains": TOKEN_ZPX}, "n_results": 1},
                    n_results=1,
                    include=["documents"],
                )
                assert top_result["ids"][0][0] == "zpx_top_5", (
                    f"[{key}] most relevant TOKEN_ZPX document must rank first, "
                    f"got {top_result['ids'][0]}"
                )
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
