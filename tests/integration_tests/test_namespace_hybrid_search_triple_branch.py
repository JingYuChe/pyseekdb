"""
Namespace hybrid_search triple-branch integration tests.

Each case activates all three signals in one ``hybrid_search`` call:
  - full-text via ``query.where_document`` ($contains / $not_contains / $and / $or)
  - search index via ``query.where`` ($eq / $ne / $lt / $lte / $gt / $gte / $in / $nin / $and / $or / $not / #id)
  - vector via ``knn`` (KNN distance ordering + optional ``knn.where``)

Operator coverage is orthogonal: one branch is verified against corpus ground truth while
the other two stay active with broad or aligned filters.

Run one case::

    pytest tests/integration_tests/test_namespace_hybrid_search_triple_branch.py \\
        -k "fts_contains_zpx and oceanbase" -v -s
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import pytest

from namespace_hybrid_search_helpers import (
    TRIPLE_BRANCH_CASES,
    TOKEN_ZPX,
    get_triple_branch_case,
    run_hybrid_triple_branch_case,
    setup_fts_namespace_with_corpus,
    setup_large_fts_collection,
    teardown_large_fts_collection,
)


class TestNamespaceHybridSearchTripleBranch:
    """Large-scale namespace vector + full-text + search-index hybrid_search tests."""

    _shared_by_mode: ClassVar[dict[str, dict[str, Any]]] = {}

    @pytest.fixture(autouse=True)
    def _bind_shared_collection(self, db_client: Any, request: pytest.FixtureRequest) -> None:
        mode = request.node.callspec.params["db_client"] if request.node.callspec else "default"
        if mode not in self._shared_by_mode:
            corpus, collection = setup_large_fts_collection(db_client)
            self._shared_by_mode[mode] = {
                "db_client": db_client,
                "corpus": corpus,
                "collection": collection,
            }
        entry = self._shared_by_mode[mode]
        self._corpus = entry["corpus"]
        self._collection = entry["collection"]

    @classmethod
    def teardown_class(cls) -> None:
        for entry in cls._shared_by_mode.values():
            teardown_large_fts_collection(entry["db_client"], entry["collection"])
        cls._shared_by_mode.clear()

    def _new_namespace(self, case_name: str) -> Any:
        return setup_fts_namespace_with_corpus(
            self._collection,
            self._corpus,
            namespace_name=f"ns_tb_{case_name}_{int(time.time() * 1000)}",
        )

    @pytest.mark.parametrize("case_name", [c.name for c in TRIPLE_BRANCH_CASES])
    def test_hybrid_search_triple_branch(self, db_client, case_name: str):
        case = get_triple_branch_case(case_name)
        namespace = self._new_namespace(case_name)
        run_hybrid_triple_branch_case(namespace, self._corpus, case)

    def test_hybrid_search_triple_branch_top1_fts_with_knn_active(self, db_client):
        """FTS top-1 ranking remains correct when KNN and search-index branches are active."""
        namespace = self._new_namespace("top1_fts_knn_si")
        top_result = namespace.hybrid_search(
            query={
                "where_document": {"$contains": TOKEN_ZPX},
                "where": {"rel_hint": {"$gte": 0}},
                "n_results": 1,
            },
            knn={
                "query_embeddings": [1.0, 1.0, 0.0],
                "where": {"rel_hint": {"$gte": 0}},
                "n_results": 20,
            },
            n_results=1,
            include=["documents"],
        )
        assert top_result["ids"][0][0] == "zpx_top_5", (
            f"most relevant TOKEN_ZPX document must rank first, got {top_result['ids'][0]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
