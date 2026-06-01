"""
Namespace hybrid_search pure vector (KNN) integration tests.

Validates KNN distance ordering and metadata filters on the ``knn`` branch.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import pytest

from namespace_hybrid_search_helpers import (
    KNN_QUERY_VECTOR,
    VECTOR_KNN_CASES,
    get_vector_knn_case,
    run_hybrid_knn_case,
    setup_fts_namespace_with_corpus,
    setup_large_fts_collection,
    skip_oceanbase_knn,
    teardown_large_fts_collection,
)


class TestNamespaceHybridSearchVector:
    """Large-scale namespace pure KNN hybrid_search tests."""

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
        ns_name = f"ns_knn_{case_name}_{int(time.time() * 1000)}"
        return setup_fts_namespace_with_corpus(
            self._collection,
            self._corpus,
            namespace_name=ns_name,
        )

    def _run_knn_case(self, request: pytest.FixtureRequest, case_name: str) -> None:
        skip_oceanbase_knn(request)
        namespace = self._new_namespace(case_name)
        run_hybrid_knn_case(namespace, self._corpus, get_vector_knn_case(case_name))

    @pytest.mark.parametrize("case_name", [c.name for c in VECTOR_KNN_CASES])
    def test_hybrid_search_vector_knn_cases(self, db_client, request, case_name: str):
        self._run_knn_case(request, case_name)

    def test_hybrid_search_vector_top1_nearest(self, db_client, request):
        """Top-1 must be the global nearest neighbor for a fixed query vector."""
        skip_oceanbase_knn(request)
        namespace = self._new_namespace("top1_nearest")
        from namespace_hybrid_search_helpers import expected_knn_ids

        result = namespace.hybrid_search(
            knn={"query_embeddings": KNN_QUERY_VECTOR, "n_results": 1},
            n_results=1,
            include=["distances"],
        )
        assert result["ids"][0][0] == expected_knn_ids(self._corpus, KNN_QUERY_VECTOR, 1)[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
