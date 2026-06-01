"""
Namespace hybrid_search combined-branch tests.

Covers:
  - full-text + search index (metadata on ``query``)
  - vector + search index (``knn.where``)
  - vector + full-text (+ RRF smoke)
  - vector + full-text + search index (+ RRF smoke)
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import pytest

from namespace_hybrid_search_helpers import (
    HYBRID_COMBINED_CASES,
    get_hybrid_combined_case,
    run_hybrid_combined_case,
    setup_fts_namespace_with_corpus,
    setup_large_fts_collection,
    skip_oceanbase_knn,
    teardown_large_fts_collection,
)


class TestNamespaceHybridSearchCombined:
    _shared_by_mode: ClassVar[dict[str, dict[str, Any]]] = {}

    @pytest.fixture(autouse=True)
    def _bind_shared_collection(self, db_client: Any, request: pytest.FixtureRequest) -> None:
        mode = request.node.callspec.params["db_client"] if request.node.callspec else "default"
        if mode not in self._shared_by_mode:
            corpus, collection = setup_large_fts_collection(db_client)
            self._shared_by_mode[mode] = {
                "corpus": corpus,
                "collection": collection,
                "db_client": db_client,
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
            namespace_name=f"ns_comb_{case_name}_{int(time.time() * 1000)}",
        )

    @pytest.mark.parametrize("case_name", [c.name for c in HYBRID_COMBINED_CASES])
    def test_hybrid_search_combined(self, db_client, request, case_name: str):
        case = get_hybrid_combined_case(case_name)
        if case.knn is not None:
            skip_oceanbase_knn(request)
        namespace = self._new_namespace(case_name)
        run_hybrid_combined_case(namespace, self._corpus, case)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
