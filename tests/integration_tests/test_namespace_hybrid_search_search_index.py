"""
Namespace hybrid_search pure search-index (metadata / JSON SEARCH INDEX) tests.

Uses ``query.where`` without ``where_document``. Ground-truth checks mirror
``test_namespace_hybrid_search_fulltext.py`` via ``namespace_hybrid_search_helpers``.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import pytest

from namespace_hybrid_search_helpers import (
    SEARCH_INDEX_CASES,
    get_search_index_case,
    run_hybrid_search_index_case,
    setup_fts_namespace_with_corpus,
    setup_large_fts_collection,
    teardown_large_fts_collection,
)


class TestNamespaceHybridSearchSearchIndex:
    """Large-scale namespace pure metadata-filter hybrid_search tests."""

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
        ns_name = f"ns_si_{case_name}_{int(time.time() * 1000)}"
        return setup_fts_namespace_with_corpus(
            self._collection,
            self._corpus,
            namespace_name=ns_name,
        )

    def _run_case(self, case_name: str) -> None:
        namespace = self._new_namespace(case_name)
        run_hybrid_search_index_case(namespace, self._corpus, get_search_index_case(case_name))

    @pytest.mark.parametrize("case_name", [c.name for c in SEARCH_INDEX_CASES])
    def test_hybrid_search_search_index_operators(self, db_client, case_name: str):
        """Cover ``$eq/$ne/$lt/$lte/$gt/$gte/$in/$nin/$and/$or/$not`` and ``#id``."""
        self._run_case(case_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
