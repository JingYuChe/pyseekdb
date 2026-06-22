"""
Unit tests for namespace upsert duplicate-record reconciliation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from pyseekdb.client.client_base import BaseClient  # noqa: E402


class TestNamespaceUpsertReconcile:
    def test_lock_name_is_stable_and_bounded(self):
        name = BaseClient._namespace_record_lock_name("c" * 32, 7, 9, "same_new_id")
        assert name.startswith("pyseekdb:nsu:")
        assert len(name) <= 64

    def test_reconcile_skips_when_single_row_exists(self):
        client = MagicMock(spec=BaseClient)
        client._namespace_record_lock.return_value.__enter__.return_value = True
        client._count_namespace_records_by_id.return_value = 1

        BaseClient._reconcile_namespace_duplicate_records(
            client,
            collection_id="c" * 32,
            collection_name="items",
            namespace_id="7",
            namespace_name="race_ns",
            ltable_id=9,
            table_name="logic_data_table",
            ids=["same_new_id"],
            documents=["doc"],
            metadatas=[{"client": 0}],
            embeddings=[[1.0, 2.0, 3.0]],
            embedding_function=None,
        )

        client._delete_namespace_records_by_id.assert_not_called()
        client._namespace_add.assert_not_called()

    def test_reconcile_collapses_duplicate_rows(self):
        client = MagicMock(spec=BaseClient)
        client._namespace_record_lock.return_value.__enter__.return_value = True
        client._count_namespace_records_by_id.return_value = 4

        BaseClient._reconcile_namespace_duplicate_records(
            client,
            collection_id="c" * 32,
            collection_name="items",
            namespace_id="7",
            namespace_name="race_ns",
            ltable_id=9,
            table_name="logic_data_table",
            ids=["same_new_id"],
            documents=["winner"],
            metadatas=[{"client": 2}],
            embeddings=[[1.0, 2.0, 3.0]],
            embedding_function=None,
        )

        client._delete_namespace_records_by_id.assert_called_once_with(
            "logic_data_table", 7, 9, "same_new_id"
        )
        client._namespace_add.assert_called_once()
        add_kwargs = client._namespace_add.call_args.kwargs
        assert add_kwargs["ids"] == ["same_new_id"]
        assert add_kwargs["documents"] == ["winner"]
        assert add_kwargs["metadatas"] == [{"client": 2}]
        assert add_kwargs["embeddings"] == [[1.0, 2.0, 3.0]]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
