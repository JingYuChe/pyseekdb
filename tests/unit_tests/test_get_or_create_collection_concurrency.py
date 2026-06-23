"""
Unit tests for concurrent-safe get_or_create_collection helpers.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from pyseekdb.client.client_base import (  # noqa: E402
    BaseClient,
    _is_collection_conflict_error,
)
from pyseekdb.client.types import _NOT_PROVIDED  # noqa: E402


class TestCollectionConflictDetection:
    def test_detects_value_error_for_existing_collection(self):
        assert _is_collection_conflict_error(ValueError("Collection 'items' already exists"))

    def test_detects_seekdb_table_exists_error(self):
        class SeekdbError(Exception):
            pass

        exc = SeekdbError("Table 'c$v2$abc' already exists failed: code=1050")
        assert _is_collection_conflict_error(exc)

    def test_ignores_unrelated_errors(self):
        assert not _is_collection_conflict_error(ValueError("invalid dimension"))

    def test_ignores_metadata_failure_without_conflict_cause(self):
        assert not _is_collection_conflict_error(
            ValueError("Failed to create collection metadata: Collection not found: 'items'")
        )

    def test_detects_conflict_in_cause_chain(self):
        inner = Exception("Table 'c$v2$abc' already exists failed: code=1050")
        outer = ValueError("Failed to create collection metadata: duplicate entry")
        outer.__cause__ = inner
        assert _is_collection_conflict_error(outer)


class TestGetOrCreateCollectionRecovery:
    def test_returns_existing_collection_after_create_conflict(self):
        client = MagicMock(spec=BaseClient)
        client.has_collection.side_effect = [False, True]
        existing = object()
        client.get_collection.return_value = existing
        client.create_collection.side_effect = ValueError("Collection 'items' already exists")

        result = BaseClient.get_or_create_collection(client, "items")

        assert result is existing
        client.get_collection.assert_called_once_with("items", embedding_function=_NOT_PROVIDED)

    def test_retries_get_after_wrapped_table_conflict(self):
        client = MagicMock(spec=BaseClient)
        client.has_collection.side_effect = [False, True]
        existing = object()
        client.get_collection.return_value = existing
        inner = Exception("Table 'c$v2$abc' already exists failed: code=1050")
        outer = ValueError("Failed to create collection metadata: duplicate entry")
        outer.__cause__ = inner
        client.create_collection.side_effect = outer

        result = BaseClient.get_or_create_collection(client, "items")

        assert result is existing
        client.get_collection.assert_called_once_with("items", embedding_function=_NOT_PROVIDED)

    def test_resumes_incomplete_namespace_collection_when_present(self):
        client = MagicMock(spec=BaseClient)
        resumed = object()
        client.has_collection.return_value = True
        client._is_incomplete_ns_collection.return_value = True
        client.create_collection.return_value = resumed

        result = BaseClient.get_or_create_collection(client, "items", use_namespace=True)

        assert result is resumed
        client.create_collection.assert_called_once()
        client.get_collection.assert_not_called()

    def test_does_not_mask_unrelated_create_errors(self):
        client = MagicMock(spec=BaseClient)
        client.has_collection.return_value = False
        client.create_collection.side_effect = ValueError("invalid dimension")

        with pytest.raises(ValueError, match="invalid dimension"):
            BaseClient.get_or_create_collection(client, "items")


class TestListNsNamespacesRecyclebinFilter:
    def test_sql_excludes_recyclebin_rows(self):
        client = MagicMock(spec=BaseClient)
        client._qtable.return_value = "`sdk_namespaces`"
        client._execute.return_value = [
            ("1", "active_ns"),
        ]

        result = BaseClient._list_ns_namespaces(client, "coll_1")

        assert result == [{"namespace_id": "1", "namespace_name": "active_ns"}]
        sql = client._execute.call_args[0][0]
        assert "__recyclebin_" in sql
        assert "LEFT(namespace_name, 13) <> '__recyclebin_'" in sql


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
