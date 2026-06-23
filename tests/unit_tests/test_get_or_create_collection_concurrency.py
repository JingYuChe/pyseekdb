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
    _is_sdk_collection_catalog_conflict_error,
)
from pyseekdb.client.types import _NOT_PROVIDED  # noqa: E402


class TestCollectionCatalogConflictDetection:
    def test_detects_integrity_error_on_sdk_collections(self):
        class IntegrityError(Exception):
            pass

        exc = IntegrityError(
            '(1062, "Duplicate entry \'my_coll\' for key \'uk_sdk_coll_name\'")'
        )
        assert _is_sdk_collection_catalog_conflict_error(exc)

    def test_ignores_unrelated_errors(self):
        assert not _is_sdk_collection_catalog_conflict_error(ValueError("invalid dimension"))


class TestCollectionCatalogInsertRecovery:
    def test_insert_conflict_reuses_existing_collection_id(self):
        client = MagicMock(spec=BaseClient)
        client._get_collection_id.side_effect = [ValueError("not found"), "existing_id"]
        conn = MagicMock()
        client._ensure_connection.return_value = conn

        class IntegrityError(Exception):
            pass

        def execute_side_effect(sql):
            if "INSERT INTO" in sql:
                raise IntegrityError(
                    '(1062, "Duplicate entry \'items\' for key \'uk_sdk_coll_name\'")'
                )
            return []

        client._execute.side_effect = execute_side_effect

        result = BaseClient._create_collection_meta_v2(client, "items", None)

        assert result["collection_id"] == "existing_id"
        conn.rollback.assert_called_once()
        assert client._get_collection_id.call_count == 2

    def test_existing_catalog_row_is_reused_without_insert(self):
        client = MagicMock(spec=BaseClient)
        client._get_collection_id.return_value = "existing_id"

        result = BaseClient._create_collection_meta_v2(client, "items", None)

        assert result["collection_id"] == "existing_id"
        insert_calls = [
            call for call in client._execute.call_args_list if "INSERT INTO" in str(call)
        ]
        assert not insert_calls


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
