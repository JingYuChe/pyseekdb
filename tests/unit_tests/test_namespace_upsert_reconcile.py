"""
Unit tests for namespace upsert duplicate-record deduplication.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from pyseekdb.client.client_base import BaseClient  # noqa: E402


class TestNamespaceUpsertDedupe:
    """TestNamespaceUpsertDedupe class."""

    def test_dedupe_skips_when_single_row_exists(self):
        """Test dedupe skips when single row exists."""
        client = MagicMock(spec=BaseClient)
        client._count_namespace_records_by_id.return_value = 1

        BaseClient._dedupe_namespace_records_by_id(
            client,
            "logic_data_table",
            7,
            9,
            "same_new_id",
        )

        client._delete_namespace_newest_duplicate_rows.assert_not_called()

    def test_dedupe_deletes_newest_duplicate_rows(self):
        """Test dedupe deletes only the extra duplicate rows."""
        client = MagicMock(spec=BaseClient)
        client._count_namespace_records_by_id.side_effect = [4, 1]
        client._delete_namespace_newest_duplicate_rows.return_value = 3

        BaseClient._dedupe_namespace_records_by_id(
            client,
            "logic_data_table",
            7,
            9,
            "same_new_id",
        )

        client._delete_namespace_newest_duplicate_rows.assert_called_once_with(
            "logic_data_table",
            7,
            9,
            "same_new_id",
            1,
            conn=client._ensure_connection.return_value,
            use_context_manager=client._use_context_manager_for_cursor.return_value,
        )

    def test_dedupe_raises_when_retries_exhausted(self):
        """Test dedupe raises when retries exhausted."""
        client = MagicMock(spec=BaseClient)
        client._count_namespace_records_by_id.return_value = 4
        client._delete_namespace_newest_duplicate_rows.return_value = 0

        with (
            patch("pyseekdb.client.client_base.time.sleep"),
            pytest.raises(ValueError, match="Failed to reconcile duplicate namespace rows"),
        ):
            BaseClient._dedupe_namespace_records_by_id(
                client,
                "logic_data_table",
                7,
                9,
                "same_new_id",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
