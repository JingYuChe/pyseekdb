"""
Integration test: rebuilding / post-creation index build on a namespace
(logic) table is not supported.

Product behavior (Agent Database / lakebase namespace mode):
- A namespace collection is backed by a logic table (``IS_LOGIC_TABLE = TRUE``,
  ``PARTITION BY KEY(namespace_id)``). Its vector index is created together with
  the table.
- Building / rebuilding the vector index *after* table creation via
  ``DBMS_VECTOR.rebuild_index`` is rejected up front.

Before the kernel fix this scenario surfaced the generic
``OB_ERR_ADD_INDEX (5703, "Add index failed")`` from the vector index DDL build
task. After the fix it is rejected early in ``resolve_rebuild_arg`` with
``OB_NOT_SUPPORTED`` — internal code -4007, wire errno 1235
(``ER_NOT_SUPPORTED_YET``), SQLSTATE ``0A000``.
"""

from __future__ import annotations

import time
from typing import Any

import pymysql.err
import pytest
from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT

from pyseekdb import IVFConfiguration
from pyseekdb.client.configuration import VectorIndexConfig
from pyseekdb.client.meta_info import NamespaceCollectionNames
from pyseekdb.client.schema import Schema

# OceanBase OB_NOT_SUPPORTED: internal code -4007, surfaced to the MySQL
# protocol as ER_NOT_SUPPORTED_YET (1235) with SQLSTATE 0A000.
OB_NOT_SUPPORTED_ERRNO = 1235
OB_NOT_SUPPORTED_SQLSTATE = "0A000"
# The pre-fix generic failure we must NOT see anymore.
OB_ERR_ADD_INDEX_ERRNO = 5703


def _vector_schema(dimension: int = 3) -> Schema:
    """IVF vector index schema (no embedding function: raw vectors)."""
    return Schema(
        vector_index=VectorIndexConfig(
            ivf=IVFConfiguration(dimension=dimension, distance="l2", centroids_fresh_mode="spfresh"),
            embedding_function=None,
        ),
    )


def _create_collection(client: Any) -> Any:
    """Create a namespace (logic table) collection with an IVF vector index."""
    name = f"test_ns_rebuild_unsupp_{int(time.time() * 1000)}"
    return client.create_collection(
        name=name,
        schema=_vector_schema(),
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )


class TestNamespaceRebuildIndexUnsupported:
    """rebuild_index on a logic table must report OB_NOT_SUPPORTED (4007), not 5703."""

    def test_rebuild_index_on_logic_table_not_supported(self, oceanbase_client):
        """post-creation rebuild_index on a namespace logic table is rejected with 4007."""
        collection = _create_collection(oceanbase_client)
        try:
            data_table = NamespaceCollectionNames.data_table_name(collection.id)
            call_sql = (
                "CALL DBMS_VECTOR.rebuild_index("
                f"'idx_vec', '{data_table}', 'embedding', 0, '', '')"
            )

            with pytest.raises(pymysql.err.Error) as exc_info:
                oceanbase_client._server._execute(call_sql)

            err = exc_info.value
            errno = err.args[0] if err.args else None
            message = str(err)

            # Must be the early NOT_SUPPORTED rejection (internal 4007 / wire 1235),
            # and explicitly NOT the old generic "Add index failed" (5703).
            assert errno == OB_NOT_SUPPORTED_ERRNO, (
                f"expected OB_NOT_SUPPORTED (wire {OB_NOT_SUPPORTED_ERRNO} / internal 4007), "
                f"got errno={errno}, message={message!r}"
            )
            assert errno != OB_ERR_ADD_INDEX_ERRNO, "regression: still failing with OB_ERR_ADD_INDEX (5703)"
            assert "logic table" in message.lower(), f"unexpected error message: {message!r}"
        finally:
            oceanbase_client.delete_collection(name=collection.name)
