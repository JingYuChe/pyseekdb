"""
Integration tests for logic_table_namespace_stats view.

The view joins sdk_collections, sdk_namespaces and sdk_namespace_stat so ops
can query human-readable collection/namespace names with monitor estimates.
"""

from __future__ import annotations

import uuid

from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT, ns_schema
from namespace_stats_test_helpers import (
    monitor_style_upsert_stat,
    query_stats_view,
    read_namespace_stat,
    upsert_namespace_stat,
)

import pyseekdb


def _raw_client():
    import os

    return pyseekdb.Client(
        host=os.environ.get("OB_HOST", "127.0.0.1"),
        port=int(os.environ.get("OB_PORT", "10902")),
        tenant=os.environ.get("OB_TENANT", "mysql"),
        database=os.environ.get("OB_DATABASE", "test"),
        user=os.environ.get("OB_USER", "root"),
        password=os.environ.get("OB_PASSWORD", ""),
    )


class TestLogicTableNamespaceStatsView:
    """Query logic_table_namespace_stats after seeding sdk_namespace_stat."""

    def test_view_returns_collection_and_namespace_names(self, oceanbase_client):
        admin = _raw_client()
        coll_name = f"vw_coll_{uuid.uuid4().hex[:10]}"
        ns_name = "vw_ns_a"
        coll = oceanbase_client.create_collection(
            name=coll_name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace(ns_name)
            ns_id = int(ns.namespace_id)
            upsert_namespace_stat(
                admin,
                collection_id=coll.id,
                namespace_id=ns_id,
                row_count=42,
                total_size=420,
                total_size_with_index=840,
            )

            rows = query_stats_view(admin, collection_name=coll_name, namespace_name=ns_name)
            assert len(rows) == 1
            row = rows[0]
            cname = row["collection_name"] if isinstance(row, dict) else row[0]
            nname = row["namespace_name"] if isinstance(row, dict) else row[1]
            rc = row["row_count"] if isinstance(row, dict) else row[2]
            ts = row["total_size"] if isinstance(row, dict) else row[3]
            tsi = row["total_size_with_index"] if isinstance(row, dict) else row[4]
            assert cname == coll_name
            assert nname == ns_name
            assert int(rc) == 42
            assert int(ts) == 420
            assert int(tsi) == 840
        finally:
            oceanbase_client.delete_collection(name=coll_name)
            if hasattr(admin, "close"):
                admin.close()

    def test_view_reflects_monitor_style_upsert(self, oceanbase_client):
        admin = _raw_client()
        coll_name = f"vw_mon_{uuid.uuid4().hex[:10]}"
        ns_name = "vw_ns_b"
        coll = oceanbase_client.create_collection(
            name=coll_name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace(ns_name)
            ns_id = int(ns.namespace_id)
            monitor_style_upsert_stat(
                admin,
                collection_id=coll.id,
                namespace_id=ns_id,
                row_count=11,
                total_size=110,
                total_size_with_index=220,
            )

            assert read_namespace_stat(admin, coll.id, ns_id, "row_count") == 11
            rows = query_stats_view(admin, collection_name=coll_name, namespace_name=ns_name)
            assert len(rows) == 1
            row = rows[0]
            rc = row["row_count"] if isinstance(row, dict) else row[2]
            assert int(rc) == 11
        finally:
            oceanbase_client.delete_collection(name=coll_name)
            if hasattr(admin, "close"):
                admin.close()

    def test_view_empty_when_no_stat_row(self, oceanbase_client):
        admin = _raw_client()
        coll_name = f"vw_empty_{uuid.uuid4().hex[:10]}"
        ns_name = "vw_ns_empty"
        coll = oceanbase_client.create_collection(
            name=coll_name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            coll.get_or_create_namespace(ns_name)
            rows = query_stats_view(admin, collection_name=coll_name, namespace_name=ns_name)
            assert not rows
        finally:
            oceanbase_client.delete_collection(name=coll_name)
            if hasattr(admin, "close"):
                admin.close()
