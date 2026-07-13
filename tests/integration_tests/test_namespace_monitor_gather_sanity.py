"""
Regression tests for kernel namespace stats gather (monitor) data sanity.

Validates the fix for invalid sdk_namespaces_stats rows where namespace_id was
UINT64_MAX (18446744073709551615) while no matching sdk_namespaces row existed.

New kernel writes sdk_namespace_stat via GATHER_NAMESPACE_STATS_INNER / the
logic_table_namespace_stats_job. These tests create real namespace data, trigger
gather, and assert no sentinel or orphan stat rows appear.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT, ns_schema
from namespace_stats_test_helpers import (
    count_namespace_uint64_max_rows,
    count_orphan_namespace_stat_rows,
    count_stat_uint64_max_rows,
    list_suspicious_namespace_stat_rows,
    read_namespace_stat,
    table_exists,
    trigger_gather_namespace_stats,
)

import pyseekdb

_INVALID_NAMESPACE_ID = 18446744073709551615


def _raw_client():
    return pyseekdb.Client(
        host=os.environ.get("OB_HOST", "127.0.0.1"),
        port=int(os.environ.get("OB_PORT", "10902")),
        tenant=os.environ.get("OB_TENANT", "mysql"),
        database=os.environ.get("OB_DATABASE", "test"),
        user=os.environ.get("OB_USER", "root"),
        password=os.environ.get("OB_PASSWORD", ""),
    )


def _close(client) -> None:
    if hasattr(client, "close"):
        client.close()


def _emb(i: int) -> list[float]:
    return [(i % 97) / 97.0, ((i * 3) % 89) / 89.0, ((i * 7) % 83) / 83.0]


def _assert_no_invalid_stat_rows(admin, *, context: str) -> None:
    bad_ns = count_namespace_uint64_max_rows(admin)
    bad_stat = count_stat_uint64_max_rows(admin)
    orphan = count_orphan_namespace_stat_rows(admin)
    suspicious = list_suspicious_namespace_stat_rows(admin)
    assert bad_ns == 0, f"{context}: sdk_namespaces has UINT64_MAX rows: {bad_ns}"
    assert bad_stat == 0, (
        f"{context}: sdk_namespace_stat has sentinel namespace_id rows: {bad_stat}"
    )
    assert orphan == 0, (
        f"{context}: orphan sdk_namespace_stat rows (no sdk_namespaces match): {orphan}; "
        f"details={suspicious}"
    )


def _seed_rows(ns, count: int = 20) -> None:
    ns.add(
        ids=[f"gather_{i:04d}" for i in range(count)],
        embeddings=[_emb(i) for i in range(count)],
        documents=[f"doc gather {i}" for i in range(count)],
        metadatas=[{"idx": i} for i in range(count)],
    )


class TestNamespaceMonitorGatherSanity:
    """Monitor gather must not write UINT64_MAX / orphan namespace stat rows."""

    def test_gather_after_insert_has_no_invalid_stat_rows(self, oceanbase_client):
        admin = _raw_client()
        coll_name = f"gather_sanity_{uuid.uuid4().hex[:10]}"
        ns_name = "ns_gather_a"
        coll = oceanbase_client.create_collection(
            name=coll_name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace(ns_name)
            ns_id = int(ns.namespace_id)
            assert ns_id > 0 and ns_id != _INVALID_NAMESPACE_ID

            _seed_rows(ns, count=30)
            time.sleep(1)

            trigger_gather_namespace_stats(admin)
            time.sleep(2)

            _assert_no_invalid_stat_rows(
                admin, context=f"after gather coll={coll_name} ns_id={ns_id}"
            )

            # If gather produced a stat row, it must belong to this namespace.
            stat_rows = admin._server._execute(
                "SELECT namespace_id, row_count FROM sdk_namespace_stat "
                f"WHERE collection_id = '{coll.id}' AND namespace_id = {ns_id}"
            )
            if stat_rows:
                rc = stat_rows[0]["row_count"] if isinstance(stat_rows[0], dict) else stat_rows[0][1]
                assert int(rc) >= 0
                assert read_namespace_stat(admin, coll.id, ns_id, "row_count") == int(rc)
        finally:
            oceanbase_client.delete_collection(name=coll_name)
            _close(admin)

    def test_gather_multiple_namespaces_no_orphans(self, oceanbase_client):
        admin = _raw_client()
        coll_name = f"gather_multi_{uuid.uuid4().hex[:10]}"
        coll = oceanbase_client.create_collection(
            name=coll_name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        ns_names = ["ns_a", "ns_b", "ns_c"]
        try:
            for i, name in enumerate(ns_names):
                ns = coll.get_or_create_namespace(name)
                _seed_rows(ns, count=10 + i * 5)

            trigger_gather_namespace_stats(admin)
            time.sleep(2)
            _assert_no_invalid_stat_rows(admin, context="multi-namespace gather")

            active_ids = {
                int(r["namespace_id"] if isinstance(r, dict) else r[0])
                for r in admin._server._execute(
                    f"SELECT namespace_id FROM sdk_namespaces WHERE collection_id = '{coll.id}'"
                )
            }
            stat_ids = {
                int(r["namespace_id"] if isinstance(r, dict) else r[0])
                for r in admin._server._execute(
                    f"SELECT namespace_id FROM sdk_namespace_stat WHERE collection_id = '{coll.id}'"
                )
            }
            assert stat_ids <= active_ids, (
                f"stat namespace_ids {stat_ids} must be subset of active {active_ids}"
            )
        finally:
            oceanbase_client.delete_collection(name=coll_name)
            _close(admin)

    def test_gather_after_namespace_deleted_removes_orphan_stat(self, oceanbase_client):
        admin = _raw_client()
        coll_name = f"gather_drop_{uuid.uuid4().hex[:10]}"
        coll = oceanbase_client.create_collection(
            name=coll_name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        keep_name, drop_name = "ns_keep", "ns_drop"
        try:
            keep_ns = coll.get_or_create_namespace(keep_name)
            drop_ns = coll.get_or_create_namespace(drop_name)
            keep_id = int(keep_ns.namespace_id)
            drop_id = int(drop_ns.namespace_id)

            _seed_rows(keep_ns, count=15)
            _seed_rows(drop_ns, count=15)

            trigger_gather_namespace_stats(admin)
            time.sleep(2)
            _assert_no_invalid_stat_rows(admin, context="before namespace delete")

            coll.delete_namespace(drop_name)
            assert count_orphan_namespace_stat_rows(admin) == 0

            trigger_gather_namespace_stats(admin)
            time.sleep(2)
            _assert_no_invalid_stat_rows(admin, context="after namespace delete + gather")

            rows = admin._server._execute(
                f"SELECT namespace_id FROM sdk_namespace_stat WHERE collection_id = '{coll.id}'"
            )
            stat_ids = {int(r["namespace_id"] if isinstance(r, dict) else r[0]) for r in rows}
            assert drop_id not in stat_ids, f"dropped namespace {drop_id} still in stat: {stat_ids}"
            if stat_ids:
                assert keep_id in stat_ids or len(stat_ids) == 0
        finally:
            oceanbase_client.delete_collection(name=coll_name)
            _close(admin)

    def test_legacy_sdk_namespaces_stats_table_absent_or_clean(self, oceanbase_client):
        """Old audit table sdk_namespaces_stats must not contain UINT64_MAX rows if present."""
        del oceanbase_client
        admin = _raw_client()
        try:
            if not table_exists(admin, "sdk_namespaces_stats"):
                pytest.skip("legacy sdk_namespaces_stats table not deployed")
            bad = admin._server._execute(
                f"SELECT COUNT(*) AS cnt FROM sdk_namespaces_stats "
                f"WHERE namespace_id = {_INVALID_NAMESPACE_ID} "
                f"OR ltable_id = {_INVALID_NAMESPACE_ID}"
            )
            row = bad[0]
            cnt = int(row["cnt"] if isinstance(row, dict) else row[0])
            assert cnt == 0, f"legacy sdk_namespaces_stats has {cnt} UINT64_MAX rows"
        finally:
            _close(admin)
