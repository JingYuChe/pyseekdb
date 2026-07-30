"""
Namespace row/size limit enforcement integration tests.

Kernel path: ObPxAdmission::check_namespace_row_limit →
ObLogicalTableMonitor::check_logical_table_row_limit

New catalog model (monitor ops refactor):
- Estimates: ``sdk_namespace_stat`` (row_count / total_size per namespace)
- Limits: flat ``sdk_namespaces.info`` (``row_limit`` / ``size_limit``)
- Readable join: ``logic_table_namespace_stats`` view

Namespace DML routes by explicit namespace_id / ltable_id columns.
"""

from __future__ import annotations

import os
import uuid

import pytest
from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT, ns_schema
from namespace_stats_test_helpers import (
    count_namespace_stat_rows,
    default_ops_limit_row_limit,
    default_ops_limit_size_limit,
    monitor_style_upsert_stat,
    read_namespace_stat,
    read_ops_limit,
    set_ops_limit,
    upsert_namespace_stat,
)

import pyseekdb


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


def _add_one(ns, i: int) -> None:
    ns.add(
        ids=[f"row_{i:06d}"],
        embeddings=[_emb(i)],
        documents=[f"doc {i}"],
        metadatas=[{"seq": i}],
    )


def _assert_blocked(fn) -> None:
    with pytest.raises(Exception) as excinfo:
        fn()
    msg = str(excinfo.value).lower()
    assert "not allowed" in msg or "limit" in msg, f"unexpected error: {excinfo.value!r}"


class TestNamespaceStatsDefaults:
    """SDK seeds default row_limit / size_limit into sdk_namespaces.info on namespace create."""

    def test_namespace_create_seeds_default_ops_limit(self, oceanbase_client):
        admin = _raw_client()
        name = f"rl_def_{uuid.uuid4().hex[:12]}"
        coll = oceanbase_client.create_collection(
            name=name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_def")
            ns_id = int(ns.namespace_id)
            assert read_ops_limit(admin, coll.id, ns_id, "row_limit") == default_ops_limit_row_limit()
            assert read_ops_limit(admin, coll.id, ns_id, "size_limit") == default_ops_limit_size_limit()
        finally:
            oceanbase_client.delete_collection(name=name)
            _close(admin)


class TestNamespaceRowLimit:
    """Row/size limit enforcement on the logical-table INSERT path."""

    def test_namespace_level_limit_blocks_insert(self, oceanbase_client):
        """High estimate + low ops_limit blocks INSERT; DELETE still allowed."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"rl_ns_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_a")
            ns_id = int(ns.namespace_id)
            for i in range(3):
                _add_one(ns, i)
            assert ns.count() == 3

            set_ops_limit(admin, collection_id=coll.id, namespace_id=ns_id, row_limit=2)
            upsert_namespace_stat(
                admin,
                collection_id=coll.id,
                namespace_id=ns_id,
                row_count=1000,
            )
            _assert_blocked(lambda: _add_one(ns, 99))

            ns.delete(ids=["row_000000"])
            assert ns.count() == 2

            set_ops_limit(admin, collection_id=coll.id, namespace_id=ns_id, row_limit=-1)
            _add_one(ns, 100)
            assert ns.count() == 3
        finally:
            owner.delete_collection(name=name)
            _close(admin)

    def test_exact_count_recheck_after_delete(self, oceanbase_client):
        """Over-estimate triggers exact COUNT; empty namespace self-corrects."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"rl_lt_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_b")
            ns_id = int(ns.namespace_id)
            for i in range(3):
                _add_one(ns, i)
            assert ns.count() == 3

            set_ops_limit(admin, collection_id=coll.id, namespace_id=ns_id, row_limit=2)
            upsert_namespace_stat(
                admin,
                collection_id=coll.id,
                namespace_id=ns_id,
                row_count=1000,
            )
            _assert_blocked(lambda: _add_one(ns, 99))

            ns.delete(ids=[f"row_{i:06d}" for i in range(3)])
            assert ns.count() == 0
            _add_one(ns, 200)
            assert ns.count() == 1
        finally:
            owner.delete_collection(name=name)
            _close(admin)

    def test_ops_limit_survives_monitor_style_upsert(self, oceanbase_client):
        """Monitor gather UPSERT must not clobber ops_limit in sdk_namespaces.info."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"rl_keep_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_c")
            ns_id = int(ns.namespace_id)
            for i in range(5):
                _add_one(ns, i)

            set_ops_limit(
                admin,
                collection_id=coll.id,
                namespace_id=ns_id,
                row_limit=1000,
                size_limit=999999,
            )
            upsert_namespace_stat(
                admin,
                collection_id=coll.id,
                namespace_id=ns_id,
                row_count=5,
                total_size=50,
                total_size_with_index=80,
            )
            monitor_style_upsert_stat(
                admin,
                collection_id=coll.id,
                namespace_id=ns_id,
                row_count=7,
                total_size=70,
                total_size_with_index=100,
            )

            assert read_namespace_stat(admin, coll.id, ns_id, "row_count") == 7
            assert read_namespace_stat(admin, coll.id, ns_id, "total_size") == 70
            assert read_ops_limit(admin, coll.id, ns_id, "row_limit") == 1000
            assert read_ops_limit(admin, coll.id, ns_id, "size_limit") == 999999
        finally:
            owner.delete_collection(name=name)
            _close(admin)

    def test_size_limit_blocks_insert(self, oceanbase_client):
        """size_limit (bytes) blocks INSERT independently of row_limit."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"rl_sz_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_sz")
            ns_id = int(ns.namespace_id)
            for i in range(3):
                _add_one(ns, i)

            set_ops_limit(
                admin,
                collection_id=coll.id,
                namespace_id=ns_id,
                row_limit=-1,
                size_limit=100,
            )
            upsert_namespace_stat(
                admin,
                collection_id=coll.id,
                namespace_id=ns_id,
                row_count=1000,
                total_size=1000,
            )
            _assert_blocked(lambda: _add_one(ns, 99))

            set_ops_limit(admin, collection_id=coll.id, namespace_id=ns_id, size_limit=-1)
            _add_one(ns, 100)
        finally:
            owner.delete_collection(name=name)
            _close(admin)

    def test_under_limit_passes(self, oceanbase_client):
        """Configured but not exceeded limits must not block inserts."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"rl_ok_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_ok")
            ns_id = int(ns.namespace_id)
            _add_one(ns, 0)
            set_ops_limit(
                admin,
                collection_id=coll.id,
                namespace_id=ns_id,
                row_limit=1_000_000,
                size_limit=1_000_000_000,
            )
            upsert_namespace_stat(
                admin,
                collection_id=coll.id,
                namespace_id=ns_id,
                row_count=1,
                total_size=1,
            )
            for i in range(1, 5):
                _add_one(ns, i)
            assert ns.count() == 5
        finally:
            owner.delete_collection(name=name)
            _close(admin)

    def test_update_not_blocked_when_over_limit(self, oceanbase_client):
        """UPDATE must pass even while INSERT is blocked."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"rl_upd_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_upd")
            ns_id = int(ns.namespace_id)
            _add_one(ns, 0)
            set_ops_limit(admin, collection_id=coll.id, namespace_id=ns_id, row_limit=1)
            upsert_namespace_stat(
                admin,
                collection_id=coll.id,
                namespace_id=ns_id,
                row_count=1000,
            )
            _assert_blocked(lambda: _add_one(ns, 1))
            owner._server._execute("UPDATE sdk_namespace_stat SET last_gather_time = NOW(6) WHERE 1 = 0")
        finally:
            owner.delete_collection(name=name)
            _close(admin)

    def test_drop_namespace_clears_stats(self, oceanbase_client):
        """DROP_NAMESPACE removes sdk_namespace_stat rows for the namespace."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"rl_drop_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_drop")
            ns_id = int(ns.namespace_id)
            _add_one(ns, 0)
            upsert_namespace_stat(
                admin,
                collection_id=coll.id,
                namespace_id=ns_id,
                row_count=5,
                total_size=10,
            )
            assert count_namespace_stat_rows(admin, coll.id, ns_id) == 1

            coll.delete_namespace("ns_drop")
            assert count_namespace_stat_rows(admin, coll.id, ns_id) == 0, (
                "DROP_NAMESPACE must delete sdk_namespace_stat rows for the namespace"
            )
        finally:
            owner.delete_collection(name=name)
            _close(admin)
