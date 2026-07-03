"""
Namespace row/size limit enforcement integration tests.

Covers the INSERT pre-check wired into ObPxAdmission::enter_query_admission →
ObLogicalTableMonitor::check_logical_table_row_limit:

- namespace-level (ltable_id=0 aggregate) limit blocks INSERT (estimate-based).
- ltable-level limit blocks INSERT with exact-count fallback; over-estimate is
  self-corrected by the exact COUNT so a since-emptied namespace inserts again.
- DELETE / UPDATE are never blocked (only INSERT-like statements are gated),
  so an over-limit namespace can always recover.
- row_limit / size_limit set by ops survive a monitor-style UPSERT (INSERT ...
  ON DUPLICATE KEY UPDATE no longer clobbers them back to -1).

The limit check is scoped by the *session* namespace context (@collection_id /
@namespace_id / @ltable_id), exactly like the RU limit — it fires for any
INSERT-like statement issued on a session that carries a namespace context,
regardless of the physical target table. Therefore all sdk_namespaces_stats
reads/writes here go through a dedicated ADMIN connection that never touched a
namespace (no context → not gated), which is also how ops would edit the table.

These require the kernel build that reads per-namespace row_limit/size_limit
columns and enforces them on the write path. On an older binary every add()
just succeeds and the block-expecting asserts fail — that is the intended
signal that the feature is not present.
"""

from __future__ import annotations

import os
import uuid

import pytest

from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT, ns_schema

import pyseekdb
from pyseekdb.client.meta_info import NamespaceStatsDefaults


def _raw_client():
    """Fresh client with NO namespace context — used only for sdk_namespaces_stats admin SQL."""
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


def _upsert_stat(admin, *, collection_id: str, namespace_id: int, ltable_id: int,
                 estimated_rows: int, row_limit: int, size_limit: int = -1,
                 average_row_size: int = 0, included_index: int = 0) -> None:
    """Seed / override a sdk_namespaces_stats row (via context-free admin conn)."""
    admin._server._execute(
        "INSERT INTO sdk_namespaces_stats "
        "(collection_id, namespace_id, ltable_id, estimated_rows, average_row_size, "
        " row_limit, size_limit, last_estimate_time, included_index) VALUES "
        f"('{collection_id}', {int(namespace_id)}, {int(ltable_id)}, {int(estimated_rows)}, "
        f" {int(average_row_size)}, {int(row_limit)}, {int(size_limit)}, NOW(6), {int(included_index)}) "
        "ON DUPLICATE KEY UPDATE estimated_rows=VALUES(estimated_rows), "
        "average_row_size=VALUES(average_row_size), row_limit=VALUES(row_limit), "
        "size_limit=VALUES(size_limit)"
    )


def _monitor_style_upsert(admin, *, collection_id: str, namespace_id: int, ltable_id: int,
                          estimated_rows: int, average_row_size: int, included_index: int = 0) -> None:
    """Replay the exact UPSERT the monitor now uses (update_namespace_audit_): only estimate
    columns in the UPDATE clause, so row_limit / size_limit must be preserved."""
    admin._server._execute(
        "INSERT INTO sdk_namespaces_stats "
        "(collection_id, namespace_id, ltable_id, estimated_rows, average_row_size, "
        " last_estimate_time, included_index) VALUES "
        f"('{collection_id}', {int(namespace_id)}, {int(ltable_id)}, {int(estimated_rows)}, "
        f" {int(average_row_size)}, NOW(6), {int(included_index)}) "
        "ON DUPLICATE KEY UPDATE estimated_rows=VALUES(estimated_rows), "
        "average_row_size=VALUES(average_row_size), last_estimate_time=VALUES(last_estimate_time)"
    )


def _get_ltable_id(admin, collection_id: str, namespace_id: int) -> int:
    rows = admin._server._execute(
        "SELECT ltable_id FROM sdk_ltables "
        f"WHERE collection_id = '{collection_id}' AND namespace_id = {int(namespace_id)} "
        "AND ltable_name = 'default' LIMIT 1"
    )
    assert rows, "default ltable row not found in sdk_ltables"
    row = rows[0]
    return int(row["ltable_id"] if isinstance(row, dict) else row[0])


def _read_stat(admin, collection_id: str, namespace_id: int, ltable_id: int, col: str,
               included_index: int = 0):
    rows = admin._server._execute(
        f"SELECT {col} AS v FROM sdk_namespaces_stats "
        f"WHERE collection_id = '{collection_id}' AND namespace_id = {int(namespace_id)} "
        f"AND ltable_id = {int(ltable_id)} AND included_index = {int(included_index)} LIMIT 1"
    )
    if not rows:
        return None
    row = rows[0]
    return int(row["v"] if isinstance(row, dict) else row[0])


def _count_stats(admin, collection_id: str, namespace_id: int) -> int:
    rows = admin._server._execute(
        "SELECT COUNT(*) AS cnt FROM sdk_namespaces_stats "
        f"WHERE collection_id = '{collection_id}' AND namespace_id = {int(namespace_id)}"
    )
    row = rows[0]
    return int(row["cnt"] if isinstance(row, dict) else row[0])


def _assert_blocked(fn) -> None:
    with pytest.raises(Exception) as excinfo:
        fn()
    msg = str(excinfo.value).lower()
    assert "not allowed" in msg or "limit" in msg, f"unexpected error: {excinfo.value!r}"


def _ensure_stats_ddl_defaults(admin) -> None:
    """Align live table column defaults with SDK DDL (CREATE IF NOT EXISTS is no-op on upgrade)."""
    admin._server._execute(
        "ALTER TABLE sdk_namespaces_stats "
        f"MODIFY row_limit BIGINT NOT NULL DEFAULT {NamespaceStatsDefaults.ROW_LIMIT} "
        "COMMENT 'row count limit, -1 means unlimited', "
        f"MODIFY size_limit BIGINT NOT NULL DEFAULT {NamespaceStatsDefaults.SIZE_LIMIT} "
        "COMMENT 'storage size limit in bytes, -1 means unlimited'"
    )


class TestNamespaceStatsDefaults:
    """Monitor-style INSERT must pick up SDK DDL defaults (100w rows / 20GB)."""

    def test_monitor_insert_uses_ddl_default_limits(self, oceanbase_client):
        admin = _raw_client()
        _ensure_stats_ddl_defaults(admin)
        name = f"rl_def_{uuid.uuid4().hex[:12]}"
        coll = oceanbase_client.create_collection(
            name=name, schema=ns_schema(), use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_def")
            ns_id = int(ns.namespace_id)
            ltable_id = _get_ltable_id(admin, coll.id, ns_id)
            for lt in (0, ltable_id):
                _monitor_style_upsert(
                    admin, collection_id=coll.id, namespace_id=ns_id, ltable_id=lt,
                    estimated_rows=1, average_row_size=100,
                )
                assert _read_stat(admin, coll.id, ns_id, lt, "row_limit") == NamespaceStatsDefaults.ROW_LIMIT
                assert _read_stat(admin, coll.id, ns_id, lt, "size_limit") == NamespaceStatsDefaults.SIZE_LIMIT
        finally:
            oceanbase_client.delete_collection(name=name)
            _close(admin)


class TestNamespaceRowLimit:
    """Row/size limit enforcement on the logical-table INSERT path."""

    def test_namespace_level_limit_blocks_insert(self, oceanbase_client):
        """Seeding the ltable_id=0 aggregate row with a small row_limit blocks INSERT."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"rl_ns_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name, schema=ns_schema(), use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_a")
            ns_id = int(ns.namespace_id)
            for i in range(3):  # no stat row yet -> unlimited
                _add_one(ns, i)
            assert ns.count() == 3

            # namespace-level aggregate: estimate 1000 >= row_limit 2 -> block (estimate-only).
            _upsert_stat(admin, collection_id=coll.id, namespace_id=ns_id, ltable_id=0,
                         estimated_rows=1000, row_limit=2)
            _assert_blocked(lambda: _add_one(ns, 99))

            # DELETE must NOT be blocked (recovery path).
            ns.delete(ids=["row_000000"])
            assert ns.count() == 2

            # Lift the limit -> INSERT allowed again.
            _upsert_stat(admin, collection_id=coll.id, namespace_id=ns_id, ltable_id=0,
                         estimated_rows=1000, row_limit=-1)
            _add_one(ns, 100)
            assert ns.count() == 3
        finally:
            owner.delete_collection(name=name)
            _close(admin)

    def test_ltable_level_limit_with_exact_count_recheck(self, oceanbase_client):
        """ltable-level limit triggers exact COUNT; over-estimate self-corrects after delete."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"rl_lt_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name, schema=ns_schema(), use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_b")
            ns_id = int(ns.namespace_id)
            for i in range(3):
                _add_one(ns, i)
            assert ns.count() == 3

            ltable_id = _get_ltable_id(admin, coll.id, ns_id)
            # estimate 1000 >= 2 -> exact count runs; actual 3 >= 2 -> block.
            _upsert_stat(admin, collection_id=coll.id, namespace_id=ns_id, ltable_id=ltable_id,
                         estimated_rows=1000, row_limit=2)
            _assert_blocked(lambda: _add_one(ns, 99))

            # Empty the namespace: exact COUNT now 0 < 2 -> over-estimate corrected, INSERT allowed.
            ns.delete(ids=[f"row_{i:06d}" for i in range(3)])
            assert ns.count() == 0
            _add_one(ns, 200)  # must not raise
            assert ns.count() == 1
        finally:
            owner.delete_collection(name=name)
            _close(admin)

    def test_ops_limit_survives_monitor_style_upsert(self, oceanbase_client):
        """A2: monitor's UPSERT (estimate columns only) must not reset ops row_limit/size_limit."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"rl_keep_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name, schema=ns_schema(), use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_c")
            ns_id = int(ns.namespace_id)
            for i in range(5):
                _add_one(ns, i)
            ltable_id = _get_ltable_id(admin, coll.id, ns_id)

            # ops configures limits.
            _upsert_stat(admin, collection_id=coll.id, namespace_id=ns_id, ltable_id=ltable_id,
                         estimated_rows=5, average_row_size=10, row_limit=1000, size_limit=999999)
            # a monitor round re-estimates the same row (row_limit/size_limit NOT in UPDATE list).
            _monitor_style_upsert(admin, collection_id=coll.id, namespace_id=ns_id,
                                  ltable_id=ltable_id, estimated_rows=7, average_row_size=20)

            # estimate columns refreshed by the "monitor" ...
            assert _read_stat(admin, coll.id, ns_id, ltable_id, "estimated_rows") == 7
            assert _read_stat(admin, coll.id, ns_id, ltable_id, "average_row_size") == 20
            # ... but ops-configured limits are preserved (A2 fix).
            assert _read_stat(admin, coll.id, ns_id, ltable_id, "row_limit") == 1000, (
                "row_limit clobbered — REPLACE INTO must be INSERT ... ON DUPLICATE KEY UPDATE"
            )
            assert _read_stat(admin, coll.id, ns_id, ltable_id, "size_limit") == 999999
        finally:
            owner.delete_collection(name=name)
            _close(admin)

    def test_size_limit_blocks_insert(self, oceanbase_client):
        """size_limit (bytes) blocks INSERT independently of row_limit."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"rl_sz_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name, schema=ns_schema(), use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_sz")
            ns_id = int(ns.namespace_id)
            for i in range(3):
                _add_one(ns, i)
            # namespace-level: est_rows*avg = 1000*1000 = 1e6 >= size_limit 100; row_limit unlimited.
            _upsert_stat(admin, collection_id=coll.id, namespace_id=ns_id, ltable_id=0,
                         estimated_rows=1000, average_row_size=1000, row_limit=-1, size_limit=100)
            _assert_blocked(lambda: _add_one(ns, 99))

            # lift size_limit -> INSERT allowed again.
            _upsert_stat(admin, collection_id=coll.id, namespace_id=ns_id, ltable_id=0,
                         estimated_rows=1000, average_row_size=1000, row_limit=-1, size_limit=-1)
            _add_one(ns, 100)  # must not raise
        finally:
            owner.delete_collection(name=name)
            _close(admin)

    def test_under_limit_passes(self, oceanbase_client):
        """A configured-but-not-exceeded limit must NOT block inserts (normal path)."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"rl_ok_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name, schema=ns_schema(), use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_ok")
            ns_id = int(ns.namespace_id)
            _add_one(ns, 0)
            ltable_id = _get_ltable_id(admin, coll.id, ns_id)
            # generous limits on both levels: nothing should block.
            for lt in (0, ltable_id):
                _upsert_stat(admin, collection_id=coll.id, namespace_id=ns_id, ltable_id=lt,
                             estimated_rows=1, average_row_size=1,
                             row_limit=1_000_000, size_limit=1_000_000_000)
            for i in range(1, 5):
                _add_one(ns, i)  # must not raise
            assert ns.count() == 5
        finally:
            owner.delete_collection(name=name)
            _close(admin)

    def test_update_not_blocked_when_over_limit(self, oceanbase_client):
        """T_UPDATE must pass even while an INSERT block is active (recovery path)."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"rl_upd_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name, schema=ns_schema(), use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_upd")
            ns_id = int(ns.namespace_id)
            _add_one(ns, 0)
            _upsert_stat(admin, collection_id=coll.id, namespace_id=ns_id, ltable_id=0,
                         estimated_rows=1000, row_limit=1)
            _assert_blocked(lambda: _add_one(ns, 1))
            # A T_UPDATE issued on the same (ns-context) session must NOT be gated.
            owner._server._execute(
                "UPDATE sdk_namespaces_stats SET last_estimate_time = NOW(6) WHERE 1 = 0"
            )
        finally:
            owner.delete_collection(name=name)
            _close(admin)

    def test_drop_namespace_clears_stats(self, oceanbase_client):
        """DROP_NAMESPACE synchronously removes the namespace's sdk_namespaces_stats rows."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"rl_drop_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name, schema=ns_schema(), use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_drop")
            ns_id = int(ns.namespace_id)
            _add_one(ns, 0)
            ltable_id = _get_ltable_id(admin, coll.id, ns_id)
            _upsert_stat(admin, collection_id=coll.id, namespace_id=ns_id, ltable_id=ltable_id,
                         estimated_rows=5, row_limit=10)
            _upsert_stat(admin, collection_id=coll.id, namespace_id=ns_id, ltable_id=0,
                         estimated_rows=5, row_limit=10)
            assert _count_stats(admin, coll.id, ns_id) >= 2

            coll.delete_namespace("ns_drop")
            assert _count_stats(admin, coll.id, ns_id) == 0, (
                "DROP_NAMESPACE must delete sdk_namespaces_stats rows for the namespace"
            )
        finally:
            owner.delete_collection(name=name)
            _close(admin)
