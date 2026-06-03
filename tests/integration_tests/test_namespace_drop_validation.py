import threading
import time

import pytest

import pyseekdb
from pyseekdb import IVFConfiguration
from pyseekdb.client.configuration import VectorIndexConfig
from pyseekdb.client.meta_info import NamespaceCollectionNames
from pyseekdb.client.schema import Schema


RECYCLEBIN_PREFIX = "__recyclebin_"


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_collection(client, suffix: str = ""):
    """Create a namespace-mode collection with an IVF index (so we get a full
    set of physical tables: logic_data / kv_data / logic_schema / hot_table)."""
    name = f"test_ns_drop_{int(time.time() * 1000)}{suffix}"
    schema = Schema(
        vector_index=VectorIndexConfig(
            ivf=IVFConfiguration(dimension=3, distance="cosine", fresh_mode="spfresh"),
            embedding_function=None,
        ),
    )
    return client.create_collection(name=name, schema=schema, use_namespace=True)


def _execute(client, sql: str):
    return client._server._execute(sql)


def _is_ss_mode(client) -> bool:
    """Best-effort detect whether the connected OB cluster is in SS mode."""
    try:
        rows = _execute(
            client,
            "SHOW PARAMETERS LIKE 'enable_logservice'",
        )
        for r in rows:
            value = (r.get("value") or r.get("VALUE") or "").lower()
            if value in ("true", "1", "on"):
                return True
    except Exception:
        pass
    return False


def _fetch_namespace_name(client, collection_id: str, namespace_id: int):
    rows = _execute(
        client,
        f"SELECT namespace_name FROM sdk_namespaces "
        f"WHERE collection_id = '{collection_id}' AND namespace_id = {namespace_id}",
    )
    if not rows:
        return None
    return rows[0].get("namespace_name") or rows[0].get("NAMESPACE_NAME")


def _count_ltables(client, collection_id: str, namespace_id: int) -> int:
    rows = _execute(
        client,
        f"SELECT COUNT(*) AS c FROM sdk_ltables "
        f"WHERE collection_id = '{collection_id}' AND namespace_id = {namespace_id}",
    )
    return int(rows[0]["c"] if "c" in rows[0] else rows[0]["C"])


def _count_logic_schema_rows(client, collection_id: str, namespace_id: int) -> int:
    tbl = NamespaceCollectionNames.logic_schema_table_name(collection_id)
    rows = _execute(
        client,
        f"SELECT COUNT(*) AS c FROM `{tbl}` WHERE namespace_id = {namespace_id}",
    )
    return int(rows[0]["c"] if "c" in rows[0] else rows[0]["C"])


def _count_hot_table_rows(client, collection_id: str, namespace_id: int) -> int:
    tbl = NamespaceCollectionNames.hot_table_name(collection_id)
    try:
        rows = _execute(
            client,
            f"SELECT COUNT(*) AS c FROM `{tbl}` WHERE namespace_id = {namespace_id}",
        )
    except Exception:
        return -1  # table missing
    return int(rows[0]["c"] if "c" in rows[0] else rows[0]["C"])


def _count_logic_data_rows(client, collection_id: str, namespace_id: int, ltable_id: int | None = None) -> int:
    tbl = NamespaceCollectionNames.data_table_name(collection_id)
    if ltable_id is not None:
        where = f"namespace_id = {namespace_id} AND ltable_id = {ltable_id}"
    else:
        where = f"namespace_id = {namespace_id}"
    rows = _execute(
        client,
        f"SELECT COUNT(*) AS c FROM `{tbl}` WHERE {where}",
    )
    return int(rows[0]["c"] if "c" in rows[0] else rows[0]["C"])


def _count_kv_data_rows(client, collection_id: str, namespace_id: int) -> int:
    tbl = NamespaceCollectionNames.kv_data_table_name(collection_id)
    try:
        rows = _execute(
            client,
            f"SELECT COUNT(*) AS c FROM `{tbl}` WHERE namespace_id = {namespace_id}",
        )
    except Exception:
        return -1
    return int(rows[0]["c"] if "c" in rows[0] else rows[0]["C"])


def _seed_logic_data(client, collection_id: str, namespace_id: int, ltable_id: int, count: int = 3):
    tbl = NamespaceCollectionNames.data_table_name(collection_id)
    values = []
    for i in range(count):
        values.append(
            f"({namespace_id}, {ltable_id}, 'doc_{i}', "
            f"X'0000803f0000000000000000', "
            f"'{{\"id\": \"id_{i}\", \"metadata\": {{\"k\": \"v\"}}}}')"
        )
    _execute(client, f"INSERT INTO `{tbl}` (namespace_id, ltable_id, document, embedding, data_content) VALUES {', '.join(values)}")


def _fetch_ltable_id(client, collection_id: str, namespace_id: int) -> int | None:
    rows = _execute(
        client,
        f"SELECT ltable_id FROM sdk_ltables "
        f"WHERE collection_id = '{collection_id}' AND namespace_id = {namespace_id} "
        f"ORDER BY ltable_id LIMIT 1",
    )
    if not rows:
        return None
    return int(rows[0]["ltable_id"] if "ltable_id" in rows[0] else rows[0]["LTABLE_ID"])


def _seed_hot_table(client, collection_id: str, namespace_id: int):
    """In SS mode insert a synthetic hot_table row so the DELETE has something
    to remove and we can verify the cleanup."""
    tbl = NamespaceCollectionNames.hot_table_name(collection_id)
    try:
        _execute(
            client,
            f"INSERT INTO `{tbl}` (namespace_id, last_access_time) "
            f"VALUES ({namespace_id}, NOW(6))",
        )
        return True
    except Exception:
        return False


def _drop_namespace_via_pl(client, collection_id: str, namespace_id: int):
    """Call DBMS_LOGIC_TABLE.DROP_NAMESPACE directly (raw SQL, bypassing
    SDK's pre-existence guard)."""
    _execute(
        client,
        f"CALL DBMS_LOGIC_TABLE.DROP_NAMESPACE('{collection_id}', {namespace_id})",
    )


def _second_oceanbase_client():
    """Second pymysql-backed client (separate TCP connection) for concurrency tests."""
    import os

    client = pyseekdb.Client(
        host=os.environ.get("OB_HOST", "localhost"),
        port=int(os.environ.get("OB_PORT", "11202")),
        tenant=os.environ.get("OB_TENANT", "mysql"),
        database=os.environ.get("OB_DATABASE", "test"),
        user=os.environ.get("OB_USER", "root"),
        password=os.environ.get("OB_PASSWORD", ""),
    )
    result = client._server._execute("SELECT 1 as test")
    assert result and result[0].get("test") == 1
    return client


# --------------------------------------------------------------------------- #
# tests                                                                       #
# --------------------------------------------------------------------------- #


class TestDropNamespaceCatalogValidation:
    """Validate everything the DROP_NAMESPACE synchronous transaction must do."""

    # ------------------------------------------------------------- #
    # 1. Happy path: rename + cleanup all relevant rows              #
    # ------------------------------------------------------------- #
    def test_drop_namespace_renames_and_cleans_catalog(self, oceanbase_client):
        client = oceanbase_client
        is_ss = _is_ss_mode(client)
        collection = _make_collection(client)
        coll_id = collection.id
        try:
            ns = collection.create_namespace("ns_validate")
            ns_id = int(ns.namespace_id)

            # Pre-conditions
            orig_name = _fetch_namespace_name(client, coll_id, ns_id)
            assert orig_name == "ns_validate"
            assert _count_ltables(client, coll_id, ns_id) >= 1, (
                "create_namespace should have inserted a default ltable"
            )
            assert _count_logic_schema_rows(client, coll_id, ns_id) >= 1, (
                "create_namespace should have inserted a logic_schema row"
            )
            if is_ss:
                # Seed a hot_table row so we can verify the SS-only delete.
                assert _seed_hot_table(client, coll_id, ns_id) is True
                assert _count_hot_table_rows(client, coll_id, ns_id) == 1

            # Seed some logic_data rows so we can verify they survive the sync
            # drop phase (async background task cleans them up later).
            lt_id = _fetch_ltable_id(client, coll_id, ns_id)
            assert lt_id is not None, "should have a default ltable"
            _seed_logic_data(client, coll_id, ns_id, lt_id, count=3)
            assert _count_logic_data_rows(client, coll_id, ns_id, lt_id) == 3

            # Action
            _drop_namespace_via_pl(client, coll_id, ns_id)

            # Post-conditions: synchronous phase
            new_name = _fetch_namespace_name(client, coll_id, ns_id)
            assert new_name is not None, "namespace row must still exist (renamed, not deleted)"
            assert new_name.startswith(RECYCLEBIN_PREFIX), (
                f"namespace must be renamed to __recyclebin_ prefix, got {new_name!r}"
            )
            assert "ns_validate" in new_name, (
                f"original name should be embedded in recyclebin name, got {new_name!r}"
            )
            assert _count_ltables(client, coll_id, ns_id) == 0, (
                "sdk_ltables rows must be deleted"
            )
            assert _count_logic_schema_rows(client, coll_id, ns_id) == 0, (
                "logic_schema_table rows must be deleted"
            )
            if is_ss:
                assert _count_hot_table_rows(client, coll_id, ns_id) == 0, (
                    "hot_table row must be deleted in SS mode"
                )

            # Logic data rows still exist — async background task handles physical cleanup.
            assert _count_logic_data_rows(client, coll_id, ns_id, lt_id) == 3, (
                "logic_data rows must survive the sync drop phase"
            )
        finally:
            client.delete_collection(name=collection.name)

    # ------------------------------------------------------------- #
    # 1b. Async cleanup: wait 32s, verify physical data is gone      #
    # ------------------------------------------------------------- #
    def test_drop_namespace_async_cleanup(self, oceanbase_client):
        """After DROP_NAMESPACE renames the namespace, a background task (30s
        interval) picks up __recyclebin_ entries and removes physical data.
        Wait 32s and verify the renamed namespace and its data are gone."""
        client = oceanbase_client
        collection = _make_collection(client)
        coll_id = collection.id
        try:
            ns = collection.create_namespace("ns_async")
            ns_id = int(ns.namespace_id)
            lt_id = _fetch_ltable_id(client, coll_id, ns_id)
            assert lt_id is not None

            # Seed data
            _seed_logic_data(client, coll_id, ns_id, lt_id, count=3)
            assert _count_logic_data_rows(client, coll_id, ns_id, lt_id) == 3

            # Drop — sync phase renames namespace, removes catalog entries
            _drop_namespace_via_pl(client, coll_id, ns_id)

            recycled_name = _fetch_namespace_name(client, coll_id, ns_id)
            assert recycled_name is not None
            assert recycled_name.startswith(RECYCLEBIN_PREFIX)
            assert "ns_async" in recycled_name

            # Data still present immediately after drop
            assert _count_logic_data_rows(client, coll_id, ns_id, lt_id) == 3

            # Wait for async background task (scans every 30s, 32s should suffice)
            time.sleep(32)

            # After background task: renamed namespace entry should be gone
            final_name = _fetch_namespace_name(client, coll_id, ns_id)
            assert final_name is None, (
                f"async task should have removed the __recyclebin_ namespace row, "
                f"got {final_name!r}"
            )

            # Logic data should also be gone
            data_rows = _count_logic_data_rows(client, coll_id, ns_id, lt_id)
            assert data_rows == 0, (
                f"async task should have cleaned up logic_data rows, "
                f"found {data_rows}"
            )

        finally:
            client.delete_collection(name=collection.name)

    # ------------------------------------------------------------- #
    # 2. Namespace-name rename format                                #
    # ------------------------------------------------------------- #
    def test_recyclebin_name_format(self, oceanbase_client):
        client = oceanbase_client
        collection = _make_collection(client)
        try:
            ns = collection.create_namespace("fmt_ns")
            ns_id = int(ns.namespace_id)
            _drop_namespace_via_pl(client, collection.id, ns_id)
            new_name = _fetch_namespace_name(client, collection.id, ns_id)
            # Expected format: __recyclebin_<orig>_<ts_us>
            assert new_name.startswith("__recyclebin_fmt_ns_")
            ts_suffix = new_name[len("__recyclebin_fmt_ns_"):]
            assert ts_suffix.isdigit() and len(ts_suffix) >= 16, (
                f"trailing timestamp_us looks malformed: {new_name!r}"
            )
        finally:
            client.delete_collection(name=collection.name)

    # ------------------------------------------------------------- #
    # 3. Multiple ltables under one namespace all get deleted        #
    # ------------------------------------------------------------- #
    def test_multiple_ltables_all_deleted(self, oceanbase_client):
        client = oceanbase_client
        collection = _make_collection(client)
        try:
            ns = collection.create_namespace("ns_multi")
            ns_id = int(ns.namespace_id)
            # Insert two extra ltable rows directly so we can verify bulk delete.
            _execute(
                client,
                f"INSERT INTO sdk_ltables (collection_id, namespace_id, ltable_name) "
                f"VALUES ('{collection.id}', {ns_id}, 'extra_lt_a'),"
                f"       ('{collection.id}', {ns_id}, 'extra_lt_b')",
            )
            assert _count_ltables(client, collection.id, ns_id) >= 3

            _drop_namespace_via_pl(client, collection.id, ns_id)

            assert _count_ltables(client, collection.id, ns_id) == 0
        finally:
            client.delete_collection(name=collection.name)

    # ------------------------------------------------------------- #
    # 4. Atomic rollback when an in-transaction DELETE fails         #
    # ------------------------------------------------------------- #
    def test_atomic_rollback_on_logic_schema_missing(self, oceanbase_client):
        """If we pre-DROP <coll>_logic_schema_table the in-trans DELETE on it
        fails with OB_TABLE_NOT_EXIST, which must roll back the namespace
        rename and the sdk_ltables delete that ran earlier in the same
        transaction."""
        client = oceanbase_client
        collection = _make_collection(client)
        coll_id = collection.id
        schema_tbl = NamespaceCollectionNames.logic_schema_table_name(coll_id)
        try:
            ns = collection.create_namespace("ns_rollback")
            ns_id = int(ns.namespace_id)
            orig_name = _fetch_namespace_name(client, coll_id, ns_id)
            assert orig_name == "ns_rollback"
            ltable_cnt_before = _count_ltables(client, coll_id, ns_id)
            assert ltable_cnt_before >= 1

            # Sabotage: drop the logic_schema_table so step 6 inside the
            # DROP_NAMESPACE transaction will fail with OB_TABLE_NOT_EXIST.
            _execute(client, f"DROP TABLE `{schema_tbl}`")

            with pytest.raises(Exception):
                _drop_namespace_via_pl(client, coll_id, ns_id)

            # Everything before the failed step must have been rolled back.
            after_name = _fetch_namespace_name(client, coll_id, ns_id)
            assert after_name == "ns_rollback", (
                f"namespace_name must be rolled back to original, got {after_name!r}"
            )
            assert _count_ltables(client, coll_id, ns_id) == ltable_cnt_before, (
                "sdk_ltables delete must be rolled back"
            )
        finally:
            # Recreate the schema table (empty) so cleanup_namespace_physical_tables
            # / delete_collection runs cleanly.
            try:
                _execute(
                    client,
                    f"CREATE TABLE IF NOT EXISTS `{schema_tbl}` ("
                    f"  namespace_id BIGINT UNSIGNED NOT NULL,"
                    f"  ltable_id BIGINT UNSIGNED NOT NULL,"
                    f"  schema_content JSON NOT NULL,"
                    f"  PRIMARY KEY (namespace_id, ltable_id)) "
                    f"PARTITION BY KEY(namespace_id) PARTITIONS 8",
                )
            except Exception:
                pass
            client.delete_collection(name=collection.name)

    # ------------------------------------------------------------- #
    # 5. Idempotent: second drop on the same ns is a no-op success   #
    # ------------------------------------------------------------- #
    def test_drop_namespace_is_idempotent(self, oceanbase_client):
        client = oceanbase_client
        collection = _make_collection(client)
        coll_id = collection.id
        try:
            ns = collection.create_namespace("ns_idem")
            ns_id = int(ns.namespace_id)
            _drop_namespace_via_pl(client, coll_id, ns_id)
            first_name = _fetch_namespace_name(client, coll_id, ns_id)
            assert first_name.startswith(RECYCLEBIN_PREFIX)

            # Second call must succeed without raising and must NOT re-rename
            # (the row name stays exactly the same — no double prefix).
            _drop_namespace_via_pl(client, coll_id, ns_id)
            second_name = _fetch_namespace_name(client, coll_id, ns_id)
            assert second_name == first_name, (
                f"second DROP_NAMESPACE must be no-op, got {second_name!r} vs {first_name!r}"
            )
            # And LEFT(name, 26) must not be '__recyclebin___recyclebin_'
            assert not second_name.startswith("__recyclebin___recyclebin_")
        finally:
            client.delete_collection(name=collection.name)

    # ------------------------------------------------------------- #
    # 6. Concurrent drop from two SDK connections                    #
    # ------------------------------------------------------------- #
    def test_concurrent_drop_from_two_sdks(self, oceanbase_client):
        """Two independent SDK connections race to DROP_NAMESPACE on the same
        (collection_id, namespace_id).

        Expected: exactly one transaction performs the rename + deletes; the
        other observes the __recyclebin_ row inside its own short trans and
        commits an empty trans (no exception, no double prefix).  Both calls
        return success."""
        client_a = oceanbase_client
        client_b = _second_oceanbase_client()

        collection = _make_collection(client_a, suffix="_conc")
        coll_id = collection.id
        try:
            ns = collection.create_namespace("ns_concurrent")
            ns_id = int(ns.namespace_id)

            results = {}
            barrier = threading.Barrier(2)

            def _worker(tag, c):
                try:
                    barrier.wait(timeout=10)
                    _drop_namespace_via_pl(c, coll_id, ns_id)
                    results[tag] = "ok"
                except Exception as exc:  # noqa: BLE001
                    results[tag] = f"err: {exc!r}"

            ta = threading.Thread(target=_worker, args=("A", client_a))
            tb = threading.Thread(target=_worker, args=("B", client_b))
            ta.start(); tb.start()
            ta.join(timeout=30); tb.join(timeout=30)

            # Both calls must succeed (one does the work, the other no-ops).
            assert results.get("A") == "ok", f"A failed: {results.get('A')!r}"
            assert results.get("B") == "ok", f"B failed: {results.get('B')!r}"

            final_name = _fetch_namespace_name(client_a, coll_id, ns_id)
            assert final_name.startswith(RECYCLEBIN_PREFIX)
            assert "ns_concurrent" in final_name
            # The pattern must be __recyclebin_<orig>_<ts> — never doubled.
            assert not final_name.startswith("__recyclebin___recyclebin_"), (
                f"concurrent drops produced double-prefixed name: {final_name!r}"
            )
            # ltables and logic_schema must be empty (one of the trans cleared them).
            assert _count_ltables(client_a, coll_id, ns_id) == 0
            assert _count_logic_schema_rows(client_a, coll_id, ns_id) == 0
        finally:
            try:
                client_a.delete_collection(name=collection.name)
            finally:
                if hasattr(client_b, "close"):
                    try:
                        client_b.close()
                    except Exception:
                        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
