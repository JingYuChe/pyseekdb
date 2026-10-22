"""Namespace collection creation must not be destroyed by concurrent SDK readers."""

from __future__ import annotations

import contextlib
import os
import threading
import uuid

import pytest
from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT

import pyseekdb
from pyseekdb import IVFConfiguration, Schema, VectorIndexConfig
from pyseekdb.client.meta_info import NamespaceCollectionNames


def _schema():
    return Schema(
        fulltext_index=None,
        vector_index=VectorIndexConfig(
            ivf=IVFConfiguration(dimension=3, distance="cosine", centroids_fresh_mode="spfresh"),
            embedding_function=None,
        ),
    )


def _new_client_like(client, *, port=None):
    """Use the fixture's exact database/tenant with an independent connection."""
    server = client._server
    connection_options = dict(server.kwargs)
    connection_options.setdefault("connect_timeout", 10)
    connection_options.setdefault("read_timeout", 75)
    connection_options.setdefault("write_timeout", 75)
    result = pyseekdb.Client(
        host=server.host,
        port=server.port if port is None else port,
        tenant=server.tenant,
        database=server.database,
        user=server.user,
        password=server.password,
        charset=server.charset,
        **connection_options,
    )
    result._server._execute("SET SESSION ob_query_timeout = 60000000")
    return result


def test_get_and_duplicate_create_preserve_in_progress_collection(oceanbase_client, monkeypatch):
    """Real catalog/DDL operations across clients preserve a creator paused before its main CREATE."""
    name = f"test_ns_create_race_{uuid.uuid4().hex}"
    creator = _new_client_like(oceanbase_client)
    alternate_port = os.environ.get("OB_ALTERNATE_PORT")
    competitor = _new_client_like(oceanbase_client, port=int(alternate_port) if alternate_port else None)
    reader = oceanbase_client
    create_paused = threading.Event()
    allow_create = threading.Event()
    creator_results = []
    creator_errors = []
    paused_sql = []
    reader_sql = []
    competitor_sql = []
    creator_execute = creator._server._execute
    reader_execute = reader._server._execute
    competitor_execute = competitor._server._execute

    def pause_main_create(sql):
        if "CREATE TABLE" in sql.upper() and "IS_LOGIC_TABLE = TRUE" in sql.upper():
            paused_sql.append(sql)
            create_paused.set()
            assert allow_create.wait(timeout=60), "Reader did not release the paused collection CREATE"
        return creator_execute(sql)

    def record_reader(sql):
        reader_sql.append(sql)
        return reader_execute(sql)

    def record_competitor(sql):
        competitor_sql.append(sql)
        return competitor_execute(sql)

    def create_worker():
        try:
            creator_results.append(
                creator.create_collection(
                    name=name,
                    schema=_schema(),
                    use_namespace=True,
                    partition_count=NAMESPACE_TEST_PARTITION_COUNT,
                )
            )
        except Exception as exc:
            creator_errors.append(exc)

    monkeypatch.setattr(creator._server, "_execute", pause_main_create)
    thread = threading.Thread(target=create_worker, daemon=True, name="namespace-collection-creator")
    thread.start()
    try:
        assert create_paused.wait(timeout=60), f"Creator did not reach its main CREATE: {creator_errors!r}"
        meta = reader._server._get_ns_collection_meta(name)
        assert meta is not None
        cid = meta["collection_id"]
        data_table = NamespaceCollectionNames.data_table_name(cid)
        assert f"`{data_table}`" in paused_sql[0]
        assert not reader._server._table_exists(data_table)

        with monkeypatch.context() as concurrent_patch:
            concurrent_patch.setattr(reader._server, "_execute", record_reader)
            for _ in range(2):
                with pytest.raises(ValueError, match="not ready or incomplete"):
                    reader.get_collection(name)
            with pytest.raises(ValueError, match="not ready or incomplete"):
                reader.get_or_create_collection(name, schema=_schema(), use_namespace=True)

            concurrent_patch.setattr(competitor._server, "_execute", record_competitor)
            # Also exercise the actual unique-key conflict after stale prechecks.
            concurrent_patch.setattr(competitor._server, "has_collection", lambda _name: False)
            concurrent_patch.setattr(competitor._server, "_get_ns_collection_meta", lambda _name: None)
            with pytest.raises(ValueError, match="already exists"):
                competitor.create_collection(
                    name=name,
                    schema=_schema(),
                    use_namespace=True,
                    partition_count=NAMESPACE_TEST_PARTITION_COUNT,
                )

        assert reader_sql
        assert all(sql.lstrip().upper().startswith(("SELECT ", "SHOW ", "USE ")) for sql in reader_sql)
        assert any(sql.lstrip().upper().startswith("INSERT INTO ") for sql in competitor_sql)
        assert not any(sql.lstrip().upper().startswith(("DELETE ", "DROP ")) for sql in competitor_sql)
        assert not any("CREATE TABLE" in sql.upper() and "IS_LOGIC_TABLE" in sql.upper() for sql in competitor_sql)
        assert reader._server._get_ns_collection_meta(name)["collection_id"] == cid
        assert not reader._server._table_exists(data_table)

        allow_create.set()
        thread.join(timeout=90)
        assert not thread.is_alive(), "Creator did not finish after its main CREATE was released"
        assert not creator_errors, repr(creator_errors)
        assert len(creator_results) == 1
        assert creator_results[0].id == cid
        reopened = reader.get_collection(name)
        assert reopened.id == cid
        assert not reader._server._is_incomplete_ns_collection(name)
        namespace = reopened.create_namespace("after_create")
        namespace.add(ids="winner", embeddings=[1.0, 2.0, 3.0])
        assert namespace.get(ids=["winner"], include=[])["ids"] == ["winner"]
    finally:
        allow_create.set()
        thread.join(timeout=90)
        with contextlib.suppress(Exception):
            competitor.close()
        if not thread.is_alive():
            with contextlib.suppress(Exception):
                reader.delete_collection(name)
            with contextlib.suppress(Exception):
                creator.close()
        assert not thread.is_alive(), f"Creator still active; resources for {name} were left for inspection"


def test_recreate_uses_new_id_and_rejects_stale_handles(oceanbase_client):
    """An old collection/namespace cannot act on the new generation with the same name."""
    name = f"test_ns_recreate_generation_{uuid.uuid4().hex}"
    client = oceanbase_client
    old_collection = client.create_collection(
        name=name,
        schema=_schema(),
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )
    try:
        old_namespace = old_collection.create_namespace("same_namespace")
        old_namespace.add(ids="old", embeddings=[1.0, 2.0, 3.0])
        client.delete_collection(name)

        new_collection = client.create_collection(
            name=name,
            schema=_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        assert new_collection.id != old_collection.id
        new_namespace = new_collection.create_namespace("same_namespace")
        new_namespace.add(ids="new", embeddings=[3.0, 2.0, 1.0])

        with pytest.raises(ValueError, match="no longer exists"):
            old_collection.create_namespace("stale_collection_write")
        with pytest.raises(ValueError, match="no longer exists"):
            old_namespace.add(ids="stale_namespace_write", embeddings=[1.0, 1.0, 1.0])

        assert client.get_collection(name).id == new_collection.id
        assert new_namespace.get(include=[])["ids"] == ["new"]
        assert new_namespace.count() == 1
    finally:
        with contextlib.suppress(Exception):
            client.delete_collection(name)


def _assert_generation_removed(server, collection_id):
    """Confirm physical and catalog cleanup without hiding probe failures."""
    for table_name in (
        NamespaceCollectionNames.data_table_name(collection_id),
        NamespaceCollectionNames.logic_schema_table_name(collection_id),
        NamespaceCollectionNames.kv_data_table_name(collection_id),
        NamespaceCollectionNames.hot_table_name(collection_id),
    ):
        assert not server._table_exists(table_name, strict=True)
    assert not server._tablegroup_exists(NamespaceCollectionNames.tablegroup_name(collection_id), strict=True)
    for catalog in ("sdk_collections", "sdk_namespaces", "sdk_ltables", "sdk_namespace_stat"):
        assert not server._execute(f"SELECT 1 FROM {server._qtable(catalog)} WHERE collection_id = '{collection_id}'")


def test_failed_delete_preserves_catalog_for_another_client_retry(oceanbase_client, monkeypatch):
    """A persistent DROP failure is reported and leaves the name available for explicit retry."""
    name = f"test_ns_delete_retry_{uuid.uuid4().hex}"
    client = oceanbase_client
    server = client._server
    collection = client.create_collection(
        name=name,
        schema=_schema(),
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )
    cid = collection.id
    retry_client = None
    try:
        namespace = collection.create_namespace("before_delete")
        namespace.add(ids="original", embeddings=[1.0, 2.0, 3.0])
        failed_table = NamespaceCollectionNames.kv_data_table_name(cid)
        failed_drop = f"DROP TABLE IF EXISTS `{failed_table}`"
        attempted_drops = []
        execute = server._execute

        def fail_one_drop(sql):
            if sql.lstrip().upper().startswith("DROP "):
                attempted_drops.append(sql.strip())
            if sql.strip() == failed_drop:
                raise RuntimeError("injected persistent KV DROP failure")
            return execute(sql)

        with monkeypatch.context() as failure_patch:
            failure_patch.setattr(server, "_execute", fail_one_drop)
            with pytest.raises(RuntimeError, match="cleanup incomplete") as failure:
                client.delete_collection(name)

        assert "injected persistent KV DROP failure" in str(failure.value)
        assert attempted_drops.count(failed_drop) >= 2
        for table_name in (
            NamespaceCollectionNames.data_table_name(cid),
            NamespaceCollectionNames.logic_schema_table_name(cid),
            NamespaceCollectionNames.hot_table_name(cid),
        ):
            assert f"DROP TABLE IF EXISTS `{table_name}`" in attempted_drops
        assert f"DROP TABLEGROUP IF EXISTS `{NamespaceCollectionNames.tablegroup_name(cid)}`" in attempted_drops
        assert server._get_ns_collection_meta(name)["collection_id"] == cid
        assert server._table_exists(failed_table, strict=True)

        retry_client = _new_client_like(client)
        assert retry_client._server is not server
        retry_client.delete_collection(name)
        _assert_generation_removed(retry_client._server, cid)
        assert not retry_client.has_collection(name)
    finally:
        with contextlib.suppress(Exception):
            server._delete_ns_collection_meta(name, collection_id=cid)
        if retry_client is not None:
            with contextlib.suppress(Exception):
                retry_client.close()


def test_fixed_id_cleanup_survives_missing_catalog_and_same_name_recreate(oceanbase_client):
    """Repeated old-ID cleanup removes orphans without resolving the new object by name."""
    name = f"test_ns_orphan_cleanup_{uuid.uuid4().hex}"
    client = oceanbase_client
    server = client._server
    old_collection = client.create_collection(
        name=name,
        schema=_schema(),
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )
    old_id = old_collection.id
    owned_ids = [old_id]
    try:
        old_namespace = old_collection.create_namespace("same_namespace")
        old_namespace.add(ids="old", embeddings=[1.0, 2.0, 3.0])
        # Simulate a previous attempt that lost only this generation's root catalog.
        server._execute(f"DELETE FROM {server._qtable('sdk_collections')} WHERE collection_id = '{old_id}'")
        assert server._get_ns_collection_meta(name) is None
        assert server._table_exists(NamespaceCollectionNames.data_table_name(old_id), strict=True)

        new_collection = client.create_collection(
            name=name,
            schema=_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        new_id = new_collection.id
        owned_ids.append(new_id)
        assert new_id != old_id
        new_namespace = new_collection.create_namespace("same_namespace")
        new_namespace.add(ids="new", embeddings=[3.0, 2.0, 1.0])

        for attempt in range(2):
            server._delete_ns_collection_meta(name, collection_id=old_id)
            _assert_generation_removed(server, old_id)
            assert server._get_ns_collection_meta(name)["collection_id"] == new_id
            assert client.get_collection(name).id == new_id
            assert not server._is_incomplete_ns_collection(name)
            new_namespace.add(ids=f"after_cleanup_{attempt}", embeddings=[1.0, 1.0, 1.0])

        assert set(new_namespace.get(include=[])["ids"]) == {"new", "after_cleanup_0", "after_cleanup_1"}
        assert new_namespace.count() == 3
    finally:
        for cid in owned_ids:
            with contextlib.suppress(Exception):
                server._delete_ns_collection_meta(name, collection_id=cid)
