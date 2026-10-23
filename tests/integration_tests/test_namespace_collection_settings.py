"""Real JSON catalog rows must preserve standard/namespace collection dispatch."""

from __future__ import annotations

import contextlib
import json
import uuid

from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT

import pyseekdb
from pyseekdb import HNSWConfiguration, IVFConfiguration, Schema, VectorIndexConfig


def _standard_schema():
    return Schema(
        fulltext_index=None,
        vector_index=VectorIndexConfig(
            hnsw=HNSWConfiguration(dimension=3, distance="l2"),
            embedding_function=None,
        ),
    )


def _namespace_schema():
    return Schema(
        fulltext_index=None,
        vector_index=VectorIndexConfig(
            ivf=IVFConfiguration(dimension=3, distance="cosine", centroids_fresh_mode="spfresh"),
            embedding_function=None,
        ),
    )


def _new_client_like(client):
    """Reopen the same endpoint and database with an independent SDK connection."""
    server = client._server
    return pyseekdb.Client(
        host=server.host,
        port=server.port,
        tenant=server.tenant,
        database=server.database,
        user=server.user,
        password=server.password,
        charset=server.charset,
        **server.kwargs,
    )


def _read_raw_settings(client, collection_id):
    server = client._server
    rows = server._execute(
        f"SELECT settings FROM {server._qtable('sdk_collections')} WHERE collection_id = '{collection_id}'"
    )
    assert len(rows) == 1
    row = rows[0]
    settings = row["settings"] if isinstance(row, dict) else row[0]
    assert isinstance(settings, str), f"Expected the driver's JSON string, got {type(settings).__name__}"
    return settings


def test_standard_v2_get_accepts_json_string_settings_and_reconnects(oceanbase_client):
    """Explicit v2 get works with raw JSON text and its handle supports actual DML."""
    client = oceanbase_client
    name = f"test_catalog_standard_{uuid.uuid4().hex[:16]}"
    created = client.create_collection(name=name, schema=_standard_schema(), _collection_version=2)
    fresh_client = None
    try:
        raw_settings = _read_raw_settings(client, created.id)
        assert not json.loads(raw_settings).get("use_namespace", False)

        reopened = client.get_collection(name, embedding_function=None)
        assert reopened.use_namespace is False
        assert reopened.id == created.id
        assert reopened.dimension == 3
        assert reopened.embedding_function is None
        reopened.add(ids="original", embeddings=[1.0, 2.0, 3.0], metadatas={"owner": "standard"})
        result = reopened.get(ids=["original"], include=["metadatas"])
        assert result["ids"] == ["original"]
        assert result["metadatas"] == [{"owner": "standard"}]

        fresh_client = _new_client_like(client)
        assert fresh_client._server is not client._server
        fresh = fresh_client.get_collection(name, embedding_function=None)
        assert fresh.use_namespace is False
        assert fresh.id == created.id
        assert fresh.get(ids=["original"], include=["metadatas"]) == result
        fresh.add(ids="reconnected", embeddings=[3.0, 2.0, 1.0])
        assert set(reopened.get(include=[])["ids"]) == {"original", "reconnected"}
    finally:
        if fresh_client is not None:
            with contextlib.suppress(Exception):
                fresh_client.close()
        with contextlib.suppress(Exception):
            client.delete_collection(name)


def test_mixed_standard_and_namespace_catalog_types_and_data_are_isolated(oceanbase_client):
    """The shared catalog restores both collection types without mixing their data."""
    client = oceanbase_client
    suffix = uuid.uuid4().hex[:16]
    standard_name = f"test_catalog_mixed_standard_{suffix}"
    namespace_name = f"test_catalog_mixed_namespace_{suffix}"
    created_names = []
    fresh_client = None
    try:
        standard = client.create_collection(name=standard_name, schema=_standard_schema(), _collection_version=2)
        created_names.append(standard_name)
        namespace_collection = client.create_collection(
            name=namespace_name,
            schema=_namespace_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        created_names.append(namespace_name)
        namespace_collection.create_namespace("records")
        assert standard.id != namespace_collection.id
        assert not json.loads(_read_raw_settings(client, standard.id)).get("use_namespace", False)
        assert json.loads(_read_raw_settings(client, namespace_collection.id))["use_namespace"] is True

        standard = client.get_collection(standard_name, embedding_function=None)
        namespace_collection = client.get_collection(namespace_name, embedding_function=None)
        assert standard.use_namespace is False
        assert namespace_collection.use_namespace is True
        namespace = namespace_collection.get_namespace("records")
        standard.add(ids="shared_id", embeddings=[1.0, 2.0, 3.0], metadatas={"owner": "standard"})
        namespace.add(ids="shared_id", embeddings=[3.0, 2.0, 1.0], metadatas={"owner": "namespace"})

        fresh_client = _new_client_like(client)
        fresh_standard = fresh_client.get_collection(standard_name, embedding_function=None)
        fresh_namespace_collection = fresh_client.get_collection(namespace_name, embedding_function=None)
        assert fresh_standard.use_namespace is False
        assert fresh_standard.id == standard.id
        assert fresh_namespace_collection.use_namespace is True
        assert fresh_namespace_collection.id == namespace_collection.id
        fresh_namespace = fresh_namespace_collection.get_namespace("records")
        standard_result = fresh_standard.get(ids=["shared_id"], include=["metadatas"])
        namespace_result = fresh_namespace.get(ids=["shared_id"], include=["metadatas"])
        assert standard_result["ids"] == namespace_result["ids"] == ["shared_id"]
        assert standard_result["metadatas"] == [{"owner": "standard"}]
        assert namespace_result["metadatas"] == [{"owner": "namespace"}]

        fresh_standard.add(ids="standard_only", embeddings=[1.0, 1.0, 1.0])
        assert set(standard.get(include=[])["ids"]) == {"shared_id", "standard_only"}
        assert namespace.get(include=[])["ids"] == ["shared_id"]
    finally:
        if fresh_client is not None:
            with contextlib.suppress(Exception):
                fresh_client.close()
        for name in reversed(created_names):
            with contextlib.suppress(Exception):
                client.delete_collection(name)


def test_v2_fallback_decodes_namespace_settings_and_uses_captured_id(oceanbase_client, monkeypatch):
    """A missed initial namespace probe still dispatches from the real v2 JSON row."""
    client = oceanbase_client
    name = f"test_catalog_fallback_{uuid.uuid4().hex[:16]}"
    created = client.create_collection(
        name=name,
        schema=_namespace_schema(),
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )
    reader = None
    try:
        namespace = created.create_namespace("records")
        namespace.add(ids="before_fallback", embeddings=[1.0, 2.0, 3.0])
        reader = _new_client_like(client)
        server = reader._server
        probe_names = []
        captured_metadata = []
        executed_sql = []
        resolve_metadata = server._resolve_collection_metadata_from_sdk_collections
        execute = server._execute

        def miss_initial_probe(collection_name):
            probe_names.append(collection_name)
            assert probe_names == [name], "Namespace dispatch must not resolve the name again"
            return None

        def capture_real_metadata(collection_name):
            meta = resolve_metadata(collection_name)
            assert meta is not None
            assert isinstance(meta.settings, str)
            captured_metadata.append(meta)
            return meta

        def record_sql(sql):
            executed_sql.append(sql)
            return execute(sql)

        with monkeypatch.context() as fallback_patch:
            fallback_patch.setattr(server, "_get_ns_collection_meta", miss_initial_probe)
            fallback_patch.setattr(server, "_resolve_collection_metadata_from_sdk_collections", capture_real_metadata)
            fallback_patch.setattr(server, "_execute", record_sql)
            reopened = reader.get_collection(name, embedding_function=None)

        assert probe_names == [name]
        assert len(captured_metadata) == 1
        assert captured_metadata[0].collection_id == reopened.id == created.id
        assert isinstance(captured_metadata[0].settings, str)
        assert reopened.use_namespace is True
        assert reopened.dimension == 3
        catalog_reads = [sql for sql in executed_sql if "sdk_collections" in sql.lower()]
        assert len(catalog_reads) == 1, f"Unexpected catalog re-resolution: {catalog_reads!r}"
        assert all(sql.lstrip().upper().startswith(("SELECT ", "SHOW ", "USE ", "DESCRIBE ")) for sql in executed_sql)

        reopened_namespace = reopened.get_namespace("records")
        assert reopened_namespace.get(include=[])["ids"] == ["before_fallback"]
        reopened_namespace.add(ids="after_fallback", embeddings=[3.0, 2.0, 1.0])
        assert set(namespace.get(include=[])["ids"]) == {"before_fallback", "after_fallback"}
    finally:
        if reader is not None:
            with contextlib.suppress(Exception):
                reader.close()
        with contextlib.suppress(Exception):
            client.delete_collection(name)
