"""
Namespace lifecycle integration tests.
Tests collection creation with use_namespace=True, namespace CRUD, and collection deletion.
"""

import time

import pytest

import pyseekdb
from pyseekdb import IVFConfiguration
from pyseekdb.client.configuration import VectorIndexConfig
from pyseekdb.client.schema import Schema



class TestNamespaceLifecycle:

    def _create_ns_collection(self, client, suffix=""):
        name = f"test_ns_lc_{int(time.time() * 1000)}{suffix}"
        schema = Schema(
            vector_index=VectorIndexConfig(
                ivf=IVFConfiguration(dimension=3, distance="cosine"),
                embedding_function=None,
            ),
        )
        collection = client.create_collection(name=name, schema=schema, use_namespace=True)
        return collection

    def test_create_namespace_collection(self, db_client):
        collection = self._create_ns_collection(db_client)
        try:
            assert collection.use_namespace is True
            assert collection.dimension == 3
            assert collection.name.startswith("test_ns_lc_")
        finally:
            db_client.delete_collection(name=collection.name)

    def test_get_collection_preserves_namespace_flag(self, db_client):
        collection = self._create_ns_collection(db_client)
        try:
            retrieved = db_client.get_collection(collection.name)
            assert retrieved.use_namespace is True
            assert retrieved.dimension == collection.dimension
        finally:
            db_client.delete_collection(name=collection.name)

    def test_create_and_get_namespace(self, db_client):
        collection = self._create_ns_collection(db_client)
        try:
            ns = collection.create_namespace("ns_a")
            assert ns.name == "ns_a"
            assert ns.namespace_id is not None

            ns2 = collection.get_namespace("ns_a")
            assert ns2.name == "ns_a"
            assert ns2.namespace_id == ns.namespace_id
        finally:
            db_client.delete_collection(name=collection.name)

    def test_get_or_create_namespace(self, db_client):
        collection = self._create_ns_collection(db_client)
        try:
            ns1 = collection.get_or_create_namespace("ns_goc")
            assert ns1.name == "ns_goc"

            ns2 = collection.get_or_create_namespace("ns_goc")
            assert ns2.namespace_id == ns1.namespace_id
        finally:
            db_client.delete_collection(name=collection.name)

    def test_has_namespace(self, db_client):
        collection = self._create_ns_collection(db_client)
        try:
            assert collection.has_namespace("nonexistent") is False
            collection.create_namespace("ns_check")
            assert collection.has_namespace("ns_check") is True
        finally:
            db_client.delete_collection(name=collection.name)

    def test_list_namespaces(self, db_client):
        collection = self._create_ns_collection(db_client)
        try:
            collection.create_namespace("ns_x")
            collection.create_namespace("ns_y")
            ns_list = collection.list_namespaces()
            names = {ns.name for ns in ns_list}
            assert "ns_x" in names
            assert "ns_y" in names
            assert len(ns_list) >= 2
        finally:
            db_client.delete_collection(name=collection.name)

    def test_delete_namespace(self, db_client):
        collection = self._create_ns_collection(db_client)
        try:
            collection.create_namespace("ns_del")
            assert collection.has_namespace("ns_del") is True

            collection.delete_namespace("ns_del")
            assert collection.has_namespace("ns_del") is False
        finally:
            db_client.delete_collection(name=collection.name)

    def test_get_nonexistent_namespace_raises(self, db_client):
        collection = self._create_ns_collection(db_client)
        try:
            with pytest.raises(ValueError):
                collection.get_namespace("does_not_exist")
        finally:
            db_client.delete_collection(name=collection.name)

    def test_delete_collection_cleans_namespaces(self, db_client):
        collection = self._create_ns_collection(db_client)
        coll_name = collection.name
        collection.create_namespace("ns_cleanup")
        db_client.delete_collection(name=coll_name)

        assert db_client.has_collection(coll_name) is False

    def test_collection_data_api_blocked_when_namespace_enabled(self, db_client):
        collection = self._create_ns_collection(db_client)
        try:
            with pytest.raises(ValueError, match="namespace enabled"):
                collection.add(ids="1", embeddings=[1.0, 2.0, 3.0])
        finally:
            db_client.delete_collection(name=collection.name)

    def test_namespace_on_non_namespace_collection_raises(self, db_client):
        name = f"test_nons_{int(time.time() * 1000)}"
        collection = db_client.create_collection(
            name=name,
            configuration=pyseekdb.HNSWConfiguration(dimension=3),
            embedding_function=None,
        )
        try:
            with pytest.raises(ValueError, match="not enabled"):
                collection.create_namespace("ns1")
        finally:
            try:
                db_client.delete_collection(name=name)
            except Exception:
                db_client._server._execute(f"DROP TABLE IF EXISTS `{name}`")
                db_client._server._execute(
                    f"DELETE FROM `sdk_collections` WHERE COLLECTION_NAME = '{name}'"
                )


    def test_create_namespace_collection_with_hnsw_raises(self, db_client):
        from pyseekdb.client.configuration import HNSWConfiguration
        name = f"test_ns_hnsw_{int(time.time() * 1000)}"
        schema = Schema(
            vector_index=VectorIndexConfig(
                hnsw=HNSWConfiguration(dimension=3),
                embedding_function=None,
            ),
        )
        with pytest.raises(ValueError, match="HNSW is not allowed"):
            db_client.create_collection(name=name, schema=schema, use_namespace=True)

    def test_ss_mode_creates_hot_table(self, oceanbase_client):
        """Verify hot_table is created when _is_shared_storage_mode returns True (SS mode)."""
        import json
        from unittest.mock import patch
        from pyseekdb.client.meta_info import NamespaceCollectionNames

        client = oceanbase_client

        with patch.object(
            type(client._server), "_is_shared_storage_mode", return_value=True
        ):
            name = f"test_ns_ss_{int(time.time() * 1000)}"
            schema = Schema(
                vector_index=VectorIndexConfig(
                    ivf=IVFConfiguration(dimension=3, distance="cosine"),
                    embedding_function=None,
                ),
            )
            collection = client.create_collection(
                name=name, schema=schema, use_namespace=True
            )

        try:
            collection_id = collection.id

            # Verify settings has storage_mode=ss
            rows = client._server._execute(
                f"SELECT settings FROM sdk_collections WHERE collection_id = '{collection_id}'"
            )
            settings = json.loads(rows[0]["settings"])
            assert settings["storage_mode"] == "ss", f"Expected ss, got {settings.get('storage_mode')}"

            # Verify hot_table was actually created
            hot_table = NamespaceCollectionNames.hot_table_name(collection_id)
            rows = client._server._execute(f"DESCRIBE `{hot_table}`")
            assert rows is not None and len(rows) > 0, "hot_table should exist in SS mode"

            col_names = {r["Field"] for r in rows}
            assert "namespace_id" in col_names
            assert "last_access_time" in col_names
        finally:
            client.delete_collection(name=name)


    def test_custom_namespace_partition_count(self, oceanbase_client):
        """Verify set_namespace_partition_count controls the PARTITIONS clause."""
        from pyseekdb import get_namespace_partition_count, set_namespace_partition_count
        from pyseekdb.client.meta_info import NamespaceCollectionNames

        original = get_namespace_partition_count()
        try:
            set_namespace_partition_count(4)
            collection = self._create_ns_collection(oceanbase_client, suffix="_pc")
            try:
                data_table = NamespaceCollectionNames.data_table_name(collection.id)
                rows = oceanbase_client._server._execute(f"SHOW CREATE TABLE `{data_table}`")
                create_sql = rows[0].get("Create Table", "") if rows else ""
                import re
                partitions = re.findall(r"partition `p\d+`", create_sql)
                assert len(partitions) == 4, (
                    f"Expected 4 partitions, found {len(partitions)}: {create_sql[-300:]}"
                )
            finally:
                oceanbase_client.delete_collection(name=collection.name)
        finally:
            set_namespace_partition_count(original)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
