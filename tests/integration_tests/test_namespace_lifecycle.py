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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
