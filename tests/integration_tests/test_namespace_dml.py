"""
Namespace DML integration tests.
Tests namespace.add, update, upsert, delete, get, count, peek operations.
"""

import time

import pytest

from pyseekdb import IVFConfiguration
from pyseekdb.client.configuration import VectorIndexConfig
from pyseekdb.client.schema import Schema


class TestNamespaceDML:

    def _setup(self, client):
        name = f"test_ns_dml_{int(time.time() * 1000)}"
        schema = Schema(
            vector_index=VectorIndexConfig(
                ivf=IVFConfiguration(dimension=3, distance="l2"),
                embedding_function=None,
            ),
        )
        collection = client.create_collection(name=name, schema=schema, use_namespace=True)
        namespace = collection.create_namespace("dml_ns")
        return collection, namespace

    def test_add_single(self, db_client):
        collection, ns = self._setup(db_client)
        try:
            ns.add(ids="d1", embeddings=[1.0, 2.0, 3.0], documents="Hello", metadatas={"tag": "a"})
            result = ns.get(ids="d1")
            assert len(result["ids"]) == 1
            assert result["ids"][0] == "d1"
        finally:
            db_client.delete_collection(name=collection.name)

    def test_add_batch(self, db_client):
        collection, ns = self._setup(db_client)
        try:
            ns.add(
                ids=["d1", "d2", "d3"],
                embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                documents=["Doc A", "Doc B", "Doc C"],
                metadatas=[{"tag": "a"}, {"tag": "b"}, {"tag": "c"}],
            )
            result = ns.get(ids=["d1", "d2", "d3"])
            assert len(result["ids"]) == 3
        finally:
            db_client.delete_collection(name=collection.name)

    def test_get_by_id(self, db_client):
        collection, ns = self._setup(db_client)
        try:
            ns.add(
                ids=["g1", "g2"],
                embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                documents=["First", "Second"],
                metadatas=[{"k": 1}, {"k": 2}],
            )
            result = ns.get(ids="g1", include=["documents", "metadatas"])
            assert len(result["ids"]) == 1
            assert result["documents"][0] == "First"
            assert result["metadatas"][0]["k"] == 1
        finally:
            db_client.delete_collection(name=collection.name)

    def test_get_with_limit(self, db_client):
        collection, ns = self._setup(db_client)
        try:
            ns.add(
                ids=["l1", "l2", "l3"],
                embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            )
            result = ns.get(limit=2)
            assert len(result["ids"]) == 2
        finally:
            db_client.delete_collection(name=collection.name)

    def test_update_metadata(self, db_client):
        collection, ns = self._setup(db_client)
        try:
            ns.add(ids="u1", embeddings=[1.0, 2.0, 3.0], metadatas={"score": 10})
            ns.update(ids="u1", metadatas={"score": 99})
            result = ns.get(ids="u1", include=["metadatas"])
            assert result["metadatas"][0]["score"] == 99
        finally:
            db_client.delete_collection(name=collection.name)

    def test_update_document_and_embedding(self, db_client):
        collection, ns = self._setup(db_client)
        try:
            ns.add(ids="u2", embeddings=[1.0, 2.0, 3.0], documents="Original")
            ns.update(ids="u2", embeddings=[9.0, 8.0, 7.0], documents="Updated")
            result = ns.get(ids="u2", include=["documents", "embeddings"])
            assert result["documents"][0] == "Updated"
        finally:
            db_client.delete_collection(name=collection.name)

    def test_upsert_existing(self, db_client):
        collection, ns = self._setup(db_client)
        try:
            ns.add(ids="up1", embeddings=[1.0, 2.0, 3.0], metadatas={"v": 1})
            ns.upsert(ids="up1", embeddings=[4.0, 5.0, 6.0], metadatas={"v": 2})
            result = ns.get(ids="up1", include=["metadatas"])
            assert result["metadatas"][0]["v"] == 2
        finally:
            db_client.delete_collection(name=collection.name)

    def test_upsert_new(self, db_client):
        collection, ns = self._setup(db_client)
        try:
            ns.upsert(ids="up_new", embeddings=[1.0, 1.0, 1.0], metadatas={"fresh": True})
            result = ns.get(ids="up_new", include=["metadatas"])
            assert len(result["ids"]) == 1
            assert result["metadatas"][0]["fresh"] is True
        finally:
            db_client.delete_collection(name=collection.name)

    def test_delete_by_ids(self, db_client):
        collection, ns = self._setup(db_client)
        try:
            ns.add(
                ids=["del1", "del2"],
                embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            )
            ns.delete(ids="del1")
            result = ns.get(ids="del1")
            assert len(result["ids"]) == 0

            result = ns.get(ids="del2")
            assert len(result["ids"]) == 1
        finally:
            db_client.delete_collection(name=collection.name)

    def test_count(self, db_client):
        collection, ns = self._setup(db_client)
        try:
            assert ns.count() == 0
            ns.add(
                ids=["c1", "c2", "c3"],
                embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            )
            assert ns.count() == 3
        finally:
            db_client.delete_collection(name=collection.name)

    def test_peek(self, db_client):
        collection, ns = self._setup(db_client)
        try:
            ns.add(
                ids=["p1", "p2"],
                embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                documents=["Peek A", "Peek B"],
                metadatas=[{"x": 1}, {"x": 2}],
            )
            result = ns.peek(limit=10)
            assert len(result["ids"]) == 2
            assert "documents" in result
            assert "metadatas" in result
        finally:
            db_client.delete_collection(name=collection.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
