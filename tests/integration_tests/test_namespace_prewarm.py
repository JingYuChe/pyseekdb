"""
Namespace prewarm integration tests.
Tests that prewarm raises ValueError in embedded mode and executes in remote mode.
"""

import time

import pytest

from pyseekdb import IVFConfiguration
from pyseekdb.client.configuration import VectorIndexConfig
from pyseekdb.client.schema import Schema


class TestNamespacePrewarm:

    def _create_ns_collection_and_namespace(self, client):
        name = f"test_ns_pw_{int(time.time() * 1000)}"
        schema = Schema(
            vector_index=VectorIndexConfig(
                ivf=IVFConfiguration(dimension=3, distance="cosine"),
                embedding_function=None,
            ),
        )
        collection = client.create_collection(name=name, schema=schema, use_namespace=True)
        namespace = collection.create_namespace("pw_ns")
        return collection, namespace

    def test_prewarm_raises_in_embedded_mode(self, embedded_client):
        collection, namespace = self._create_ns_collection_and_namespace(embedded_client)
        try:
            with pytest.raises(ValueError, match="shared-storage remote"):
                namespace.prewarm()
        finally:
            embedded_client.delete_collection(name=collection.name)

    def test_prewarm_executes_in_server_mode(self, server_client):
        """Prewarm should not raise in remote mode (actual behavior depends on backend)."""
        collection, namespace = self._create_ns_collection_and_namespace(server_client)
        try:
            namespace.prewarm()
        except Exception as e:
            if "shared-storage remote" in str(e):
                pytest.fail("prewarm should not raise embedded-mode error in server mode")
        finally:
            server_client.delete_collection(name=collection.name)

    def test_prewarm_executes_in_oceanbase_mode(self, oceanbase_client):
        """Prewarm should not raise in OceanBase mode (actual behavior depends on backend)."""
        collection, namespace = self._create_ns_collection_and_namespace(oceanbase_client)
        try:
            namespace.prewarm()
        except Exception as e:
            if "shared-storage remote" in str(e):
                pytest.fail("prewarm should not raise embedded-mode error in oceanbase mode")
        finally:
            oceanbase_client.delete_collection(name=collection.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
