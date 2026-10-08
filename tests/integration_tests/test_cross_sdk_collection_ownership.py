"""A JSON Table SDK catalog row must not be adopted by pyseekdb."""

import json
import uuid

import pytest

from pyseekdb import HNSWConfiguration, IVFConfiguration
from pyseekdb.client.configuration import VectorIndexConfig
from pyseekdb.client.schema import Schema


def test_json_table_collection_name_is_reserved_for_its_owner(oceanbase_client):
    client = oceanbase_client
    backend = client._server
    backend._create_sdk_collections_if_not_exists()
    name = f"cross_sdk_json_{uuid.uuid4().hex[:12]}"
    collection_id = uuid.uuid4().hex
    settings = json.dumps({"version": 1, "sdk_type": "json-table-sdk", "indexes": ["json_search"]})
    backend._execute_catalog(
        f"INSERT INTO sdk_collections (collection_id, collection_name, settings) "
        f"VALUES ('{collection_id}', '{name}', '{settings}')"
    )
    schema = Schema(
        vector_index=VectorIndexConfig(
            ivf=IVFConfiguration(dimension=3, distance="l2"), embedding_function=None
        )
    )
    try:
        with pytest.raises(ValueError, match="belongs to JSON Table SDK"):
            client.create_collection(name, schema=schema, use_namespace=True, partition_count=1)
        with pytest.raises(ValueError, match="belongs to JSON Table SDK"):
            client.create_collection(name, configuration=HNSWConfiguration(dimension=3), embedding_function=None)
        with pytest.raises(ValueError, match="belongs to JSON Table SDK"):
            client.get_or_create_collection(name, schema=schema, use_namespace=True)
        with pytest.raises(ValueError, match="belongs to JSON Table SDK"):
            client.get_collection(name, embedding_function=None)
        with pytest.raises(ValueError, match="belongs to JSON Table SDK"):
            client.delete_collection(name)
        rows = backend._execute_catalog(
            f"SELECT settings FROM sdk_collections WHERE collection_id = '{collection_id}'"
        )
        assert rows and json.loads(rows[0]["settings"])["sdk_type"] == "json-table-sdk"
    finally:
        backend._execute_catalog(f"DELETE FROM sdk_collections WHERE collection_id = '{collection_id}'")
