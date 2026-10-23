"""Read raw catalog JSON through the real v2 and namespace routing methods."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from pyseekdb.client.client_base import BaseClient, _CollectionMeta  # noqa: E402
from pyseekdb.client.meta_info import CollectionNames, NamespaceCollectionNames  # noqa: E402
from tests.unit_tests.test_namespace import FakeClient  # noqa: E402


class CatalogRoutingClient(FakeClient):
    """Return driver-shaped SQL rows without replacing either collection getter."""

    get_collection = BaseClient.get_collection

    def __init__(self, raw_settings, row_shape, *, first_namespace_probe_missing=False):
        super().__init__()
        self.raw_settings = raw_settings
        self.row_shape = row_shape
        self.collection_id = "a" * 32
        self.collection_name = "items"
        self.first_namespace_probe_missing = first_namespace_probe_missing
        self.namespace_probes = 0
        self.v2_catalog_reads = 0
        self.forbid_catalog_read_after_v2 = False
        self.missing_table = None

    def _catalog_row(self, *, uppercase):
        values = (self.collection_id, self.collection_name, self.raw_settings)
        if self.row_shape == "tuple":
            return values
        keys = ("collection_id", "collection_name", "settings")
        if uppercase:
            keys = tuple(key.upper() for key in keys)
        return dict(zip(keys, values, strict=True))

    def _execute(self, sql, params=None):
        assert params is None
        self.executed_sqls.append(sql)
        if "sdk_collections" in sql and self.forbid_catalog_read_after_v2 and self.v2_catalog_reads:
            raise AssertionError("The captured generation must not be resolved by name again")
        if sql.startswith("SELECT collection_id, collection_name, settings "):
            self.namespace_probes += 1
            if self.first_namespace_probe_missing and self.namespace_probes == 1:
                return []
            return [self._catalog_row(uppercase=False)]
        if sql.startswith("SELECT COLLECTION_ID, COLLECTION_NAME, SETTINGS "):
            self.v2_catalog_reads += 1
            return [self._catalog_row(uppercase=True)]
        if sql.startswith("SELECT settings FROM "):
            return [{"settings": self.raw_settings}] if self.row_shape == "dict" else [(self.raw_settings,)]
        if sql.startswith("DESCRIBE "):
            if CollectionNames.table_name_v2(self.collection_id) not in sql:
                raise RuntimeError(1146, "legacy collection table does not exist")
            return [{"Field": "embedding", "Type": "vector(3)"}]
        if sql.startswith("SHOW CREATE TABLE "):
            return [
                {
                    "Create Table": "CREATE TABLE items (embedding vector(3), VECTOR INDEX idx_vec "
                    "(embedding) WITH(distance=cosine))"
                }
            ]
        if "information_schema.TABLES" in sql or "DBA_OB_TABLEGROUPS" in sql:
            return [] if self.missing_table and self.missing_table in sql else [(1,)]
        if sql.startswith("USE "):
            return None
        raise AssertionError(f"Unexpected SQL in collection getter: {sql}")


@pytest.mark.parametrize("row_shape", ["dict", "tuple"])
@pytest.mark.parametrize("raw_settings", ["{}", '{"version": 2}', '{"use_namespace": false}', None])
@pytest.mark.parametrize("entrypoint", ["get_collection", "_get_collection_v2"])
def test_standard_v2_reads_raw_catalog_settings(row_shape, raw_settings, entrypoint):
    client = CatalogRoutingClient(raw_settings, row_shape)
    client._resolve_embedding_function = MagicMock(wraps=client._resolve_embedding_function)
    client._resolve_sparse_vector_index_config = MagicMock(wraps=client._resolve_sparse_vector_index_config)

    collection = getattr(client, entrypoint)("items", embedding_function=None)

    assert collection.id == client.collection_id
    assert collection._use_namespace is False
    assert collection.dimension == 3
    assert client.v2_catalog_reads == 1
    client._resolve_embedding_function.assert_called_once_with(raw_settings)
    client._resolve_sparse_vector_index_config.assert_called_once_with(raw_settings)
    # Keep the established raw str|None contract for all other catalog consumers.
    assert _CollectionMeta.from_row(client._catalog_row(uppercase=True)).settings == raw_settings


@pytest.mark.parametrize("row_shape", ["dict", "tuple"])
@pytest.mark.parametrize("entrypoint", ["get_collection", "_get_collection_v2"])
def test_empty_catalog_string_retains_existing_json_validation(row_shape, entrypoint):
    """An empty string is invalid JSON; do not broaden the owner helper's contract."""
    client = CatalogRoutingClient("", row_shape)

    with pytest.raises(ValueError):
        getattr(client, entrypoint)("items", embedding_function=None)


@pytest.mark.parametrize("row_shape", ["dict", "tuple"])
@pytest.mark.parametrize("missing_main_table", [False, True])
def test_namespace_appearing_after_initial_probe_uses_captured_json_generation(row_shape, missing_main_table):
    settings = {
        "use_namespace": True,
        "storage_mode": "ss",
        "dimension": 3,
        "distance": "cosine",
        "partition_count": 4,
        "dense_index_type": "ivf",
    }
    client = CatalogRoutingClient(json.dumps(settings), row_shape, first_namespace_probe_missing=True)
    client.forbid_catalog_read_after_v2 = True
    client._get_ready_ns_collection = MagicMock(wraps=client._get_ready_ns_collection)
    client._resolve_embedding_function = MagicMock(wraps=client._resolve_embedding_function)
    if missing_main_table:
        client.missing_table = NamespaceCollectionNames.data_table_name(client.collection_id)
        with pytest.raises(ValueError, match="not ready or incomplete"):
            client.get_collection("items", embedding_function=None)
    else:
        collection = client.get_collection("items", embedding_function=None)
        assert collection.id == client.collection_id
        assert collection._use_namespace is True
        assert collection.dimension == 3

    assert client.namespace_probes == 1
    assert client.v2_catalog_reads == 1
    client._get_ready_ns_collection.assert_called_once()
    captured_meta = client._get_ready_ns_collection.call_args.args[0]
    assert captured_meta["settings"] == settings
    assert isinstance(captured_meta["settings"], dict)
    assert captured_meta["collection_id"] == client.collection_id
    client._resolve_embedding_function.assert_not_called()
    assert all(sql.startswith(("SELECT ", "DESCRIBE ", "SHOW ", "USE ")) for sql in client.executed_sqls)


@pytest.mark.parametrize("row_shape", ["dict", "tuple"])
@pytest.mark.parametrize("raw_settings", ["{broken", "not json"])
def test_v2_rejects_malformed_catalog_json(row_shape, raw_settings):
    client = CatalogRoutingClient(raw_settings, row_shape)

    with pytest.raises(ValueError):
        client._get_collection_v2("items", embedding_function=None)


@pytest.mark.parametrize("row_shape", ["dict", "tuple"])
@pytest.mark.parametrize("raw_settings", ["[]", "null", "true", "1", '"text"'])
def test_v2_rejects_non_object_catalog_json(row_shape, raw_settings):
    client = CatalogRoutingClient(raw_settings, row_shape)

    with pytest.raises(ValueError, match="expected a JSON object"):
        client._get_collection_v2("items", embedding_function=None)
