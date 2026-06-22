"""
Unit tests for namespace-related classes:
- IVFConfiguration validation
- VectorIndexConfig mutual exclusion
- Collection namespace guard
- Namespace object properties
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from pyseekdb import IVFConfiguration  # noqa: E402
from pyseekdb.client.client_base import BaseClient  # noqa: E402
from pyseekdb.client.collection import Collection  # noqa: E402
from pyseekdb.client.configuration import HNSWConfiguration, IVFIndexType, VectorIndexConfig  # noqa: E402
from pyseekdb.client.namespace import Namespace  # noqa: E402


# ==================== IVFConfiguration Tests ====================


class TestIVFConfiguration:

    def test_valid_defaults(self):
        config = IVFConfiguration()
        assert config.dimension == 384
        assert config.distance == "cosine"
        assert config.centroids_fresh_mode is None
        assert config.properties is None

    def test_valid_custom(self):
        config = IVFConfiguration(dimension=128, distance="l2", centroids_fresh_mode="spfresh")
        assert config.dimension == 128
        assert config.distance == "l2"
        assert config.centroids_fresh_mode == "spfresh"

    def test_valid_inner_product(self):
        config = IVFConfiguration(dimension=1024, distance="inner_product")
        assert config.distance == "inner_product"

    def test_dimension_boundary_min(self):
        config = IVFConfiguration(dimension=1)
        assert config.dimension == 1

    def test_dimension_boundary_max(self):
        config = IVFConfiguration(dimension=4096)
        assert config.dimension == 4096

    def test_invalid_dimension_zero(self):
        with pytest.raises(ValueError, match="must be between"):
            IVFConfiguration(dimension=0)

    def test_invalid_dimension_negative(self):
        with pytest.raises(ValueError, match="must be between"):
            IVFConfiguration(dimension=-1)

    def test_invalid_dimension_too_large(self):
        with pytest.raises(ValueError, match="must be between"):
            IVFConfiguration(dimension=4097)

    def test_invalid_dimension_type(self):
        with pytest.raises(TypeError, match="dimension must be an integer"):
            IVFConfiguration(dimension="128")

    def test_invalid_dimension_bool(self):
        with pytest.raises(TypeError, match="dimension must be an integer"):
            IVFConfiguration(dimension=True)

    def test_invalid_distance(self):
        with pytest.raises(ValueError, match="distance must be one of"):
            IVFConfiguration(distance="invalid")

    def test_invalid_centroids_fresh_mode_type(self):
        with pytest.raises(TypeError, match="centroids_fresh_mode must be a str"):
            IVFConfiguration(centroids_fresh_mode=True)

    def test_properties_valid(self):
        config = IVFConfiguration(properties={"nlist": 128, "nprobe": 16})
        assert config.properties["nlist"] == 128

    def test_valid_type_default(self):
        config = IVFConfiguration()
        assert config.type == "ivf_flat"

    def test_valid_type_ivf_sq8(self):
        config = IVFConfiguration(type="ivf_sq8")
        assert config.type == "ivf_sq8"

    def test_valid_type_ivf_pq(self):
        config = IVFConfiguration(type="ivf_pq")
        assert config.type == "ivf_pq"

    def test_valid_type_enum(self):
        config = IVFConfiguration(type=IVFIndexType.IVF_SQ8)
        assert config.type == "ivf_sq8"

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="type must be one of"):
            IVFConfiguration(type="invalid")

    def test_valid_lib_default(self):
        config = IVFConfiguration()
        assert config.lib == "ob"

    def test_valid_lib_vsag(self):
        config = IVFConfiguration(lib="vsag")
        assert config.lib == "vsag"

    def test_valid_lib_enum(self):
        from pyseekdb.client.configuration import IVFIndexLib
        config = IVFConfiguration(lib=IVFIndexLib.VSAG)
        assert config.lib == "vsag"

    def test_invalid_lib(self):
        with pytest.raises(ValueError, match="lib must be one of"):
            IVFConfiguration(lib="invalid")

    def test_properties_invalid_nested(self):
        with pytest.raises(TypeError):
            IVFConfiguration(properties={"bad": {"nested": True}})


# ==================== VectorIndexConfig Tests ====================


class TestVectorIndexConfig:

    def test_ivf_only(self):
        ivf = IVFConfiguration(dimension=128)
        config = VectorIndexConfig(ivf=ivf, embedding_function=None)
        assert config.ivf is ivf
        assert config.hnsw is None

    def test_hnsw_only(self):
        hnsw = HNSWConfiguration(dimension=128)
        config = VectorIndexConfig(hnsw=hnsw, embedding_function=None)
        assert config.hnsw is hnsw
        assert config.ivf is None

    def test_ivf_and_hnsw_mutually_exclusive(self):
        ivf = IVFConfiguration(dimension=128)
        hnsw = HNSWConfiguration(dimension=128)
        with pytest.raises(ValueError, match="Only one of ivf or hnsw"):
            VectorIndexConfig(ivf=ivf, hnsw=hnsw, embedding_function=None)

    def test_neither_ivf_nor_hnsw(self):
        config = VectorIndexConfig(embedding_function=None)
        assert config.ivf is None
        assert config.hnsw is None


# ==================== Collection Namespace Guard Tests ====================


class TestCollectionNamespaceGuard:

    def _make_collection(self, use_namespace: bool) -> Collection:
        mock_client = MagicMock()
        return Collection(
            client=mock_client,
            name="test_coll",
            collection_id="1",
            dimension=3,
            use_namespace=use_namespace,
        )

    def test_guard_blocks_add_when_namespace_enabled(self):
        coll = self._make_collection(use_namespace=True)
        with pytest.raises(ValueError, match="namespace enabled"):
            coll.add(ids="1", embeddings=[1.0, 2.0, 3.0])

    def test_guard_blocks_update_when_namespace_enabled(self):
        coll = self._make_collection(use_namespace=True)
        with pytest.raises(ValueError, match="namespace enabled"):
            coll.update(ids="1", metadatas={"k": "v"})

    def test_guard_blocks_upsert_when_namespace_enabled(self):
        coll = self._make_collection(use_namespace=True)
        with pytest.raises(ValueError, match="namespace enabled"):
            coll.upsert(ids="1", embeddings=[1.0, 2.0, 3.0])

    def test_guard_blocks_delete_when_namespace_enabled(self):
        coll = self._make_collection(use_namespace=True)
        with pytest.raises(ValueError, match="namespace enabled"):
            coll.delete(ids="1")

    def test_guard_blocks_query_when_namespace_enabled(self):
        coll = self._make_collection(use_namespace=True)
        with pytest.raises(ValueError, match="namespace enabled"):
            coll.query(query_embeddings=[1.0, 2.0, 3.0])

    def test_guard_blocks_get_when_namespace_enabled(self):
        coll = self._make_collection(use_namespace=True)
        with pytest.raises(ValueError, match="namespace enabled"):
            coll.get(ids="1")

    def test_guard_blocks_count_when_namespace_enabled(self):
        coll = self._make_collection(use_namespace=True)
        with pytest.raises(ValueError, match="namespace enabled"):
            coll.count()

    def test_guard_blocks_peek_when_namespace_enabled(self):
        coll = self._make_collection(use_namespace=True)
        with pytest.raises(ValueError, match="namespace enabled"):
            coll.peek()


# ==================== Collection Namespace Management Guard Tests ====================


class TestCollectionNamespaceManagementGuard:

    def _make_collection(self, use_namespace: bool) -> Collection:
        mock_client = MagicMock()
        return Collection(
            client=mock_client,
            name="test_coll",
            collection_id="1",
            dimension=3,
            use_namespace=use_namespace,
        )

    def test_create_namespace_requires_namespace_enabled(self):
        coll = self._make_collection(use_namespace=False)
        with pytest.raises(ValueError, match="not enabled"):
            coll.create_namespace("ns1")

    def test_get_namespace_requires_namespace_enabled(self):
        coll = self._make_collection(use_namespace=False)
        with pytest.raises(ValueError, match="not enabled"):
            coll.get_namespace("ns1")

    def test_get_or_create_namespace_requires_namespace_enabled(self):
        coll = self._make_collection(use_namespace=False)
        with pytest.raises(ValueError, match="not enabled"):
            coll.get_or_create_namespace("ns1")

    def test_delete_namespace_requires_namespace_enabled(self):
        coll = self._make_collection(use_namespace=False)
        with pytest.raises(ValueError, match="not enabled"):
            coll.delete_namespace("ns1")

    def test_list_namespaces_requires_namespace_enabled(self):
        coll = self._make_collection(use_namespace=False)
        with pytest.raises(ValueError, match="not enabled"):
            coll.list_namespaces()

    def test_has_namespace_requires_namespace_enabled(self):
        coll = self._make_collection(use_namespace=False)
        with pytest.raises(ValueError, match="not enabled"):
            coll.has_namespace("ns1")


# ==================== Namespace Object Tests ====================


class TestNamespaceObject:

    def _make_namespace(self) -> Namespace:
        mock_client = MagicMock()
        mock_ef = MagicMock()
        coll = Collection(
            client=mock_client,
            name="test_coll",
            collection_id="1",
            dimension=3,
            embedding_function=mock_ef,
            use_namespace=True,
        )
        return Namespace(client=mock_client, collection=coll, name="ns1", namespace_id="100")

    def test_name_property(self):
        ns = self._make_namespace()
        assert ns.name == "ns1"

    def test_namespace_id_property(self):
        ns = self._make_namespace()
        assert ns.namespace_id == "100"

    def test_collection_property(self):
        ns = self._make_namespace()
        assert ns.collection.name == "test_coll"

    def test_embedding_function_inherited(self):
        ns = self._make_namespace()
        assert ns.embedding_function is ns.collection.embedding_function

    def test_repr(self):
        ns = self._make_namespace()
        r = repr(ns)
        assert "ns1" in r
        assert "100" in r
        assert "test_coll" in r

    def test_add_delegates_to_client(self):
        ns = self._make_namespace()
        ns.add(ids="d1", embeddings=[1.0, 2.0, 3.0])
        ns._client._namespace_add.assert_called_once()

    def test_update_delegates_to_client(self):
        ns = self._make_namespace()
        ns.update(ids="d1", metadatas={"k": "v"})
        ns._client._namespace_update.assert_called_once()

    def test_upsert_delegates_to_client(self):
        ns = self._make_namespace()
        ns.upsert(ids="d1", embeddings=[1.0, 2.0, 3.0])
        ns._client._namespace_upsert.assert_called_once()

    def test_delete_delegates_to_client(self):
        ns = self._make_namespace()
        ns.delete(ids="d1")
        ns._client._namespace_delete.assert_called_once()

    def test_query_delegates_to_client(self):
        ns = self._make_namespace()
        ns._client._namespace_query.return_value = {"ids": [["d1"]]}
        ns.query(query_embeddings=[1.0, 2.0, 3.0])
        ns._client._namespace_query.assert_called_once()

    def test_get_delegates_to_client(self):
        ns = self._make_namespace()
        ns._client._namespace_get.return_value = {"ids": ["d1"]}
        ns.get(ids="d1")
        ns._client._namespace_get.assert_called_once()

    def test_count_delegates_to_client(self):
        ns = self._make_namespace()
        ns._client._namespace_count.return_value = 42
        assert ns.count() == 42

    def test_peek_delegates_to_client(self):
        ns = self._make_namespace()
        ns._client._namespace_peek.return_value = {"ids": ["d1"]}
        ns.peek(limit=5)
        ns._client._namespace_peek.assert_called_once()

    def test_prewarm_delegates_to_client(self):
        ns = self._make_namespace()
        ns.prewarm()
        ns._client._namespace_prewarm.assert_called_once()


# ==================== Namespace SQL Generation Tests ====================


class FakeClient(BaseClient):
    """Concrete BaseClient subclass that captures SQL without executing."""

    def __init__(self):
        self.database = "test"
        self.executed_sqls = []
        self.query_sqls = []
        self.query_return_value = []

    def _ensure_connection(self):
        mock_conn = MagicMock()
        client = self

        class CaptureCursor:
            def execute(self, sql, params=None):
                resolved = sql
                if params:
                    for p in params:
                        resolved = resolved.replace("%s", repr(p), 1)
                if os.environ.get("PYSEEKDB_PRINT_SQL", "").lower() in ("1", "true", "yes"):
                    print(f"[pyseekdb SQL] {resolved}", flush=True)
                client.executed_sqls.append(resolved)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def close(self):
                pass

        mock_conn.cursor.return_value = CaptureCursor()
        return mock_conn

    def _use_context_manager_for_cursor(self):
        return True

    def _execute(self, sql, params=None):
        if params:
            for p in params:
                sql = sql.replace("%s", repr(p), 1)
        if os.environ.get("PYSEEKDB_PRINT_SQL", "").lower() in ("1", "true", "yes"):
            print(f"[pyseekdb SQL] {sql}", flush=True)
        self.executed_sqls.append(sql)
        return None

    # Bypass the sdk_ltables lookup in unit tests: SQL-generation tests don't
    # have a real database, so return a fixed ltable_id matching test assertions.
    def _resolve_namespace_ltable_id(self, collection_id, namespace_id):
        return 1

    def _execute_query_with_cursor(self, conn, sql, params, use_context_manager=True):
        resolved = sql
        for p in params:
            resolved = resolved.replace("%s", repr(p), 1)
        if os.environ.get("PYSEEKDB_PRINT_SQL", "").lower() in ("1", "true", "yes"):
            print(f"[pyseekdb SQL] {resolved}", flush=True)
        self.query_sqls.append(resolved)
        return self.query_return_value

    def is_connected(self):
        return True

    def _cleanup(self):
        pass

    def get_raw_connection(self):
        return None

    @property
    def mode(self):
        return "FakeClient"

    def create_collection(self, *a, **kw):
        pass

    def get_collection(self, *a, **kw):
        pass

    def delete_collection(self, *a, **kw):
        pass

    def list_collections(self):
        return []

    def has_collection(self, name):
        return False

    def create_database(self, *a, **kw):
        pass

    def get_database(self, *a, **kw):
        pass

    def delete_database(self, *a, **kw):
        pass

    def list_databases(self, *a, **kw):
        return []

    def fork_database(self, *a, **kw):
        pass


class TestNamespaceSQLGeneration:
    """Verify SQL statements generated by BaseClient._namespace_* methods."""

    COLLECTION_ID = "42"
    NAMESPACE_ID = "7"
    TABLE = "42_logic_data_table"

    def _client(self):
        return FakeClient()

    def _common_kwargs(self):
        return dict(
            collection_id=self.COLLECTION_ID,
            collection_name="test_coll",
            namespace_id=self.NAMESPACE_ID,
            namespace_name="test_ns",
        )

    # ---- ADD ----

    def test_add_single_sql(self):
        c = self._client()
        c._namespace_add(
            **self._common_kwargs(),
            ids="d1",
            embeddings=[1.0, 2.0, 3.0],
            documents="hello",
            metadatas={"tag": "a"},
        )
        sql = c.executed_sqls[-1]
        assert sql.startswith("INSERT INTO")
        assert f"`{self.TABLE}`" in sql
        assert "(namespace_id, ltable_id, document, embedding, data_content)" in sql
        assert "(7, 1, 'hello'," in sql
        assert '\\"id\\": \\"d1\\"' in sql or '"id": "d1"' in sql
        assert '\\"tag\\": \\"a\\"' in sql or '"tag": "a"' in sql

    def test_add_batch_sql(self):
        c = self._client()
        c._namespace_add(
            **self._common_kwargs(),
            ids=["d1", "d2", "d3"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            documents=["A", "B", "C"],
            metadatas=[{"k": 1}, {"k": 2}, {"k": 3}],
        )
        sql = c.executed_sqls[-1]
        assert sql.count("(7, 1,") == 3
        assert '\\"id\\": \\"d1\\"' in sql or '"id": "d1"' in sql
        assert '\\"id\\": \\"d2\\"' in sql or '"id": "d2"' in sql
        assert '\\"id\\": \\"d3\\"' in sql or '"id": "d3"' in sql

    def test_add_without_metadata_sql(self):
        c = self._client()
        c._namespace_add(
            **self._common_kwargs(),
            ids="d1",
            embeddings=[1.0, 2.0, 3.0],
        )
        sql = c.executed_sqls[-1]
        assert '\\"id\\": \\"d1\\"' in sql or '"id": "d1"' in sql
        assert "metadata" not in sql.replace("data_content", "")

    # ---- UPDATE ----

    def test_update_metadata_sql(self):
        c = self._client()
        c._namespace_update(
            **self._common_kwargs(),
            ids="d1",
            metadatas={"score": 99},
        )
        sql = c.executed_sqls[-1]
        assert sql.startswith("UPDATE")
        assert f"`{self.TABLE}`" in sql
        assert "CASE" in sql
        assert "JSON_SET(data_content, '$.metadata'," in sql
        assert "CAST(" in sql
        assert "AS JSON)" in sql
        assert "namespace_id = 7" in sql
        assert "'d1'" in sql

    def test_update_embedding_and_document_sql(self):
        c = self._client()
        c._namespace_update(
            **self._common_kwargs(),
            ids="d1",
            embeddings=[9.0, 8.0, 7.0],
            documents="Updated",
        )
        emb_sql = c.executed_sqls[-2]
        assert "embedding = X'" in emb_sql
        assert "namespace_id = 7" in emb_sql
        doc_sql = c.executed_sqls[-1]
        assert "CASE" in doc_sql
        assert "'Updated'" in doc_sql
        assert "namespace_id = 7" in doc_sql

    # ---- DELETE ----

    def test_delete_by_ids_sql(self):
        c = self._client()
        c._namespace_delete(
            **self._common_kwargs(),
            ids="d1",
        )
        sql = c.executed_sqls[-1]
        assert sql.startswith("DELETE FROM")
        assert f"`{self.TABLE}`" in sql
        assert "namespace_id = 7" in sql
        assert "JSON_UNQUOTE(JSON_EXTRACT(data_content, '$.id')) = " in sql
        assert "'d1'" in sql

    def test_delete_by_where_sql(self):
        c = self._client()
        c._namespace_delete(
            **self._common_kwargs(),
            where={"category": "AI"},
        )
        sql = c.executed_sqls[-1]
        assert "DELETE FROM" in sql
        assert "namespace_id = 7" in sql
        assert "JSON_EXTRACT" in sql or "JSON_OVERLAPS" in sql
        assert "metadata.category" in sql

    def test_delete_by_where_document_sql(self):
        c = self._client()
        c._namespace_delete(
            **self._common_kwargs(),
            where_document={"$contains": "obsolete"},
        )
        sql = c.executed_sqls[-1]
        assert "DELETE FROM" in sql
        assert "namespace_id = 7" in sql
        assert "MATCH(document) AGAINST" in sql
        assert "obsolete" in sql

    # ---- QUERY ----

    def test_query_basic_dsl(self):
        c = self._client()
        c.query_return_value = []
        c._namespace_query(
            **self._common_kwargs(),
            query_embeddings=[1.0, 0.0, 0.0],
            n_results=3,
            distance="l2",
        )
        sql = c.query_sqls[-1]
        assert "hybrid_search(TABLE" in sql
        assert f"`{self.TABLE}`" in sql
        assert '"knn"' in sql
        assert '"query_vector"' in sql
        assert "l2_distance(embedding," not in sql
        assert "APPROXIMATE LIMIT" not in sql

    def test_query_with_where_dsl(self):
        c = self._client()
        c.query_return_value = []
        c._namespace_query(
            **self._common_kwargs(),
            query_embeddings=[1.0, 0.0, 0.0],
            n_results=5,
            where={"category": "AI"},
            distance="cosine",
        )
        sql = c.query_sqls[-1]
        assert "hybrid_search(TABLE" in sql
        assert "data_content.metadata.category" in sql
        assert "cosine_distance(embedding," not in sql

    # ---- GET ----

    def test_get_by_ids_sql(self):
        c = self._client()
        c.query_return_value = []
        c._namespace_get(
            **self._common_kwargs(),
            ids="g1",
        )
        sql = c.query_sqls[-1]
        assert "SELECT" in sql
        assert f"`{self.TABLE}`" in sql
        assert "namespace_id = 7" in sql
        assert """JSON_UNQUOTE(JSON_EXTRACT(data_content, '$.id')) = 'g1'""" in sql

    def test_get_with_limit_sql(self):
        c = self._client()
        c.query_return_value = []
        c._namespace_get(
            **self._common_kwargs(),
            limit=10,
        )
        sql = c.query_sqls[-1]
        assert "LIMIT 10" in sql

    # ---- COUNT ----

    def test_count_sql(self):
        c = self._client()
        c.query_return_value = [{"cnt": 0}]
        result = c._namespace_count(**self._common_kwargs())
        sql = c.query_sqls[-1]
        assert "SELECT COUNT(*) AS cnt" in sql
        assert f"`{self.TABLE}`" in sql
        assert "namespace_id = 7" in sql
        assert "ltable_id = 1" in sql
        assert result == 0


    # ---- IVF TYPE in SQL ----

    def test_ivf_type_default_in_sql(self):
        """Verify _get_ivf_vector_index_sql uses TYPE=IVF_FLAT and LIB=OB by default."""
        from pyseekdb.client.client_base import _get_ivf_vector_index_sql

        config = IVFConfiguration(dimension=3, distance="cosine")
        sql = _get_ivf_vector_index_sql(config)
        assert "TYPE=IVF_FLAT" in sql
        assert "LIB=OB" in sql
        assert "TYPE=ivf," not in sql

    def test_ivf_type_sq8_in_sql(self):
        """Verify _get_ivf_vector_index_sql uses TYPE=IVF_SQ8 when specified."""
        from pyseekdb.client.client_base import _get_ivf_vector_index_sql

        config = IVFConfiguration(dimension=3, distance="l2", type="ivf_sq8")
        sql = _get_ivf_vector_index_sql(config)
        assert "TYPE=IVF_SQ8" in sql

    def test_ivf_type_pq_in_sql(self):
        """Verify _get_ivf_vector_index_sql uses TYPE=IVF_PQ when specified."""
        from pyseekdb.client.client_base import _get_ivf_vector_index_sql

        config = IVFConfiguration(dimension=3, distance="inner_product", type="ivf_pq")
        sql = _get_ivf_vector_index_sql(config)
        assert "TYPE=IVF_PQ" in sql

    def test_ivf_lib_vsag_in_sql(self):
        """Verify _get_ivf_vector_index_sql uses LIB=VSAG when specified."""
        from pyseekdb.client.client_base import _get_ivf_vector_index_sql

        config = IVFConfiguration(dimension=3, distance="cosine", lib="vsag")
        sql = _get_ivf_vector_index_sql(config)
        assert "LIB=VSAG" in sql

    def test_ivf_lib_ob_in_sql(self):
        """Verify _get_ivf_vector_index_sql uses LIB=OB by default."""
        from pyseekdb.client.client_base import _get_ivf_vector_index_sql

        config = IVFConfiguration(dimension=3, distance="cosine")
        sql = _get_ivf_vector_index_sql(config)
        assert "LIB=OB" in sql

    def test_create_namespace_physical_tables_sn_inline_vector_index(self):
        """SN logic_data_table: inline VECTOR INDEX in CREATE TABLE (same as SS)."""
        from pyseekdb.client.configuration import FulltextIndexConfig

        c = self._client()
        ivf_config = IVFConfiguration(dimension=3, distance="l2", centroids_fresh_mode="spfresh")
        c._create_namespace_physical_tables(
            collection_id=self.COLLECTION_ID,
            dimension=3,
            ivf_config=ivf_config,
            fulltext_config=FulltextIndexConfig(analyzer="ik"),
            is_shared_storage=False,
        )
        data_create = next(
            s for s in c.executed_sqls if "CREATE TABLE" in s and self.TABLE in s
        )
        assert "VECTOR INDEX idx_vec(embedding)" in data_create
        assert "FULLTEXT INDEX idx_fts(document) WITH PARSER ik" in data_create
        assert "SEARCH INDEX idx_json(data_content)" in data_create
        assert "centroids_fresh_mode=spfresh" in data_create
        assert not any(s.startswith("CREATE VECTOR INDEX") for s in c.executed_sqls)
        assert not any("_hot_table" in s for s in c.executed_sqls if s.startswith("CREATE TABLE"))

    def test_create_namespace_physical_tables_ss_inline_vector_index(self):
        """SS logic_data_table: inline VECTOR INDEX in CREATE TABLE."""
        c = self._client()
        ivf_config = IVFConfiguration(dimension=3, distance="l2", centroids_fresh_mode="spfresh")
        c._create_namespace_physical_tables(
            collection_id=self.COLLECTION_ID,
            dimension=3,
            ivf_config=ivf_config,
            is_shared_storage=True,
        )
        data_create = next(
            s for s in c.executed_sqls if "CREATE TABLE" in s and self.TABLE in s
        )
        assert "VECTOR INDEX idx_vec(embedding)" in data_create
        assert "FULLTEXT INDEX" not in data_create
        assert "SEARCH INDEX idx_json(data_content)" in data_create
        assert "centroids_fresh_mode=spfresh" in data_create
        assert not any(s.startswith("CREATE VECTOR INDEX") for s in c.executed_sqls)
        hot_create = next(
            s for s in c.executed_sqls if "CREATE TABLE" in s and "_hot_table" in s
        )
        assert hot_create

    def test_create_namespace_physical_tables_search_index_only(self):
        """Without ivf/fulltext config, only SEARCH INDEX is created."""
        c = self._client()
        c._create_namespace_physical_tables(
            collection_id=self.COLLECTION_ID,
            dimension=3,
            is_shared_storage=False,
        )
        data_create = next(
            s for s in c.executed_sqls if "CREATE TABLE" in s and self.TABLE in s
        )
        assert "SEARCH INDEX idx_json(data_content)" in data_create
        assert "VECTOR INDEX" not in data_create
        assert "FULLTEXT INDEX" not in data_create

    def test_create_namespace_physical_tables_ivf_without_centroids_fresh_mode(self):
        """IVF without centroids_fresh_mode omits centroids_fresh_mode from VECTOR INDEX DDL."""
        c = self._client()
        ivf_config = IVFConfiguration(dimension=3, distance="l2")
        c._create_namespace_physical_tables(
            collection_id=self.COLLECTION_ID,
            dimension=3,
            ivf_config=ivf_config,
            is_shared_storage=False,
        )
        data_create = next(
            s for s in c.executed_sqls if "CREATE TABLE" in s and self.TABLE in s
        )
        assert "VECTOR INDEX idx_vec(embedding)" in data_create
        assert "centroids_fresh_mode" not in data_create


# ==================== Namespace Name Validation Tests ====================


class TestValidateNamespaceName:

    def test_valid_simple_name(self):
        from pyseekdb.client.client_base import _validate_namespace_name
        _validate_namespace_name("my_namespace")

    def test_valid_with_digits_and_underscore(self):
        from pyseekdb.client.client_base import _validate_namespace_name
        _validate_namespace_name("tenant_123_abc")

    def test_valid_single_char(self):
        from pyseekdb.client.client_base import _validate_namespace_name
        _validate_namespace_name("a")

    def test_valid_boundary_256_chars(self):
        from pyseekdb.client.client_base import _validate_namespace_name
        _validate_namespace_name("a" * 256)

    def test_empty_name_raises(self):
        from pyseekdb.client.client_base import _validate_namespace_name
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_namespace_name("")

    def test_non_string_raises(self):
        from pyseekdb.client.client_base import _validate_namespace_name
        with pytest.raises(TypeError, match="must be a string"):
            _validate_namespace_name(123)

    def test_too_long_raises(self):
        from pyseekdb.client.client_base import _validate_namespace_name
        with pytest.raises(ValueError, match="too long"):
            _validate_namespace_name("a" * 257)

    def test_hyphen_raises(self):
        from pyseekdb.client.client_base import _validate_namespace_name
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_namespace_name("my-namespace")

    def test_space_raises(self):
        from pyseekdb.client.client_base import _validate_namespace_name
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_namespace_name("my namespace")

    def test_dot_raises(self):
        from pyseekdb.client.client_base import _validate_namespace_name
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_namespace_name("my.namespace")

    def test_chinese_raises(self):
        from pyseekdb.client.client_base import _validate_namespace_name
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_namespace_name("命名空间")

    def test_collection_create_namespace_validates_name(self):
        mock_client = MagicMock()
        coll = Collection(client=mock_client, name="c", collection_id="1", dimension=3, use_namespace=True)
        with pytest.raises(ValueError, match="invalid characters"):
            coll.create_namespace("bad-name")

    def test_collection_get_namespace_validates_name(self):
        mock_client = MagicMock()
        coll = Collection(client=mock_client, name="c", collection_id="1", dimension=3, use_namespace=True)
        with pytest.raises(ValueError, match="must not be empty"):
            coll.get_namespace("")

    def test_collection_has_namespace_validates_name(self):
        mock_client = MagicMock()
        coll = Collection(client=mock_client, name="c", collection_id="1", dimension=3, use_namespace=True)
        with pytest.raises(TypeError, match="must be a string"):
            coll.has_namespace(42)

    def test_partition_count_property(self):
        mock_client = MagicMock()
        ns_coll = Collection(
            client=mock_client, name="c", collection_id="1", dimension=3,
            use_namespace=True, partition_count=4,
        )
        assert ns_coll.partition_count == 4
        # Non-namespace collections are not partitioned.
        plain = Collection(client=mock_client, name="p", collection_id="2", dimension=3)
        assert plain.partition_count is None


# ==================== Record ID Validation Tests ====================


class TestValidateRecordIds:

    def test_valid_single_id(self):
        from pyseekdb.client.client_base import _validate_record_ids
        _validate_record_ids(["doc_1"])

    def test_valid_multiple_ids(self):
        from pyseekdb.client.client_base import _validate_record_ids
        _validate_record_ids(["id1", "id2", "id_3"])

    def test_valid_boundary_512_chars(self):
        from pyseekdb.client.client_base import _validate_record_ids
        _validate_record_ids(["a" * 512])

    def test_empty_id_raises(self):
        from pyseekdb.client.client_base import _validate_record_ids
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_record_ids([""])

    def test_non_string_id_raises(self):
        from pyseekdb.client.client_base import _validate_record_ids
        with pytest.raises(TypeError, match="must be a string"):
            _validate_record_ids([123])

    def test_too_long_id_raises(self):
        from pyseekdb.client.client_base import _validate_record_ids
        with pytest.raises(ValueError, match="too long"):
            _validate_record_ids(["a" * 513])

    def test_invalid_chars_raises(self):
        from pyseekdb.client.client_base import _validate_record_ids
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_record_ids(["doc-1"])

    def test_space_in_id_raises(self):
        from pyseekdb.client.client_base import _validate_record_ids
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_record_ids(["doc 1"])

    def test_mixed_valid_and_invalid_raises_on_first_bad(self):
        from pyseekdb.client.client_base import _validate_record_ids
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_record_ids(["good_id", "bad-id"])


# ==================== Namespace Batch Limit Tests ====================


class TestNamespaceBatchLimit:

    def _client(self):
        return FakeClient()

    def _common_kwargs(self):
        return dict(
            collection_id="42",
            collection_name="test_coll",
            namespace_id="7",
            namespace_name="test_ns",
        )

    def test_add_single_record_ok(self):
        c = self._client()
        c._namespace_add(**self._common_kwargs(), ids="d1", embeddings=[1.0, 2.0, 3.0])

    def test_add_100_records_ok(self):
        c = self._client()
        ids = [f"d{i}" for i in range(100)]
        embeddings = [[float(i)] * 3 for i in range(100)]
        c._namespace_add(**self._common_kwargs(), ids=ids, embeddings=embeddings)

    def test_add_101_records_raises(self):
        c = self._client()
        ids = [f"d{i}" for i in range(101)]
        embeddings = [[float(i)] * 3 for i in range(101)]
        with pytest.raises(ValueError, match="exceeds maximum allowed 100"):
            c._namespace_add(**self._common_kwargs(), ids=ids, embeddings=embeddings)

    def test_update_101_records_raises(self):
        c = self._client()
        ids = [f"d{i}" for i in range(101)]
        with pytest.raises(ValueError, match="exceeds maximum allowed 100"):
            c._namespace_update(**self._common_kwargs(), ids=ids, metadatas=[{"k": "v"}] * 101)

    def test_upsert_101_records_raises(self):
        c = self._client()
        ids = [f"d{i}" for i in range(101)]
        embeddings = [[float(i)] * 3 for i in range(101)]
        with pytest.raises(ValueError, match="exceeds maximum allowed 100"):
            c._namespace_upsert(**self._common_kwargs(), ids=ids, embeddings=embeddings)

    def test_add_invalid_id_raises(self):
        c = self._client()
        with pytest.raises(ValueError, match="invalid characters"):
            c._namespace_add(**self._common_kwargs(), ids="bad-id", embeddings=[1.0, 2.0, 3.0])

    def test_delete_invalid_id_raises(self):
        c = self._client()
        with pytest.raises(ValueError, match="invalid characters"):
            c._namespace_delete(**self._common_kwargs(), ids="bad-id")


# ==================== Physical Table Names Tests ====================


class TestPhysicalTableNames:

    def test_data_table_name(self):
        from pyseekdb.client.meta_info import NamespaceCollectionNames
        assert NamespaceCollectionNames.data_table_name("my_coll") == "my_coll_logic_data_table"

    def test_hot_table_name(self):
        from pyseekdb.client.meta_info import NamespaceCollectionNames
        assert NamespaceCollectionNames.hot_table_name("my_coll") == "my_coll_hot_table"

    def test_kv_data_table_name(self):
        from pyseekdb.client.meta_info import NamespaceCollectionNames
        assert NamespaceCollectionNames.kv_data_table_name("my_coll") == "my_coll_kv_data_table"

    def test_logic_schema_table_name(self):
        from pyseekdb.client.meta_info import NamespaceCollectionNames
        assert NamespaceCollectionNames.logic_schema_table_name("my_coll") == "my_coll_logic_schema_table"

    def test_tablegroup_name(self):
        from pyseekdb.client.meta_info import NamespaceCollectionNames
        assert NamespaceCollectionNames.tablegroup_name("my_coll") == "my_coll_tg"

    def test_namespace_catalog_table_names(self):
        from pyseekdb.client.meta_info import NamespaceCollectionNames
        assert NamespaceCollectionNames.sdk_namespaces_table() == "sdk_namespaces"
        assert NamespaceCollectionNames.sdk_ltables_table() == "sdk_ltables"
        assert NamespaceCollectionNames.sdk_namespaces_stats_table() == "sdk_namespaces_stats"

    def test_is_ns_data_table_true(self):
        from pyseekdb.client.meta_info import NamespaceCollectionNames
        assert NamespaceCollectionNames.is_ns_data_table("my_coll_logic_data_table") is True

    def test_is_ns_data_table_false(self):
        from pyseekdb.client.meta_info import NamespaceCollectionNames
        assert NamespaceCollectionNames.is_ns_data_table("my_coll_hot_table") is False


# ==================== Namespace Catalog Tests ====================


class TestNamespaceCatalogs:

    def test_ensure_namespace_catalogs_creates_all_catalog_tables(self):
        c = FakeClient()
        c._ensure_namespace_catalogs()

        sql = "\n".join(c.executed_sqls)
        assert "CREATE TABLE IF NOT EXISTS `test`.`sdk_namespaces`" in sql
        assert "PRIMARY KEY (namespace_id)" in sql
        assert "UNIQUE KEY uk_sdk_ns_coll_name (collection_id, namespace_name)" in sql

        assert "CREATE TABLE IF NOT EXISTS `test`.`sdk_ltables`" in sql
        assert "PRIMARY KEY (ltable_id)" in sql
        assert "UNIQUE KEY uk_sdk_lt_coll_ns_name (collection_id, namespace_id, ltable_name)" in sql

        assert "CREATE TABLE IF NOT EXISTS `test`.`sdk_namespaces_stats`" in sql
        assert "collection_id CHAR(32) NOT NULL" in sql
        assert "namespace_id BIGINT UNSIGNED NOT NULL" in sql
        assert "ltable_id BIGINT UNSIGNED NOT NULL" in sql
        assert "included_index BOOL" in sql
        assert "PRIMARY KEY (namespace_id, ltable_id, included_index)" in sql
        assert "PARTITION BY KEY(namespace_id) PARTITIONS 1000" in sql

    def test_ensure_namespace_catalogs_creates_catalog_tables_in_order(self):
        c = FakeClient()
        c._ensure_namespace_catalogs()

        assert len(c.executed_sqls) == 5
        assert c.executed_sqls[0] == "USE `test`"
        assert "`test`.`sdk_namespaces`" in c.executed_sqls[2]
        assert "`test`.`sdk_ltables`" in c.executed_sqls[3]
        assert "`test`.`sdk_namespaces_stats`" in c.executed_sqls[4]

    def test_delete_ns_collection_meta_cleans_namespaces_stats_table(self):
        c = FakeClient()
        c._get_ns_collection_meta = MagicMock(return_value={"collection_id": "abc123"})
        c._cleanup_namespace_physical_tables = MagicMock()
        c._execute = MagicMock()

        c._delete_ns_collection_meta("coll")

        calls = [str(call) for call in c._execute.call_args_list]
        assert any("DELETE FROM `sdk_namespaces_stats`" in s for s in calls)
        assert any("WHERE collection_id = 'abc123'" in s for s in calls)


# ==================== UseNamespace Validation Tests ====================


class TestUseNamespaceValidation:

    def test_hnsw_raises(self):
        c = FakeClient()
        from pyseekdb.client.schema import Schema
        from pyseekdb.client.configuration import VectorIndexConfig, HNSWConfiguration
        hnsw = HNSWConfiguration(dimension=128)
        schema = Schema(vector_index=VectorIndexConfig(hnsw=hnsw, embedding_function=None))
        with pytest.raises(ValueError, match="HNSW is not allowed"):
            c._create_namespace_collection("test", schema)

    def test_ob_type_validation(self):
        c = FakeClient()
        c.detect_db_type_and_version = MagicMock(return_value=("mysql", "8.0"))
        from pyseekdb.client.schema import Schema
        from pyseekdb.client.configuration import VectorIndexConfig
        schema = Schema(vector_index=VectorIndexConfig(ivf=IVFConfiguration(dimension=3, centroids_fresh_mode="spfresh"), embedding_function=None))
        with pytest.raises(ValueError, match="only supported on OceanBase"):
            c._create_namespace_collection("test", schema)

    def test_create_namespace_collection_without_ivf_skips_vector_index(self):
        c = FakeClient()
        c.detect_db_type_and_version = MagicMock(return_value=("oceanbase", "4.3"))
        c._is_shared_storage_mode = MagicMock(return_value=False)
        c._create_ns_collection_meta = MagicMock(return_value={"collection_id": "abc123"})
        c._ensure_namespace_catalogs = MagicMock()
        from pyseekdb.client.schema import Schema
        from pyseekdb.client.configuration import VectorIndexConfig

        schema = Schema(vector_index=VectorIndexConfig(embedding_function=None))
        c._create_namespace_collection("test", schema)
        data_create = next(
            s for s in c.executed_sqls if "CREATE TABLE" in s and "logic_data_table" in s
        )
        assert "VECTOR INDEX" not in data_create
        assert "SEARCH INDEX idx_json(data_content)" in data_create
        settings = c._create_ns_collection_meta.call_args[0][1]
        assert "dense_index_type" not in settings
        assert "centroids_fresh_mode" not in settings

    def test_create_namespace_collection_ivf_without_centroids_fresh_mode(self):
        c = FakeClient()
        c.detect_db_type_and_version = MagicMock(return_value=("oceanbase", "4.3"))
        c._is_shared_storage_mode = MagicMock(return_value=False)
        c._create_ns_collection_meta = MagicMock(return_value={"collection_id": "abc123"})
        c._ensure_namespace_catalogs = MagicMock()
        from pyseekdb.client.schema import Schema
        from pyseekdb.client.configuration import VectorIndexConfig

        schema = Schema(
            vector_index=VectorIndexConfig(
                ivf=IVFConfiguration(dimension=3, distance="l2"),
                embedding_function=None,
            )
        )
        c._create_namespace_collection("test", schema)
        data_create = next(
            s for s in c.executed_sqls if "CREATE TABLE" in s and "logic_data_table" in s
        )
        assert "VECTOR INDEX idx_vec(embedding)" in data_create
        assert "centroids_fresh_mode" not in data_create
        settings = c._create_ns_collection_meta.call_args[0][1]
        assert settings["dense_index_type"] == "ivf"
        assert "centroids_fresh_mode" not in settings


# ==================== Delete Namespace Uses Kernel ====================


class TestDeleteNamespaceUsesKernel:

    def test_delete_namespace_calls_dbms_logic_table(self):
        c = FakeClient()
        c._execute = MagicMock(side_effect=[
            [{"got": 1}],  # GET_LOCK
            [{"namespace_id": 10, "namespace_name": "ns1", "ltable_id": 7}],
            None,  # _get_ns_namespace_meta -> SET @namespace_id
            None,  # _get_ns_namespace_meta -> SET @ltable_id
            None,  # _delete_ns_namespace_meta -> USE catalog database
            None,  # _delete_ns_namespace_meta -> SET @collection_id
            None,  # _delete_ns_namespace_meta -> SET @namespace_id
            None,  # _delete_ns_namespace_meta -> SET @ltable_id
            None,  # CALL DBMS_LOGIC_TABLE.DROP_NAMESPACE
            [{"got": 1}],  # RELEASE_LOCK
        ])
        c._delete_ns_namespace_meta("abc123", "ns1")
        calls = [str(call) for call in c._execute.call_args_list]
        assert any("GET_LOCK" in s and "pyseekdb:nslc:" in s for s in calls)
        assert any("RELEASE_LOCK" in s for s in calls)
        assert any("DBMS_LOGIC_TABLE.DROP_NAMESPACE" in s for s in calls)
        # session context must be set BEFORE DROP_NAMESPACE so the kernel sees the
        # right @collection_id / @namespace_id / @ltable_id for this call.
        drop_idx = next(i for i, s in enumerate(calls) if "DBMS_LOGIC_TABLE.DROP_NAMESPACE" in s)
        before_drop = " | ".join(calls[:drop_idx])
        assert "SET @collection_id" in before_drop
        assert "SET @namespace_id" in before_drop
        assert "SET @ltable_id" in before_drop

    def test_get_ns_namespace_meta_sets_ltable_id(self):
        """get_namespace must populate @ltable_id along with @namespace_id."""
        c = FakeClient()
        c._execute = MagicMock(side_effect=[
            [{"namespace_id": 10, "namespace_name": "ns1", "ltable_id": 7}],
            None,  # SET @namespace_id
            None,  # SET @ltable_id
        ])
        meta = c._get_ns_namespace_meta("abc123", "ns1")
        assert meta == {"namespace_id": "10", "namespace_name": "ns1", "ltable_id": "7"}
        calls = [str(call) for call in c._execute.call_args_list]
        # JOIN against sdk_ltables so we can resolve the default ltable in one round trip.
        assert any("sdk_ltables" in s for s in calls)
        assert any("SET @namespace_id" in s for s in calls)
        assert any("SET @ltable_id" in s for s in calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
