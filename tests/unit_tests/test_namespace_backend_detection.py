"""Namespace backend guards recognize current and legacy OceanLake product banners."""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from pyseekdb import IVFConfiguration, Schema, VectorIndexConfig  # noqa: E402
from pyseekdb.client.client_base import BaseClient, is_lakebase_version_string  # noqa: E402
from pyseekdb.client.version import Version  # noqa: E402
from tests.unit_tests.test_namespace import FakeClient  # noqa: E402


class BackendDetectionClient(FakeClient):
    """Provide database responses while retaining the real detection and guard methods."""

    create_collection = BaseClient.create_collection

    def __init__(self, banner, product_version, row_shape):
        super().__init__()
        self.banner = banner
        self.product_version = product_version
        self.row_shape = row_shape

    def _rows(self, key, value):
        return [{key: value}] if self.row_shape == "dict" else [(value,)]

    def _execute(self, sql, params=None):
        assert params is None
        self.executed_sqls.append(sql)
        normalized = sql.strip().lower()
        if normalized == "select version() as version":
            return self._rows("version", self.banner)
        if normalized == "select ob_version() as ob_version":
            return self._rows("ob_version", self.product_version)
        if "ob_startup_mode" in normalized:
            return self._rows("VALUE", "SHARED_STORAGE")
        if normalized.startswith("select "):
            return []
        return None


def _create_namespace_collection(client):
    schema = Schema(
        fulltext_index=None,
        vector_index=VectorIndexConfig(
            ivf=IVFConfiguration(dimension=3, distance="cosine"),
            embedding_function=None,
        ),
    )
    return client.create_collection("backend_guard", schema=schema, use_namespace=True, partition_count=4)


@pytest.mark.parametrize(
    ("banner", "expected"),
    [
        ("5.7.25-OceanSeek-v4.6.3.0", True),
        ("5.7.25-oceanseek-v4.6.3.0", True),
        ("5.7.25-OCEANSEEK-v4.6.3.0", True),
        ("5.7.25-OceanBase Database AI-v4.6.3.0", True),
        ("5.7.25-oceanbase database ai-v4.6.3.0", True),
        ("5.7.25-OCEANBASE DATABASE AI-v4.6.3.0", True),
        ("5.7.25-OceanBase AI Database-v4.6.3.0", True),
        ("5.7.25-oceanbase ai database-v4.6.3.0", True),
        ("5.7.25-OCEANBASE AI DATABASE-v4.6.3.0", True),
        ("5.7.25-Other AI Database-v4.6.3.0", False),
        ("5.7.25-OceanBase_CE-v9.0.0.0", False),
        ("5.7.25-OceanBase-v4.6.3.0", False),
        ("5.7.25-seekdb-v1.4.0.0", False),
        ("5.7.25-SeekDB-v1.4.0.0", False),
        ("8.0.36-MySQL Community Server", False),
        ("", False),
    ],
)
def test_lakebase_product_banner_recognition_is_case_insensitive(banner, expected):
    assert is_lakebase_version_string(banner) is expected


@pytest.mark.parametrize("row_shape", ["dict", "tuple"])
@pytest.mark.parametrize("product", ["OceanSeek", "OceanBase Database AI", "OceanBase AI Database"])
@pytest.mark.parametrize("product_version", ["4.6.3.0", "4.6.2.1"])
def test_namespace_create_uses_ob_product_version_and_accepts_supported_backends(row_shape, product, product_version):
    client = BackendDetectionClient(f"5.7.25-{product}-v{product_version}", product_version, row_shape)

    backend, version = client.detect_db_type_and_version()
    assert backend == "oceanbase"
    assert version == Version(product_version)
    assert version != Version("5.7.25")
    assert client._is_lakebase_cluster() is True
    client._validate_ob_database_type()

    collection = _create_namespace_collection(client)

    assert collection._use_namespace is True
    assert collection.dimension == 3
    assert any(sql.lower() == "select ob_version() as ob_version" for sql in client.executed_sqls)
    assert any(sql.startswith("INSERT INTO") and "sdk_collections" in sql for sql in client.executed_sqls)
    assert any(
        sql.startswith("CREATE TABLE") and f"{collection.id}_logic_data_table" in sql for sql in client.executed_sqls
    )


@pytest.mark.parametrize("row_shape", ["dict", "tuple"])
@pytest.mark.parametrize("product", ["OceanSeek", "OceanBase Database AI", "OceanBase AI Database"])
def test_namespace_create_rejects_older_product_version_before_writes(row_shape, product):
    client = BackendDetectionClient(f"5.7.25-{product}-v4.6.2.0", "4.6.2.0", row_shape)
    assert client.detect_db_type_and_version() == ("oceanbase", Version("4.6.2.0"))

    with pytest.raises(ValueError, match=r"version >= 4\.6\.2\.1.*current version is 4\.6\.2\.0"):
        _create_namespace_collection(client)

    assert client.executed_sqls
    assert all(sql.lower().startswith("select ") for sql in client.executed_sqls)


@pytest.mark.parametrize("row_shape", ["dict", "tuple"])
def test_high_version_standard_oceanbase_is_not_a_namespace_backend(row_shape):
    client = BackendDetectionClient("5.7.25-OceanBase_CE-v9.0.0.0", "9.0.0.0", row_shape)
    assert client.detect_db_type_and_version() == ("oceanbase", Version("9.0.0.0"))
    assert client._is_lakebase_cluster() is False

    with pytest.raises(ValueError, match="standard OceanBase"):
        _create_namespace_collection(client)

    assert all(sql.lower().startswith("select ") for sql in client.executed_sqls)


@pytest.mark.parametrize("row_shape", ["dict", "tuple"])
@pytest.mark.parametrize("product", ["seekdb", "SeekDB"])
def test_seekdb_is_not_confused_with_oceanseek(row_shape, product):
    client = BackendDetectionClient(f"5.7.25-{product}-v1.4.0.0", "9.0.0.0", row_shape)
    assert client.detect_db_type_and_version() == ("seekdb", Version("1.4.0.0"))
    assert client._is_lakebase_cluster() is False

    with pytest.raises(ValueError, match="only supported on OceanLake"):
        _create_namespace_collection(client)

    assert all(sql.lower().startswith("select ") for sql in client.executed_sqls)
    assert not any("ob_version()" in sql.lower() for sql in client.executed_sqls)


@pytest.mark.parametrize("row_shape", ["dict", "tuple"])
@pytest.mark.parametrize("product", ["OceanSeek", "OceanBase Database AI", "OceanBase AI Database"])
def test_lakebase_banner_detection_result_is_cached(row_shape, product):
    client = BackendDetectionClient(f"5.7.25-{product}-v4.6.3.0", "4.6.3.0", row_shape)

    assert client._is_lakebase_cluster() is True
    assert client._is_lakebase_cluster() is True

    assert len(client.executed_sqls) == 1
