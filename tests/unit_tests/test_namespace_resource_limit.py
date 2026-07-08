"""Unit tests for namespace RU config validation and SDK API wiring."""

from __future__ import annotations

import pytest

from pyseekdb.client.meta_info import NamespaceResourceLimitKeys
from pyseekdb.client.validators import (
    _validate_namespace_resource_limit,
    _validate_namespace_resource_limit_key,
    _validate_namespace_resource_limit_value,
)


class TestNamespaceResourceLimitValidation:
    def test_accepts_ops_limit_keys(self):
        for key in (NamespaceResourceLimitKeys.ROW_LIMIT, NamespaceResourceLimitKeys.SIZE_LIMIT):
            _validate_namespace_resource_limit_key(key)

    def test_accepts_ru_limit_keys(self):
        for key in (
            NamespaceResourceLimitKeys.RATE_LIMIT_ENABLE,
            NamespaceResourceLimitKeys.QPS_BURST,
            NamespaceResourceLimitKeys.QPS_REFILL,
            NamespaceResourceLimitKeys.TPS_BURST,
            NamespaceResourceLimitKeys.TPS_REFILL,
            NamespaceResourceLimitKeys.DATA_BURST,
            NamespaceResourceLimitKeys.DATA_REFILL,
        ):
            _validate_namespace_resource_limit_key(key)

    def test_rejects_unknown_key(self):
        with pytest.raises(ValueError, match="Invalid config_key"):
            _validate_namespace_resource_limit_key("not_a_key")
        with pytest.raises(ValueError, match="Invalid config_key"):
            _validate_namespace_resource_limit_key("enabled")

    def test_rate_limit_enable_must_be_zero_or_one(self):
        _validate_namespace_resource_limit_value(NamespaceResourceLimitKeys.RATE_LIMIT_ENABLE, 0)
        _validate_namespace_resource_limit_value(NamespaceResourceLimitKeys.RATE_LIMIT_ENABLE, 1)
        with pytest.raises(ValueError, match="rate_limit_enable"):
            _validate_namespace_resource_limit_value(NamespaceResourceLimitKeys.RATE_LIMIT_ENABLE, 2)

    def test_row_limit_accepts_minus_one_zero_and_positive(self):
        _validate_namespace_resource_limit_value(NamespaceResourceLimitKeys.ROW_LIMIT, -1)
        _validate_namespace_resource_limit_value(NamespaceResourceLimitKeys.ROW_LIMIT, 0)
        _validate_namespace_resource_limit_value(NamespaceResourceLimitKeys.ROW_LIMIT, 100)
        with pytest.raises(ValueError, match="row_limit"):
            _validate_namespace_resource_limit_value(NamespaceResourceLimitKeys.ROW_LIMIT, -2)

    def test_ru_rate_must_be_non_negative(self):
        with pytest.raises(ValueError, match="qps_burst"):
            _validate_namespace_resource_limit_value(NamespaceResourceLimitKeys.QPS_BURST, -1)

    def test_config_value_must_be_int(self):
        with pytest.raises(TypeError):
            _validate_namespace_resource_limit_value("row_limit", True)  # type: ignore[arg-type]


class TestNamespaceResourceLimitBatchValidation:
    def test_accepts_merged_json(self):
        _validate_namespace_resource_limit({"row_limit": 2, "tps_burst": 100, "rate_limit_enable": 1})

    def test_rejects_negative_burst(self):
        with pytest.raises(ValueError, match="qps_burst"):
            _validate_namespace_resource_limit({"qps_burst": -1})

    def test_rejects_invalid_row_limit(self):
        with pytest.raises(ValueError, match="row_limit"):
            _validate_namespace_resource_limit({"row_limit": -2})

    def test_rejects_unknown_key(self):
        with pytest.raises(ValueError, match="Invalid config_key"):
            _validate_namespace_resource_limit({"bogus_key": 1})


class TestNamespaceResourceLimitClient:
    def test_set_namespace_resource_limit_calls_pl(self):
        from unittest.mock import MagicMock

        from pyseekdb.client.client_base import BaseClient

        executed: list[str] = []
        client = MagicMock()
        client._use_catalog_database = MagicMock()
        client._execute = lambda sql, *args, **kwargs: executed.append(sql)
        BaseClient._set_namespace_resource_limit(client, "my_coll", "my_ns", {"row_limit": 99, "tps_burst": 50})
        assert len(executed) == 1
        assert "SET_NAMESPACE_RESOURCE_LIMIT" in executed[0]
        assert "'my_coll'" in executed[0]
        assert "'my_ns'" in executed[0]
        assert "row_limit" in executed[0]
        assert "99" in executed[0]
        assert "tps_burst" in executed[0]
