"""Unit tests for namespace ops/RU config validation and SDK API wiring."""

from __future__ import annotations

import pytest

from pyseekdb.client.meta_info import NamespaceOpsConfigKeys
from pyseekdb.client.validators import (
    _validate_namespace_ops_config_key,
    _validate_namespace_ops_config_value,
)


class TestNamespaceOpsConfigValidation:
    def test_accepts_ops_limit_keys(self):
        for key in (NamespaceOpsConfigKeys.ROW_LIMIT, NamespaceOpsConfigKeys.SIZE_LIMIT):
            _validate_namespace_ops_config_key(key)

    def test_accepts_ru_limit_keys(self):
        for key in (
            NamespaceOpsConfigKeys.RU_ENABLED,
            NamespaceOpsConfigKeys.QPS_BURST,
            NamespaceOpsConfigKeys.QPS_REFILL,
            NamespaceOpsConfigKeys.TPS_BURST,
            NamespaceOpsConfigKeys.TPS_REFILL,
            NamespaceOpsConfigKeys.DATA_BURST,
            NamespaceOpsConfigKeys.DATA_REFILL,
        ):
            _validate_namespace_ops_config_key(key)

    def test_rejects_unknown_key(self):
        with pytest.raises(ValueError, match="Invalid config_key"):
            _validate_namespace_ops_config_key("not_a_key")
        with pytest.raises(ValueError, match="Invalid config_key"):
            _validate_namespace_ops_config_key("enabled")

    def test_ru_enabled_must_be_zero_or_one(self):
        _validate_namespace_ops_config_value(NamespaceOpsConfigKeys.RU_ENABLED, 0)
        _validate_namespace_ops_config_value(NamespaceOpsConfigKeys.RU_ENABLED, 1)
        with pytest.raises(ValueError, match="ru_enabled"):
            _validate_namespace_ops_config_value(NamespaceOpsConfigKeys.RU_ENABLED, 2)

    def test_config_value_must_be_int(self):
        with pytest.raises(TypeError):
            _validate_namespace_ops_config_value("row_limit", True)  # type: ignore[arg-type]


class TestNamespaceOpsConfigClient:
    def test_set_namespace_ops_config_calls_pl(self):
        from unittest.mock import MagicMock

        from pyseekdb.client.client_base import BaseClient

        executed: list[str] = []
        client = MagicMock()
        client._use_catalog_database = MagicMock()
        client._execute_catalog = lambda sql, *args, **kwargs: executed.append(sql)
        BaseClient._set_namespace_ops_config(client, "my_coll", "my_ns", "row_limit", 99)
        assert len(executed) == 1
        assert "JSON_MERGE_PATCH" in executed[0]
        assert "'my_coll'" in executed[0]
        assert "'my_ns'" in executed[0]
        assert "row_limit" in executed[0]
        assert "99" in executed[0]
