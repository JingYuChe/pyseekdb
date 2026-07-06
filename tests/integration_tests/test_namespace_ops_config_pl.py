"""
Integration tests for namespace ops/RU config via SDK public API.

Exercises Collection/Namespace convenience config methods and
DBMS_LOGIC_TABLE.SET_NAMESPACE_OPS_CONFIG under the hood.
"""

from __future__ import annotations

import time
import uuid

import pytest

from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT, ns_schema
from namespace_stats_test_helpers import read_ops_limit, read_ru_limit, upsert_namespace_stat

import pyseekdb
from pyseekdb.client.meta_info import NamespaceOpsConfigKeys


def _raw_client():
    import os

    return pyseekdb.Client(
        host=os.environ.get("OB_HOST", "127.0.0.1"),
        port=int(os.environ.get("OB_PORT", "10902")),
        tenant=os.environ.get("OB_TENANT", "mysql"),
        database=os.environ.get("OB_DATABASE", "test"),
        user=os.environ.get("OB_USER", "root"),
        password=os.environ.get("OB_PASSWORD", ""),
    )


def _close(client) -> None:
    if hasattr(client, "close"):
        client.close()


def _emb(i: int) -> list[float]:
    return [(i % 97) / 97.0, ((i * 3) % 89) / 89.0, ((i * 7) % 83) / 83.0]


def _is_throttle(exc: Exception) -> bool:
    s = str(exc).lower()
    return "4039" in s or "throttl" in s


def _hammer(ns, start: int, n: int) -> list[bool]:
    blocked: list[bool] = []
    for i in range(start, start + n):
        try:
            ns.add(ids=[f"h_{i:06d}"], embeddings=[_emb(i)], documents=[f"d{i}"], metadatas=[{"seq": i}])
            blocked.append(False)
        except Exception as exc:
            if _is_throttle(exc):
                blocked.append(True)
            else:
                raise
    return blocked


def _add_one(ns, i: int) -> None:
    ns.add(
        ids=[f"row_{i:06d}"],
        embeddings=[_emb(i)],
        documents=[f"doc {i}"],
        metadatas=[{"seq": i}],
    )


def _assert_blocked(fn) -> None:
    with pytest.raises(Exception) as excinfo:
        fn()
    msg = str(excinfo.value).lower()
    assert "not allowed" in msg or "limit" in msg, f"unexpected error: {excinfo.value!r}"


class TestNamespaceOpsConfigViaSdk:
    def test_namespace_set_row_limit(self, oceanbase_client):
        admin = _raw_client()
        coll_name = f"ops_sdk_{uuid.uuid4().hex[:10]}"
        ns_name = "ops_ns"
        coll = oceanbase_client.create_collection(
            name=coll_name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace(ns_name)
            ns_id = int(ns.namespace_id)
            ns.set_row_limit(42_000)
            assert read_ops_limit(admin, coll.id, ns_id, "row_limit") == 42_000
        finally:
            oceanbase_client.delete_collection(name=coll_name)
            _close(admin)

    def test_collection_set_size_limit(self, oceanbase_client):
        admin = _raw_client()
        coll_name = f"ops_coll_{uuid.uuid4().hex[:10]}"
        ns_name = "ops_ns_b"
        coll = oceanbase_client.create_collection(
            name=coll_name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace(ns_name)
            ns_id = int(ns.namespace_id)
            coll.set_namespace_size_limit(ns_name, 8_000_000)
            assert read_ops_limit(admin, coll.id, ns_id, "size_limit") == 8_000_000
        finally:
            oceanbase_client.delete_collection(name=coll_name)
            _close(admin)

    def test_set_ru_enabled(self, oceanbase_client):
        admin = _raw_client()
        coll_name = f"ops_ru_{uuid.uuid4().hex[:10]}"
        coll = oceanbase_client.create_collection(
            name=coll_name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ops_ns_ru")
            ns_id = int(ns.namespace_id)
            ns.set_ru_enabled(0)
            assert read_ru_limit(admin, coll.id, ns_id, "ru_enabled") == 0
            ns.set_ru_enabled(1)
            assert read_ru_limit(admin, coll.id, ns_id, "ru_enabled") == 1
        finally:
            oceanbase_client.delete_collection(name=coll_name)
            _close(admin)

    @pytest.mark.parametrize(
        ("method_name", "config_key", "config_value"),
        [
            ("set_qps_burst", NamespaceOpsConfigKeys.QPS_BURST, 300),
            ("set_qps_refill", NamespaceOpsConfigKeys.QPS_REFILL, 150),
            ("set_tps_burst", NamespaceOpsConfigKeys.TPS_BURST, 200),
            ("set_tps_refill", NamespaceOpsConfigKeys.TPS_REFILL, 80),
            ("set_data_burst", NamespaceOpsConfigKeys.DATA_BURST, 60 * 1024 * 1024),
            ("set_data_refill", NamespaceOpsConfigKeys.DATA_REFILL, 15 * 1024 * 1024),
        ],
    )
    def test_ru_token_bucket_setters(self, oceanbase_client, method_name, config_key, config_value):
        admin = _raw_client()
        coll_name = f"ops_{config_key}_{uuid.uuid4().hex[:8]}"
        coll = oceanbase_client.create_collection(
            name=coll_name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ru_bucket_ns")
            ns_id = int(ns.namespace_id)
            getattr(ns, method_name)(config_value)
            assert read_ru_limit(admin, coll.id, ns_id, config_key) == config_value
        finally:
            oceanbase_client.delete_collection(name=coll_name)
            _close(admin)

    def test_invalid_config_key_rejected_before_pl(self, oceanbase_client):
        coll_name = f"ops_bad_{uuid.uuid4().hex[:10]}"
        coll = oceanbase_client.create_collection(
            name=coll_name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("bad_ns")
            with pytest.raises(ValueError, match="Invalid config_key"):
                ns.set_ops_config("bogus_key", 1)
            with pytest.raises(ValueError, match="Invalid config_key"):
                ns.set_ops_config("enabled", 1)
            with pytest.raises(ValueError, match="Invalid config_key"):
                coll.set_namespace_ops_config("bad_ns", "row_limit_extra", 1)
        finally:
            oceanbase_client.delete_collection(name=coll_name)

    def test_invalid_ru_enabled_value_rejected_before_pl(self, oceanbase_client):
        coll_name = f"ops_en_{uuid.uuid4().hex[:10]}"
        coll = oceanbase_client.create_collection(
            name=coll_name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("en_ns")
            with pytest.raises(ValueError, match="ru_enabled"):
                ns.set_ru_enabled(2)
        finally:
            oceanbase_client.delete_collection(name=coll_name)


class TestNamespaceOpsConfigEnforcement:
    def test_row_limit_via_sdk_blocks_insert(self, oceanbase_client):
        admin = _raw_client()
        coll_name = f"ops_rl_{uuid.uuid4().hex[:10]}"
        coll = oceanbase_client.create_collection(
            name=coll_name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("rl_ns")
            ns_id = int(ns.namespace_id)
            for i in range(3):
                _add_one(ns, i)
            assert ns.count() == 3

            ns.set_row_limit(2)
            upsert_namespace_stat(
                admin,
                collection_id=coll.id,
                namespace_id=ns_id,
                row_count=1000,
            )
            _assert_blocked(lambda: _add_one(ns, 99))

            ns.set_row_limit(-1)
            _add_one(ns, 100)
            assert ns.count() == 4
        finally:
            oceanbase_client.delete_collection(name=coll_name)
            _close(admin)

    def test_tps_throttling_via_sdk_config(self, oceanbase_client):
        coll_name = f"ops_tps_{uuid.uuid4().hex[:10]}"
        coll = oceanbase_client.create_collection(
            name=coll_name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("tps_ns")
            ns.set_ru_enabled(1)
            ns.set_tps_burst(10)
            ns.set_tps_refill(1)
            blocked = _hammer(ns, 0, 150)
            assert sum(blocked) > 0, (
                f"expected TPS throttling after SDK config, got 0/{len(blocked)} blocked"
            )
        finally:
            time.sleep(6)
            oceanbase_client.delete_collection(name=coll_name)

    def test_ru_enabled_zero_via_sdk_disables_throttling(self, oceanbase_client):
        admin = _raw_client()
        coll_name = f"ops_ru0_{uuid.uuid4().hex[:10]}"
        coll = oceanbase_client.create_collection(
            name=coll_name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ru0_ns")
            ns_id = int(ns.namespace_id)
            ns.set_ru_enabled(0)
            assert read_ru_limit(admin, coll.id, ns_id, "ru_enabled") == 0

            blocked = _hammer(ns, 0, 300)
            assert sum(blocked) >= 1, "expected the default limit to trip at least once (which triggers refresh)"
            tail = blocked[-100:]
            assert sum(tail) == 0, (
                f"after refresh loaded ru_enabled=0 the tail must stop throttling; "
                f"tail_blocked={sum(tail)}/{len(tail)}, total_blocked={sum(blocked)}"
            )
        finally:
            time.sleep(6)
            oceanbase_client.delete_collection(name=coll_name)
            _close(admin)
