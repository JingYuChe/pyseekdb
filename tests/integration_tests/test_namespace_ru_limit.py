"""
Namespace RU (rate) limit integration tests.

Exercises the local token-bucket limiter wired into
ObPxAdmission::check_namespace_rate_limit → ObNamespaceRUManager::acquire:

- TPS throttling: a burst of write statements on one namespace eventually gets
  OB_KILLED_BY_THROTTLING (errno 4039) once the per-namespace TPS bucket
  (default burst 100 / refill 50/s) is exhausted.
- refresh_config: writing {"ru_limit":{"ru_enabled":0}} into sdk_namespaces.info
  and then tripping a throttle makes the controller reload config from the
  internal table (lazy, on-throttle) and lift the limit — proving
  controller->refresh_config reads sdk_namespaces.info and applies it.

RU limiting is rate/time dependent, so throttling counts are not asserted
exactly; the tests assert the qualitative behaviour (throttle happens; after
ru_enabled=0 + a throttle, the tail of a burst stops failing).
"""

from __future__ import annotations

import os
import time
import uuid

from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT, ns_schema

import pyseekdb


def _raw_client():
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
    """Do n single-row adds as fast as possible; return per-add blocked(True)/ok(False)."""
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


def _set_info(admin, collection_id: str, namespace_id: int, info_json: str) -> None:
    """Overwrite sdk_namespaces.info with the given JSON literal (admin conn, no ns context)."""
    admin._server._execute(
        f"UPDATE sdk_namespaces SET info = CAST('{info_json}' AS JSON) "
        f"WHERE collection_id = '{collection_id}' AND namespace_id = {int(namespace_id)}"
    )


def _set_info_ru_disabled(admin, collection_id: str, namespace_id: int) -> None:
    _set_info(admin, collection_id, namespace_id, '{"ru_limit": {"ru_enabled": 0}}')


class TestNamespaceRuLimit:
    """Per-namespace TPS/QPS token-bucket rate limiting."""

    def test_tps_throttling_kicks_in(self, oceanbase_client):
        """A rapid burst of writes on one namespace trips the TPS limiter (4039)."""
        owner = oceanbase_client
        name = f"ru_tps_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_tps")
            # default TPS bucket: burst 100 / refill 50/s -> 300 rapid single writes must throttle.
            blocked = _hammer(ns, 0, 300)
            n_blocked = sum(blocked)
            assert n_blocked > 0, (
                f"expected some writes throttled by TPS limit, got 0/{len(blocked)} (RU limiter not enforcing?)"
            )
        finally:
            # Let the per-namespace buckets refill so teardown queries (which carry the
            # session ns context) are not themselves throttled.
            time.sleep(6)
            owner.delete_collection(name=name)

    def test_refresh_config_from_info_disables_limit(self, oceanbase_client):
        """ru_enabled=0 in sdk_namespaces.info, loaded lazily on the first throttle, lifts the limit.

        Note the limiter's 2-minute refresh cooldown: only the FIRST throttle on a fresh
        controller refreshes (last_refresh_time==0). So ru_enabled=0 must already be in the
        internal table before the burst — then the first throttle loads it and the tail clears.
        """
        owner = oceanbase_client
        admin = _raw_client()
        name = f"ru_refresh_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_ref")
            ns_id = int(ns.namespace_id)

            # Ops disables RU limit for this namespace BEFORE any traffic/refresh.
            _set_info_ru_disabled(admin, coll.id, ns_id)

            # Burst: first ~100 pass the default burst, ~101 trips a throttle which triggers
            # refresh_config -> reads ru_enabled=0 -> buckets unlimited -> the rest all pass.
            blocked = _hammer(ns, 0, 300)
            assert sum(blocked) >= 1, "expected the default limit to trip at least once (which triggers refresh)"
            tail = blocked[-100:]
            assert sum(tail) == 0, (
                f"after refresh loaded ru_enabled=0 the tail must stop throttling; "
                f"tail_blocked={sum(tail)}/{len(tail)}, total_blocked={sum(blocked)}"
            )
        finally:
            time.sleep(6)
            owner.delete_collection(name=name)
            _close(admin)

    def test_refresh_config_numeric_override_lifts_limit(self, oceanbase_client):
        """Large qps/tps burst+refill in info.ru_limit (not ru_enabled=0) also lifts the limit."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"ru_num_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_num")
            ns_id = int(ns.namespace_id)
            # numeric field overrides (ru_enabled stays 1) — proves per-field parsing, not just ru_enabled.
            _set_info(
                admin,
                coll.id,
                ns_id,
                '{"ru_limit": {"ru_enabled": 1, "qps_burst": 1000000, "qps_refill": 1000000, '
                '"tps_burst": 1000000, "tps_refill": 1000000}}',
            )
            blocked = _hammer(ns, 0, 300)
            assert sum(blocked) >= 1, "default limit should trip once to trigger refresh"
            tail = blocked[-100:]
            assert sum(tail) == 0, f"large tps/qps override must lift the limit; tail_blocked={sum(tail)}/{len(tail)}"
        finally:
            time.sleep(6)
            owner.delete_collection(name=name)
            _close(admin)

    def test_refresh_config_malformed_info_is_safe(self, oceanbase_client):
        """Malformed ru_limit must not crash refresh and must not disable the default limit."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"ru_bad_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_bad")
            ns_id = int(ns.namespace_id)
            # ru_limit is a string, not an object -> JSON_EXTRACT('$.ru_limit.ru_enabled') is NULL
            # -> IFNULL defaults keep the safe built-in limits; must not raise a non-throttle error.
            _set_info(admin, coll.id, ns_id, '{"ru_limit": "garbage"}')
            # _hammer re-raises any NON-throttle exception, so reaching the assert means no crash.
            blocked = _hammer(ns, 0, 300)
            assert sum(blocked) > 0, (
                f"malformed config must fall back to the default limit (still throttling), got 0/{len(blocked)} blocked"
            )
        finally:
            time.sleep(6)
            owner.delete_collection(name=name)
            _close(admin)
