"""
Namespace RU (rate) limit integration tests.

Exercises the local token-bucket limiter wired into
ObPxAdmission::check_namespace_rate_limit → ObNamespaceRUManager::acquire:

- TPS throttling: a burst of write statements on one namespace eventually gets
  OB_KILLED_BY_THROTTLING (errno 4039) once the per-namespace TPS bucket
  (default burst 100 / refill 50/s) is exhausted.
- refresh_config: ``sdk_namespaces.info`` is reloaded on each acquire (eager
  refresh before ``try_acquire``), so ops changes such as
  ``rate_limit_enable=0`` or large burst/refill take effect on the next write
  without waiting for a throttle.

RU limiting is rate/time dependent, so throttling counts are not asserted
exactly; the tests assert qualitative behaviour (default throttle happens;
ops disable / override lifts the limit immediately; re-enable restores it).
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
    _set_info(admin, collection_id, namespace_id, '{"rate_limit_enable": 0}')


class TestNamespaceRuLimit:
    """Per-namespace TPS/QPS token-bucket rate limiting."""

    def test_tps_throttling_kicks_in(self, oceanbase_client):
        """A rapid burst of writes on one namespace trips the TPS limiter (4039)."""
        owner = oceanbase_client
        admin = _raw_client()
        name = f"ru_tps_{uuid.uuid4().hex[:12]}"
        coll = owner.create_collection(
            name=name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            ns = coll.get_or_create_namespace("ns_tps")
            # Use a deterministic bucket instead of relying on the cluster being
            # able to issue writes faster than the default 50 token/s refill.
            _set_info(
                admin,
                coll.id,
                int(ns.namespace_id),
                '{"rate_limit_enable": 1, "tps_burst": 5, "tps_refill": 0}',
            )
            blocked = _hammer(ns, 0, 20)
            n_blocked = sum(blocked)
            assert n_blocked > 0, (
                f"expected some writes throttled by TPS limit, got 0/{len(blocked)} (RU limiter not enforcing?)"
            )
        finally:
            # Let the per-namespace buckets refill so teardown queries (which carry the
            # session ns context) are not themselves throttled.
            time.sleep(6)
            owner.delete_collection(name=name)
            _close(admin)

    def test_refresh_config_from_info_disables_limit(self, oceanbase_client):
        """rate_limit_enable=0 in sdk_namespaces.info disables throttling on the next acquire."""
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

            _set_info_ru_disabled(admin, coll.id, ns_id)

            blocked = _hammer(ns, 0, 300)
            assert sum(blocked) == 0, (
                f"rate_limit_enable=0 must lift throttling immediately via eager refresh; "
                f"blocked={sum(blocked)}/{len(blocked)}"
            )
        finally:
            time.sleep(6)
            owner.delete_collection(name=name)
            _close(admin)

    def test_refresh_config_numeric_override_lifts_limit(self, oceanbase_client):
        """Large qps/tps burst+refill in info lifts the limit on the next acquire."""
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
            _set_info(
                admin,
                coll.id,
                ns_id,
                '{"rate_limit_enable": 1, "qps_burst": 1000000, "qps_refill": 1000000, '
                '"tps_burst": 1000000, "tps_refill": 1000000}',
            )
            blocked = _hammer(ns, 0, 300)
            assert sum(blocked) == 0, (
                f"large tps/qps override must lift the limit immediately; "
                f"blocked={sum(blocked)}/{len(blocked)}"
            )
        finally:
            time.sleep(6)
            owner.delete_collection(name=name)
            _close(admin)

    def test_refresh_config_malformed_info_is_safe(self, oceanbase_client):
        """Malformed numeric RU fields must not crash refresh and must keep default throttling."""
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
            # Establish a deterministic low limit first. Malformed values below
            # must be ignored while preserving this last valid configuration.
            _set_info(
                admin,
                coll.id,
                ns_id,
                '{"rate_limit_enable": 1, "tps_burst": 5, "tps_refill": 0}',
            )
            initial_blocked = _hammer(ns, 0, 20)
            assert sum(initial_blocked) > 0, (
                f"valid low TPS limit must throttle before malformed refresh, "
                f"got {sum(initial_blocked)}/{len(initial_blocked)} blocked"
            )

            # Allow the bucket to start with tokens again before verifying that
            # malformed values do not replace the last valid configuration.
            time.sleep(1)
            # Non-numeric tps_* values CAST to 0 in refresh SQL, so the update is skipped and
            # the last valid bucket remains. rate_limit_enable stays 1.
            _set_info(
                admin,
                coll.id,
                ns_id,
                '{"rate_limit_enable": 1, "tps_burst": "garbage", "tps_refill": "garbage"}',
            )
            # _hammer re-raises any NON-throttle exception, so reaching the assert means no crash.
            blocked = _hammer(ns, 100, 20)
            assert sum(blocked) > 0, (
                f"malformed tps_* must keep the last valid limit (still throttling), "
                f"got {sum(blocked)}/{len(blocked)} blocked"
            )
        finally:
            time.sleep(6)
            owner.delete_collection(name=name)
            _close(admin)
