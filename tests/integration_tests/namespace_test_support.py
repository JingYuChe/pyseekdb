"""Helpers to skip namespace integration tests on unsupported backends/versions."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest

from pyseekdb.client.client_base import NAMESPACE_MIN_LAKEBASE_VERSION

_OB_NAMESPACE_SUPPORT: tuple[bool, str] | None = None

# Tests that must not be short-circuited by the OB version gate.
_VERSION_SKIP_EXEMPT_TEST_NAMES = frozenset({
    "test_old_lakebase_version_rejected",
    "test_standard_oceanbase_rejected",
    "test_min_version_constant",
})


def is_namespace_integration_test(nodeid: str, fspath: str | Path) -> bool:
    """Return whether the collected item belongs to the namespace integration suite."""
    stem = Path(fspath).stem
    return stem.startswith("test_namespace") or stem == "test_logic_table_monitoring_p1"


def should_exempt_from_namespace_version_skip(item) -> bool:
    """Return whether the test validates version handling itself."""
    return item.name in _VERSION_SKIP_EXEMPT_TEST_NAMES


def infer_integration_test_mode(item) -> str | None:
    """Infer embedded/server/oceanbase mode from parametrized fixtures."""
    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        for key in ("db_client", "admin_client", "_mode"):
            if key in callspec.params:
                return str(callspec.params[key])

    fixturenames = getattr(item, "fixturenames", ())
    if "oceanbase_client" in fixturenames or "oceanbase_admin_client" in fixturenames:
        return "oceanbase"
    if "embedded_client" in fixturenames or "embedded_admin_client" in fixturenames:
        return "embedded"
    if "server_client" in fixturenames or "server_admin_client" in fixturenames:
        return "server"

    # Files like test_logic_table_monitoring_p1.py build OB clients inline.
    if is_namespace_integration_test(item.nodeid, item.fspath):
        return "oceanbase"
    return None


def probe_oceanbase_namespace_support() -> tuple[bool, str]:
    """Probe whether the configured OceanBase instance supports namespace collections."""
    global _OB_NAMESPACE_SUPPORT
    if _OB_NAMESPACE_SUPPORT is not None:
        return _OB_NAMESPACE_SUPPORT

    import pyseekdb

    host = os.environ.get("OB_HOST", "localhost")
    port = int(os.environ.get("OB_PORT", "11202"))
    tenant = os.environ.get("OB_TENANT", "mysql")
    database = os.environ.get("OB_DATABASE", "test")
    user = os.environ.get("OB_USER", "root")
    password = os.environ.get("OB_PASSWORD", "")

    client = None
    try:
        client = pyseekdb.Client(
            host=host,
            port=port,
            tenant=tenant,
            database=database,
            user=user,
            password=password,
        )
        db_type, version = client._server.detect_db_type_and_version()
        is_lakebase = client._server._is_lakebase_cluster()
    except Exception as exc:
        _OB_NAMESPACE_SUPPORT = (False, f"OceanBase unavailable for namespace tests ({host}:{port}): {exc}")
        return _OB_NAMESPACE_SUPPORT
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()

    if db_type.lower() != "oceanbase":
        _OB_NAMESPACE_SUPPORT = (
            False,
            f"namespace tests require LakeBase (OceanBase Database AI), got {db_type}",
        )
    elif not is_lakebase:
        _OB_NAMESPACE_SUPPORT = (
            False,
            "namespace tests require LakeBase (OceanBase Database AI); connected cluster is standard OceanBase",
        )
    elif version < NAMESPACE_MIN_LAKEBASE_VERSION:
        _OB_NAMESPACE_SUPPORT = (
            False,
            f"namespace tests require LakeBase >= {NAMESPACE_MIN_LAKEBASE_VERSION}, current {version}",
        )
    else:
        _OB_NAMESPACE_SUPPORT = (True, "")
    return _OB_NAMESPACE_SUPPORT


def maybe_skip_namespace_integration_test(item) -> None:
    """Skip namespace integration tests when the active mode/kernel cannot run them."""
    if not is_namespace_integration_test(item.nodeid, item.fspath):
        return

    mode = infer_integration_test_mode(item)
    if mode in ("embedded", "server"):
        pytest.skip("namespace collections require OceanBase (skip embedded/server integration modes)")

    if should_exempt_from_namespace_version_skip(item):
        return

    supported, reason = probe_oceanbase_namespace_support()
    if not supported:
        pytest.skip(reason)
