"""
Integration tests for tenant parameter logic_table_namespace_stats_gather_interval.

The parameter is OB_TENANT_PARAMETER (scope = TENANT): each tenant has its own
effective value. ObLogicalTableMonitor reads TENANT_CONF for the tenant that owns
the logical tables (typically ``mysql``), not cluster-wide sys settings.

Ops usage on the logical-table tenant::

  -- obclient -uroot@mysql ...
  ALTER SYSTEM SET logic_table_namespace_stats_gather_interval = '10m';
  SHOW PARAMETERS LIKE 'logic_table_namespace_stats_gather_interval';

From ``root@sys`` you must target the tenant explicitly::

  ALTER SYSTEM SET logic_table_namespace_stats_gather_interval = '10m' TENANT = 'mysql';

When reading SHOW PARAMETERS, use the ``value`` column (current effective value).
``default_value`` is always the compile-time default (60s) and must not be used
to verify ALTER results.
"""

from __future__ import annotations

import os
import time

import pytest

_PARAM = "logic_table_namespace_stats_gather_interval"
_DEFAULT = "60s"
_LOWER_OVERHEAD = "10m"


def _logic_table_tenant() -> str:
    """Tenant where namespace / logical-table workloads run (OB_TENANT in CI)."""
    return os.environ.get("OB_TENANT", "mysql")


def _raw_client():
    import pyseekdb

    tenant = _logic_table_tenant()
    return pyseekdb.Client(
        host=os.environ.get("OB_HOST", "127.0.0.1"),
        port=int(os.environ.get("OB_PORT", "10902")),
        tenant=tenant,
        database=os.environ.get("OB_DATABASE", "test"),
        user=os.environ.get("OB_USER", "root"),
        password=os.environ.get("OB_PASSWORD", ""),
    )


def _close(client) -> None:
    if hasattr(client, "close"):
        client.close()


def _row_field(row, field: str) -> str:
    if isinstance(row, dict):
        return str(row.get(field) or row.get(field.upper()) or "")
    columns = (
        "zone",
        "svr_type",
        "svr_ip",
        "svr_port",
        "name",
        "data_type",
        "value",
        "info",
        "section",
        "scope",
        "source",
        "edit_level",
        "default_value",
        "isdefault",
    )
    try:
        return str(row[columns.index(field)])
    except (ValueError, IndexError) as exc:
        raise AssertionError(f"cannot read {field!r} from SHOW PARAMETERS row: {row!r}") from exc


def _show_parameter_rows(client):
    return client._server._execute(f"SHOW PARAMETERS LIKE '{_PARAM}'")


def _read_gather_interval(client, *, retries: int = 5, delay_s: float = 0.2) -> str | None:
    last_rows = None
    for attempt in range(retries):
        last_rows = _show_parameter_rows(client)
        if not last_rows:
            return None
        value = _row_field(last_rows[0], "value")
        if attempt == retries - 1 or value:
            return value
        time.sleep(delay_s)
    return None


def _set_gather_interval(client, interval: str) -> None:
    # Must run in the logical-table tenant session (see module docstring).
    client._server._execute(f"ALTER SYSTEM SET {_PARAM} = '{interval}'")


def _assert_gather_interval(client, expected: str) -> None:
    rows = _show_parameter_rows(client)
    if expected == _DEFAULT:
        if not rows:
            return
        assert _row_field(rows[0], "value") == _DEFAULT
        return
    assert rows, f"expected SHOW PARAMETERS row for {_PARAM}={expected!r}"
    row = rows[0]
    assert _row_field(row, "scope").upper() == "TENANT", (
        f"{_PARAM} must be TENANT-scoped, got scope={_row_field(row, 'scope')!r}"
    )
    assert _row_field(row, "value") == expected, (
        f"read value column only (not default_value={_row_field(row, 'default_value')!r})"
    )


def _parameter_available(client) -> bool:
    if _logic_table_tenant() == "sys":
        return False
    try:
        rows = _show_parameter_rows(client)
        if rows and _row_field(rows[0], "scope").upper() != "TENANT":
            return False
    except Exception:
        return False

    previous = _read_gather_interval(client)
    try:
        _set_gather_interval(client, _LOWER_OVERHEAD)
        return _read_gather_interval(client) == _LOWER_OVERHEAD
    except Exception:
        return False
    finally:
        try:
            _set_gather_interval(client, previous if previous is not None else _DEFAULT)
        except Exception:
            pass


class TestNamespaceStatsGatherInterval:
    def test_alter_gather_interval_10m_then_restore_60s(self, oceanbase_client):
        tenant = _logic_table_tenant()
        if tenant == "sys":
            pytest.skip(
                f"{_PARAM} is TENANT-scoped; connect as root@{tenant!r} or use "
                f"ALTER SYSTEM SET ... TENANT = 'mysql' from sys"
            )

        admin = _raw_client()
        if not _parameter_available(admin):
            _close(admin)
            pytest.skip(f"{_PARAM} is not available on tenant {tenant!r}")

        try:
            _set_gather_interval(admin, _LOWER_OVERHEAD)
            _assert_gather_interval(admin, _LOWER_OVERHEAD)

            _set_gather_interval(admin, _DEFAULT)
            _assert_gather_interval(admin, _DEFAULT)
        finally:
            try:
                _set_gather_interval(admin, _DEFAULT)
            except Exception:
                pass
            _close(admin)
