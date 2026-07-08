"""
Integration tests for logic_table_namespace_stats_job (DBMS_Scheduler internal job).

Namespace stats gathering interval is configured via DBMS_SCHEDULER.SET_ATTRIBUTE on
``logic_table_namespace_stats_job`` (see SET_NAMESPACE_RESOURCE_LIMIT interface doc §二).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

from namespace_stats_test_helpers import (
    disable_namespace_stats_job,
    enable_namespace_stats_job,
    query_namespace_stats_job,
    read_scheduler_enabled,
    read_scheduler_repeat_interval,
    read_scheduler_start_date,
    restore_namespace_stats_job,
    set_scheduler_attribute,
)

_JOB_NAME = "logic_table_namespace_stats_job"
_DEFAULT_REPEAT = "FREQ=SECONDLY; INTERVAL=60"


def _raw_client():
    import pyseekdb

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


def _row_field(row, field: str) -> str:
    if isinstance(row, dict):
        return str(row.get(field) or row.get(field.upper()) or "")
    idx = {"job_name": 0, "enabled": 1, "repeat_interval": 2, "start_date": 3}.get(field, 0)
    return str(row[idx])


def _require_stats_job(admin) -> None:
    rows = query_namespace_stats_job(admin)
    if not rows:
        pytest.skip(f"{_JOB_NAME} not found (cluster may predate scheduler migration)")


def _assert_scheduler_rejected(exc: Exception) -> None:
    """Scheduler rejects invalid repeat_interval (code varies by OB build)."""
    msg = str(exc).lower()
    assert (
        "4002" in str(exc)
        or "1235" in str(exc)
        or "1064" in str(exc)
        or "invalid" in msg
        or "argument" in msg
        or "not supported" in msg
        or "parse error" in msg
    )


@pytest.fixture
def stats_job_admin():
    admin = _raw_client()
    _require_stats_job(admin)
    try:
        yield admin
    finally:
        restore_namespace_stats_job(admin)
        _close(admin)


class TestNamespaceStatsSchedulerJob:
    def test_namespace_stats_scheduler_job_exists(self, oceanbase_client, stats_job_admin):
        del oceanbase_client
        rows = query_namespace_stats_job(stats_job_admin)
        row = rows[0]
        assert _row_field(row, "job_name") == _JOB_NAME
        assert read_scheduler_enabled(stats_job_admin) in ("1", "True", "true")
        assert read_scheduler_repeat_interval(stats_job_admin) == _DEFAULT_REPEAT


class TestNamespaceStatsSchedulerSetAttribute:
    def test_set_attribute_repeat_interval_minutely_2(self, oceanbase_client, stats_job_admin):
        del oceanbase_client
        set_scheduler_attribute(stats_job_admin, "repeat_interval", "FREQ=MINUTELY; INTERVAL=2")
        assert read_scheduler_repeat_interval(stats_job_admin) == "FREQ=MINUTELY; INTERVAL=2"

    def test_set_attribute_repeat_interval_secondly_30(self, oceanbase_client, stats_job_admin):
        del oceanbase_client
        set_scheduler_attribute(stats_job_admin, "repeat_interval", "FREQ=SECONDLY; INTERVAL=30")
        assert read_scheduler_repeat_interval(stats_job_admin) == "FREQ=SECONDLY; INTERVAL=30"

    def test_set_attribute_start_date(self, oceanbase_client, stats_job_admin):
        del oceanbase_client
        before = read_scheduler_start_date(stats_job_admin)
        target = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        set_scheduler_attribute(stats_job_admin, "start_date", target)
        after = read_scheduler_start_date(stats_job_admin)
        assert after != before
        assert target[:10] in after or after.startswith(target[:10])

    def test_disable_and_enable_job(self, oceanbase_client, stats_job_admin):
        del oceanbase_client
        disable_namespace_stats_job(stats_job_admin)
        assert read_scheduler_enabled(stats_job_admin) in ("0", "False", "false")
        enable_namespace_stats_job(stats_job_admin)
        assert read_scheduler_enabled(stats_job_admin) in ("1", "True", "true")

    def test_set_attribute_invalid_repeat_interval_rejected(self, oceanbase_client, stats_job_admin):
        del oceanbase_client
        with pytest.raises(Exception) as excinfo:
            set_scheduler_attribute(stats_job_admin, "repeat_interval", "FREQ=MINUTELY; INTERVAL=0")
        _assert_scheduler_rejected(excinfo.value)
        assert read_scheduler_repeat_interval(stats_job_admin) == _DEFAULT_REPEAT
