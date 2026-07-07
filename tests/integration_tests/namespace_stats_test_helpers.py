"""
Helpers for namespace stats / ops-limit integration tests (sdk_namespace_stat era).

The kernel monitor reads:
- row_count / total_size from ``sdk_namespace_stat``
- row_limit / size_limit / RU settings from flat ``sdk_namespaces.info`` JSON

Ops limits are configured via ``SET NAMESPACE RU CONFIG`` or direct
JSON updates on ``sdk_namespaces.info``.  All admin SQL here uses a context-free
client so catalog edits are not gated by namespace admission.
"""

from __future__ import annotations

import json

from pyseekdb.client.meta_info import NamespaceCollectionNames, NamespaceStatsDefaults


def ensure_namespace_stats_catalog(admin) -> None:
    """Create sdk_namespace_stat + logic_table_namespace_stats if missing."""
    stat = NamespaceCollectionNames.sdk_namespace_stat_table()
    view = NamespaceCollectionNames.logic_table_namespace_stats_view()
    ns = NamespaceCollectionNames.sdk_namespaces_table()
    coll = "sdk_collections"
    admin._server._execute(
        f"""CREATE TABLE IF NOT EXISTS {stat} (
            collection_id CHAR(32) NOT NULL,
            namespace_id BIGINT UNSIGNED NOT NULL,
            row_count BIGINT NOT NULL DEFAULT 0,
            total_size BIGINT NOT NULL DEFAULT 0,
            total_size_with_index BIGINT NOT NULL DEFAULT 0,
            last_gather_time TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
            PRIMARY KEY (namespace_id),
            KEY idx_sdk_ns_stat_by_collection (collection_id)
        ) DEFAULT CHARSET=utf8mb4 PARTITION BY KEY(namespace_id) PARTITIONS 8"""
    )
    admin._server._execute(
        f"""CREATE OR REPLACE VIEW {view} AS
            SELECT c.collection_name,
                   n.namespace_name,
                   s.row_count,
                   s.total_size,
                   s.total_size_with_index,
                   s.last_gather_time
            FROM {stat} s
            JOIN {ns} n ON s.namespace_id = n.namespace_id
            JOIN {coll} c ON n.collection_id = c.collection_id"""
    )


def set_ops_limit(
    admin,
    *,
    collection_id: str,
    namespace_id: int,
    row_limit: int | None = None,
    size_limit: int | None = None,
) -> None:
    """Patch flat row_limit / size_limit fields in sdk_namespaces.info."""
    patches: dict[str, int] = {}
    if row_limit is not None:
        patches["row_limit"] = int(row_limit)
    if size_limit is not None:
        patches["size_limit"] = int(size_limit)
    if not patches:
        return
    patch = json.dumps(patches, separators=(",", ":"))
    patch_escaped = patch.replace("'", "''")
    admin._server._execute(
        "UPDATE sdk_namespaces SET info = JSON_MERGE_PATCH("
        "COALESCE(info, CAST('{}' AS JSON)), "
        f"CAST('{patch_escaped}' AS JSON)) "
        f"WHERE collection_id = '{collection_id}' AND namespace_id = {int(namespace_id)}"
    )


def set_ru_config_via_pl(
    admin,
    *,
    collection_name: str,
    namespace_name: str,
    config: dict,
) -> None:
    """Configure limits/RU via .SET NAMESPACE PS CONFIG (batch JSON)."""
    from pyseekdb.client.validators import _validate_namespace_ru_config

    _validate_namespace_ru_config(config)
    config_json = json.dumps(config, separators=(",", ":"))
    config_escaped = config_json.replace("'", "''")
    admin._server._execute(
        "CALL DBMS_LOGIC_TABLE.SET_NAMESPACE_RU_CONFIG("
        f"'{collection_name}', '{namespace_name}', '{config_escaped}')"
    )


def upsert_namespace_stat(
    admin,
    *,
    collection_id: str,
    namespace_id: int,
    row_count: int,
    total_size: int = 0,
    total_size_with_index: int = 0,
) -> None:
    """Seed / override a sdk_namespace_stat row."""
    ensure_namespace_stats_catalog(admin)
    admin._server._execute(
        "INSERT INTO sdk_namespace_stat "
        "(collection_id, namespace_id, row_count, total_size, total_size_with_index, last_gather_time) "
        f"VALUES ('{collection_id}', {int(namespace_id)}, {int(row_count)}, "
        f"{int(total_size)}, {int(total_size_with_index)}, NOW(6)) "
        "ON DUPLICATE KEY UPDATE row_count=VALUES(row_count), "
        "total_size=VALUES(total_size), "
        "total_size_with_index=VALUES(total_size_with_index), "
        "last_gather_time=VALUES(last_gather_time)"
    )


def monitor_style_upsert_stat(
    admin,
    *,
    collection_id: str,
    namespace_id: int,
    row_count: int,
    total_size: int,
    total_size_with_index: int | None = None,
) -> None:
    """Replay monitor gather UPSERT (estimate columns only; limits live in info JSON)."""
    if total_size_with_index is None:
        total_size_with_index = total_size
    admin._server._execute(
        "INSERT INTO sdk_namespace_stat "
        "(collection_id, namespace_id, row_count, total_size, total_size_with_index, last_gather_time) "
        f"VALUES ('{collection_id}', {int(namespace_id)}, {int(row_count)}, "
        f"{int(total_size)}, {int(total_size_with_index)}, NOW(6)) "
        "ON DUPLICATE KEY UPDATE row_count=VALUES(row_count), "
        "total_size=VALUES(total_size), "
        "total_size_with_index=VALUES(total_size_with_index), "
        "last_gather_time=VALUES(last_gather_time)"
    )


def _read_info_field(admin, collection_id: str, namespace_id: int, key: str) -> int | None:
    rows = admin._server._execute(
        "SELECT CAST(JSON_UNQUOTE(JSON_EXTRACT(info, "
        f"'$.{key}')) AS SIGNED) AS v "
        f"FROM sdk_namespaces WHERE collection_id = '{collection_id}' "
        f"AND namespace_id = {int(namespace_id)} LIMIT 1"
    )
    if not rows:
        return None
    row = rows[0]
    val = row["v"] if isinstance(row, dict) else row[0]
    return None if val is None else int(val)


def read_ops_limit(admin, collection_id: str, namespace_id: int, key: str) -> int | None:
    return _read_info_field(admin, collection_id, namespace_id, key)


def read_ru_limit(admin, collection_id: str, namespace_id: int, key: str) -> int | None:
    return _read_info_field(admin, collection_id, namespace_id, key)


def read_namespace_stat(admin, collection_id: str, namespace_id: int, col: str) -> int | None:
    rows = admin._server._execute(
        f"SELECT {col} AS v FROM sdk_namespace_stat "
        f"WHERE collection_id = '{collection_id}' AND namespace_id = {int(namespace_id)} LIMIT 1"
    )
    if not rows:
        return None
    row = rows[0]
    return int(row["v"] if isinstance(row, dict) else row[0])


def query_stats_view(admin, *, collection_name: str | None = None, namespace_name: str | None = None):
    view = NamespaceCollectionNames.logic_table_namespace_stats_view()
    sql = (
        "SELECT collection_name, namespace_name, row_count, total_size, "
        f"total_size_with_index, last_gather_time FROM {view} WHERE 1=1"
    )
    if collection_name is not None:
        sql += f" AND collection_name = '{collection_name}'"
    if namespace_name is not None:
        sql += f" AND namespace_name = '{namespace_name}'"
    return admin._server._execute(sql)


def count_namespace_stat_rows(admin, collection_id: str, namespace_id: int) -> int:
    rows = admin._server._execute(
        "SELECT COUNT(*) AS cnt FROM sdk_namespace_stat "
        f"WHERE collection_id = '{collection_id}' AND namespace_id = {int(namespace_id)}"
    )
    row = rows[0]
    return int(row["cnt"] if isinstance(row, dict) else row[0])


def default_ops_limit_row_limit() -> int:
    return NamespaceStatsDefaults.ROW_LIMIT


def default_ops_limit_size_limit() -> int:
    return NamespaceStatsDefaults.SIZE_LIMIT


_NAMESPACE_STATS_JOB_NAME = "logic_table_namespace_stats_job"
_DEFAULT_STATS_JOB_REPEAT = "FREQ=SECONDLY; INTERVAL=60"


def _scheduler_row_field(row, field: str) -> str:
    if isinstance(row, dict):
        for key in (field, field.upper(), field.lower()):
            if key in row and row[key] is not None:
                return str(row[key])
        return ""
    columns = field.split(",")
    if len(columns) == 1:
        idx_map = {
            "job_name": 0,
            "enabled": 1,
            "repeat_interval": 2,
            "start_date": 3,
            "max_run_duration": 4,
        }
        return str(row[idx_map.get(field, 0)])
    return str(row[0])


def query_namespace_stats_job(
    admin,
    *,
    columns: str = "job_name, enabled, repeat_interval, start_date, max_run_duration",
):
    """Return scheduler metadata row for logic_table_namespace_stats_job."""
    return admin._server._execute(
        f"SELECT {columns} FROM oceanbase.__all_tenant_scheduler_job "
        f"WHERE job_name = '{_NAMESPACE_STATS_JOB_NAME}' LIMIT 1"
    )


def read_scheduler_repeat_interval(admin) -> str:
    rows = query_namespace_stats_job(admin, columns="repeat_interval")
    assert rows, f"{_NAMESPACE_STATS_JOB_NAME} not found"
    return _scheduler_row_field(rows[0], "repeat_interval")


def read_scheduler_enabled(admin) -> str:
    rows = query_namespace_stats_job(admin, columns="enabled")
    assert rows, f"{_NAMESPACE_STATS_JOB_NAME} not found"
    return _scheduler_row_field(rows[0], "enabled")


def read_scheduler_start_date(admin) -> str:
    rows = query_namespace_stats_job(admin, columns="start_date")
    assert rows, f"{_NAMESPACE_STATS_JOB_NAME} not found"
    return _scheduler_row_field(rows[0], "start_date")


def set_scheduler_attribute(admin, attribute: str, value: str) -> None:
    """CALL DBMS_SCHEDULER.SET_ATTRIBUTE for the namespace stats gather job."""
    escaped = value.replace("'", "''")
    admin._server._execute(
        "CALL DBMS_SCHEDULER.SET_ATTRIBUTE("
        f"'{_NAMESPACE_STATS_JOB_NAME}', '{attribute}', '{escaped}')"
    )


def disable_namespace_stats_job(admin) -> None:
    admin._server._execute(f"CALL DBMS_SCHEDULER.DISABLE('{_NAMESPACE_STATS_JOB_NAME}')")


def enable_namespace_stats_job(admin) -> None:
    admin._server._execute(f"CALL DBMS_SCHEDULER.ENABLE('{_NAMESPACE_STATS_JOB_NAME}')")


def restore_namespace_stats_job(admin) -> None:
    """Best-effort restore default scheduler settings after ops tests."""
    rows = query_namespace_stats_job(admin, columns="job_name")
    if not rows:
        return
    enable_namespace_stats_job(admin)
    set_scheduler_attribute(admin, "repeat_interval", _DEFAULT_STATS_JOB_REPEAT)


def call_ops_config_pl_raw(
    admin,
    *,
    collection_name: str,
    namespace_name: str,
    config_json: str,
) -> None:
    """Invoke SET NAMESPACE RU CONFIG without SDK-side validation (kernel tests)."""
    config_escaped = config_json.replace("'", "''")
    admin._server._execute(
        "CALL DBMS_LOGIC_TABLE.SET_NAMESPACE_RU_CONFIG("
        f"'{collection_name}', '{namespace_name}', '{config_escaped}')"
    )
