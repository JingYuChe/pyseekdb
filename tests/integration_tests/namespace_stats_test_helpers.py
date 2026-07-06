"""
Helpers for namespace stats / ops-limit integration tests (sdk_namespace_stat era).

The kernel monitor reads:
- row_count / total_size from ``sdk_namespace_stat``
- row_limit / size_limit from ``sdk_namespaces.info.ops_limit``

Ops limits are configured via ``DBMS_LOGIC_TABLE.SET_NAMESPACE_OPS_CONFIG`` or direct
JSON updates on ``sdk_namespaces.info``.  All admin SQL here uses a context-free
client so catalog edits are not gated by namespace admission.
"""

from __future__ import annotations

from pyseekdb.client.meta_info import NamespaceCollectionNames, NamespaceStatsDefaults


def ensure_namespace_stats_catalog(admin) -> None:
    """Create sdk_namespace_stat + logic_table_namespaces_stats if missing."""
    stat = NamespaceCollectionNames.sdk_namespace_stat_table()
    view = NamespaceCollectionNames.logic_table_namespaces_stats_view()
    ns = NamespaceCollectionNames.sdk_namespaces_table()
    coll = "sdk_collections"
    admin._server._execute(
        f"""CREATE TABLE IF NOT EXISTS {stat} (
            collection_id CHAR(32) NOT NULL,
            namespace_id BIGINT UNSIGNED NOT NULL,
            row_count BIGINT NOT NULL DEFAULT 0,
            total_size BIGINT NOT NULL DEFAULT 0,
            total_size_included_index BIGINT NOT NULL DEFAULT 0,
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
                   s.total_size_included_index,
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
    """Patch sdk_namespaces.info.ops_limit for a namespace."""
    patches: dict[str, int] = {}
    if row_limit is not None:
        patches["row_limit"] = int(row_limit)
    if size_limit is not None:
        patches["size_limit"] = int(size_limit)
    if not patches:
        return
    import json

    patch = json.dumps({"ops_limit": patches})
    patch_escaped = patch.replace("'", "''")
    admin._server._execute(
        "UPDATE sdk_namespaces SET info = JSON_MERGE_PATCH("
        "COALESCE(info, CAST('{}' AS JSON)), "
        f"CAST('{patch_escaped}' AS JSON)) "
        f"WHERE collection_id = '{collection_id}' AND namespace_id = {int(namespace_id)}"
    )


def set_ops_limit_via_pl(
    admin,
    *,
    collection_name: str,
    namespace_name: str,
    config_key: str,
    config_value: int,
) -> None:
    """Configure limits/RU via DBMS_LOGIC_TABLE.SET_NAMESPACE_OPS_CONFIG."""
    from pyseekdb.client.validators import (
        _validate_namespace_ops_config_key,
        _validate_namespace_ops_config_value,
    )

    _validate_namespace_ops_config_key(config_key)
    _validate_namespace_ops_config_value(config_key, config_value)
    admin._server._execute(
        "CALL DBMS_LOGIC_TABLE.SET_NAMESPACE_OPS_CONFIG("
        f"'{collection_name}', '{namespace_name}', '{config_key}', {int(config_value)})"
    )


def upsert_namespace_stat(
    admin,
    *,
    collection_id: str,
    namespace_id: int,
    row_count: int,
    total_size: int = 0,
    total_size_included_index: int = 0,
) -> None:
    """Seed / override a sdk_namespace_stat row."""
    ensure_namespace_stats_catalog(admin)
    admin._server._execute(
        "INSERT INTO sdk_namespace_stat "
        "(collection_id, namespace_id, row_count, total_size, total_size_included_index, last_gather_time) "
        f"VALUES ('{collection_id}', {int(namespace_id)}, {int(row_count)}, "
        f"{int(total_size)}, {int(total_size_included_index)}, NOW(6)) "
        "ON DUPLICATE KEY UPDATE row_count=VALUES(row_count), "
        "total_size=VALUES(total_size), "
        "total_size_included_index=VALUES(total_size_included_index), "
        "last_gather_time=VALUES(last_gather_time)"
    )


def monitor_style_upsert_stat(
    admin,
    *,
    collection_id: str,
    namespace_id: int,
    row_count: int,
    total_size: int,
    total_size_included_index: int | None = None,
) -> None:
    """Replay monitor gather UPSERT (estimate columns only; ops_limit lives in info JSON)."""
    if total_size_included_index is None:
        total_size_included_index = total_size
    admin._server._execute(
        "INSERT INTO sdk_namespace_stat "
        "(collection_id, namespace_id, row_count, total_size, total_size_included_index, last_gather_time) "
        f"VALUES ('{collection_id}', {int(namespace_id)}, {int(row_count)}, "
        f"{int(total_size)}, {int(total_size_included_index)}, NOW(6)) "
        "ON DUPLICATE KEY UPDATE row_count=VALUES(row_count), "
        "total_size=VALUES(total_size), "
        "total_size_included_index=VALUES(total_size_included_index), "
        "last_gather_time=VALUES(last_gather_time)"
    )


def read_ops_limit(admin, collection_id: str, namespace_id: int, key: str) -> int | None:
    rows = admin._server._execute(
        "SELECT CAST(JSON_UNQUOTE(JSON_EXTRACT(info, "
        f"'$.ops_limit.{key}')) AS SIGNED) AS v "
        f"FROM sdk_namespaces WHERE collection_id = '{collection_id}' "
        f"AND namespace_id = {int(namespace_id)} LIMIT 1"
    )
    if not rows:
        return None
    row = rows[0]
    val = row["v"] if isinstance(row, dict) else row[0]
    return None if val is None else int(val)


def read_ru_limit(admin, collection_id: str, namespace_id: int, key: str) -> int | None:
    rows = admin._server._execute(
        "SELECT CAST(JSON_UNQUOTE(JSON_EXTRACT(info, "
        f"'$.ru_limit.{key}')) AS SIGNED) AS v "
        f"FROM sdk_namespaces WHERE collection_id = '{collection_id}' "
        f"AND namespace_id = {int(namespace_id)} LIMIT 1"
    )
    if not rows:
        return None
    row = rows[0]
    val = row["v"] if isinstance(row, dict) else row[0]
    return None if val is None else int(val)


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
    sql = (
        "SELECT collection_name, namespace_name, row_count, total_size, "
        "total_size_included_index, last_gather_time "
        "FROM logic_table_namespaces_stats WHERE 1=1"
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
