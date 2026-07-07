"""
Metadata information for collection fields and namespace catalog naming.
"""

from typing import ClassVar


class CollectionFieldNames:
    """Standard field names used in collection record payloads."""

    ID = "_id"
    DOCUMENT = "document"
    EMBEDDING = "embedding"
    SPARSE_EMBEDDING = "sparse_embedding"
    METADATA = "metadata"

    ALL_FIELDS: ClassVar[list[str]] = [ID, DOCUMENT, EMBEDDING, METADATA]


class CollectionNames:
    """Helpers for mapping between collection names and physical table names."""

    # Version prefix for collection tables
    _PREFIX = "c$v1$"
    _PREFIX_V2 = "c$v2$"

    @staticmethod
    def table_name(collection_name: str) -> str:
        """Convert collection name to table name."""
        return f"{CollectionNames._PREFIX}{collection_name}"

    @staticmethod
    def table_name_v2(collection_id: str) -> str:
        """Convert collection id to table name."""
        return f"{CollectionNames._PREFIX_V2}{collection_id}"

    @staticmethod
    def collection_name(table_name: str) -> str:
        """Extract collection name from table name."""
        if table_name.startswith(CollectionNames._PREFIX):
            return table_name[len(CollectionNames._PREFIX) :]
        return table_name

    @staticmethod
    def is_collection_table(table_name: str) -> bool:
        """Check if a table name is a collection table."""
        return table_name.startswith(CollectionNames._PREFIX)

    @staticmethod
    def table_pattern() -> str:
        """Get SQL LIKE pattern for collection tables."""
        return f"{CollectionNames._PREFIX}%"

    @staticmethod
    def prefix() -> str:
        """Get the collection table prefix."""
        return CollectionNames._PREFIX

    @staticmethod
    def sdk_collections_table_name() -> str:
        """Return the SDK catalog table that stores collection metadata."""
        return "sdk_collections"


class NamespaceCollectionNames:
    """Naming helpers for namespace-enabled collection physical tables."""

    _LOGIC_DATA_SUFFIX = "_logic_data_table"
    _HOT_SUFFIX = "_hot_table"
    _KV_DATA_SUFFIX = "_kv_data_table"
    _LOGIC_SCHEMA_SUFFIX = "_logic_schema_table"
    _TG_SUFFIX = "_tg"

    @staticmethod
    def sdk_namespaces_table() -> str:
        """Return the SDK catalog table that stores namespace metadata."""
        return "sdk_namespaces"

    @staticmethod
    def sdk_ltables_table() -> str:
        """Return the SDK catalog table that stores logic-table metadata."""
        return "sdk_ltables"

    @staticmethod
    def sdk_namespace_stat_table() -> str:
        """Return the namespace-level stats table used by ObLogicalTableMonitor.

        Table ``sdk_namespace_stat`` stores per-namespace **measurements** only:
        ``row_count``, ``total_size``, ``total_size_with_index``,
        ``last_gather_time``. Row/size **limits** live in
        ``sdk_namespaces.info`` (flat JSON: ``row_limit``, ``size_limit``, RU fields).

        See ``doc/agent_db监控运维/sdk_namespace_stat表说明.md``.
        """
        return "sdk_namespace_stat"

    @staticmethod
    def logic_table_namespace_stats_view() -> str:
        """Return the SDK view joining collection/namespace names with namespace stats.

        Read-only ops view over sdk_namespace_stat + sdk_namespaces + sdk_collections.
        Created by ``_ensure_namespace_catalogs()`` (CREATE OR REPLACE VIEW).
        """
        return "logic_table_namespace_stats"

    @staticmethod
    def data_table_name(collection_id: str) -> str:
        """Build the logic data table name for a namespace-enabled collection."""
        return f"{collection_id}{NamespaceCollectionNames._LOGIC_DATA_SUFFIX}"

    @staticmethod
    def hot_table_name(collection_id: str) -> str:
        """Build the hot data table name for a namespace-enabled collection."""
        return f"{collection_id}{NamespaceCollectionNames._HOT_SUFFIX}"

    @staticmethod
    def kv_data_table_name(collection_id: str) -> str:
        """Build the KV data table name for a namespace-enabled collection."""
        return f"{collection_id}{NamespaceCollectionNames._KV_DATA_SUFFIX}"

    @staticmethod
    def logic_schema_table_name(collection_id: str) -> str:
        """Build the logic schema table name for a namespace-enabled collection."""
        return f"{collection_id}{NamespaceCollectionNames._LOGIC_SCHEMA_SUFFIX}"

    @staticmethod
    def tablegroup_name(collection_id: str) -> str:
        """Build the table group name for a namespace-enabled collection."""
        return f"{collection_id}{NamespaceCollectionNames._TG_SUFFIX}"

    @staticmethod
    def is_ns_data_table(table_name: str) -> bool:
        """Return True if ``table_name`` is a namespace logic data table."""
        return table_name.endswith(NamespaceCollectionNames._LOGIC_DATA_SUFFIX)


class NamespaceStatsDefaults:
    """Default row/size limits stored in sdk_namespaces.info (flat JSON)."""

    ROW_LIMIT = 1_000_000
    SIZE_LIMIT = 20 * 1024 * 1024 * 1024  # 20GB

    @staticmethod
    def default_ops_limit_json() -> str:
        """JSON fragment for default row_limit / size_limit values."""
        return (
            f'{{"row_limit": {NamespaceStatsDefaults.ROW_LIMIT}, '
            f'"size_limit": {NamespaceStatsDefaults.SIZE_LIMIT}}}'
        )


class NamespaceRuConfigKeys:
    """Keys accepted by SET NAMESPACE RU CONFIG / SDK RU config APIs."""

    ROW_LIMIT = "row_limit"
    SIZE_LIMIT = "size_limit"
    RU_ENABLED = "ru_enabled"
    QPS_BURST = "qps_burst"
    QPS_REFILL = "qps_refill"
    TPS_BURST = "tps_burst"
    TPS_REFILL = "tps_refill"
    DATA_BURST = "data_burst"
    DATA_REFILL = "data_refill"

    @classmethod
    def all_keys(cls) -> frozenset[str]:
        return frozenset({
            cls.ROW_LIMIT,
            cls.SIZE_LIMIT,
            cls.RU_ENABLED,
            cls.QPS_BURST,
            cls.QPS_REFILL,
            cls.TPS_BURST,
            cls.TPS_REFILL,
            cls.DATA_BURST,
            cls.DATA_REFILL,
        })


class NamespaceRuLimitDefaults:
    """Kernel default RU token-bucket settings when RU fields are absent in info."""

    RU_ENABLED = 1
    QPS_BURST = 200
    QPS_REFILL = 100
    TPS_BURST = 100
    TPS_REFILL = 50
    DATA_BURST = 50 * 1024 * 1024  # 50MB
    DATA_REFILL = 10 * 1024 * 1024  # 10MB/s


class NamespaceFieldNames:
    """Standard field names used in namespace record payloads."""

    NAMESPACE_ID = "namespace_id"
    LTABLE_ID = "ltable_id"
    DOCUMENT = "document"
    EMBEDDING = "embedding"
    DATA_CONTENT = "data_content"
