"""
Metadata information for collection fields.
"""

from typing import ClassVar


class CollectionFieldNames:
    ID = "_id"
    DOCUMENT = "document"
    EMBEDDING = "embedding"
    SPARSE_EMBEDDING = "sparse_embedding"
    METADATA = "metadata"

    ALL_FIELDS: ClassVar[list[str]] = [ID, DOCUMENT, EMBEDDING, METADATA]


class CollectionNames:
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
        return "sdk_collections"


class NamespaceCollectionNames:
    _LOGIC_DATA_SUFFIX = "_logic_data_table"
    _HOT_SUFFIX = "_hot_table"
    _KV_DATA_SUFFIX = "_kv_data_table"
    _LOGIC_SCHEMA_SUFFIX = "_logic_schema_table"
    _TG_SUFFIX = "_tg"

    @staticmethod
    def sdk_ns_namespaces_table() -> str:
        return "sdk_ns_namespaces"

    @staticmethod
    def sdk_ns_ltables_table() -> str:
        return "sdk_ns_ltables"

    @staticmethod
    def data_table_name(collection_id: str) -> str:
        return f"{collection_id}{NamespaceCollectionNames._LOGIC_DATA_SUFFIX}"

    @staticmethod
    def hot_table_name(collection_id: str) -> str:
        return f"{collection_id}{NamespaceCollectionNames._HOT_SUFFIX}"

    @staticmethod
    def kv_data_table_name(collection_id: str) -> str:
        return f"{collection_id}{NamespaceCollectionNames._KV_DATA_SUFFIX}"

    @staticmethod
    def logic_schema_table_name(collection_id: str) -> str:
        return f"{collection_id}{NamespaceCollectionNames._LOGIC_SCHEMA_SUFFIX}"

    @staticmethod
    def tablegroup_name(collection_id: str) -> str:
        return f"{collection_id}{NamespaceCollectionNames._TG_SUFFIX}"

    @staticmethod
    def is_ns_data_table(table_name: str) -> bool:
        return table_name.endswith(NamespaceCollectionNames._LOGIC_DATA_SUFFIX)


class NamespaceFieldNames:
    NAMESPACE_ID = "namespace_id"
    LTABLE_ID = "ltable_id"
    DOCUMENT = "document"
    EMBEDDING = "embedding"
    DATA_CONTENT = "data_content"
