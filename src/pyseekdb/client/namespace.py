"""
Namespace class - lightweight facade for namespace-level data operations.

All operations are delegated to the client that created it via
``self._client._namespace_*()`` methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .meta_info import NamespaceRuConfigKeys
from .validators import (
    _validate_include,
    _validate_n_results,
    _validate_namespace_ru_config,
)

if TYPE_CHECKING:
    from .collection import Collection
    from .embedding_function import EmbeddingFunction


class Namespace:
    """Scoped view of a single namespace within a namespace-enabled collection."""

    def __init__(
        self,
        client: Any,
        collection: Collection,
        name: str,
        namespace_id: str,
    ):
        """Bind a namespace handle to its parent collection and catalog identifiers."""
        self._client = client
        self._collection = collection
        self._name = name
        self._namespace_id = namespace_id

    @property
    def name(self) -> str:
        """Human-readable namespace name."""
        return self._name

    @property
    def namespace_id(self) -> str:
        """Stable namespace identifier assigned by the catalog."""
        return self._namespace_id

    @property
    def collection(self) -> Collection:
        """Parent collection that owns this namespace."""
        return self._collection

    @property
    def embedding_function(self) -> EmbeddingFunction | None:
        """Embedding function inherited from the parent collection."""
        return self._collection.embedding_function

    def __repr__(self) -> str:
        """Return a debug-friendly representation of this namespace."""
        return (
            f"Namespace(name='{self._name}', namespace_id={self._namespace_id}, collection='{self._collection.name}')"
        )

    def _dml_collection_context(self) -> dict[str, Any]:
        """Collection metadata required for namespace DML/DQL validation."""
        return {
            "has_vector_index": self._collection.has_vector_index,
            "collection_dimension": self._collection.dimension,
            "dimension": self._collection.dimension,
        }

    def _guard_exists(self) -> None:
        """Raise if this namespace or its collection was deleted."""
        if not self._client._ns_collection_exists_by_id(self._collection.id):
            raise ValueError(
                f"Collection '{self._collection.name}' no longer exists (it may have been deleted). "
                "Namespace operations are not allowed on a deleted collection."
            )
        if not self._client._ns_namespace_exists_by_id(self._collection.id, self._namespace_id):
            raise ValueError(
                f"Namespace '{self._name}' no longer exists (it or its collection may have been deleted). "
                "Operations are not allowed on a deleted namespace."
            )

    # ==================== DML Operations ====================

    def add(
        self,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        **kwargs,
    ) -> None:
        """Insert records into this namespace."""
        self._guard_exists()
        return self._client._namespace_add(
            collection_id=self._collection.id,
            collection_name=self._collection.name,
            namespace_id=self._namespace_id,
            namespace_name=self._name,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            embedding_function=self._collection.embedding_function,
            **self._dml_collection_context(),
            **kwargs,
        )

    def update(
        self,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        **kwargs,
    ) -> None:
        """Update existing records in this namespace."""
        self._guard_exists()
        return self._client._namespace_update(
            collection_id=self._collection.id,
            collection_name=self._collection.name,
            namespace_id=self._namespace_id,
            namespace_name=self._name,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            embedding_function=self._collection.embedding_function,
            **self._dml_collection_context(),
            **kwargs,
        )

    def upsert(
        self,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        **kwargs,
    ) -> None:
        """Insert or update records in this namespace."""
        self._guard_exists()
        return self._client._namespace_upsert(
            collection_id=self._collection.id,
            collection_name=self._collection.name,
            namespace_id=self._namespace_id,
            namespace_name=self._name,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            embedding_function=self._collection.embedding_function,
            **self._dml_collection_context(),
            **kwargs,
        )

    def delete(
        self,
        ids: str | list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Delete records from this namespace by ids or filters."""
        self._guard_exists()
        return self._client._namespace_delete(
            collection_id=self._collection.id,
            collection_name=self._collection.name,
            namespace_id=self._namespace_id,
            namespace_name=self._name,
            ids=ids,
            where=where,
            where_document=where_document,
            **kwargs,
        )

    # ==================== DQL Operations ====================

    def query(
        self,
        query_embeddings: list[float] | list[list[float]] | None = None,
        query_texts: str | list[str] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run vector similarity search within this namespace."""
        self._guard_exists()
        _validate_n_results(n_results)
        _validate_include(include)
        return self._client._namespace_query(
            collection_id=self._collection.id,
            collection_name=self._collection.name,
            namespace_id=self._namespace_id,
            namespace_name=self._name,
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
            embedding_function=self._collection.embedding_function,
            distance=self._collection.distance,
            **self._dml_collection_context(),
            **kwargs,
        )

    def hybrid_search(
        self,
        query: dict[str, Any] | None = None,
        knn: dict[str, Any] | None = None,
        rank: dict[str, Any] | None = None,
        n_results: int = 10,
        include: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run hybrid fulltext + vector search within this namespace."""
        self._guard_exists()
        _validate_n_results(n_results)
        _validate_include(include)
        if include is None and not query and not knn:
            include = []
        return self._client._namespace_hybrid_search(
            collection_id=self._collection.id,
            collection_name=self._collection.name,
            namespace_id=self._namespace_id,
            namespace_name=self._name,
            query=query,
            knn=knn,
            rank=rank,
            n_results=n_results,
            include=include,
            embedding_function=self._collection.embedding_function,
            dimension=self._collection.dimension,
            **kwargs,
        )

    def get(
        self,
        ids: str | list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Fetch records from this namespace by ids or filters."""
        self._guard_exists()
        _validate_include(include)
        return self._client._namespace_get(
            collection_id=self._collection.id,
            collection_name=self._collection.name,
            namespace_id=self._namespace_id,
            namespace_name=self._name,
            ids=ids,
            where=where,
            where_document=where_document,
            limit=limit,
            offset=offset,
            include=include,
            **kwargs,
        )

    def count(self) -> int:
        """Return the number of records in this namespace."""
        self._guard_exists()
        return self._client._namespace_count(
            collection_id=self._collection.id,
            collection_name=self._collection.name,
            namespace_id=self._namespace_id,
            namespace_name=self._name,
        )

    def peek(self, limit: int = 10) -> dict[str, Any]:
        """Return up to ``limit`` records from this namespace."""
        self._guard_exists()
        return self._client._namespace_peek(
            collection_id=self._collection.id,
            collection_name=self._collection.name,
            namespace_id=self._namespace_id,
            namespace_name=self._name,
            limit=limit,
        )

    def prewarm(self) -> None:
        """Prewarm namespace physical tables to reduce cold-start latency."""
        self._guard_exists()
        return self._client._namespace_prewarm(
            collection_id=self._collection.id,
            collection_name=self._collection.name,
            namespace_id=self._namespace_id,
            namespace_name=self._name,
        )

    def set_ru_config(self, config: dict) -> None:
        """Configure row/size limits or RU token-bucket settings for this namespace.

        ``config`` is a flat JSON object, e.g. ``{"row_limit": 1000, "tps_burst": 50}``.
        See :meth:`Collection.set_namespace_ru_config` for valid keys.
        """
        self._guard_exists()
        _validate_namespace_ru_config(config)
        self._client._set_namespace_ru_config(self._collection.name, self._name, config)

    def set_row_limit(self, row_limit: int) -> None:
        """Set row count limit for this namespace. ``-1`` means unlimited; ``0`` blocks writes."""
        self.set_ru_config({NamespaceRuConfigKeys.ROW_LIMIT: row_limit})

    def set_size_limit(self, size_limit: int) -> None:
        """Set storage size limit in bytes for this namespace. ``-1`` means unlimited; ``0`` blocks writes."""
        self.set_ru_config({NamespaceRuConfigKeys.SIZE_LIMIT: size_limit})

    def set_ru_enabled(self, ru_enabled: int) -> None:
        """Enable (``1``) or disable (``0``) RU throttling for this namespace."""
        self.set_ru_config({NamespaceRuConfigKeys.RU_ENABLED: ru_enabled})

    def set_qps_burst(self, qps_burst: int) -> None:
        """Set read QPS token-bucket capacity."""
        if qps_burst < 0:
            return
        self.set_ru_config({NamespaceRuConfigKeys.QPS_BURST: qps_burst})

    def set_qps_refill(self, qps_refill: int) -> None:
        """Set read QPS token-bucket refill per second."""
        if qps_refill < 0:
            return
        self.set_ru_config({NamespaceRuConfigKeys.QPS_REFILL: qps_refill})

    def set_tps_burst(self, tps_burst: int) -> None:
        """Set write TPS token-bucket capacity."""
        if tps_burst < 0:
            return
        self.set_ru_config({NamespaceRuConfigKeys.TPS_BURST: tps_burst})

    def set_tps_refill(self, tps_refill: int) -> None:
        """Set write TPS token-bucket refill per second."""
        if tps_refill < 0:
            return
        self.set_ru_config({NamespaceRuConfigKeys.TPS_REFILL: tps_refill})

    def set_data_burst(self, data_burst: int) -> None:
        """Set data-volume token-bucket capacity in bytes."""
        if data_burst < 0:
            return
        self.set_ru_config({NamespaceRuConfigKeys.DATA_BURST: data_burst})

    def set_data_refill(self, data_refill: int) -> None:
        """Set data-volume token-bucket refill bytes per second."""
        if data_refill < 0:
            return
        self.set_ru_config({NamespaceRuConfigKeys.DATA_REFILL: data_refill})
