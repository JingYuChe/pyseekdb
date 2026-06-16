"""
Namespace class - lightweight facade for namespace-level data operations.

All operations are delegated to the client that created it via
``self._client._namespace_*()`` methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .collection import Collection
    from .embedding_function import EmbeddingFunction


class Namespace:
    def __init__(
        self,
        client: Any,
        collection: "Collection",
        name: str,
        namespace_id: str,
    ):
        self._client = client
        self._collection = collection
        self._name = name
        self._namespace_id = namespace_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def namespace_id(self) -> str:
        return self._namespace_id

    @property
    def collection(self) -> "Collection":
        return self._collection

    @property
    def embedding_function(self) -> "EmbeddingFunction | None":
        return self._collection.embedding_function

    def __repr__(self) -> str:
        return (
            f"Namespace(name='{self._name}', namespace_id={self._namespace_id}, "
            f"collection='{self._collection.name}')"
        )

    def _guard_exists(self) -> None:
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
            **kwargs,
        )

    def delete(
        self,
        ids: str | list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
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
        self._guard_exists()
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
        self._guard_exists()
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
        self._guard_exists()
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
        self._guard_exists()
        return self._client._namespace_count(
            collection_id=self._collection.id,
            collection_name=self._collection.name,
            namespace_id=self._namespace_id,
            namespace_name=self._name,
        )

    def peek(self, limit: int = 10) -> dict[str, Any]:
        self._guard_exists()
        return self._client._namespace_peek(
            collection_id=self._collection.id,
            collection_name=self._collection.name,
            namespace_id=self._namespace_id,
            namespace_name=self._name,
            limit=limit,
        )

    def prewarm(self) -> None:
        self._guard_exists()
        return self._client._namespace_prewarm(
            collection_id=self._collection.id,
            collection_name=self._collection.name,
            namespace_id=self._namespace_id,
            namespace_name=self._name,
        )
