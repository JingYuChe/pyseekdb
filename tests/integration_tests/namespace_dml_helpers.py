"""
Shared helpers for namespace DML integration tests.
"""

from __future__ import annotations

import time
from typing import Any

from pyseekdb import IVFConfiguration
from pyseekdb.client.configuration import VectorIndexConfig
from pyseekdb.client.schema import Schema


def ns_schema() -> Schema:
    return Schema(
        vector_index=VectorIndexConfig(
            ivf=IVFConfiguration(dimension=3, distance="l2"),
            embedding_function=None,
        ),
    )


def create_ns_collection(client: Any, suffix: str = "") -> Any:
    name = f"test_ns_dml_{int(time.time() * 1000)}{suffix}"
    return client.create_collection(name=name, schema=ns_schema(), use_namespace=True)


def cleanup(client: Any, *collections: Any) -> None:
    for collection in collections:
        try:
            client.delete_collection(name=collection.name)
        except Exception:
            pass


def _default_include(
    documents: str | None,
    metadatas: dict | None,
    embeddings: list[float] | None,
    include: list[str] | None,
) -> list[str] | None:
    if include is not None:
        return include
    fields: list[str] = []
    if documents is not None:
        fields.append("documents")
    if metadatas is not None:
        fields.append("metadatas")
    if embeddings is not None:
        fields.append("embeddings")
    return fields or None


def assert_get_present(
    ns: Any,
    doc_id: str,
    *,
    documents: str | None = None,
    metadatas: dict | None = None,
    embeddings: list[float] | None = None,
    include: list[str] | None = None,
) -> dict[str, Any]:
    result = ns.get(ids=doc_id, include=_default_include(documents, metadatas, embeddings, include))
    indices = [i for i, rid in enumerate(result["ids"]) if rid == doc_id]
    assert indices, f"expected doc {doc_id!r}, got {result['ids']}"
    idx = indices[0]
    if documents is not None:
        assert result["documents"][idx] == documents
    if metadatas is not None:
        assert result["metadatas"][idx] == metadatas
    if embeddings is not None:
        assert result["embeddings"][idx] == embeddings
    return result


def assert_get_absent(ns: Any, doc_id: str) -> None:
    result = ns.get(ids=doc_id)
    assert len(result["ids"]) == 0, f"expected doc {doc_id!r} absent, got {result['ids']}"
