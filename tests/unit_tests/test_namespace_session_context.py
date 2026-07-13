"""Unit tests for namespace session variable set/clear gating."""

from __future__ import annotations

from pyseekdb.client.client_base import BaseClient


class _StubClient:
    def __init__(self) -> None:
        self._executed: list[str] = []
        self._ns_session_context_active = False
        self.collection_exists = True
        self.namespace_exists = True

    def _execute(self, sql: str):
        self._executed.append(sql)
        return []

    def _ns_collection_exists_by_id(self, collection_id: str) -> bool:
        return self.collection_exists

    def _ns_namespace_exists_by_id(self, collection_id: str, namespace_id: str) -> bool:
        return self.namespace_exists

    def _ensure_namespace_live(self, collection_id: str, namespace_id: int) -> None:
        BaseClient._ensure_namespace_live(self, collection_id, namespace_id)


class TestNamespaceSessionContext:
    def test_clear_skips_when_namespace_context_never_set(self):
        client = _StubClient()
        BaseClient._clear_session_ns_context(client)
        assert client._executed == []

    def test_clear_after_namespace_context_uses_empty_collection_id(self):
        client = _StubClient()
        BaseClient._set_session_ns_context(client, collection_id="abc123", namespace_id=1, ltable_id=2)
        client._executed.clear()
        BaseClient._clear_session_ns_context(client)
        assert client._executed == [
            "SET @collection_id = ''",
            "SET @namespace_id = NULL",
            "SET @ltable_id = NULL",
        ]
        assert client._ns_session_context_active is False

    def test_partial_namespace_context_is_tracked_and_cleared(self):
        client = _StubClient()
        BaseClient._set_session_ns_context(client, namespace_id=9, ltable_id=10)
        assert client._ns_session_context_active is True
        client._executed.clear()
        BaseClient._clear_session_ns_context(client)
        assert client._executed[0] == "SET @collection_id = ''"

    def test_set_session_ns_context_rejects_deleted_namespace(self):
        client = _StubClient()
        client.namespace_exists = False
        try:
            BaseClient._set_session_ns_context(client, collection_id="abc123", namespace_id=7, ltable_id=1)
            raised = False
        except ValueError as exc:
            raised = True
            assert "no longer exists" in str(exc)
        assert raised
        assert client._executed == []
