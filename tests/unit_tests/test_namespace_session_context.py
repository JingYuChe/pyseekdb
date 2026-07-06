"""Unit tests for namespace session variable set/clear gating."""

from __future__ import annotations

from pyseekdb.client.client_base import BaseClient


class _StubClient:
    def __init__(self) -> None:
        self._executed: list[str] = []
        self._ns_session_context_active = False

    def _execute(self, sql: str):
        self._executed.append(sql)
        return []


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
