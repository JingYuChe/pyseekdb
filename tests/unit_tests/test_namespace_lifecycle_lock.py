"""Unit tests for namespace lifecycle GET_LOCK serialization."""

import contextlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from pyseekdb.client.client_base import BaseClient  # noqa: E402


class TestNamespaceLifecycleLockName:
    def test_lock_name_is_stable_and_bounded(self):
        name_a = BaseClient._namespace_lifecycle_lock_name("cid", "ns1")
        name_b = BaseClient._namespace_lifecycle_lock_name("cid", "ns1")
        name_c = BaseClient._namespace_lifecycle_lock_name("cid", "ns2")
        assert name_a == name_b
        assert name_a != name_c
        assert name_a.startswith("pyseekdb:nslc:")
        assert len(name_a) <= 64


class TestNamespaceLifecycleGuard:
    def test_has_namespace_acquires_and_releases_lock(self):
        client = MagicMock(spec=BaseClient)
        client._get_ns_namespace_meta.return_value = None

        @contextlib.contextmanager
        def _lock_cm(*_args, **_kwargs):
            yield True

        client._namespace_lifecycle_lock.side_effect = _lock_cm
        client._namespace_lifecycle_guard = (
            BaseClient._namespace_lifecycle_guard.__get__(client, BaseClient)
        )

        assert BaseClient._has_ns_namespace(client, "cid", "ns1") is False
        client._namespace_lifecycle_lock.assert_called_once_with(
            "cid", "ns1", total_wait_seconds=30.0
        )
        client._get_ns_namespace_meta.assert_called_once_with("cid", "ns1")

    def test_guard_raises_when_lock_times_out(self):
        client = MagicMock(spec=BaseClient)

        @contextlib.contextmanager
        def _no_lock():
            yield False

        client._namespace_lifecycle_lock.return_value = _no_lock()
        client._namespace_lifecycle_guard = (
            BaseClient._namespace_lifecycle_guard.__get__(client, BaseClient)
        )

        with pytest.raises(TimeoutError, match="namespace lifecycle lock"):
            BaseClient._has_ns_namespace(client, "cid", "ns1")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
