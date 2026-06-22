"""Concurrent namespace lifecycle tests (drop vs has across clients)."""

from __future__ import annotations

import contextlib
import os
import threading
import time
from unittest.mock import patch

import pytest

import pyseekdb
from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT, cleanup, ns_schema


def _unique_name(prefix: str) -> str:
    return f"{prefix}_{time.time_ns()}"


def _new_oceanbase_client():
    return pyseekdb.Client(
        host=os.environ.get("OB_HOST", "127.0.0.1"),
        port=int(os.environ.get("OB_PORT", "10902")),
        tenant=os.environ.get("OB_TENANT", "mysql"),
        database=os.environ.get("OB_DATABASE", "test"),
        user=os.environ.get("OB_USER", "root"),
        password=os.environ.get("OB_PASSWORD", ""),
    )


def test_multi_client_has_false_after_drop_returns(oceanbase_client):
    """After delete_namespace returns, has_namespace on another client is always False."""
    collection = oceanbase_client.create_collection(
        name=_unique_name("test_ns_lc_drop_has"),
        schema=ns_schema(),
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )
    clients = [_new_oceanbase_client() for _ in range(2)]
    ns_name = "drop_has_ns"
    drop_done = threading.Event()
    errors: list[Exception] = []
    post_drop_has: list[bool] = []

    try:
        collection.create_namespace(ns_name)

        def _drop_worker() -> None:
            try:
                clients[0].get_collection(collection.name).delete_namespace(ns_name)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                drop_done.set()

        def _has_worker() -> None:
            try:
                coll = clients[1].get_collection(collection.name)
                drop_done.wait(timeout=60)
                for _ in range(30):
                    post_drop_has.append(coll.has_namespace(ns_name))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        drop_thread = threading.Thread(target=_drop_worker)
        has_thread = threading.Thread(target=_has_worker)
        drop_thread.start()
        has_thread.start()
        drop_thread.join(timeout=120)
        has_thread.join(timeout=120)

        assert errors == []
        assert drop_thread.is_alive() is False
        assert has_thread.is_alive() is False
        assert post_drop_has
        assert all(value is False for value in post_drop_has)
    finally:
        for client in clients:
            with contextlib.suppress(Exception):
                client.close()
        cleanup(oceanbase_client, collection)


def test_multi_client_has_blocks_until_drop_finishes(oceanbase_client):
    """has_namespace on client B waits for client A drop to release lifecycle lock."""
    collection = oceanbase_client.create_collection(
        name=_unique_name("test_ns_lc_drop_block"),
        schema=ns_schema(),
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )
    clients = [_new_oceanbase_client() for _ in range(2)]
    ns_name = "block_ns"
    drop_started = threading.Event()
    drop_can_finish = threading.Event()
    has_started = threading.Event()
    errors: list[Exception] = []
    has_result: list[bool] = []

    try:
        collection.create_namespace(ns_name)
        dropper = clients[0].get_collection(collection.name)
        checker = clients[1].get_collection(collection.name)
        real_execute = clients[0]._server._execute

        def _slow_drop_execute(sql, *args, **kwargs):
            if "DROP_NAMESPACE" in str(sql):
                drop_started.set()
                assert drop_can_finish.wait(timeout=30), "drop blocked too long"
            return real_execute(sql, *args, **kwargs)

        def _drop_worker() -> None:
            try:
                with patch.object(clients[0]._server, "_execute", side_effect=_slow_drop_execute):
                    dropper.delete_namespace(ns_name)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def _has_worker() -> None:
            try:
                assert drop_started.wait(timeout=30), "drop did not start"
                has_started.set()
                has_result.append(checker.has_namespace(ns_name))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        drop_thread = threading.Thread(target=_drop_worker)
        has_thread = threading.Thread(target=_has_worker)
        drop_thread.start()
        has_thread.start()

        assert has_started.wait(timeout=30), "has did not start while drop in progress"
        time.sleep(0.2)
        assert drop_thread.is_alive(), "drop finished before lifecycle lock blocked has"
        drop_can_finish.set()

        drop_thread.join(timeout=120)
        has_thread.join(timeout=120)

        assert errors == []
        assert has_result == [False]
    finally:
        for client in clients:
            with contextlib.suppress(Exception):
                client.close()
        cleanup(oceanbase_client, collection)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
