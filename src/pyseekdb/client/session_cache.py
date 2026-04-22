class SessionCache:
    def __init__(self):
        self._namespace_id_cache: dict[tuple[str, str], int] = {}
        self._ltable_id_cache: dict[tuple[str, int, str], int] = {}

    def get_namespace_id(self, collection_id: str, namespace_name: str) -> int | None:
        return self._namespace_id_cache.get((collection_id, namespace_name))

    def set_namespace_id(self, collection_id: str, namespace_name: str, namespace_id: int) -> None:
        self._namespace_id_cache[(collection_id, namespace_name)] = namespace_id

    def invalidate_namespace(self, collection_id: str, namespace_name: str) -> None:
        self._namespace_id_cache.pop((collection_id, namespace_name), None)
        keys_to_remove = [
            k for k in self._ltable_id_cache
            if k[0] == collection_id and k[1] == self._namespace_id_cache.get((collection_id, namespace_name))
        ]
        for k in keys_to_remove:
            self._ltable_id_cache.pop(k, None)

    def invalidate_collection(self, collection_id: str) -> None:
        keys_to_remove = [k for k in self._namespace_id_cache if k[0] == collection_id]
        for k in keys_to_remove:
            self._namespace_id_cache.pop(k, None)
        keys_to_remove = [k for k in self._ltable_id_cache if k[0] == collection_id]
        for k in keys_to_remove:
            self._ltable_id_cache.pop(k, None)

    def get_ltable_id(self, collection_id: str, namespace_id: int, ltable_name: str) -> int | None:
        return self._ltable_id_cache.get((collection_id, namespace_id, ltable_name))

    def set_ltable_id(self, collection_id: str, namespace_id: int, ltable_name: str, ltable_id: int) -> None:
        self._ltable_id_cache[(collection_id, namespace_id, ltable_name)] = ltable_id

    def invalidate_ltable(self, collection_id: str, namespace_id: int, ltable_name: str) -> None:
        self._ltable_id_cache.pop((collection_id, namespace_id, ltable_name), None)
