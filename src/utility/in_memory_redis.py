"""Minimal Redis-like client for KB manifest keys when Redis is unreachable."""


class InMemoryRedis:
    """Implements get / set / delete used by KnowledgeMetadataStore."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self._data.get(name)

    def set(self, name: str, value: str, *args: object, **kwargs: object) -> bool:
        self._data[name] = value
        return True

    def delete(self, name: str) -> int:
        return 1 if self._data.pop(name, None) is not None else 0

    def ping(self, **kwargs: object) -> bool:
        return True
