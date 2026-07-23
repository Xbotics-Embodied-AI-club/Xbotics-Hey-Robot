"""Re-entrant resource ownership for nested skills."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class ResourceManager:
    def __init__(self) -> None:
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._owners: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)

    @asynccontextmanager
    async def acquire(
        self,
        robot_id: str,
        resources: tuple[str, ...],
        *,
        owner: str,
    ) -> AsyncIterator[None]:
        keys = tuple((robot_id, resource) for resource in sorted(set(resources)))
        acquired: list[tuple[str, str]] = []
        try:
            for key in keys:
                owners = self._owners[key]
                if owner not in owners:
                    lock = self._locks.setdefault(key, asyncio.Lock())
                    await lock.acquire()
                owners[owner] = owners.get(owner, 0) + 1
                acquired.append(key)
            yield
        finally:
            for key in reversed(acquired):
                owners = self._owners[key]
                count = owners.get(owner, 0) - 1
                if count > 0:
                    owners[owner] = count
                    continue
                owners.pop(owner, None)
                if not owners:
                    self._owners.pop(key, None)
                    self._locks[key].release()
