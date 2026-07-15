"""供应用服务依赖的传输无关消息总线契约。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

MessageHandler = Callable[[str, dict[str, Any]], Awaitable[None]]
RawMessageHandler = Callable[[str, bytes], Awaitable[None]]


class MessageBus(Protocol):
    """进程内与 NATS 传输实现共同满足的最小消息接口。"""

    async def connect(self) -> None: ...

    async def publish(self, topic: str, payload: dict[str, Any]) -> None: ...

    async def publish_raw(self, topic: str, payload: bytes) -> None: ...

    async def subscribe(
        self, topics: list[str], on_message: MessageHandler
    ) -> None: ...

    async def subscribe_raw(
        self, topics: list[str], on_message: RawMessageHandler
    ) -> None: ...

    async def unsubscribe(self, topics: list[str]) -> None: ...

    async def close(self) -> None: ...
