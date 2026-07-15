"""供单进程开发运行器使用的确定性进程内消息总线。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from hey_robot.bus.types import MessageHandler, RawMessageHandler


class InMemoryBusHub:
    """同一 Python 进程内客户端共享的订阅注册表。"""

    def __init__(self) -> None:
        self.messages: dict[str, list[MessageHandler]] = defaultdict(list)
        self.raw_messages: dict[str, list[RawMessageHandler]] = defaultdict(list)


class InMemoryBusClient:
    """与 ``BusClient`` 兼容的进程内同步投递实现。

    它刻意不提供持久化、网络边界或跨进程可见性。拆分服务、仿真联调和
    真机运行应使用 NATS。
    """

    def __init__(self, hub: InMemoryBusHub) -> None:
        self._hub = hub
        self._connected = False
        self._message_subscriptions: list[tuple[str, MessageHandler]] = []
        self._raw_subscriptions: list[tuple[str, RawMessageHandler]] = []

    async def connect(self) -> None:
        self._connected = True

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self._require_connection()
        for handler in tuple(self._hub.messages[topic]):
            await handler(topic, payload)

    async def publish_raw(self, topic: str, payload: bytes) -> None:
        self._require_connection()
        for handler in tuple(self._hub.raw_messages[topic]):
            await handler(topic, payload)

    async def subscribe(self, topics: list[str], on_message: MessageHandler) -> None:
        self._require_connection()
        for topic in topics:
            self._hub.messages[topic].append(on_message)
            self._message_subscriptions.append((topic, on_message))

    async def subscribe_raw(
        self, topics: list[str], on_message: RawMessageHandler
    ) -> None:
        self._require_connection()
        for topic in topics:
            self._hub.raw_messages[topic].append(on_message)
            self._raw_subscriptions.append((topic, on_message))

    async def unsubscribe(self, topics: list[str]) -> None:
        for topic, handler in tuple(self._message_subscriptions):
            if topic not in topics:
                continue
            if handler in self._hub.messages[topic]:
                self._hub.messages[topic].remove(handler)
            self._message_subscriptions.remove((topic, handler))
        for raw_topic, raw_handler in tuple(self._raw_subscriptions):
            if raw_topic not in topics:
                continue
            if raw_handler in self._hub.raw_messages[raw_topic]:
                self._hub.raw_messages[raw_topic].remove(raw_handler)
            self._raw_subscriptions.remove((raw_topic, raw_handler))

    async def close(self) -> None:
        await self.unsubscribe(
            [
                topic
                for topic, _ in self._message_subscriptions + self._raw_subscriptions
            ]
        )
        self._connected = False

    def _require_connection(self) -> None:
        if not self._connected:
            raise RuntimeError("InMemoryBusClient is not connected")
