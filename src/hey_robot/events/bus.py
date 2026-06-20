from __future__ import annotations

from typing import Protocol

from hey_robot.bus.client import BusClient
from hey_robot.events.event import RuntimeEvent
from hey_robot.protocol import Topics


class EventPublisher(Protocol):
    async def publish(self, event: RuntimeEvent) -> None: ...


class BusEventPublisher:
    def __init__(self, bus: BusClient, topics: Topics | None = None) -> None:
        self.bus = bus
        self.topics = topics or Topics()

    async def publish(self, event: RuntimeEvent) -> None:
        await self.bus.publish(self.topics.runtime_event, event.to_dict())


class NullEventPublisher:
    async def publish(self, _event: RuntimeEvent) -> None:
        return None
