from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from hey_robot.config import ChannelSpec
from hey_robot.events import RuntimeEvent
from hey_robot.protocol import AgentReply, UserTurn

InboundHandler = Callable[[UserTurn], Awaitable[None]]


@dataclass(frozen=True)
class ChannelContext:
    name: str
    spec: ChannelSpec
    deployment_id: str


class Channel(Protocol):
    name: str

    async def start(self, handler: InboundHandler) -> None: ...

    async def send(self, reply: AgentReply) -> None: ...

    async def on_event(self, event: RuntimeEvent) -> None: ...

    async def stop(self) -> None: ...


class ChannelManager:
    def __init__(self) -> None:
        self._channels: dict[str, Channel] = {}

    def register(self, channel: Channel) -> None:
        if channel.name in self._channels:
            raise ValueError(f"duplicate channel: {channel.name}")
        self._channels[channel.name] = channel

    def get(self, name: str) -> Channel | None:
        return self._channels.get(name)

    def items(self):
        return self._channels.items()

    async def start_all(self, handler: InboundHandler) -> None:
        await asyncio.gather(
            *(channel.start(handler) for channel in self._channels.values())
        )

    async def stop_all(self) -> None:
        await asyncio.gather(
            *(channel.stop() for channel in self._channels.values()),
            return_exceptions=True,
        )

    async def send(self, reply: AgentReply) -> None:
        channel_name = reply.envelope.channel
        if not channel_name:
            return
        channel = self.get(channel_name)
        if channel is not None:
            await channel.send(reply)

    async def publish_event(self, event: RuntimeEvent) -> None:
        await asyncio.gather(
            *(
                getattr(channel, "on_event", _noop_event)(event)
                for channel in self._channels.values()
            ),
            return_exceptions=True,
        )


async def _noop_event(_event: RuntimeEvent) -> None:
    return None
