from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from hey_robot.channels import ChannelContext, ChannelManager, WebChannel
from hey_robot.config import ChannelSpec
from hey_robot.events import EventKind, RuntimeEvent
from hey_robot.protocol import AgentReply, Envelope, SkillIntent


@dataclass
class FakeAgentIO:
    skills: list[SkillIntent] = field(default_factory=list)
    replies: list[AgentReply] = field(default_factory=list)
    task_results: list[tuple[bool, str]] = field(default_factory=list)

    async def submit_skill(self, skill: SkillIntent) -> None:
        self.skills.append(skill)

    async def publish_reply(self, reply: AgentReply) -> None:
        self.replies.append(reply)

    async def publish_task_result(self, *, success: bool, summary: str) -> None:
        self.task_results.append((success, summary))


def test_channel_manager_forwards_events_to_channels() -> None:
    channel = WebChannel(
        ChannelContext(name="web", spec=ChannelSpec(type="web"), deployment_id="d1")
    )
    manager = ChannelManager()
    manager.register(channel)

    asyncio.run(
        manager.publish_event(RuntimeEvent.make(EventKind.ROBOT_STATUS, source="robot"))
    )

    assert channel._events[-1]["kind"] == "robot.status"


def test_channel_manager_sends_reply_only_to_matching_channel() -> None:
    web = WebChannel(
        ChannelContext(name="web", spec=ChannelSpec(type="web"), deployment_id="d1")
    )
    voice = WebChannel(
        ChannelContext(name="voice", spec=ChannelSpec(type="web"), deployment_id="d1")
    )
    manager = ChannelManager()
    manager.register(web)
    manager.register(voice)

    asyncio.run(
        manager.send(AgentReply(envelope=Envelope(channel="voice"), text="spoken"))
    )
    asyncio.run(
        manager.send(AgentReply(envelope=Envelope(channel=None), text="ignored"))
    )

    assert web._replies == []
    assert voice._replies[-1]["text"] == "spoken"


def test_channel_manager_start_stop_duplicate_register_and_event_fallback() -> None:
    started: list[str] = []
    stopped: list[str] = []

    class ChannelWithoutEvent:
        def __init__(self, name: str) -> None:
            self.name = name

        async def start(self, _handler) -> None:
            started.append(self.name)

        async def send(self, _reply) -> None:
            return None

        async def stop(self) -> None:
            stopped.append(self.name)

    manager = ChannelManager()
    channel = ChannelWithoutEvent("ops")
    manager.register(cast(Any, channel))

    with pytest.raises(ValueError, match="duplicate channel"):
        manager.register(cast(Any, channel))

    async def handler(_turn) -> None:
        return None

    asyncio.run(manager.start_all(handler))
    asyncio.run(manager.send(AgentReply(envelope=Envelope(), text="skip")))
    asyncio.run(
        manager.publish_event(RuntimeEvent.make(EventKind.ROBOT_STATUS, source="robot"))
    )
    asyncio.run(manager.stop_all())

    assert list(manager.items())
    assert started == ["ops"]
    assert stopped == ["ops"]
