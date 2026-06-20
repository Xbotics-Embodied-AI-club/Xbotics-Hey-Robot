from __future__ import annotations

import asyncio

from hey_robot.channels import ChannelContext, WebChannel
from hey_robot.config import ChannelSpec
from hey_robot.events import EventKind, RuntimeEvent
from hey_robot.protocol import AgentReply, Envelope


def test_web_channel_records_events_and_replies() -> None:
    channel = WebChannel(
        ChannelContext(name="web", spec=ChannelSpec(type="web"), deployment_id="d1")
    )
    event = RuntimeEvent.make(
        EventKind.AGENT_TURN_START, source="agent", trace_id="tr1"
    )
    reply = AgentReply(envelope=Envelope(channel="web"), text="ok")

    asyncio.run(channel.on_event(event))
    asyncio.run(channel.send(reply))

    assert channel._events[-1]["kind"] == "agent.turn.start"
    assert channel._replies[-1]["text"] == "ok"
