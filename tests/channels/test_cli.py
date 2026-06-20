from __future__ import annotations

import asyncio
from typing import cast

from hey_robot.channels import ChannelContext, CLIChannel
from hey_robot.config import ChannelSpec
from hey_robot.protocol import AgentReply, Envelope, UserTurn


def test_cli_channel_send_and_input_loop(monkeypatch, capsys) -> None:
    channel = CLIChannel(
        ChannelContext(
            name="cli",
            deployment_id="d1",
            spec=ChannelSpec(
                type="cli",
                account_id="cli-account",
                settings={
                    "prompt": "robot> ",
                    "reply_prefix": "bot",
                    "sender_id": "local-user",
                    "chat_id": "cli-chat",
                },
            ),
        )
    )
    turns: list[UserTurn] = []
    inputs = iter(["", " hello ", EOFError()])

    def fake_input(prompt: str) -> str:
        value = next(inputs)
        if isinstance(value, BaseException):
            raise value
        assert prompt == "robot> "
        return cast(str, value)

    async def handler(turn: UserTurn) -> None:
        turns.append(turn)

    monkeypatch.setattr("builtins.input", fake_input)

    asyncio.run(channel.send(AgentReply(envelope=Envelope(channel="cli"), text="done")))
    asyncio.run(channel._input_loop(handler))

    out = capsys.readouterr().out

    assert "bot> done" in out
    assert turns[0].text == "hello"
    assert turns[0].envelope.chat_id == "cli-chat"
    assert turns[0].envelope.sender_id == "local-user"
    assert turns[0].envelope.robot_id is None


def test_cli_channel_formats_notifications(capsys) -> None:
    channel = CLIChannel(
        ChannelContext(name="cli", deployment_id="d1", spec=ChannelSpec(type="cli"))
    )

    asyncio.run(
        channel.send(
            AgentReply(
                envelope=Envelope(channel="cli"),
                text="watchdog stale",
                metadata={
                    "notification": True,
                    "severity": "warning",
                    "notification_kind": "task_watchdog",
                },
            )
        )
    )

    out = capsys.readouterr().out
    assert "assistant> [WARNING] task watchdog: watchdog stale" in out


def test_cli_channel_start_stop_and_on_event(monkeypatch) -> None:
    channel = CLIChannel(
        ChannelContext(name="cli", deployment_id="d1", spec=ChannelSpec(type="cli"))
    )
    invoked: list[str] = []

    async def fake_loop(_handler) -> None:
        invoked.append("loop")

    monkeypatch.setattr(channel, "_input_loop", fake_loop)

    async def run() -> None:
        async def handler(_turn: UserTurn) -> None:
            return None

        await channel.start(handler)
        await asyncio.sleep(0)
        await channel.on_event(object())  # type: ignore[arg-type]
        await channel.stop()

    asyncio.run(run())

    assert invoked == ["loop"]


def test_cli_channel_start_skips_input_loop_without_tty(monkeypatch) -> None:
    channel = CLIChannel(
        ChannelContext(name="cli", deployment_id="d1", spec=ChannelSpec(type="cli"))
    )
    invoked: list[str] = []

    class _FakeStdin:
        @staticmethod
        def isatty() -> bool:
            return False

    async def fake_loop(_handler) -> None:
        invoked.append("loop")

    monkeypatch.setattr("sys.stdin", _FakeStdin())
    monkeypatch.setattr(channel, "_input_loop", fake_loop)

    async def run() -> None:
        async def handler(_turn: UserTurn) -> None:
            return None

        await channel.start(handler)
        await channel.stop()

    asyncio.run(run())

    assert invoked == []
