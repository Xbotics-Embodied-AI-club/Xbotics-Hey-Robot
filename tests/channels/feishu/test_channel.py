from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from hey_robot.channels import ChannelContext, FeishuChannel
from hey_robot.config import ChannelSpec
from hey_robot.protocol import AgentReply, Envelope, UserTurn


def test_feishu_channel_imports_without_sdk() -> None:
    channel = FeishuChannel(
        ChannelContext(
            name="feishu", spec=ChannelSpec(type="feishu"), deployment_id="d1"
        )
    )
    assert channel.name == "feishu"


def test_feishu_channel_start_raises_clear_import_error(monkeypatch) -> None:
    monkeypatch.setattr("hey_robot.channels.feishu.FEISHU_AVAILABLE", False)
    channel = FeishuChannel(
        ChannelContext(
            name="feishu", spec=ChannelSpec(type="feishu"), deployment_id="d1"
        )
    )

    async def handler(_turn: UserTurn) -> None:
        return None

    with pytest.raises(ImportError, match="lark-oapi"):
        asyncio.run(channel.start(handler))


def test_feishu_channel_send_formats_text_without_sdk() -> None:
    sent: list[tuple[str, str, str, str | None]] = []

    class TestFeishuChannel(FeishuChannel):
        def _send_or_reply_sync(
            self,
            receive_id_type: str,
            chat_id: str,
            msg_type: str,
            _content: str,
            message_id: str | None,
        ) -> None:
            sent.append((receive_id_type, chat_id, msg_type, message_id))

    channel = TestFeishuChannel(
        ChannelContext(
            name="feishu", spec=ChannelSpec(type="feishu"), deployment_id="d1"
        )
    )
    channel._client = object()

    asyncio.run(
        channel.send(
            AgentReply(
                envelope=Envelope(
                    channel="feishu", chat_id="oc_chat_1", message_id="om_1"
                ),
                text="robot ready",
            )
        )
    )

    assert sent == [("chat_id", "oc_chat_1", "text", "om_1")]


def test_feishu_channel_send_formats_notification_card_without_sdk() -> None:
    sent: list[tuple[str, str, str, str | None]] = []

    class TestFeishuChannel(FeishuChannel):
        def _send_or_reply_sync(
            self,
            receive_id_type: str,
            chat_id: str,
            msg_type: str,
            _content: str,
            message_id: str | None,
        ) -> None:
            sent.append((receive_id_type, chat_id, msg_type, message_id))

    channel = TestFeishuChannel(
        ChannelContext(
            name="feishu", spec=ChannelSpec(type="feishu"), deployment_id="d1"
        )
    )
    channel._client = object()

    asyncio.run(
        channel.send(
            AgentReply(
                envelope=Envelope(channel="feishu", chat_id="oc_chat_1"),
                text="watchdog stale",
                metadata={
                    "notification": True,
                    "severity": "warning",
                    "notification_kind": "task_watchdog",
                },
            )
        )
    )

    assert sent == [("chat_id", "oc_chat_1", "interactive", None)]


def test_feishu_channel_on_message_builds_user_turn() -> None:
    channel = FeishuChannel(
        ChannelContext(
            name="feishu",
            spec=ChannelSpec(type="feishu", settings={"allow_from": ["*"]}),
            deployment_id="d1",
        )
    )
    turns: list[UserTurn] = []

    async def handler(turn: UserTurn) -> None:
        turns.append(turn)

    channel._handler = handler
    channel._bot_open_id = "ou_bot"
    message = SimpleNamespace(
        message_id="om_1",
        chat_id="oc_chat",
        chat_type="group",
        message_type="text",
        content='{"text":"@_user_1 move forward"}',
        mentions=[
            SimpleNamespace(
                key="@_user_1", id=SimpleNamespace(open_id="ou_bot"), name="robot"
            )
        ],
        parent_id=None,
        root_id=None,
        thread_id=None,
    )
    event = SimpleNamespace(
        message=message,
        sender=SimpleNamespace(
            sender_type="user", sender_id=SimpleNamespace(open_id="ou_user")
        ),
    )

    asyncio.run(channel._on_message(SimpleNamespace(event=event)))

    assert len(turns) == 1
    assert turns[0].envelope.channel == "feishu"
    assert turns[0].envelope.chat_id == "oc_chat"
    assert turns[0].envelope.message_id == "om_1"
    assert "move forward" in turns[0].text
    assert turns[0].envelope.robot_id is None


def test_feishu_channel_start_builds_client_with_fake_sdk(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeBuilder:
        def __init__(self) -> None:
            self.data: dict[str, object] = {}

        def app_id(self, value):
            self.data["app_id"] = value
            return self

        def app_secret(self, value):
            self.data["app_secret"] = value
            return self

        def domain(self, value):
            self.data["domain"] = value
            return self

        def log_level(self, value):
            self.data["log_level"] = value
            return self

        def build(self):
            return SimpleNamespace(
                **self.data, request=lambda _req: SimpleNamespace(success=lambda: False)
            )

    class FakeDispatcherBuilder:
        def __init__(self) -> None:
            self.handler = None

        def register_p2_im_message_receive_v1(self, handler):
            self.handler = handler
            return self

        def build(self):
            return SimpleNamespace(handler=self.handler)

    class FakeWsClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["ws_args"] = args
            captured["ws_kwargs"] = kwargs

        def start(self) -> None:
            return None

    fake_lark = SimpleNamespace(
        LogLevel=SimpleNamespace(INFO="INFO"),
        Client=SimpleNamespace(builder=lambda: FakeBuilder()),
        EventDispatcherHandler=SimpleNamespace(
            builder=lambda *_args: FakeDispatcherBuilder()
        ),
        ws=SimpleNamespace(Client=FakeWsClient),
    )
    monkeypatch.setattr("hey_robot.channels.feishu.FEISHU_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "lark_oapi", fake_lark)
    monkeypatch.setitem(
        sys.modules,
        "lark_oapi.core.const",
        SimpleNamespace(FEISHU_DOMAIN="feishu", LARK_DOMAIN="lark"),
    )
    monkeypatch.setitem(
        sys.modules, "lark_oapi.ws", SimpleNamespace(client=SimpleNamespace(loop=None))
    )
    monkeypatch.setitem(sys.modules, "lark_oapi.ws.client", SimpleNamespace(loop=None))

    channel = FeishuChannel(
        ChannelContext(
            name="feishu",
            spec=ChannelSpec(
                type="feishu",
                settings={"app_id": "app", "app_secret": "secret", "allow_from": ["*"]},
            ),
            deployment_id="d1",
        )
    )

    async def handler(_turn: UserTurn) -> None:
        return None

    async def run() -> None:
        task = asyncio.create_task(channel.start(handler))
        await asyncio.sleep(0.05)
        await channel.stop()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())

    assert getattr(channel._client, "app_id") == "app"
    ws_args = captured["ws_args"]
    assert isinstance(ws_args, tuple)
    assert ws_args[0] == "app"
