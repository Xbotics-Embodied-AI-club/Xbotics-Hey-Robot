from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest

from hey_robot.channels import ChannelContext, WebChannel
from hey_robot.config import ChannelSpec, DeploymentConfig
from hey_robot.events import EventKind, RuntimeEvent
from hey_robot.frontend_paths import frontend_root
from hey_robot.gateway import GatewayService
from hey_robot.protocol import AgentReply, Envelope, UserTurn


def test_web_channel_payload_and_live_broadcasts() -> None:
    channel = WebChannel(
        ChannelContext(
            name="web",
            spec=ChannelSpec(
                type="web", settings={"chat_id": "chat-a", "sender_id": "user-a"}
            ),
            deployment_id="d1",
        )
    )

    turn = channel._payload_to_turn(
        {"text": "pick", "sender_id": "u1", "chat_id": "c1"}
    )

    asyncio.run(channel.send(AgentReply(envelope=Envelope(channel="web"), text="ok")))
    asyncio.run(
        channel.on_event(RuntimeEvent.make(EventKind.ROBOT_STATUS, source="robot"))
    )

    assert turn.text == "pick"
    assert turn.envelope.channel == "web"
    assert turn.envelope.sender_id == "u1"
    assert turn.envelope.robot_id is None
    html_body = channel._chat_html_response().body
    if isinstance(html_body, bytes):
        html_body = html_body.decode()
    assert "/static/shared/js/api.js" in html_body
    assert "/static/shared/js/ws.js" in html_body
    assert "/static/views/chat/chat.js" in html_body

    api_js = (
        frontend_root().joinpath("shared", "js", "api.js").read_text(encoding="utf-8")
    )
    assert "/turn" in api_js

    ws_js = (
        frontend_root().joinpath("shared", "js", "ws.js").read_text(encoding="utf-8")
    )
    assert "/ws" in ws_js
    assert channel._replies[-1]["text"] == "ok"
    assert channel._events[-1]["kind"] == "robot.status"


def test_interaction_ui_treats_progress_replies_as_non_terminal() -> None:
    script = WebChannel(
        ChannelContext(name="web", spec=ChannelSpec(type="web"), deployment_id="d1")
    )._chat_html_response()

    html_body = script.body
    if isinstance(html_body, bytes):
        html_body = html_body.decode()
    assert "/static/views/chat/chat.js" in html_body

    static_js = (
        frontend_root().joinpath("views", "chat", "chat.js").read_text(encoding="utf-8")
    )
    assert "handleRuntimeEvent" not in static_js
    assert "updateRobotStatus" not in static_js
    assert "sendQuickAction" not in static_js

    assert "WS.connect" in static_js
    assert "bindStore" in static_js
    assert "loadLastMessages" in static_js


def test_web_history_restores_persisted_episode(tmp_path) -> None:
    config = DeploymentConfig.from_dict(
        {
            "deployment": {"id": "d1"},
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "robots": {"mock0": {"type": "mock"}},
            "agents": {"main": {"type": "robot_agent", "robot_id": "mock0"}},
            "channels": {"web": {"type": "web", "enabled": True}},
        }
    )
    gateway = GatewayService(config, episode_dir=tmp_path / "episodes")
    envelope = Envelope(channel="web", chat_id="c1", chat_type="web", sender_id="u1")

    first = asyncio.run(gateway._web_history(envelope, 10))
    gateway.episodes.append_user_turn(
        first["episode_id"],
        UserTurn(
            envelope=envelope.child(episode_id=first["episode_id"], agent_id="main"),
            text="hello",
        ),
    )

    history = asyncio.run(gateway._web_history(envelope, 10))

    assert history["episode_id"] == first["episode_id"]
    assert history["records"][0]["role"] == "user"
    assert history["records"][0]["content"] == "hello"


def test_web_channel_payload_defaults_and_socket_cleanup() -> None:
    channel = WebChannel(
        ChannelContext(
            name="web",
            spec=ChannelSpec(type="web", settings={"chat_id": "default-chat"}),
            deployment_id="d1",
        )
    )

    turn = channel._payload_to_turn(
        {
            "message": " inspect now ",
            "episode": "ep-7",
            "sender_id": "user-7",
            "metadata": {"source": "browser"},
        }
    )

    assert turn.text == "inspect now"
    assert turn.envelope.chat_id == "ep-7"
    assert turn.envelope.sender_id == "user-7"
    assert turn.metadata["source"] == "browser"

    envelope = channel._payload_to_envelope({"timestamp": 123.0})

    assert envelope.chat_id == "default-chat"
    assert envelope.robot_id is None
    assert envelope.timestamp == 123.0

    with pytest.raises(ValueError, match="requires text"):
        channel._payload_to_turn({"text": "  "})

    sent: list[dict[str, object]] = []

    class GoodSocket:
        async def send_json(self, payload):
            sent.append(payload)

    class BadSocket:
        async def send_json(self, _payload):
            raise RuntimeError("stale")

    channel._websockets = {GoodSocket(), BadSocket()}
    asyncio.run(
        channel.send(AgentReply(envelope=Envelope(channel="web"), text="reply-1"))
    )
    asyncio.run(
        channel.on_event(RuntimeEvent.make(EventKind.ROBOT_STATUS, source="robot"))
    )

    assert len(sent) == 2
    assert len(channel._websockets) == 1
    assert sent[0]["type"] == "agent.reply"
    assert sent[1]["type"] == "runtime.event"


def test_web_channel_stop_requests_server_exit() -> None:
    channel = WebChannel(
        ChannelContext(name="web", spec=ChannelSpec(type="web"), deployment_id="d1")
    )
    channel._server = SimpleNamespace(should_exit=False)

    async def runner() -> None:
        channel._server_task = asyncio.create_task(asyncio.sleep(0))
        await channel.stop()

    asyncio.run(runner())

    assert channel._server.should_exit is True


def test_web_channel_start_registers_routes_and_handles_http_and_ws(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    handled_turns: list[UserTurn] = []
    binding_calls: list[tuple[str | None, float]] = []
    cockpit_calls: list[str] = []

    def binding_provider(envelope: Envelope, ttl_sec: float) -> dict[str, object]:
        binding_calls.append((envelope.sender_id, ttl_sec))
        return {"code": "ABC123", "status": "pending"}

    def cockpit_provider(episode_id: str) -> dict[str, object] | None:
        cockpit_calls.append(episode_id)
        if episode_id == "missing":
            return None
        return {"episode_id": episode_id, "view": {"status": "active"}}

    class FakeHTMLResponse:
        def __init__(self, body: str, headers: dict[str, str] | None = None) -> None:
            self.body = body
            self.headers = headers or {}

    class FakeRedirectResponse(FakeHTMLResponse):
        def __init__(self, url: str) -> None:
            super().__init__(url)

    class FakeHTTPError(Exception):
        def __init__(self, *, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FakeWebSocketDisconnectError(Exception):
        pass

    class FakeStaticFiles:
        def __init__(self, *, directory: str) -> None:
            self.directory = directory

    class FakeFastAPI:
        def __init__(self, title: str) -> None:
            self.title = title
            self.routes: dict[tuple[str, str], Callable[..., Awaitable[Any]]] = {}
            self.mounts: list[tuple[str, object, str | None]] = []
            self.middlewares: list[tuple[str, Callable[..., Awaitable[Any]]]] = []
            captured["app"] = self

        def mount(self, path: str, app: object, name: str | None = None) -> None:
            self.mounts.append((path, app, name))

        def middleware(self, middleware_type: str):
            def decorator(func):
                self.middlewares.append((middleware_type, func))
                return func

            return decorator

        def get(self, path: str, **_kwargs):
            def decorator(func):
                self.routes[("GET", path)] = func
                return func

            return decorator

        def post(self, path: str, **_kwargs):
            def decorator(func):
                self.routes[("POST", path)] = func
                return func

            return decorator

        def websocket(self, path: str):
            def decorator(func):
                self.routes[("WS", path)] = func
                return func

            return decorator

    class FakeConfig:
        def __init__(
            self, app: object, host: str, port: int, log_level: str, access_log: bool
        ) -> None:
            self.app = app
            self.host = host
            self.port = port
            self.log_level = log_level
            self.access_log = access_log

    class FakeServer:
        def __init__(self, config: FakeConfig) -> None:
            self.config = config
            self.should_exit = False

        async def serve(self) -> None:
            return None

    fake_fastapi = SimpleNamespace(
        FastAPI=FakeFastAPI,
        HTTPException=FakeHTTPError,
        WebSocket=object,
        WebSocketDisconnect=FakeWebSocketDisconnectError,
    )
    fake_responses = SimpleNamespace(
        HTMLResponse=FakeHTMLResponse, RedirectResponse=FakeRedirectResponse
    )
    fake_staticfiles = SimpleNamespace(StaticFiles=FakeStaticFiles)
    fake_uvicorn = SimpleNamespace(Config=FakeConfig, Server=FakeServer)

    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.responses", fake_responses)
    monkeypatch.setitem(sys.modules, "fastapi.staticfiles", fake_staticfiles)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    channel = WebChannel(
        ChannelContext(
            name="web",
            spec=ChannelSpec(
                type="web",
                settings={
                    "chat_id": "chat-a",
                    "sender_id": "user-a",
                    "port": 9001,
                    "access_log": True,
                    "uvicorn_log_level": "info",
                },
            ),
            deployment_id="d1",
        ),
        history_provider=lambda envelope, limit: {
            "episode_id": envelope.chat_id,
            "records": [{"role": "user", "content": f"history:{limit}"}],
        },
        binding_provider=binding_provider,
        binding_status_provider=lambda code: {"code": code, "status": "claimed"},
        cockpit_provider=cockpit_provider,
    )

    async def handler(turn: UserTurn) -> None:
        handled_turns.append(turn)

    async def exercise() -> None:
        await channel.start(handler)
        app = cast(FakeFastAPI, captured["app"])
        assert [kind for kind, _ in app.middlewares] == ["http"]

        dashboard = app.routes[("GET", "/")]
        health = app.routes[("GET", "/health")]
        config_json = app.routes[("GET", "/config.json")]
        cockpit_data = app.routes[("GET", "/cockpit/{episode_id}")]
        turn = app.routes[("POST", "/turn")]
        replies_recent = app.routes[("GET", "/replies/recent")]
        events_recent = app.routes[("GET", "/events/recent")]
        history = app.routes[("GET", "/history")]
        create_binding = app.routes[("POST", "/identity/binding")]
        binding_status = app.routes[("GET", "/identity/binding/{code}")]
        ws = app.routes[("WS", "/ws")]

        html = await dashboard()
        assert isinstance(html, FakeRedirectResponse)
        chat_html = await app.routes[("GET", "/chat")]()
        assert isinstance(chat_html, FakeHTMLResponse)
        assert chat_html.headers["Cache-Control"] == "no-store"
        assert (await health()) == {"status": "ok", "channel": "web"}
        features = (await config_json())["features"]
        assert features["history"] is True
        assert features["identity_binding"] is True
        assert features["cockpit"] is True
        assert "quick_actions" not in features
        identity = (await config_json())["identity"]
        assert identity["binding_enabled"] is True
        assert identity["binding_status_enabled"] is True

        accepted = await turn({"text": "hello web", "sender_id": "u1"})
        assert accepted["accepted"] is True
        assert handled_turns[-1].text == "hello web"
        assert (await cockpit_data("ep1"))["view"]["status"] == "active"
        with pytest.raises(FakeHTTPError, match="no task found"):
            await cockpit_data("missing")

        assert ("POST", "/actions/quick") not in app.routes

        with pytest.raises(FakeHTTPError, match="web turn requires text"):
            await turn({"text": "  "})
        await channel.send(AgentReply(envelope=Envelope(channel="web"), text="done"))
        await channel.on_event(
            RuntimeEvent.make(EventKind.ROBOT_STATUS, source="robot")
        )
        assert len((await replies_recent(limit=1))["replies"]) == 1
        assert len((await events_recent(limit=1))["events"]) == 1
        history_payload = await history(chat_id="ep-sync", user_id="owner", limit=2)
        assert history_payload["episode_id"] == "ep-sync"
        assert history_payload["records"][0]["content"] == "history:2"
        binding_payload = await create_binding(
            {"sender_id": "web-user", "user_id": "owner", "ttl_sec": 90}
        )
        assert binding_payload["code"] == "ABC123"
        assert (await binding_status("ABC123"))["status"] == "claimed"
        assert binding_calls == [("web-user", 90.0)]
        assert cockpit_calls == ["ep1", "missing"]
        assert handled_turns[0].envelope.user_id is None

        class FakeSocket:
            def __init__(self) -> None:
                self.accepted = False
                self.sent: list[dict[str, object]] = []
                self._payloads = iter([{"text": ""}, {"text": "ws message"}])

            async def accept(self) -> None:
                self.accepted = True

            async def receive_json(self) -> dict[str, object]:
                try:
                    return cast(dict[str, object], next(self._payloads))
                except StopIteration as exc:
                    raise FakeWebSocketDisconnectError() from exc

            async def send_json(self, payload: dict[str, object]) -> None:
                self.sent.append(payload)

        socket = FakeSocket()
        await ws(socket)

        assert socket.accepted is True
        assert socket.sent[0]["type"] == "error"
        assert socket.sent[1]["accepted"] is True
        assert handled_turns[-1].text == "ws message"
        await channel.stop()

    asyncio.run(exercise())
    assert channel._server is not None
    assert channel._server.config.port == 9001  # type: ignore[union-attr]


def test_web_channel_binding_routes_can_be_disabled(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeHTTPError(Exception):
        def __init__(self, *, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FakeStaticFiles:
        def __init__(self, *, directory: str) -> None:
            self.directory = directory

    class FakeFastAPI:
        def __init__(self, *, title: str) -> None:
            self.title = title
            self.routes: dict[tuple[str, str], Callable[..., Awaitable[Any]]] = {}
            self.middlewares: list[tuple[str, Callable[..., Awaitable[Any]]]] = []
            captured["app"] = self

        def mount(self, _path: str, _app: object, name: str | None = None) -> None:
            self.name = name

        def middleware(self, middleware_type: str):
            def decorator(func):
                self.middlewares.append((middleware_type, func))
                return func

            return decorator

        def get(self, path: str, **_kwargs):
            def decorator(func):
                self.routes[("GET", path)] = func
                return func

            return decorator

        def post(self, path: str, **_kwargs):
            def decorator(func):
                self.routes[("POST", path)] = func
                return func

            return decorator

        def websocket(self, path: str):
            def decorator(func):
                self.routes[("WS", path)] = func
                return func

            return decorator

    class FakeConfig:
        def __init__(
            self, app: object, host: str, port: int, log_level: str, access_log: bool
        ) -> None:
            self.app = app
            self.host = host
            self.port = port
            self.log_level = log_level
            self.access_log = access_log

    class FakeServer:
        def __init__(self, config: FakeConfig) -> None:
            self.config = config
            self.should_exit = False

        async def serve(self) -> None:
            return None

    fake_fastapi = SimpleNamespace(
        FastAPI=FakeFastAPI,
        HTTPException=FakeHTTPError,
        WebSocket=object,
        WebSocketDisconnect=RuntimeError,
    )
    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)
    monkeypatch.setitem(
        sys.modules,
        "fastapi.responses",
        SimpleNamespace(HTMLResponse=object, RedirectResponse=object),
    )
    monkeypatch.setitem(
        sys.modules, "fastapi.staticfiles", SimpleNamespace(StaticFiles=FakeStaticFiles)
    )
    monkeypatch.setitem(
        sys.modules, "uvicorn", SimpleNamespace(Config=FakeConfig, Server=FakeServer)
    )

    channel = WebChannel(
        ChannelContext(name="web", spec=ChannelSpec(type="web"), deployment_id="d1")
    )

    async def handler(_turn: UserTurn) -> None:
        return None

    async def exercise() -> None:
        await channel.start(handler)
        app = cast(FakeFastAPI, captured["app"])
        create_binding = app.routes[("POST", "/identity/binding")]
        binding_status = app.routes[("GET", "/identity/binding/{code}")]

        with pytest.raises(FakeHTTPError, match="disabled"):
            await create_binding({})
        with pytest.raises(FakeHTTPError, match="disabled"):
            await binding_status("ABC123")
        await channel.stop()

    asyncio.run(exercise())


def test_web_channel_start_raises_clear_import_error(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "fastapi", raising=False)
    monkeypatch.delitem(sys.modules, "fastapi.responses", raising=False)
    monkeypatch.delitem(sys.modules, "fastapi.staticfiles", raising=False)
    monkeypatch.delitem(sys.modules, "uvicorn", raising=False)

    original_import = __import__

    def fake_import(
        name, module_globals=None, module_locals=None, fromlist=(), level=0
    ):  # type: ignore[no-untyped-def]
        if name in {"fastapi", "uvicorn"}:
            raise ImportError("missing dependency")
        return original_import(name, module_globals, module_locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    channel = WebChannel(
        ChannelContext(name="web", spec=ChannelSpec(type="web"), deployment_id="d1")
    )

    async def handler(_turn: UserTurn) -> None:
        return None

    with pytest.raises(ImportError, match="requires fastapi and uvicorn"):
        asyncio.run(channel.start(handler))
