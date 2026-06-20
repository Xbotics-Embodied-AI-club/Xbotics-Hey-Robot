import asyncio
import inspect
import socket
import time
from collections.abc import Callable
from typing import Any, cast

from hey_robot.channels.base import ChannelContext, InboundHandler
from hey_robot.events import RuntimeEvent
from hey_robot.frontend_paths import frontend_root
from hey_robot.protocol import AgentReply, Envelope, UserTurn
from hey_robot.protocol.messages import to_payload

HistoryPayload = dict[str, Any]
HistoryProvider = Callable[[Envelope, int], HistoryPayload | Any]
BindingPayload = dict[str, Any]
BindingProvider = Callable[[Envelope, float], BindingPayload | Any]
BindingStatusProvider = Callable[[str], BindingPayload | Any]
CockpitPayload = dict[str, Any] | None
CockpitProvider = Callable[[str], CockpitPayload | Any]
TasksListPayload = dict[str, Any]
TasksListProvider = Callable[[int], TasksListPayload | Any]
EpisodeTaskPayload = dict[str, Any] | None
EpisodeTaskProvider = Callable[[str], EpisodeTaskPayload | Any]
RuntimeSummaryPayload = dict[str, Any]
RuntimeSummaryProvider = Callable[[int], RuntimeSummaryPayload | Any]


class WebChannel:
    """HTTP/WebSocket channel for direct interaction only."""

    def __init__(
        self,
        context: ChannelContext,
        *,
        history_provider: HistoryProvider | None = None,
        binding_provider: BindingProvider | None = None,
        binding_status_provider: BindingStatusProvider | None = None,
        cockpit_provider: CockpitProvider | None = None,
        tasks_list_provider: TasksListProvider | None = None,
        episode_task_provider: EpisodeTaskProvider | None = None,
        runtime_summary_provider: RuntimeSummaryProvider | None = None,
    ) -> None:
        self.context = context
        self.name = context.name
        self.history_provider = history_provider
        self.binding_provider = binding_provider
        self.binding_status_provider = binding_status_provider
        self.cockpit_provider = cockpit_provider
        self.tasks_list_provider = tasks_list_provider
        self.episode_task_provider = episode_task_provider
        self.runtime_summary_provider = runtime_summary_provider
        self.host = str(context.spec.settings.get("host", "127.0.0.1"))
        self.port = int(context.spec.settings.get("port", 8080))
        self._server: Any | None = None
        self._server_task: asyncio.Task | None = None
        self._websockets: set[Any] = set()
        self._events: list[dict[str, Any]] = []
        self._replies: list[dict[str, Any]] = []
        self._cached_access_urls: list[str] | None = None

    async def start(self, handler: InboundHandler) -> None:
        try:
            import uvicorn
            from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
            from fastapi.responses import HTMLResponse, RedirectResponse
            from fastapi.staticfiles import StaticFiles
        except ImportError as exc:
            raise ImportError(
                "WebChannel requires fastapi and uvicorn. Install `hey-robot[agent]`."
            ) from exc

        app = FastAPI(title="Hey Robot Gateway")
        app.mount("/static", StaticFiles(directory=str(_static_root())), name="static")

        @app.middleware("http")
        async def _static_cache(request, call_next):
            response = await call_next(request)
            if request.url.path.startswith("/static/"):
                response.headers["Cache-Control"] = "public, max-age=3600"
            return response

        @app.get("/", response_class=RedirectResponse)
        async def dashboard() -> RedirectResponse:
            return RedirectResponse(url="/chat")

        @app.get("/chat", response_class=HTMLResponse)
        async def chat_page() -> HTMLResponse:
            return self._chat_html_response()  # type: ignore[no-any-return]

        # Legacy routes — redirect to new chat
        @app.get("/console", response_class=RedirectResponse)
        async def console_page() -> RedirectResponse:
            return RedirectResponse(url="/chat")

        @app.get("/control", response_class=RedirectResponse)
        async def control_page() -> RedirectResponse:
            return RedirectResponse(url="/chat")

        @app.get("/account", response_class=RedirectResponse)
        async def account_page() -> RedirectResponse:
            return RedirectResponse(url="/chat")

        @app.get("/admin", response_class=HTMLResponse)
        async def admin_page() -> HTMLResponse:
            return self._views_html_response("admin", "index.html")  # type: ignore[no-any-return]

        @app.get("/tasks", response_class=HTMLResponse)
        async def tasks_list_page() -> HTMLResponse:
            return self._views_html_response("tasks", "index.html")  # type: ignore[no-any-return]

        @app.get("/tasks/{episode_id}", response_class=HTMLResponse)
        async def tasks_detail_page(episode_id: str) -> HTMLResponse:  # noqa: ARG001
            return self._views_html_response("tasks", "detail.html")  # type: ignore[no-any-return]

        @app.get("/cockpit", response_class=RedirectResponse)
        async def cockpit_page() -> RedirectResponse:
            return RedirectResponse(url="/tasks")

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok", "channel": self.name}

        @app.get("/config.json")
        async def config_json() -> dict[str, Any]:
            primary_url = self._primary_access_url()
            return {
                "deployment_id": self.context.deployment_id,
                "channel": self.name,
                "listen_host": self.host,
                "listen_port": self.port,
                "access_url": primary_url,
                "default_chat": str(self.context.spec.settings.get("chat_id") or "web"),
                "default_sender": str(
                    self.context.spec.settings.get("sender_id") or "web-user"
                ),
                "features": {
                    "history": self.history_provider is not None,
                    "identity_binding": self.binding_provider is not None,
                    "recent_replies": True,
                    "runtime_events": True,
                    "websocket": True,
                    "execution_feedback": True,
                    "cockpit": self.cockpit_provider is not None,
                },
                "identity": {
                    "binding_enabled": self.binding_provider is not None,
                    "binding_status_enabled": self.binding_status_provider is not None,
                },
            }

        @app.post("/turn")
        async def turn(payload: dict[str, Any]) -> dict[str, Any]:
            try:
                user_turn = self._payload_to_turn(payload)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            await handler(user_turn)
            return {"accepted": True, "trace_id": user_turn.envelope.trace_id}

        @app.get("/cockpit/{episode_id}")
        async def cockpit_data(episode_id: str) -> dict[str, Any]:
            if self.cockpit_provider is None:
                raise HTTPException(status_code=404, detail="cockpit data is disabled")
            result = self.cockpit_provider(episode_id)
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"no task found for episode {episode_id}",
                )
            return cast(dict[str, Any], result)

        @app.get("/api/tasks")
        async def tasks_list(limit: int = 50) -> dict[str, Any]:
            if self.tasks_list_provider is None:
                raise HTTPException(status_code=404, detail="task list is disabled")
            result = self.tasks_list_provider(max(1, min(int(limit), 200)))
            if inspect.isawaitable(result):
                return cast(TasksListPayload, await result)
            return cast(TasksListPayload, result)

        @app.get("/api/episodes/{episode_id}/task")
        async def episode_task_detail(episode_id: str) -> dict[str, Any]:
            if self.episode_task_provider is None:
                raise HTTPException(
                    status_code=404, detail="episode task detail is disabled"
                )
            result = self.episode_task_provider(episode_id)
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"no task report for episode {episode_id}",
                )
            return cast(dict[str, Any], result)

        @app.get("/api/runtime-summary")
        async def runtime_summary(limit: int = 50) -> dict[str, Any]:
            if self.runtime_summary_provider is None:
                raise HTTPException(
                    status_code=404, detail="runtime summary is disabled"
                )
            result = self.runtime_summary_provider(max(1, min(int(limit), 200)))
            if inspect.isawaitable(result):
                return cast(RuntimeSummaryPayload, await result)
            return cast(RuntimeSummaryPayload, result)

        @app.websocket("/ws")
        async def ws(socket: WebSocket) -> None:
            await socket.accept()
            self._websockets.add(socket)
            try:
                while True:
                    payload = await socket.receive_json()
                    try:
                        user_turn = self._payload_to_turn(payload)
                    except ValueError as exc:
                        await socket.send_json({"type": "error", "detail": str(exc)})
                        continue
                    await handler(user_turn)
                    await socket.send_json(
                        {"accepted": True, "trace_id": user_turn.envelope.trace_id}
                    )
            except WebSocketDisconnect:
                return
            finally:
                self._websockets.discard(socket)

        @app.get("/replies/recent")
        async def recent_replies(
            limit: int = 100,
            chat_id: str | None = None,
            sender_id: str | None = None,
            user_id: str | None = None,
            trace_id: str | None = None,
        ) -> dict[str, Any]:
            if not any((chat_id, sender_id, user_id, trace_id)):
                return {"replies": self._replies[-max(1, min(int(limit), 500)) :]}
            envelope = self._payload_to_envelope(
                {
                    "chat_id": chat_id,
                    "sender_id": sender_id,
                    "user_id": user_id,
                }
            )
            replies = self._filter_frames(
                self._replies,
                envelope=envelope,
                trace_id=trace_id,
            )
            return {"replies": replies[-max(1, min(int(limit), 500)) :]}

        @app.get("/events/recent")
        async def recent_events(
            limit: int = 100,
            chat_id: str | None = None,
            sender_id: str | None = None,
            user_id: str | None = None,
            trace_id: str | None = None,
        ) -> dict[str, Any]:
            if not any((chat_id, sender_id, user_id, trace_id)):
                return {"events": self._events[-max(1, min(int(limit), 500)) :]}
            envelope = self._payload_to_envelope(
                {
                    "chat_id": chat_id,
                    "sender_id": sender_id,
                    "user_id": user_id,
                }
            )
            events = self._filter_frames(
                self._events,
                envelope=envelope,
                trace_id=trace_id,
            )
            return {"events": events[-max(1, min(int(limit), 500)) :]}

        @app.get("/history")
        async def history(
            chat_id: str | None = None,
            sender_id: str | None = None,
            user_id: str | None = None,
            limit: int = 100,
        ) -> dict[str, Any]:
            if self.history_provider is None:
                return {"episode_id": None, "records": []}
            envelope = self._payload_to_envelope(
                {
                    "chat_id": chat_id,
                    "sender_id": sender_id,
                    "user_id": user_id,
                }
            )
            result = self.history_provider(envelope, max(1, min(int(limit), 500)))
            if inspect.isawaitable(result):
                return cast(HistoryPayload, await result)
            return cast(HistoryPayload, result)

        @app.post("/identity/binding")
        async def create_binding(payload: dict[str, Any]) -> dict[str, Any]:
            if self.binding_provider is None:
                raise HTTPException(
                    status_code=404, detail="identity binding is disabled"
                )
            envelope = self._payload_to_envelope(payload)
            ttl_sec = float(payload.get("ttl_sec") or 600.0)
            result = self.binding_provider(envelope, ttl_sec)
            if inspect.isawaitable(result):
                return cast(BindingPayload, await result)
            return cast(BindingPayload, result)

        @app.get("/identity/binding/{code}")
        async def binding_status(code: str) -> dict[str, Any]:
            if self.binding_status_provider is None:
                raise HTTPException(
                    status_code=404, detail="identity binding is disabled"
                )
            result = self.binding_status_provider(code)
            if inspect.isawaitable(result):
                return cast(BindingPayload, await result)
            return cast(BindingPayload, result)

        access_log = bool(self.context.spec.settings.get("access_log", False))
        log_level = str(self.context.spec.settings.get("uvicorn_log_level", "warning"))
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level=log_level,
            access_log=access_log,
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())

    async def send(self, reply: AgentReply) -> None:
        payload = to_payload(reply)
        self._replies.append(payload)
        self._replies = self._replies[-500:]
        stale = []
        for sock in self._websockets:
            try:
                await sock.send_json({"type": "agent.reply", "payload": payload})
            except Exception:
                stale.append(sock)
        for sock in stale:
            self._websockets.discard(sock)

    async def on_event(self, event: RuntimeEvent) -> None:
        payload = event.to_dict()
        self._events.append(payload)
        self._events = self._events[-1000:]
        stale = []
        for sock in self._websockets:
            try:
                await sock.send_json({"type": "runtime.event", "payload": payload})
            except Exception:
                stale.append(sock)
        for sock in stale:
            self._websockets.discard(sock)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            await asyncio.gather(self._server_task, return_exceptions=True)

    def _payload_to_turn(self, payload: dict[str, Any]) -> UserTurn:
        text = str(payload.get("text") or payload.get("message") or "").strip()
        if not text:
            raise ValueError("web turn requires text")
        return UserTurn(
            envelope=self._payload_to_envelope(payload),
            text=text,
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def _payload_to_envelope(self, payload: dict[str, Any]) -> Envelope:
        chat_id = (
            payload.get("chat_id")
            or payload.get("episode")
            or self.context.spec.settings.get("chat_id")
            or "web"
        )
        envelope = Envelope(
            channel=self.name,
            account_id=self.context.spec.account_id or self.name,
            user_id=(str(payload.get("user_id")).strip() or None)
            if payload.get("user_id") is not None
            else None,
            chat_id=str(chat_id),
            chat_type=str(payload.get("chat_type") or "web"),
            sender_id=str(
                payload.get("sender_id")
                or self.context.spec.settings.get("sender_id")
                or "web-user"
            ),
            message_id=payload.get("message_id"),
            reply_to_id=payload.get("reply_to_id"),
            deployment_id=self.context.deployment_id,
            timestamp=float(payload.get("timestamp") or time.time()),
        )
        return envelope

    def _html_response(self, page: str) -> Any:
        from fastapi.responses import HTMLResponse

        return HTMLResponse(
            self._page_html(page),
            headers={
                "Cache-Control": "no-store",
                "Pragma": "no-cache",
            },
        )

    def _chat_html_response(self) -> Any:
        from fastapi.responses import HTMLResponse

        html = (
            frontend_root()
            .joinpath("views", "chat", "index.html")
            .read_text(encoding="utf-8")
        )
        return HTMLResponse(
            html,
            headers={
                "Cache-Control": "no-store",
                "Pragma": "no-cache",
            },
        )

    def _views_html_response(self, view: str, page: str) -> Any:
        from fastapi.responses import HTMLResponse

        html = frontend_root().joinpath("views", view, page).read_text(encoding="utf-8")
        return HTMLResponse(
            html,
            headers={
                "Cache-Control": "no-store",
                "Pragma": "no-cache",
            },
        )

    def _page_html(self, page: str) -> str:
        return frontend_root().joinpath("interaction", page).read_text(encoding="utf-8")

    def _access_urls(self) -> list[str]:
        if self._cached_access_urls is not None:
            return self._cached_access_urls
        host = self.host.strip()
        if host and host not in {"0.0.0.0", "::"}:  # noqa: S104
            self._cached_access_urls = [self._format_url(host)]
            return self._cached_access_urls
        addresses: set[str] = set()
        try:
            hostname = socket.gethostname()
            for family, *_rest, sockaddr in socket.getaddrinfo(hostname, None):
                if family not in {socket.AF_INET, socket.AF_INET6}:
                    continue
                address = str(sockaddr[0])
                if (
                    not address
                    or address.startswith("127.")
                    or address == "::1"
                    or "%" in address
                ):
                    continue
                addresses.add(address)
        except OSError:
            pass
        self._cached_access_urls = [
            self._format_url(address) for address in sorted(addresses)
        ]
        return self._cached_access_urls

    def _primary_access_url(self) -> str | None:
        urls = self._access_urls()
        if not urls:
            return None
        scored = sorted(urls, key=self._access_url_rank)
        return scored[0]

    @staticmethod
    def _access_url_rank(url: str) -> tuple[int, str]:
        host = url.removeprefix("http://")
        if host.startswith("["):
            address = host[1:].split("]", 1)[0]
        else:
            address = host.rsplit(":", 1)[0]
        if address.startswith("192.168."):
            return (0, address)
        if address.startswith("10."):
            return (1, address)
        if address.startswith("172."):
            try:
                second = int(address.split(".")[1])
            except (IndexError, ValueError):
                second = -1
            if 16 <= second <= 31:
                return (2, address)
        if ":" not in address:
            return (3, address)
        if address.startswith("fd"):
            return (4, address)
        return (5, address)

    def _format_url(self, host: str) -> str:
        if ":" in host and not host.startswith("["):
            return f"http://[{host}]:{self.port}"
        return f"http://{host}:{self.port}"

    def _filter_frames(
        self,
        frames: list[dict[str, Any]],
        *,
        envelope: Envelope,
        trace_id: str | None,
    ) -> list[dict[str, Any]]:
        return [
            frame
            for frame in frames
            if self._matches_frame_scope(frame, envelope=envelope, trace_id=trace_id)
        ]

    def _matches_frame_scope(
        self,
        frame: dict[str, Any],
        *,
        envelope: Envelope,
        trace_id: str | None,
    ) -> bool:
        raw_envelope = frame.get("envelope")
        frame_envelope: dict = raw_envelope if isinstance(raw_envelope, dict) else {}
        frame_trace_id = frame_envelope.get("trace_id", frame.get("trace_id"))
        frame_channel = frame_envelope.get("channel", frame.get("channel"))
        frame_user_id = frame_envelope.get("user_id", frame.get("user_id"))
        frame_chat_id = frame_envelope.get("chat_id", frame.get("chat_id"))
        frame_sender_id = frame_envelope.get("sender_id", frame.get("sender_id"))
        if trace_id and str(frame_trace_id or "") != trace_id:
            return False
        if envelope.channel and str(frame_channel or "") != envelope.channel:
            return False
        if envelope.user_id and str(frame_user_id or "") != envelope.user_id:
            return False
        if envelope.chat_id and str(frame_chat_id or "") != envelope.chat_id:
            return False
        return not (
            envelope.sender_id and str(frame_sender_id or "") != envelope.sender_id
        )


def _static_root() -> Any:
    return frontend_root()
