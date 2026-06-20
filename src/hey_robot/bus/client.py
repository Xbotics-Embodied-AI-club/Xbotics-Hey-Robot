"""NATS 消息总线客户端（生产级）。"""

from __future__ import annotations

import json
import ssl
from collections.abc import Awaitable, Callable
from typing import Any


class BusClient:
    """NATS 客户端封装，提供 publish/subscribe 接口。"""

    def __init__(
        self,
        url: str,
        *,
        tls_ca_file: str | None = None,
        tls_cert_file: str | None = None,
        tls_key_file: str | None = None,
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
        reconnect: bool = True,
        max_reconnect_attempts: int = 60,
        reconnect_time_wait_ms: int = 2000,
        use_jetstream: bool = False,
        js_stream: str = "hey_robot",
    ):
        self.url = url
        self.tls_ca_file = tls_ca_file
        self.tls_cert_file = tls_cert_file
        self.tls_key_file = tls_key_file
        self.username = username
        self.password = password
        self.token = token
        self.reconnect = reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_time_wait_ms = reconnect_time_wait_ms
        self.use_jetstream = use_jetstream
        self.js_stream = js_stream
        self._client: Any | None = None
        self._subscriptions: dict[str, Any] = {}
        self._connected = False
        self._js: Any | None = None
        self._js_stream_ready = False

    async def connect(self):
        try:
            import nats
        except ImportError as e:
            raise ImportError("未安装 nats。可用: pip install nats-py") from e

        tls_ctx = None
        if self.tls_ca_file or self.tls_cert_file or self.tls_key_file:
            tls_ctx = ssl.create_default_context(cafile=self.tls_ca_file)
            if self.tls_cert_file and self.tls_key_file:
                tls_ctx.load_cert_chain(self.tls_cert_file, self.tls_key_file)

        self._client = await nats.connect(
            servers=[self.url],
            tls=tls_ctx,
            user=self.username,
            password=self.password,
            token=self.token,
            allow_reconnect=self.reconnect,
            max_reconnect_attempts=self.max_reconnect_attempts,
            reconnect_time_wait=self.reconnect_time_wait_ms / 1000.0,
        )
        self._connected = True

        if self.use_jetstream:
            assert self._client is not None
            self._js = self._client.jetstream()

    async def publish(self, topic: str, payload: dict):
        if not self._connected or self._client is None:
            raise RuntimeError("BusClient 未连接。请先调用 connect()。")
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.use_jetstream and self._js:
            await self._js.publish(topic, data)
        else:
            assert self._client is not None
            await self._client.publish(topic, data)

    async def publish_raw(self, topic: str, payload: bytes) -> None:
        if not self._connected or self._client is None:
            raise RuntimeError("BusClient is not connected")
        if self.use_jetstream and self._js:
            await self._js.publish(topic, payload)
        else:
            await self._client.publish(topic, payload)

    async def subscribe(
        self,
        topics: list[str],
        on_message: Callable[[str, dict], Awaitable[None]],
    ):
        if not self._connected or self._client is None:
            raise RuntimeError("BusClient 未连接。请先调用 connect()。")

        async def _handler(msg):
            try:
                payload = json.loads(msg.data.decode("utf-8"))
            except json.JSONDecodeError:
                payload = {}
            await on_message(msg.subject, payload)

        for topic in topics:
            if self.use_jetstream and self._js:
                await self._ensure_js_stream(topic)
                assert self._js is not None
                sub = await self._js.subscribe(
                    topic,
                    cb=_handler,
                    durable=f"durable_{topic.replace('.', '_')}",
                )
                self._subscriptions[topic] = sub
            else:
                assert self._client is not None
                sub = await self._client.subscribe(topic, cb=_handler)
                self._subscriptions[topic] = sub

    async def subscribe_raw(
        self,
        topics: list[str],
        on_message: Callable[[str, bytes], Awaitable[None]],
    ) -> None:
        if not self._connected or self._client is None:
            raise RuntimeError("BusClient is not connected")

        async def _handler(msg):
            await on_message(msg.subject, bytes(msg.data))

        for topic in topics:
            sub = await self._client.subscribe(topic, cb=_handler)
            self._subscriptions[topic] = sub

    async def unsubscribe(self, topics: list[str]):
        if not self._connected or self._client is None:
            return
        for topic in topics:
            sub = self._subscriptions.pop(topic, None)
            if sub is not None:
                await sub.unsubscribe()

    async def close(self):
        if self._client is not None:
            await self._client.drain()
            await self._client.close()
        self._connected = False

    async def _ensure_js_stream(self, topic: str):
        if self._js is None or self._js_stream_ready:
            return
        try:
            await self._js.stream_info(self.js_stream)
        except Exception:
            await self._js.add_stream(name=self.js_stream, subjects=[topic])
        self._js_stream_ready = True
