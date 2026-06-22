from __future__ import annotations

import sys
from types import SimpleNamespace

from hey_robot.channels.feishu import transport
from hey_robot.channels.feishu.transport import FeishuTransport


class _Response:
    def __init__(
        self,
        *,
        success: bool,
        code: object = None,
        status_code: int | None = None,
    ) -> None:
        self.code = code
        self.raw = SimpleNamespace(status_code=status_code, content=b"")

        self._success = success

    def success(self) -> bool:
        return self._success


class _CreateMessageRequestBody:
    @classmethod
    def builder(cls) -> _Builder:
        return _Builder()


class _CreateMessageRequest:
    @classmethod
    def builder(cls) -> _Builder:
        return _Builder()


class _Builder:
    def receive_id_type(self, _value: object) -> _Builder:
        return self

    def request_body(self, _value: object) -> _Builder:
        return self

    def receive_id(self, _value: object) -> _Builder:
        return self

    def msg_type(self, _value: object) -> _Builder:
        return self

    def content(self, _value: object) -> _Builder:
        return self

    def build(self) -> object:
        return object()


def _install_fake_lark_message_module(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "lark_oapi.api.im.v1",
        SimpleNamespace(
            CreateMessageRequest=_CreateMessageRequest,
            CreateMessageRequestBody=_CreateMessageRequestBody,
        ),
    )


def test_retryable_response_handles_absent_code() -> None:
    assert not transport._is_retryable_response(_Response(success=False))


def test_retryable_response_accepts_http_429_or_5xx() -> None:
    assert transport._is_retryable_response(_Response(success=False, status_code=429))
    assert transport._is_retryable_response(_Response(success=False, status_code=503))


def test_retryable_response_rejects_permission_error_code() -> None:
    assert not transport._is_retryable_response(_Response(success=False, code=99991663))


def test_send_message_retries_transient_exception(monkeypatch) -> None:
    _install_fake_lark_message_module(monkeypatch)
    monkeypatch.setattr(transport.time, "sleep", lambda _delay: None)
    calls = {"count": 0}

    class Message:
        def create(self, _request: object) -> _Response:
            calls["count"] += 1
            if calls["count"] == 1:
                raise OSError("temporary network failure")
            return _Response(success=True)

    client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=Message())))

    assert FeishuTransport(client).send_message("open_id", "ou_user", "text", "{}")
    assert calls["count"] == 2


def test_send_message_does_not_retry_non_retryable_response(monkeypatch) -> None:
    _install_fake_lark_message_module(monkeypatch)
    monkeypatch.setattr(transport.time, "sleep", lambda _delay: None)
    calls = {"count": 0}

    class Message:
        def create(self, _request: object) -> _Response:
            calls["count"] += 1
            return _Response(success=False, code=99991663)

    client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=Message())))

    assert not FeishuTransport(client).send_message("open_id", "ou_user", "text", "{}")
    assert calls["count"] == 1
