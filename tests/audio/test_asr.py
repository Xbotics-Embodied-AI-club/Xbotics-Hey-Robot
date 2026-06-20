from __future__ import annotations

import asyncio
import gzip
import io
import json
import struct
import wave
from dataclasses import dataclass

import pytest

from hey_robot.audio.asr import (
    ASRMessage,
    DoubaoASRClient,
    MsgFlag,
    MsgType,
    SherpaONNXASRClient,
    build_asr_client,
)
from hey_robot.audio.config import ASRConfig


def test_asr_transcribe_wav_uses_websocket_protocol(monkeypatch) -> None:
    sent: list[bytes] = []
    headers_seen: dict[str, str] = {}

    class FakeSocket:
        def __init__(self, messages: list[bytes]) -> None:
            self._messages = messages

        async def send(self, _data: bytes) -> None:
            sent.append(_data)

        async def recv(self) -> bytes:
            if self._messages:
                return self._messages.pop(0)
            raise TimeoutError

    @dataclass
    class FakeConnect:
        socket: FakeSocket

        async def __aenter__(self) -> FakeSocket:
            return self.socket

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_connect(
        _url: str, *, additional_headers: dict[str, str], max_size: int
    ) -> FakeConnect:
        assert _url == "wss://example.invalid/asr"
        assert max_size == 10 * 1024 * 1024
        headers_seen.update(additional_headers)
        return FakeConnect(
            FakeSocket(
                [
                    _build_server_response(
                        "小白 打开", flag=MsgFlag.POSITIVE_SEQ, sequence=1
                    ),
                    _build_server_response(
                        "小白 打开夹爪", flag=MsgFlag.LAST_NEGATIVE_SEQ, sequence=-1
                    ),
                ]
            )
        )

    import websockets

    monkeypatch.setattr(websockets, "connect", fake_connect)

    client = DoubaoASRClient(
        ASRConfig(
            api_key="test-key",
            endpoint="wss://example.invalid/asr",
            resource_id="volc.seedasr.sauc.duration",
            chunk_ms=200,
        )
    )

    result = asyncio.run(client.transcribe_wav(_build_wav_bytes(duration_ms=400)))

    assert result == "小白 打开夹爪"
    assert headers_seen["X-Api-Key"] == "test-key"
    assert headers_seen["X-Api-Resource-Id"] == "volc.seedasr.sauc.duration"
    assert headers_seen["X-Api-Sequence"] == "-1"
    assert len(sent) == 3

    full_request = ASRMessage.from_bytes(sent[0])
    assert full_request.msg_type == MsgType.FULL_CLIENT_REQUEST
    assert full_request.flag == MsgFlag.POSITIVE_SEQ
    assert full_request.sequence == 1
    assert full_request.payload_json["request"]["model_name"] == "bigmodel"

    first_audio = ASRMessage.from_bytes(sent[1])
    assert first_audio.msg_type == MsgType.AUDIO_ONLY_REQUEST
    assert first_audio.flag == MsgFlag.POSITIVE_SEQ
    assert first_audio.sequence == 2
    last_audio = ASRMessage.from_bytes(sent[2])
    assert last_audio.flag == MsgFlag.LAST_NEGATIVE_SEQ
    assert last_audio.sequence == -3


def test_asr_error_frame_raises_runtime_error(monkeypatch) -> None:
    class FakeSocket:
        def __init__(self) -> None:
            self._messages = [_build_server_error(45000001, "request invalid")]

        async def send(self, _data: bytes) -> None:
            return None

        async def recv(self) -> bytes:
            if self._messages:
                return self._messages.pop(0)
            raise TimeoutError

    class FakeConnect:
        async def __aenter__(self) -> FakeSocket:
            return FakeSocket()

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_connect(_url: str, **_kwargs: object) -> FakeConnect:
        return FakeConnect()

    import websockets

    monkeypatch.setattr(websockets, "connect", fake_connect)

    client = DoubaoASRClient(
        ASRConfig(
            api_key="test-key",
            endpoint="wss://example.invalid/asr",
            resource_id="volc.seedasr.sauc.duration",
        )
    )

    with pytest.raises(RuntimeError, match="ASR error code=45000001"):
        asyncio.run(client.transcribe_wav(_build_wav_bytes(duration_ms=200)))


def test_build_asr_client_returns_doubao_provider() -> None:
    client = build_asr_client(ASRConfig(provider="doubao"))

    assert isinstance(client, DoubaoASRClient)


def test_build_asr_client_returns_single_sherpa_provider() -> None:
    client = build_asr_client(ASRConfig(provider="sherpa_onnx"))

    assert isinstance(client, SherpaONNXASRClient)


def _build_server_response(text: str, *, flag: MsgFlag, sequence: int) -> bytes:
    payload = json.dumps({"result": {"text": text}}, ensure_ascii=False).encode("utf-8")
    compressed = gzip.compress(payload)
    header = bytes([0x11, (MsgType.FULL_SERVER_RESPONSE << 4) | flag, 0x11, 0x00])
    return (
        header
        + struct.pack(">i", sequence)
        + struct.pack(">I", len(compressed))
        + compressed
    )


def _build_server_error(code: int, message: str) -> bytes:
    payload = message.encode("utf-8")
    header = bytes([0x11, (MsgType.ERROR << 4), 0x10, 0x00])
    return header + struct.pack(">I", code) + struct.pack(">I", len(payload)) + payload


def _build_wav_bytes(*, duration_ms: int, sample_rate: int = 16000) -> bytes:
    frame_count = int(sample_rate * duration_ms / 1000)
    pcm = b"\x00\x00" * frame_count
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()
