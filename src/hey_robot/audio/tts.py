from __future__ import annotations

import copy
import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any, cast

from hey_robot.audio.config import TTSConfig
from hey_robot.audio.tts_protocol import (
    EventType,
    MsgType,
    TTSMessage,
    finish_connection_message,
    finish_episode_message,
    start_connection_message,
    start_episode_message,
    task_request_message,
)


class DoubaoTTSClient:
    """Volcengine/Doubao bidirectional TTS client."""

    def __init__(self, config: TTSConfig) -> None:
        self.config = config

    async def synthesize(self, text: str) -> bytes:
        chunks = [chunk async for chunk in self.synthesize_stream(text)]
        return b"".join(chunks)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        if not self.config.enabled or not text.strip():
            return
        api_key = self.config.resolved_api_key
        if not api_key:
            raise ValueError(
                f"TTS API key is not configured; set {self.config.api_key_env}"
            )
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                "Doubao TTS requires `websockets`. Install `hey-robot[agent]`."
            ) from exc

        headers = {
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": self.config.resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        async with websockets.connect(
            self.config.endpoint,
            additional_headers=headers,
            max_size=10 * 1024 * 1024,
        ) as socket:
            await socket.send(start_connection_message())
            await _expect(
                socket, MsgType.FULL_SERVER_RESPONSE, EventType.CONNECTION_STARTED
            )

            episode_id = str(uuid.uuid4())
            await socket.send(
                start_episode_message(
                    _start_episode_payload(self.config, episode_id), episode_id
                )
            )
            await _expect(
                socket, MsgType.FULL_SERVER_RESPONSE, EventType.EPISODE_STARTED
            )

            for chunk in _text_chunks(text):
                await socket.send(
                    task_request_message(_task_payload(self.config, chunk), episode_id)
                )
            await socket.send(finish_episode_message(episode_id))

            while True:
                message = TTSMessage.from_bytes(await _recv_bytes(socket))
                if message.msg_type == MsgType.AUDIO_ONLY_SERVER and message.payload:
                    yield message.payload
                elif (
                    message.msg_type == MsgType.FULL_SERVER_RESPONSE
                    and message.event == EventType.EPISODE_FINISHED
                ):
                    break
                elif message.msg_type == MsgType.ERROR:
                    raise RuntimeError(message.payload.decode("utf-8", errors="ignore"))

            await socket.send(finish_connection_message())


async def _expect(socket, msg_type: MsgType, event: EventType) -> TTSMessage:
    message = TTSMessage.from_bytes(await _recv_bytes(socket))
    if message.msg_type != msg_type or message.event != event:
        raise RuntimeError(
            f"unexpected TTS response: type={message.msg_type} event={message.event}"
        )
    return message


async def _recv_bytes(socket) -> bytes:
    data = await socket.recv()
    if isinstance(data, str):
        raise RuntimeError("unexpected text frame from TTS websocket")
    return cast(bytes, data)


def _start_episode_payload(config: TTSConfig, episode_id: str) -> bytes:
    payload: dict[str, Any] = {
        "user": {"uid": episode_id},
        "namespace": "BidirectionalTTS",
        "req_params": {
            "speaker": config.voice_type,
            "audio_params": {
                "format": config.encoding,
                "sample_rate": config.sample_rate,
                "enable_timestamp": True,
            },
            "additions": json.dumps({"disable_markdown_filter": False}),
        },
    }
    payload = copy.deepcopy(payload)
    payload["event"] = int(EventType.START_EPISODE)
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _task_payload(config: TTSConfig, text: str) -> bytes:
    payload: dict[str, Any] = {
        "user": {"uid": str(uuid.uuid4())},
        "namespace": "BidirectionalTTS",
        "event": int(EventType.TASK_REQUEST),
        "req_params": {
            "speaker": config.voice_type,
            "text": text,
        },
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _text_chunks(text: str, *, chunk_size: int = 2000) -> list[str]:
    cleaned = text.replace("**", "").strip()
    return [
        cleaned[index : index + chunk_size]
        for index in range(0, len(cleaned), chunk_size)
    ]
