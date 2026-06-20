from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from enum import IntEnum


class MsgType(IntEnum):
    FULL_CLIENT_REQUEST = 0b0001
    FULL_SERVER_RESPONSE = 0b1001
    AUDIO_ONLY_SERVER = 0b1011
    ERROR = 0b1111


class MsgFlag(IntEnum):
    NO_SEQ = 0
    WITH_EVENT = 0b0100


class EventType(IntEnum):
    START_CONNECTION = 1
    FINISH_CONNECTION = 2
    CONNECTION_STARTED = 50
    START_EPISODE = 100
    FINISH_EPISODE = 102
    EPISODE_STARTED = 150
    EPISODE_FINISHED = 152
    TASK_REQUEST = 200


@dataclass
class TTSMessage:
    msg_type: MsgType
    flag: MsgFlag = MsgFlag.NO_SEQ
    event: int = 0
    episode_id: str = ""
    payload: bytes = b""

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        buffer.write(bytes([0x11, (self.msg_type << 4) | self.flag, 0x10, 0x00]))
        if self.flag == MsgFlag.WITH_EVENT:
            buffer.write(struct.pack(">i", int(self.event)))
            if self.event not in {
                EventType.START_CONNECTION,
                EventType.FINISH_CONNECTION,
                EventType.CONNECTION_STARTED,
            }:
                episode = self.episode_id.encode("utf-8")
                buffer.write(struct.pack(">I", len(episode)))
                buffer.write(episode)
        buffer.write(struct.pack(">I", len(self.payload)))
        buffer.write(self.payload)
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> TTSMessage:
        if len(data) < 8:
            raise ValueError("TTS message is too short")
        msg_type = MsgType(data[1] >> 4)
        flag = MsgFlag(data[1] & 0x0F)
        offset = 4
        event = 0
        episode_id = ""
        if flag == MsgFlag.WITH_EVENT:
            event = struct.unpack(">i", data[offset : offset + 4])[0]
            offset += 4
            if event not in {
                EventType.CONNECTION_STARTED,
                EventType.START_CONNECTION,
                EventType.FINISH_CONNECTION,
            }:
                size = struct.unpack(">I", data[offset : offset + 4])[0]
                offset += 4
                episode_id = data[offset : offset + size].decode("utf-8")
                offset += size
            elif event == EventType.CONNECTION_STARTED and len(data) >= offset + 4:
                size = struct.unpack(">I", data[offset : offset + 4])[0]
                offset += 4 + size
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        return cls(
            msg_type=msg_type,
            flag=flag,
            event=event,
            episode_id=episode_id,
            payload=data[offset : offset + size],
        )


def start_connection_message() -> bytes:
    return TTSMessage(
        msg_type=MsgType.FULL_CLIENT_REQUEST,
        flag=MsgFlag.WITH_EVENT,
        event=EventType.START_CONNECTION,
        payload=b"{}",
    ).to_bytes()


def finish_connection_message() -> bytes:
    return TTSMessage(
        msg_type=MsgType.FULL_CLIENT_REQUEST,
        flag=MsgFlag.WITH_EVENT,
        event=EventType.FINISH_CONNECTION,
        payload=b"{}",
    ).to_bytes()


def start_episode_message(payload: bytes, episode_id: str) -> bytes:
    return TTSMessage(
        msg_type=MsgType.FULL_CLIENT_REQUEST,
        flag=MsgFlag.WITH_EVENT,
        event=EventType.START_EPISODE,
        episode_id=episode_id,
        payload=payload,
    ).to_bytes()


def task_request_message(payload: bytes, episode_id: str) -> bytes:
    return TTSMessage(
        msg_type=MsgType.FULL_CLIENT_REQUEST,
        flag=MsgFlag.WITH_EVENT,
        event=EventType.TASK_REQUEST,
        episode_id=episode_id,
        payload=payload,
    ).to_bytes()


def finish_episode_message(episode_id: str) -> bytes:
    return TTSMessage(
        msg_type=MsgType.FULL_CLIENT_REQUEST,
        flag=MsgFlag.WITH_EVENT,
        event=EventType.FINISH_EPISODE,
        episode_id=episode_id,
        payload=b"{}",
    ).to_bytes()
