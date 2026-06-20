from __future__ import annotations

import asyncio
import gzip
import io
import json
import struct
import uuid
import wave
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from hey_robot.audio.config import ASRConfig

_VERSION = 0x11
_HEADER_SIZE = 0x10
_RETRYABLE_TIMEOUTS = (asyncio.TimeoutError,)


class ASRClient(Protocol):
    async def transcribe_wav(
        self, wav_bytes: bytes, *, _filename: str = "utterance.wav"
    ) -> str: ...


class MsgType(IntEnum):
    FULL_CLIENT_REQUEST = 0b0001
    AUDIO_ONLY_REQUEST = 0b0010
    FULL_SERVER_RESPONSE = 0b1001
    ERROR = 0b1111


class MsgFlag(IntEnum):
    NO_SEQ = 0b0000
    POSITIVE_SEQ = 0b0001
    LAST_NO_SEQ = 0b0010
    LAST_NEGATIVE_SEQ = 0b0011


class Serialization(IntEnum):
    RAW = 0b0000
    JSON = 0b0001


class Compression(IntEnum):
    NONE = 0b0000
    GZIP = 0b0001


@dataclass(frozen=True)
class ASRMessage:
    msg_type: MsgType
    flag: MsgFlag
    serialization: Serialization
    compression: Compression
    sequence: int | None
    payload: bytes = b""
    error_code: int | None = None

    @property
    def payload_text(self) -> str:
        if not self.payload:
            return ""
        data = self.payload
        if self.compression == Compression.GZIP:
            data = gzip.decompress(data)
        return data.decode("utf-8", errors="replace")

    @property
    def payload_json(self) -> dict[str, Any]:
        text = self.payload_text
        if not text:
            return {}
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
        return {}

    @classmethod
    def from_bytes(cls, data: bytes) -> ASRMessage:
        if len(data) < 4:
            raise ValueError("ASR websocket frame is too short")
        version = data[0] >> 4
        header_size = data[0] & 0x0F
        if version != 0x1 or header_size != 0x1:
            raise ValueError(
                f"unsupported ASR header: version={version} header_size={header_size}"
            )
        msg_type = MsgType(data[1] >> 4)
        flag = MsgFlag(data[1] & 0x0F)
        serialization = Serialization(data[2] >> 4)
        compression = Compression(data[2] & 0x0F)
        offset = 4
        sequence: int | None = None
        error_code: int | None = None
        if msg_type == MsgType.ERROR:
            if len(data) < offset + 8:
                raise ValueError("ASR error frame is too short")
            error_code = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4
            size = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4
            payload = data[offset : offset + size]
            return cls(
                msg_type=msg_type,
                flag=flag,
                serialization=serialization,
                compression=compression,
                sequence=sequence,
                payload=payload,
                error_code=error_code,
            )
        if flag in {MsgFlag.POSITIVE_SEQ, MsgFlag.LAST_NEGATIVE_SEQ}:
            sequence = struct.unpack(">i", data[offset : offset + 4])[0]
            offset += 4
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        payload = data[offset : offset + size]
        return cls(
            msg_type=msg_type,
            flag=flag,
            serialization=serialization,
            compression=compression,
            sequence=sequence,
            payload=payload,
        )


class DoubaoASRClient:
    """Volcengine/Doubao WebSocket ASR client."""

    def __init__(self, config: ASRConfig) -> None:
        self.config = config

    async def transcribe_wav(
        self, wav_bytes: bytes, *, _filename: str = "utterance.wav"
    ) -> str:
        if not wav_bytes:
            return ""
        api_key = self.config.resolved_api_key
        access_key = self.config.resolved_access_key
        app_key = self.config.resolved_app_key
        resource_id = self.config.resolved_resource_id
        if not resource_id:
            raise ValueError(
                f"ASR resource id is not configured; set {self.config.resource_id_env} "
                f"or channels.voice.asr.resource_id"
            )
        if access_key:
            if not app_key:
                raise ValueError(
                    f"ASR app key is not configured; set {self.config.app_key_env} or channels.voice.asr.app_key"
                )
        elif not api_key:
            raise ValueError(
                f"ASR API key is not configured; set {self.config.api_key_env} or channels.voice.asr.api_key"
            )

        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                "Doubao ASR requires `websockets`. Install `hey-robot[agent]`."
            ) from exc

        request_id = str(uuid.uuid4())
        headers = _build_headers(
            api_key=api_key,
            access_key=access_key,
            app_key=app_key,
            resource_id=resource_id,
            request_id=request_id,
        )
        audio_chunks = list(
            _wav_bytes_to_pcm_chunks(
                wav_bytes,
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                bits=self.config.bits,
                chunk_ms=self.config.chunk_ms,
            )
        )
        if not audio_chunks:
            return ""

        async with websockets.connect(
            self.config.endpoint,
            additional_headers=headers,
            max_size=10 * 1024 * 1024,
        ) as socket:
            await socket.send(_build_full_request(self.config, request_id))
            latest_text = ""
            next_sequence = 2
            for index, chunk in enumerate(audio_chunks, start=1):
                last = index == len(audio_chunks)
                flag = MsgFlag.LAST_NEGATIVE_SEQ if last else MsgFlag.POSITIVE_SEQ
                sequence = -next_sequence if last else next_sequence
                await socket.send(
                    _build_audio_request(chunk, sequence=sequence, flag=flag)
                )
                next_sequence += 1
                latest_text = await _consume_available_text(socket, latest_text)

            while True:
                try:
                    message = ASRMessage.from_bytes(
                        cast(
                            bytes,
                            await asyncio.wait_for(
                                socket.recv(), timeout=self.config.timeout_sec
                            ),
                        )
                    )
                except TimeoutError:
                    break
                if message.msg_type == MsgType.ERROR:
                    raise RuntimeError(_format_error(message))
                if message.msg_type != MsgType.FULL_SERVER_RESPONSE:
                    continue
                text = _extract_text(message.payload_json)
                if text:
                    latest_text = text
                if message.flag == MsgFlag.LAST_NEGATIVE_SEQ:
                    break

            return latest_text.strip()


class SherpaONNXASRClient:
    """Local sherpa-onnx offline transcription using transducer models."""

    def __init__(self, config: ASRConfig) -> None:
        self.config = config
        self._recognizer: Any | None = None

    async def transcribe_wav(
        self, wav_bytes: bytes, *, _filename: str = "utterance.wav"
    ) -> str:
        if not wav_bytes:
            return ""
        return await asyncio.to_thread(self._transcribe_blocking, wav_bytes)

    def _transcribe_blocking(self, wav_bytes: bytes) -> str:
        recognizer = self._recognizer or self._build_recognizer()
        self._recognizer = recognizer
        samples = _wav_bytes_to_float32_samples(
            wav_bytes,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            bits=self.config.bits,
        )
        stream = recognizer.create_stream()
        stream.accept_waveform(self.config.sample_rate, samples)
        if hasattr(stream, "input_finished"):
            stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        result = recognizer.get_result(stream)
        if isinstance(result, str):
            return result.strip()
        text = getattr(result, "text", "")
        return str(text).strip()

    def _build_recognizer(self) -> Any:
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ImportError(
                "Sherpa ASR requires `sherpa-onnx`. Install it in the runtime environment."
            ) from exc

        model_dir = self.config.resolved_sherpa_model_dir
        if not model_dir:
            raise ValueError(
                f"Sherpa model dir is not configured; set {self.config.sherpa_model_dir_env} "
                "or channels.voice.asr.sherpa_model_dir"
            )
        model_root = Path(model_dir)
        tokens = model_root / self.config.sherpa_tokens
        encoder = model_root / self.config.sherpa_encoder
        decoder = model_root / self.config.sherpa_decoder
        joiner = model_root / self.config.sherpa_joiner
        missing = [
            str(path)
            for path in (tokens, encoder, decoder, joiner)
            if not path.exists()
        ]
        if missing:
            raise ValueError("Sherpa model files are missing: " + ", ".join(missing))
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            num_threads=max(1, int(self.config.sherpa_num_threads)),
            provider=self.config.sherpa_provider or "cpu",
            sample_rate=self.config.sample_rate,
            feature_dim=max(1, int(self.config.sherpa_feature_dim)),
        )


def build_asr_client(config: ASRConfig) -> ASRClient:
    return _build_single_asr_client(config.provider, config)


def _build_single_asr_client(provider: str, config: ASRConfig) -> ASRClient:
    normalized = provider.strip().lower()
    if normalized == "doubao":
        return DoubaoASRClient(config)
    if normalized in {"sherpa", "sherpa_onnx", "sherpa-onnx"}:
        return SherpaONNXASRClient(config)
    raise ValueError(f"unsupported ASR provider: {provider!r}")


def _build_headers(
    *,
    api_key: str,
    access_key: str,
    app_key: str,
    resource_id: str,
    request_id: str,
) -> dict[str, str]:
    headers: dict[str, str] = {
        "X-Api-Resource-Id": resource_id,
        "X-Api-Request-Id": request_id,
        "X-Api-Sequence": "-1",
    }
    if access_key:
        headers["X-Api-App-Key"] = app_key or api_key
        headers["X-Api-Access-Key"] = access_key
    else:
        headers["X-Api-Key"] = api_key
    return headers


def _build_full_request(config: ASRConfig, request_id: str) -> bytes:
    payload: dict[str, Any] = {
        "user": {"uid": request_id},
        "audio": {
            "format": "pcm",
            "codec": "raw",
            "rate": config.sample_rate,
            "bits": config.bits,
            "channel": config.channels,
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": config.enable_itn,
            "enable_punc": config.enable_punc,
            "enable_ddc": config.enable_ddc,
        },
    }
    if config.language:
        payload["audio"]["language"] = config.language
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    compressed = gzip.compress(body)
    return _pack_frame(
        msg_type=MsgType.FULL_CLIENT_REQUEST,
        flag=MsgFlag.POSITIVE_SEQ,
        serialization=Serialization.JSON,
        compression=Compression.GZIP,
        sequence=1,
        payload=compressed,
    )


def _build_audio_request(payload: bytes, *, sequence: int, flag: MsgFlag) -> bytes:
    return _pack_frame(
        msg_type=MsgType.AUDIO_ONLY_REQUEST,
        flag=flag,
        serialization=Serialization.RAW,
        compression=Compression.NONE,
        sequence=sequence,
        payload=payload,
    )


def _pack_frame(
    *,
    msg_type: MsgType,
    flag: MsgFlag,
    serialization: Serialization,
    compression: Compression,
    payload: bytes,
    sequence: int | None = None,
) -> bytes:
    buffer = io.BytesIO()
    buffer.write(
        bytes(
            [_VERSION, (msg_type << 4) | flag, (serialization << 4) | compression, 0x00]
        )
    )
    if sequence is not None:
        buffer.write(struct.pack(">i", int(sequence)))
    buffer.write(struct.pack(">I", len(payload)))
    buffer.write(payload)
    return buffer.getvalue()


async def _consume_available_text(socket: Any, latest_text: str) -> str:
    while True:
        try:
            data = await asyncio.wait_for(socket.recv(), timeout=0.01)
        except TimeoutError:
            return latest_text
        message = ASRMessage.from_bytes(cast(bytes, data))
        if message.msg_type == MsgType.ERROR:
            raise RuntimeError(_format_error(message))
        if message.msg_type != MsgType.FULL_SERVER_RESPONSE:
            continue
        text = _extract_text(message.payload_json)
        if text:
            latest_text = text
        if message.flag == MsgFlag.LAST_NEGATIVE_SEQ:
            return latest_text


def _extract_text(payload: dict[str, Any]) -> str:
    result = payload.get("result")
    if isinstance(result, dict):
        text = result.get("text")
        if isinstance(text, str):
            return text.strip()
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str):
                return text.strip()
    text = payload.get("text")
    if isinstance(text, str):
        return text.strip()
    return ""


def _format_error(message: ASRMessage) -> str:
    raw = message.payload.decode("utf-8", errors="replace")
    if message.error_code is None:
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return f"ASR error code={message.error_code} payload={parsed}"


def _wav_bytes_to_pcm_chunks(
    wav_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
    bits: int,
    chunk_ms: int,
) -> list[bytes]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        wav_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        pcm_bytes = wav_file.readframes(wav_file.getnframes())

    if frame_rate != sample_rate:
        raise ValueError(
            f"ASR wav sample rate mismatch: expected {sample_rate}, got {frame_rate}"
        )
    if wav_channels != channels:
        raise ValueError(
            f"ASR wav channel mismatch: expected {channels}, got {wav_channels}"
        )
    if bits != 16:
        raise ValueError("ASR client only supports 16-bit PCM input")
    if sample_width != 2:
        raise ValueError(
            f"ASR wav sample width mismatch: expected 2, got {sample_width}"
        )

    frame_size = max(1, int(sample_rate * chunk_ms / 1000)) * channels * (bits // 8)
    return [
        pcm_bytes[index : index + frame_size]
        for index in range(0, len(pcm_bytes), frame_size)
        if pcm_bytes[index : index + frame_size]
    ]


def _wav_bytes_to_float32_samples(
    wav_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
    bits: int,
) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        wav_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        pcm_bytes = wav_file.readframes(wav_file.getnframes())

    if frame_rate != sample_rate:
        raise ValueError(
            f"ASR wav sample rate mismatch: expected {sample_rate}, got {frame_rate}"
        )
    if wav_channels != channels:
        raise ValueError(
            f"ASR wav channel mismatch: expected {channels}, got {wav_channels}"
        )
    if bits != 16:
        raise ValueError("Sherpa ASR only supports 16-bit PCM input")
    if sample_width != 2:
        raise ValueError(
            f"ASR wav sample width mismatch: expected 2, got {sample_width}"
        )

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples
