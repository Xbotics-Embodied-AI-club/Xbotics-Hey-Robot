from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hey_robot.audio.config import ASRConfig, RecorderConfig


@dataclass(frozen=True)
class StreamingASRTurn:
    raw_text: str
    duration_sec: float
    sample_rate: int
    channels: int
    peak: float
    rms: float
    wav_bytes: bytes = b""


class SherpaStreamingVoiceEngine:
    """HomeBot-style local wakeup + streaming ASR adapted to hey-robot."""

    def __init__(self, recorder: RecorderConfig, asr: ASRConfig) -> None:
        self.recorder = recorder
        self.asr = asr
        self._recognizer: Any | None = None
        self._initialized = False
        self._block_size = max(
            1, int(self.recorder.sample_rate * self.recorder.block_ms / 1000)
        )
        self._samples_per_read = max(1, int(0.1 * self.recorder.sample_rate))

    async def listen_once(self) -> StreamingASRTurn | None:
        return await asyncio.to_thread(self._listen_once_blocking)

    def _listen_once_blocking(self) -> StreamingASRTurn | None:
        self._ensure_initialized()
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise ImportError(
                "Streaming voice input requires `sounddevice`. Install `hey-robot[agent]`."
            ) from exc

        with sd.InputStream(
            device=self.recorder.input_device,
            channels=self.recorder.channels,
            samplerate=self.recorder.sample_rate,
            dtype="float32",
            blocksize=self._block_size,
        ) as stream:
            return self._recognize_once(stream)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ImportError(
                "Streaming sherpa voice engine requires `sherpa-onnx`."
            ) from exc

        model_dir = self._require_dir(
            self.asr.resolved_sherpa_model_dir,
            self.asr.sherpa_model_dir_env,
        )
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(model_dir / self.asr.sherpa_tokens),
            encoder=str(model_dir / self.asr.sherpa_encoder),
            decoder=str(model_dir / self.asr.sherpa_decoder),
            joiner=str(model_dir / self.asr.sherpa_joiner),
            num_threads=max(1, int(self.asr.sherpa_num_threads)),
            provider=self.asr.sherpa_provider or "cpu",
            sample_rate=self.recorder.sample_rate,
            feature_dim=max(1, int(self.asr.sherpa_feature_dim)),
        )
        self._initialized = True

    def _require_dir(self, value: str, env_name: str) -> Path:
        directory = Path(value.strip()) if value.strip() else None
        if directory is None:
            raise ValueError(f"voice model dir is not configured; set {env_name}")
        if not directory.exists():
            raise ValueError(f"voice model dir does not exist: {directory}")
        return directory

    def _recognize_once(self, stream: Any) -> StreamingASRTurn | None:
        assert self._recognizer is not None
        asr_stream = self._recognizer.create_stream()
        current_text = ""
        last_speech_time = time.time()
        chunks: list[np.ndarray] = []
        while True:
            samples, _overflow = stream.read(self._samples_per_read)
            frame = np.asarray(samples, dtype=np.float32).reshape(-1)
            chunks.append(frame.copy())
            asr_stream.accept_waveform(self.recorder.sample_rate, frame)
            while self._recognizer.is_ready(asr_stream):
                self._recognizer.decode_stream(asr_stream)
            result = self._recognizer.get_result(asr_stream)
            text = (
                result if isinstance(result, str) else str(getattr(result, "text", ""))
            )
            text = text.strip()
            now = time.time()
            if text and text != current_text:
                current_text = text
                last_speech_time = now
            if current_text and (now - last_speech_time) > max(
                0.2, self.asr.listen_timeout_sec
            ):
                break
            if len(chunks) >= int(max(1.0, self.recorder.max_speech_sec) * 10):
                break
        audio = (
            np.concatenate(chunks, axis=0) if chunks else np.zeros(0, dtype=np.float32)
        )
        if audio.size == 0 or not current_text:
            return None
        return StreamingASRTurn(
            raw_text=current_text,
            duration_sec=float(audio.shape[0]) / float(self.recorder.sample_rate),
            sample_rate=self.recorder.sample_rate,
            channels=self.recorder.channels,
            peak=float(np.max(np.abs(audio))) if audio.size else 0.0,
            rms=float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0,
        )
