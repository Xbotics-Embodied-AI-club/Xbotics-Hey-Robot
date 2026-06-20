from __future__ import annotations

import io
import queue
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass

import numpy as np

from hey_robot.audio.config import RecorderConfig


@dataclass(frozen=True)
class RecordedUtterance:
    wav_bytes: bytes
    duration_sec: float
    sample_rate: int
    channels: int
    peak: float
    rms: float


class AudioRecorder:
    """Blocking microphone recorder with simple energy VAD."""

    def __init__(self, config: RecorderConfig) -> None:
        self.config = config

    def record_utterance(self) -> RecordedUtterance | None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise ImportError(
                "Voice audio input requires `sounddevice`. Install `hey-robot[agent]`."
            ) from exc

        block_size = max(1, int(self.config.sample_rate * self.config.block_ms / 1000))
        max_blocks = max(
            1, int(self.config.max_speech_sec * self.config.sample_rate / block_size)
        )
        silence_blocks = max(
            1, int(self.config.silence_sec * self.config.sample_rate / block_size)
        )
        pre_roll_blocks = max(
            0, int(self.config.pre_roll_sec * self.config.sample_rate / block_size)
        )
        min_blocks = max(
            1, int(self.config.min_speech_sec * self.config.sample_rate / block_size)
        )
        pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_blocks)
        chunks: list[np.ndarray] = []
        active = False
        quiet = 0
        with sd.InputStream(
            device=self.config.input_device,
            channels=self.config.channels,
            samplerate=self.config.sample_rate,
            dtype=self.config.dtype,
            blocksize=block_size,
        ) as stream:
            while True:
                block, _overflow = stream.read(block_size)
                frame = np.asarray(block, dtype=np.float32).copy()
                energy = (
                    float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
                )
                if not active:
                    pre_roll.append(frame)
                    if energy >= self.config.energy_threshold:
                        active = True
                        chunks.extend(list(pre_roll))
                        chunks.append(frame)
                    else:
                        time.sleep(self.config.idle_sleep_sec)
                    continue

                chunks.append(frame)
                quiet = quiet + 1 if energy < self.config.energy_threshold else 0
                if len(chunks) >= max_blocks or (
                    len(chunks) >= min_blocks and quiet >= silence_blocks
                ):
                    break

        if not chunks:
            return None
        audio = np.concatenate(chunks, axis=0)
        duration = (
            float(audio.shape[0]) / float(self.config.sample_rate)
            if audio.size
            else 0.0
        )
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        if len(chunks) < min_blocks:
            return None
        return RecordedUtterance(
            wav_bytes=float32_to_wav_bytes(audio, sample_rate=self.config.sample_rate),
            duration_sec=duration,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            peak=peak,
            rms=rms,
        )


class AudioPlayer:
    def __init__(
        self, *, sample_rate: int = 16000, output_device: int | str | None = None
    ) -> None:
        self.sample_rate = sample_rate
        self.output_device = output_device

    def open_pcm16_stream(self) -> _PCM16StreamPlayer:
        return _PCM16StreamPlayer(
            sample_rate=self.sample_rate, output_device=self.output_device
        )

    def play_pcm16(self, pcm: bytes) -> None:
        if not pcm:
            return
        with self.open_pcm16_stream() as stream:
            stream.enqueue(pcm)


class _PCM16StreamPlayer:
    def __init__(self, *, sample_rate: int, output_device: int | str | None) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise ImportError(
                "Voice audio output requires `sounddevice`. Install `hey-robot[agent]`."
            ) from exc
        self._sd = sd
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=256)
        self._stream = sd.RawOutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=0,
            device=output_device,
        )
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="hey-robot-pcm-player", daemon=True
        )

    def __enter__(self) -> _PCM16StreamPlayer:
        self._stream.start()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(drain=True)

    def _run(self) -> None:
        while True:
            chunk = self._queue.get()
            try:
                if chunk is None:
                    return
                self._stream.write(chunk)
            finally:
                self._queue.task_done()

    def enqueue(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        self._queue.put(pcm)

    def close(self, *, drain: bool) -> None:
        if self._closed:
            return
        self._closed = True
        if drain:
            self._queue.join()
        self._queue.put(None)
        self._thread.join(timeout=5)
        self._stream.stop()
        self._stream.close()


def float32_to_wav_bytes(audio: np.ndarray, *, sample_rate: int) -> bytes:
    clipped = np.clip(audio.reshape(-1), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def pcm16_to_wav_bytes(pcm: bytes, *, sample_rate: int, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()
