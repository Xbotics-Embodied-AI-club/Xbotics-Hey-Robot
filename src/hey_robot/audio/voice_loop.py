from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path

from hey_robot.audio.asr import build_asr_client
from hey_robot.audio.config import VoiceAudioConfig
from hey_robot.audio.io import AudioPlayer, AudioRecorder, pcm16_to_wav_bytes
from hey_robot.audio.session import VoiceRouteDecision, VoiceSessionRouter
from hey_robot.audio.streaming_asr import SherpaStreamingVoiceEngine
from hey_robot.audio.tts import DoubaoTTSClient
from hey_robot.logging import HeyRobotLogger

logger = HeyRobotLogger(name="voice")

VoiceTextHandler = Callable[[str, dict], Awaitable[None]]


class VoiceInteractionLoop:
    """Local microphone -> ASR -> text handler, plus optional reply TTS."""

    def __init__(self, config: VoiceAudioConfig) -> None:
        self.config = config
        self.recorder = AudioRecorder(config.recorder)
        self.asr = build_asr_client(config.asr)
        self.tts = DoubaoTTSClient(config.tts)
        self.router = VoiceSessionRouter(config.activation)
        self.player = AudioPlayer(
            sample_rate=config.tts.sample_rate, output_device=config.tts.output_device
        )
        self.streaming = SherpaStreamingVoiceEngine(config.recorder, config.asr)
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()
        self._speak_lock = asyncio.Lock()
        self._last_tts_finished_at = 0.0
        self._tts_active = False
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5
        self._capture_safe = asyncio.Event()
        self._capture_safe.set()
        self._echo_guard_task: asyncio.Task | None = None

    async def start(self, handler: VoiceTextHandler) -> None:
        self._stopped.clear()
        self._task = asyncio.create_task(self._run(handler), name="voice-interaction")

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def speak(self, text: str) -> None:
        if not self.config.reply_tts or not self.config.tts.enabled or not text.strip():
            return
        if self._echo_guard_task and not self._echo_guard_task.done():
            self._echo_guard_task.cancel()
        self._capture_safe.clear()
        async with self._speak_lock:
            self._tts_active = True
            try:
                pcm_parts: list[bytes] = []
                with self.player.open_pcm16_stream() as player:
                    async for chunk in self.tts.synthesize_stream(text):
                        if not chunk:
                            continue
                        pcm_parts.append(chunk)
                        player.enqueue(chunk)
                pcm = b"".join(pcm_parts)
                self._dump_tts_audio(text, pcm)
                logger.info(
                    f"TTS 播放完成 chars={len(text)} bytes={len(pcm)} output_device={self.config.tts.output_device!r}"
                )
            except Exception as exc:
                logger.warning(f"TTS 失败: {type(exc).__name__}: {exc}")
            finally:
                self._tts_active = False
                self._last_tts_finished_at = time.monotonic()
                guard = self.config.tts_echo_guard_sec
                if guard > 0:
                    self._echo_guard_task = asyncio.create_task(
                        self._delayed_set_capture_safe(guard)
                    )
                else:
                    self._capture_safe.set()

    async def _run(self, handler: VoiceTextHandler) -> None:
        if self.config.scripted_texts:
            await self._run_scripted(handler)
            return
        logger.info(
            "voice 交互循环启动 "
            f"sample_rate={self.config.recorder.sample_rate} "
            f"input_device={self.config.recorder.input_device!r} "
            f"activation={self.config.activation.enabled} "
            f"wake_words={self.config.activation.wake_words} "
            f"session_timeout={self.config.activation.session_timeout_sec:.1f}s"
        )
        while not self._stopped.is_set():
            try:
                await self._wait_until_capture_safe()
                if self._stopped.is_set():
                    return
                utterance = await self._capture_utterance()
                if utterance is None:
                    self._consecutive_errors = 0
                    continue
                self._consecutive_errors = 0
                if self._tts_blocks_capture():
                    continue
                logger.info(
                    "语音采集完成 "
                    f"duration={utterance.duration_sec:.2f}s peak={utterance.peak:.3f} rms={utterance.rms:.3f}"
                )
                raw_text = (
                    utterance.raw_text
                    if hasattr(utterance, "raw_text")
                    else await self.asr.transcribe_wav(utterance.wav_bytes)
                )
                logger.info(f"ASR 原始文本 text={raw_text!r}")
                decision = self.router.route(raw_text)
                if not decision.accepted:
                    logger.info(
                        "ASR 丢弃 "
                        f"reason={decision.reason} session_active={decision.session_active} "
                        f"text={decision.text!r}"
                    )
                    continue
                text = decision.text
                if getattr(utterance, "wav_bytes", b""):
                    self._dump_asr_audio(utterance.wav_bytes, raw_text)
                logger.info(
                    "ASR 路由 "
                    f"reason={decision.reason} session_active={decision.session_active} "
                    f"text_len={len(text)} text={text!r}"
                )
                await handler(
                    text,
                    {
                        **self.config.metadata,
                        "voice": _route_metadata(decision),
                        "audio": {
                            "duration_sec": utterance.duration_sec,
                            "sample_rate": utterance.sample_rate,
                            "channels": utterance.channels,
                            "peak": utterance.peak,
                            "rms": utterance.rms,
                        },
                    },
                )
            except asyncio.CancelledError:
                raise
            except (ImportError, ValueError) as exc:
                logger.error(f"voice loop 启动失败: {type(exc).__name__}: {exc}")
                return
            except Exception as exc:
                self._consecutive_errors += 1
                if self._consecutive_errors >= self._max_consecutive_errors:
                    logger.error(
                        f"voice loop 连续失败 {self._consecutive_errors} 次，自动禁用 voice 通道: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    return
                delay = min(2.0, 0.25 * self._consecutive_errors)
                logger.error(
                    f"voice loop 错误 ({self._consecutive_errors}/{self._max_consecutive_errors}): "
                    f"{type(exc).__name__}: {exc}"
                )
                await asyncio.sleep(delay)

    async def _run_scripted(self, handler: VoiceTextHandler) -> None:
        logger.info(
            "voice script 循环启动 "
            f"count={len(self.config.scripted_texts)} "
            f"wake_words={self.config.activation.wake_words}"
        )
        if self.config.scripted_start_delay_sec > 0:
            try:
                await asyncio.wait_for(
                    self._stopped.wait(), timeout=self.config.scripted_start_delay_sec
                )
                return
            except TimeoutError:
                pass
        while not self._stopped.is_set():
            for raw_text in self.config.scripted_texts:
                if self._stopped.is_set():
                    return
                decision = self.router.route(raw_text)
                if decision.accepted:
                    await handler(
                        decision.text,
                        {
                            **self.config.metadata,
                            "voice": _route_metadata(decision),
                            "audio": {
                                "source": "scripted",
                                "sample_rate": self.config.recorder.sample_rate,
                                "channels": self.config.recorder.channels,
                            },
                        },
                    )
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._stopped.wait(),
                        timeout=max(0.0, self.config.scripted_interval_sec),
                    )
            if not self.config.scripted_repeat:
                await self._stopped.wait()
                return

    def _in_tts_echo_guard(self) -> bool:
        guard = max(0.0, self.config.tts_echo_guard_sec)
        if guard <= 0:
            return False
        return (time.monotonic() - self._last_tts_finished_at) < guard

    def _tts_blocks_capture(self) -> bool:
        return self._tts_active or self._in_tts_echo_guard()

    async def _delayed_set_capture_safe(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        self._capture_safe.set()

    async def _wait_until_capture_safe(self) -> None:
        if self._capture_safe.is_set():
            return
        safe_wait = asyncio.create_task(self._capture_safe.wait())
        stop_wait = asyncio.create_task(self._stopped.wait())
        try:
            await asyncio.wait(
                [safe_wait, stop_wait],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for t in (safe_wait, stop_wait):
                if not t.done():
                    t.cancel()

    async def _capture_utterance(self):
        if self.config.asr.provider == "sherpa_onnx":
            return await self.streaming.listen_once()
        return await asyncio.to_thread(self.recorder.record_utterance)

    def _dump_tts_audio(self, text: str, pcm: bytes) -> None:
        if not pcm:
            return
        directory = self._runtime_debug_dir("voice/tts")
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        wav_path = directory / f"tts_{timestamp}.wav"
        txt_path = directory / f"tts_{timestamp}.txt"
        wav_path.write_bytes(
            pcm16_to_wav_bytes(pcm, sample_rate=self.config.tts.sample_rate)
        )
        txt_path.write_text(text, encoding="utf-8")

    def _dump_asr_audio(self, wav_bytes: bytes, text: str) -> None:
        if not wav_bytes:
            return
        directory = self._runtime_debug_dir("voice/asr")
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        wav_path = directory / f"asr_{timestamp}.wav"
        txt_path = directory / f"asr_{timestamp}.txt"
        wav_path.write_bytes(wav_bytes)
        txt_path.write_text(text, encoding="utf-8")

    def _runtime_debug_dir(self, name: str) -> Path:
        runtime_dir = os.environ.get("HEY_ROBOT_RUNTIME_DIR", "").strip()
        base_dir = (
            Path(runtime_dir) if runtime_dir else Path.cwd() / "outputs" / "diagnostic"
        )
        return base_dir / name


def _route_metadata(decision: VoiceRouteDecision) -> dict[str, object]:
    return {
        "route_reason": decision.reason,
        "session_active": decision.session_active,
        "wake_word": decision.wake_word or None,
    }
