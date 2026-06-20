from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class ASRConfig:
    provider: str = "doubao"
    api_key: str = ""
    api_key_env: str = "ARK_API_KEY"
    access_key: str = ""
    access_key_env: str = "ARK_ACCESS_KEY"
    app_key: str = ""
    app_key_env: str = "ARK_APP_KEY"
    endpoint: str = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_nostream"
    resource_id: str = "volc.seedasr.sauc.duration"
    resource_id_env: str = "DOUBAO_ASR_MODEL"
    model: str = ""
    model_env: str = "DOUBAO_ASR_MODEL"
    language: str | None = "zh-CN"
    sample_rate: int = 16000
    bits: int = 16
    channels: int = 1
    chunk_ms: int = 200
    timeout_sec: float = 60.0
    enable_itn: bool = True
    enable_punc: bool = True
    enable_ddc: bool = False
    sherpa_provider: str = "cpu"
    sherpa_model_dir: str = ""
    sherpa_model_dir_env: str = "SHERPA_ONNX_MODEL_DIR"
    sherpa_tokens: str = "tokens.txt"
    sherpa_encoder: str = "encoder.int8.onnx"
    sherpa_decoder: str = "decoder.onnx"
    sherpa_joiner: str = "joiner.int8.onnx"
    sherpa_num_threads: int = 1
    sherpa_feature_dim: int = 80
    listen_timeout_sec: float = 1.2

    @property
    def resolved_api_key(self) -> str:
        return self.api_key or os.environ.get(self.api_key_env, "")

    @property
    def resolved_access_key(self) -> str:
        return self.access_key or os.environ.get(self.access_key_env, "")

    @property
    def resolved_app_key(self) -> str:
        return self.app_key or os.environ.get(self.app_key_env, "")

    @property
    def resolved_resource_id(self) -> str:
        return (
            self.model
            or os.environ.get(self.resource_id_env, "")
            or os.environ.get(self.model_env, "")
            or self.resource_id
        )

    @property
    def resolved_model(self) -> str:
        return self.resolved_resource_id

    @property
    def resolved_sherpa_model_dir(self) -> str:
        return self.sherpa_model_dir or os.environ.get(self.sherpa_model_dir_env, "")


@dataclass(frozen=True)
class TTSConfig:
    provider: str = "doubao"
    enabled: bool = True
    api_key: str = ""
    api_key_env: str = "ARK_API_KEY"
    resource_id: str = "seed-tts-2.0"
    voice_type: str = "zh_female_vv_uranus_bigtts"
    endpoint: str = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
    encoding: str = "pcm"
    sample_rate: int = 16000
    output_device: int | str | None = None
    timeout_sec: float = 30.0

    @property
    def resolved_api_key(self) -> str:
        return self.api_key or os.environ.get(self.api_key_env, "")


@dataclass(frozen=True)
class RecorderConfig:
    input_device: int | str | None = None
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    block_ms: int = 100
    energy_threshold: float = 0.015
    silence_sec: float = 0.8
    pre_roll_sec: float = 0.3
    min_speech_sec: float = 0.4
    max_speech_sec: float = 8.0
    idle_sleep_sec: float = 0.05


@dataclass(frozen=True)
class VoiceActivationConfig:
    enabled: bool = True
    wake_words: list[str] = field(default_factory=list)
    strip_wake_word: bool = True
    session_timeout_sec: float = 60.0
    min_route_chars: int = 2


@dataclass(frozen=True)
class VoiceAudioConfig:
    sender_id: str = "voice-user"
    chat_id: str = "xlerobot-voice"
    tts_echo_guard_sec: float = 0.6
    reply_tts: bool = True
    scripted_texts: list[str] = field(default_factory=list)
    scripted_start_delay_sec: float = 0.0
    scripted_interval_sec: float = 0.1
    scripted_repeat: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    activation: VoiceActivationConfig = field(default_factory=VoiceActivationConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)


def voice_config_from_settings(settings: dict[str, Any]) -> VoiceAudioConfig:
    activation_data = dict(settings.get("activation", {}) or {})
    recorder_data = dict(settings.get("recorder", {}) or {})
    asr_data = dict(settings.get("asr", {}) or {})
    tts_data = dict(settings.get("tts", {}) or {})
    _apply_alias(tts_data, source="access_token_env", target="api_key_env")
    _apply_alias(tts_data, source="appid_env", target="api_key_env")
    return VoiceAudioConfig(
        sender_id=str(settings.get("sender_id", "voice-user")),
        chat_id=str(settings.get("chat_id", "xlerobot-voice")),
        tts_echo_guard_sec=float(settings.get("tts_echo_guard_sec", 0.6)),
        reply_tts=bool(settings.get("reply_tts", True)),
        scripted_texts=[str(item) for item in settings.get("scripted_texts", []) or []],
        scripted_start_delay_sec=float(settings.get("scripted_start_delay_sec", 0.0)),
        scripted_interval_sec=float(settings.get("scripted_interval_sec", 0.1)),
        scripted_repeat=bool(settings.get("scripted_repeat", False)),
        metadata=dict(settings.get("metadata", {}) or {}),
        activation=VoiceActivationConfig(
            **_known_fields(VoiceActivationConfig, activation_data)
        ),
        recorder=RecorderConfig(**_known_fields(RecorderConfig, recorder_data)),
        asr=ASRConfig(**_known_fields(ASRConfig, asr_data)),
        tts=TTSConfig(**_known_fields(TTSConfig, tts_data)),
    )


def _apply_alias(data: dict[str, Any], *, source: str, target: str) -> None:
    if source in data and target not in data:
        data[target] = data[source]


def _known_fields(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    return {key: value for key, value in data.items() if key in allowed}
