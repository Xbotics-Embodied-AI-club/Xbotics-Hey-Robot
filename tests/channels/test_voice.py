from __future__ import annotations

import asyncio

from hey_robot.audio.config import VoiceActivationConfig, voice_config_from_settings
from hey_robot.audio.io import RecordedUtterance
from hey_robot.audio.session import VoiceSessionRouter
from hey_robot.audio.streaming_asr import StreamingASRTurn
from hey_robot.audio.voice_loop import VoiceInteractionLoop
from hey_robot.channels import ChannelContext, VoiceChannel
from hey_robot.channels.voice import (
    _is_nonsense_asr,
    _is_specific_action_intent,
    _needs_voice_clarification,
)
from hey_robot.config import ChannelSpec, DeploymentConfig
from hey_robot.events import RuntimeEvent
from hey_robot.gateway import GatewayService
from hey_robot.protocol import AgentReply, Envelope


def test_voice_config_from_settings() -> None:
    config = voice_config_from_settings(
        {
            "sender_id": "u1",
            "chat_id": "c1",
            "activation": {
                "wake_words": ["hey", "嘿"],
                "strip_wake_word": False,
                "min_route_chars": 3,
                "session_timeout_sec": 9.0,
            },
            "tts_echo_guard_sec": 0.8,
            "recorder": {"input_device": 1, "energy_threshold": 0.02},
            "asr": {"model": "ep-asr"},
            "tts": {"enabled": False, "api_key_env": "VOICE_TOKEN"},
            "scripted_texts": ["hey 打开台灯"],
            "scripted_start_delay_sec": 0.02,
            "scripted_interval_sec": 0.01,
        }
    )

    assert config.sender_id == "u1"
    assert config.activation.wake_words == ["hey", "嘿"]
    assert config.activation.strip_wake_word is False
    assert config.activation.min_route_chars == 3
    assert config.activation.session_timeout_sec == 9.0
    assert config.tts_echo_guard_sec == 0.8
    assert config.recorder.input_device == 1
    assert config.recorder.energy_threshold == 0.02
    assert config.asr.model == "ep-asr"
    assert config.asr.resolved_model == "ep-asr"
    assert config.tts.enabled is False
    assert config.tts.api_key_env == "VOICE_TOKEN"
    assert config.scripted_texts == ["hey 打开台灯"]
    assert config.scripted_start_delay_sec == 0.02
    assert config.scripted_interval_sec == 0.01


def test_voice_config_keeps_activation_separate_from_asr() -> None:
    config = voice_config_from_settings(
        {
            "activation": {
                "wake_words": ["小白"],
                "strip_wake_word": True,
            },
            "asr": {
                "provider": "sherpa_onnx",
                "sherpa_model_dir": "models/sherpa",
            },
        }
    )

    assert config.activation.wake_words == ["小白"]
    assert config.asr.provider == "sherpa_onnx"
    assert config.asr.resolved_sherpa_model_dir == "models/sherpa"


def test_voice_session_routes_without_activation_when_no_wake_words() -> None:
    router = VoiceSessionRouter(VoiceActivationConfig(wake_words=[]))

    decision = router.route("你看到了什么")

    assert decision.accepted is True
    assert decision.text == "你看到了什么"
    assert decision.reason == "activation_disabled"


def test_voice_session_requires_wake_word_when_inactive() -> None:
    router = VoiceSessionRouter(VoiceActivationConfig(wake_words=["hey"]))

    decision = router.route("please say hey robot")

    assert decision.accepted is False
    assert decision.reason == "wake_word_required"


def test_voice_session_strips_wake_word_and_keeps_session_open() -> None:
    now = 100.0
    router = VoiceSessionRouter(
        VoiceActivationConfig(wake_words=["小白"], session_timeout_sec=12.0),
        clock=lambda: now,
    )

    wake_only = router.route("小白")
    command = router.route("你能看到什么？")

    assert wake_only.accepted is False
    assert wake_only.reason == "activated_without_command"
    assert command.accepted is True
    assert command.text == "你能看到什么？"
    assert command.reason == "active_session"


def test_voice_session_tolerates_repeated_first_wake_char() -> None:
    router = VoiceSessionRouter(VoiceActivationConfig(wake_words=["小白"]))

    decision = router.route("小小白，你好。")

    assert decision.accepted is True
    assert decision.text == "你好"
    assert decision.wake_word == "小白"


def test_voice_session_routes_emergency_stop_without_wake_word() -> None:
    router = VoiceSessionRouter(VoiceActivationConfig(wake_words=["小白"]))

    decision = router.route("停止")

    assert decision.accepted is True
    assert decision.text == "停止"
    assert decision.reason == "emergency_phrase"


def test_voice_session_routes_emergency_stop_with_punctuation() -> None:
    router = VoiceSessionRouter(VoiceActivationConfig(wake_words=["小白"]))

    decision = router.route("停止。")

    assert decision.accepted is True
    assert decision.text == "停止。"
    assert decision.reason == "emergency_phrase"


def test_voice_session_routes_emergency_stop_substring_match() -> None:
    """Emergency phrases should match anywhere in the text, not just exact."""
    router = VoiceSessionRouter(VoiceActivationConfig(wake_words=["小白"]))

    assert router.route("停止停止跟随").accepted is True
    assert router.route("快停下来").accepted is True
    assert router.route("别跟着我").accepted is True
    assert router.route("请停止移动").accepted is True


def test_voice_session_still_drops_non_emergency_substrings() -> None:
    """Normal speech containing partial matches should still be dropped."""
    router = VoiceSessionRouter(VoiceActivationConfig(wake_words=["小白"]))

    decision = router.route("今天天气不错")

    assert decision.accepted is False
    assert decision.reason == "wake_word_required"


def test_voice_session_emergency_stop_does_not_activate_session() -> None:
    router = VoiceSessionRouter(VoiceActivationConfig(wake_words=["小白"]))

    router.route("停止")
    decision = router.route("跟着我")

    assert decision.accepted is False
    assert decision.reason == "wake_word_required"


def test_voice_session_drops_non_emergency_without_wake_word() -> None:
    router = VoiceSessionRouter(VoiceActivationConfig(wake_words=["小白"]))

    decision = router.route("你好")

    assert decision.accepted is False
    assert decision.reason == "wake_word_required"


def test_xlerobot_voice_deployment_config_loads() -> None:
    config = DeploymentConfig.from_yaml("configs/xlerobot.real.windows.yaml")
    voice = voice_config_from_settings(config.channels["voice"].settings)

    assert voice.tts.api_key_env == "ARK_API_KEY"
    assert voice.asr.provider in {"doubao", "sherpa_onnx"}
    assert voice.activation.enabled is True
    assert voice.activation.wake_words
    assert "hey robot" not in voice.activation.wake_words
    assert "hey" not in voice.activation.wake_words
    assert "嘿" not in voice.activation.wake_words
    assert "hi" not in voice.activation.wake_words
    assert "hello" not in voice.activation.wake_words
    assert "嗨" not in voice.activation.wake_words
    assert "你" not in voice.activation.wake_words
    assert "你好" not in voice.activation.wake_words
    assert voice.activation.strip_wake_word is True
    assert voice.activation.min_route_chars >= 2
    assert voice.tts_echo_guard_sec >= 0.0
    assert voice.recorder.input_device is not None
    assert voice.tts.output_device is not None


def test_xlerobot_sim_windows_voice_has_local_sherpa_asr_model_dir() -> None:
    config = DeploymentConfig.from_yaml("configs/xlerobot.sim.windows.yaml")
    voice = voice_config_from_settings(config.channels["voice"].settings)

    assert voice.asr.provider == "sherpa_onnx"
    assert voice.activation.wake_words == ["小白", "机器人", "robot"]
    assert voice.asr.resolved_sherpa_model_dir == "models/asr"


def test_voice_interaction_loop_scripted_text_routes_without_audio() -> None:
    config = voice_config_from_settings(
        {
            "activation": {"wake_words": ["hey"], "strip_wake_word": False},
            "scripted_texts": ["hey 检查桌面"],
            "scripted_interval_sec": 0.01,
            "tts": {"enabled": False},
        }
    )
    loop = VoiceInteractionLoop(config)
    received: list[tuple[str, dict]] = []

    async def run() -> None:
        done = asyncio.Event()

        async def handler(text: str, metadata: dict) -> None:
            received.append((text, metadata))
            done.set()

        await loop.start(handler)
        await asyncio.wait_for(done.wait(), timeout=2.0)
        await loop.stop()

    asyncio.run(run())

    assert received[0][0] == "hey 检查桌面"
    assert received[0][1]["audio"]["source"] == "scripted"


def test_voice_interaction_loop_scripted_text_drops_short_routes() -> None:
    config = voice_config_from_settings(
        {
            "activation": {
                "wake_words": ["hey"],
                "strip_wake_word": False,
                "min_route_chars": 4,
            },
            "scripted_texts": ["hey"],
            "scripted_interval_sec": 0.01,
            "tts": {"enabled": False},
        }
    )
    loop = VoiceInteractionLoop(config)
    received: list[tuple[str, dict]] = []

    async def run() -> None:
        async def handler(text: str, metadata: dict) -> None:
            received.append((text, metadata))

        await loop.start(handler)
        await asyncio.sleep(0.05)
        await loop.stop()

    asyncio.run(run())

    assert received == []


def test_voice_interaction_loop_skips_audio_captured_during_active_tts() -> None:
    config = voice_config_from_settings(
        {
            "activation": {"wake_words": ["hey"]},
            "tts_echo_guard_sec": 0.0,
            "tts": {"enabled": False},
        }
    )
    loop = VoiceInteractionLoop(config)
    transcribed: list[bytes] = []
    received: list[str] = []

    class FakeRecorder:
        def record_utterance(self) -> RecordedUtterance:
            loop._tts_active = True
            return RecordedUtterance(
                wav_bytes=b"captured-tts",
                duration_sec=0.5,
                sample_rate=16000,
                channels=1,
                peak=0.5,
                rms=0.2,
            )

    class FakeASR:
        async def transcribe_wav(self, wav_bytes: bytes) -> str:
            transcribed.append(wav_bytes)
            return "hey echoed reply"

    loop.recorder = FakeRecorder()  # type: ignore[assignment]
    loop.asr = FakeASR()  # type: ignore[assignment]

    async def run() -> None:
        async def handler(text: str, _metadata: dict) -> None:
            received.append(text)

        await loop.start(handler)
        await asyncio.sleep(0.05)
        await loop.stop()

    asyncio.run(run())

    assert transcribed == []
    assert received == []


def test_voice_interaction_loop_capture_uses_native_streaming_turn_for_sherpa_provider() -> (
    None
):
    config = voice_config_from_settings(
        {
            "activation": {"wake_words": ["小白"], "strip_wake_word": True},
            "asr": {
                "provider": "sherpa_onnx",
                "sherpa_model_dir": "models/asr",
            },
            "tts": {"enabled": False},
        }
    )
    loop = VoiceInteractionLoop(config)

    class FakeStreaming:
        async def listen_once(self):
            return StreamingASRTurn(
                raw_text="小白 跟着我",
                duration_sec=0.8,
                sample_rate=16000,
                channels=1,
                peak=0.6,
                rms=0.2,
            )

    loop.streaming = FakeStreaming()  # type: ignore[assignment]
    utterance = asyncio.run(loop._capture_utterance())

    assert utterance is not None
    assert isinstance(utterance, StreamingASRTurn)
    assert utterance.raw_text == "小白 跟着我"
    assert utterance.channels == 1


def test_voice_streaming_turn_normalization_routes_human_follow_text() -> None:
    turn = StreamingASRTurn(
        raw_text="小白 跟着我",
        duration_sec=0.8,
        sample_rate=16000,
        channels=1,
        peak=0.6,
        rms=0.2,
    )
    router = VoiceSessionRouter(VoiceActivationConfig(wake_words=["小白"]))

    decision = router.route(turn.raw_text)

    assert decision.accepted is True
    assert decision.text == "跟着我"


def test_voice_channel_constructs_without_audio_devices() -> None:
    channel = VoiceChannel(
        ChannelContext(
            name="voice",
            deployment_id="test",
            spec=ChannelSpec(
                type="voice",
                settings={
                    "asr": {"model": "ep-asr"},
                    "tts": {"enabled": False},
                },
            ),
        )
    )

    assert channel.name == "voice"
    assert channel.config.chat_id == "xlerobot-voice"


def test_gateway_registers_voice_channel(tmp_path) -> None:
    config = DeploymentConfig.from_dict(
        {
            "resources": {"episodes": {"root": str(tmp_path / "episodes")}},
            "robots": {"xlerobot": {"type": "mock"}},
            "channels": {
                "voice": {
                    "type": "voice",
                    "enabled": True,
                    "asr": {"model": "ep-asr"},
                    "tts": {"enabled": False},
                }
            },
        }
    )
    gateway = GatewayService(config, episode_dir=tmp_path / "episodes")

    assert gateway.channels.get("voice") is not None


def test_voice_channel_filters_low_priority_notifications(monkeypatch) -> None:
    spoken: list[str] = []

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    channel = VoiceChannel(
        ChannelContext(
            name="voice",
            deployment_id="test",
            spec=ChannelSpec(
                type="voice",
                settings={
                    "asr": {"model": "ep-asr"},
                    "tts": {"enabled": False},
                    "notification_levels": ["critical"],
                },
            ),
        )
    )
    monkeypatch.setattr(channel.loop, "speak", fake_speak)

    asyncio.run(
        channel.send(
            AgentReply(
                envelope=Envelope(channel="voice"),
                text="routine status",
                metadata={
                    "notification": True,
                    "severity": "info",
                    "notification_kind": "task_update",
                },
            )
        )
    )
    asyncio.run(
        channel.send(
            AgentReply(
                envelope=Envelope(channel="voice"),
                text="operator action required",
                metadata={
                    "notification": True,
                    "severity": "critical",
                    "notification_kind": "operator_alert",
                },
            )
        )
    )

    assert spoken == ["[CRITICAL] operator alert: operator action required"]


def test_voice_channel_ignores_non_voice_replies_and_non_final_progress(
    monkeypatch,
) -> None:
    spoken: list[str] = []

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    channel = VoiceChannel(
        ChannelContext(
            name="voice",
            deployment_id="test",
            spec=ChannelSpec(
                type="voice",
                settings={
                    "asr": {"model": "ep-asr"},
                    "tts": {"enabled": False},
                },
            ),
        )
    )
    monkeypatch.setattr(channel.loop, "speak", fake_speak)

    asyncio.run(
        channel.send(
            AgentReply(
                envelope=Envelope(channel="web"),
                text="web final reply",
                final=True,
            )
        )
    )
    asyncio.run(
        channel.send(
            AgentReply(
                envelope=Envelope(channel="voice"),
                text="progress reply",
                final=False,
            )
        )
    )
    asyncio.run(
        channel.send(
            AgentReply(
                envelope=Envelope(channel="voice"),
                text="voice final reply",
                final=True,
            )
        )
    )

    assert spoken == ["voice final reply"]


def test_voice_channel_does_not_speak_internal_execution_feedback(monkeypatch) -> None:
    spoken: list[str] = []

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    channel = VoiceChannel(
        ChannelContext(
            name="voice",
            deployment_id="test",
            spec=ChannelSpec(
                type="voice",
                settings={"asr": {"model": "ep-asr"}, "tts": {"enabled": False}},
            ),
        )
    )
    monkeypatch.setattr(channel.loop, "speak", fake_speak)

    asyncio.run(
        channel.send(
            AgentReply(
                envelope=Envelope(channel="voice"),
                text="Execution feedback for skill skill1:\n- subgoal_success: True",
                final=True,
                metadata={"tool": "request_capability"},
            )
        )
    )

    assert spoken == []


def test_voice_channel_does_not_speak_internal_tool_summary(monkeypatch) -> None:
    spoken: list[str] = []

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    channel = VoiceChannel(
        ChannelContext(
            name="voice",
            deployment_id="test",
            spec=ChannelSpec(
                type="voice",
                settings={"asr": {"model": "ep-asr"}, "tts": {"enabled": False}},
            ),
        )
    )
    monkeypatch.setattr(channel.loop, "speak", fake_speak)

    asyncio.run(
        channel.send(
            AgentReply(
                envelope=Envelope(channel="voice"),
                text="scene inspected",
                final=True,
                metadata={"tool": "request_capability"},
            )
        )
    )

    assert spoken == []


def test_voice_channel_start_stop_and_event(monkeypatch) -> None:
    captured: dict[str, object] = {}
    started: list[str] = []
    stopped: list[str] = []

    async def fake_start(handler) -> None:  # type: ignore[no-untyped-def]
        started.append("start")
        await handler("inspect desk", {"source": "mic"})

    async def fake_stop() -> None:
        stopped.append("stop")

    channel = VoiceChannel(
        ChannelContext(
            name="voice",
            deployment_id="test",
            spec=ChannelSpec(
                type="voice",
                settings={"asr": {"model": "ep-asr"}, "tts": {"enabled": False}},
            ),
        )
    )
    monkeypatch.setattr(channel.loop, "start", fake_start)
    monkeypatch.setattr(channel.loop, "stop", fake_stop)

    async def run() -> None:
        async def handler(turn) -> None:  # type: ignore[no-untyped-def]
            captured["turn"] = turn

        await channel.start(handler)
        await channel.on_event(object())  # type: ignore[arg-type]
        await channel.stop()

    asyncio.run(run())

    turn = captured["turn"]
    assert started == ["start"]
    assert stopped == ["stop"]
    assert turn.text == "inspect desk"  # type: ignore[attr-defined]
    assert turn.envelope.channel == "voice"  # type: ignore[attr-defined]


def test_voice_channel_speaks_human_follow_lifecycle_events(monkeypatch) -> None:
    spoken: list[str] = []

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    channel = VoiceChannel(
        ChannelContext(
            name="voice",
            deployment_id="test",
            spec=ChannelSpec(
                type="voice",
                settings={"asr": {"model": "ep-asr"}, "tts": {"enabled": False}},
            ),
        )
    )
    monkeypatch.setattr(channel.loop, "speak", fake_speak)

    event = RuntimeEvent.make(
        "skill.lifecycle",
        source="skill",
        episode_id="ep1",
        channel="voice",
        payload={
            "skill_id": "skill1",
            "name": "human_follow",
            "phase": "executing",
            "step": "following",
            "ux": {"phase": "following"},
        },
    )
    asyncio.run(channel.on_event(event))
    asyncio.run(channel.on_event(event))

    assert spoken == ["我已经看到目标，正在跟随。"]


def test_voice_channel_does_not_speak_generic_skill_completed_event(
    monkeypatch,
) -> None:
    spoken: list[str] = []

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    channel = VoiceChannel(
        ChannelContext(
            name="voice",
            deployment_id="test",
            spec=ChannelSpec(
                type="voice",
                settings={"asr": {"model": "ep-asr"}, "tts": {"enabled": False}},
            ),
        )
    )
    monkeypatch.setattr(channel.loop, "speak", fake_speak)

    event = RuntimeEvent.make(
        "skill.lifecycle",
        source="skill",
        episode_id="ep1",
        channel="voice",
        payload={
            "skill_id": "skill1",
            "name": "inspect_scene",
            "phase": "completed",
            "summary": "scene inspected",
        },
    )
    asyncio.run(channel.on_event(event))

    assert spoken == []


def test_voice_channel_clarifies_vague_voice_turns(monkeypatch) -> None:
    captured: list[object] = []
    spoken: list[str] = []

    async def fake_start(handler) -> None:  # type: ignore[no-untyped-def]
        await handler("想想看看吧", {"source": "mic"})

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    channel = VoiceChannel(
        ChannelContext(
            name="voice",
            deployment_id="test",
            spec=ChannelSpec(
                type="voice",
                settings={"asr": {"model": "ep-asr"}, "tts": {"enabled": False}},
            ),
        )
    )
    monkeypatch.setattr(channel.loop, "start", fake_start)
    monkeypatch.setattr(channel.loop, "speak", fake_speak)

    async def handler(turn) -> None:  # type: ignore[no-untyped-def]
        captured.append(turn)

    asyncio.run(channel.start(handler))

    assert captured == []
    assert spoken == ["请说清楚要观察哪里，或要我做哪个动作。"]


def test_needs_voice_clarification_allows_follow_confirmation() -> None:
    """Messages like '可以跟随' and '可以启动跟随' are confirmation responses."""
    assert _needs_voice_clarification("可以跟随") is False
    assert _needs_voice_clarification("可以启动跟随") is False
    assert _needs_voice_clarification("可以 启动 跟随") is False


def test_needs_voice_clarification_allows_follow_messages() -> None:
    """Messages explicitly about following should not require clarification."""
    assert _needs_voice_clarification("跟着我") is False
    assert _needs_voice_clarification("跟随") is False


def test_is_specific_action_intent_detects_markers() -> None:
    assert _is_specific_action_intent("跟随") is True
    assert _is_specific_action_intent("跟着") is True
    assert _is_specific_action_intent("启动") is True
    assert _is_specific_action_intent("确认") is True
    assert _is_specific_action_intent("好的") is True
    assert _is_specific_action_intent("你好") is False
    assert _is_specific_action_intent("") is False


def test_needs_voice_clarification_still_blocks_bare_vague() -> None:
    """Bare vague markers without specific action intent should still be blocked."""
    assert _needs_voice_clarification("看看吧") is True
    assert _needs_voice_clarification("好吧") is True
    assert _needs_voice_clarification("行吧") is True
    assert _needs_voice_clarification("随便") is True


def test_needs_voice_clarification_allows_normal_questions() -> None:
    """Normal questions containing common words should not be blocked."""
    assert _needs_voice_clarification("你可以看到我吗") is False
    assert _needs_voice_clarification("你能做什么") is False
    assert _needs_voice_clarification("可以看到我吗") is False
    assert _needs_voice_clarification("你能看见前面的东西吗") is False
    assert _needs_voice_clarification("你知道我在哪里吗") is False
    assert _needs_voice_clarification("前面有什么") is False


def test_needs_voice_clarification_allows_ability_questions() -> None:
    """Questions asking about robot abilities should pass through to the agent."""
    assert _needs_voice_clarification("你可以做什么") is False
    assert _needs_voice_clarification("你能干嘛") is False
    assert _needs_voice_clarification("可以跟我走吗") is False


def test_voice_channel_does_not_clarify_follow_confirmation(monkeypatch) -> None:
    captured: list[object] = []
    spoken: list[str] = []

    async def fake_start(handler) -> None:  # type: ignore[no-untyped-def]
        await handler("可以跟随", {"source": "mic"})

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    channel = VoiceChannel(
        ChannelContext(
            name="voice",
            deployment_id="test",
            spec=ChannelSpec(
                type="voice",
                settings={"asr": {"model": "ep-asr"}, "tts": {"enabled": False}},
            ),
        )
    )
    monkeypatch.setattr(channel.loop, "start", fake_start)
    monkeypatch.setattr(channel.loop, "speak", fake_speak)

    async def handler(turn) -> None:  # type: ignore[no-untyped-def]
        captured.append(turn)

    asyncio.run(channel.start(handler))

    assert len(captured) == 1
    assert captured[0].text == "可以跟随"  # type: ignore[attr-defined]
    assert spoken == []


def test_voice_channel_does_not_clarify_normal_questions(monkeypatch) -> None:
    """Normal questions like '你可以看到我吗' should reach the agent, not be blocked."""
    captured: list[object] = []
    spoken: list[str] = []

    async def fake_start(handler) -> None:  # type: ignore[no-untyped-def]
        await handler("你可以看到我吗", {"source": "mic"})

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    channel = VoiceChannel(
        ChannelContext(
            name="voice",
            deployment_id="test",
            spec=ChannelSpec(
                type="voice",
                settings={"asr": {"model": "ep-asr"}, "tts": {"enabled": False}},
            ),
        )
    )
    monkeypatch.setattr(channel.loop, "start", fake_start)
    monkeypatch.setattr(channel.loop, "speak", fake_speak)

    async def handler(turn) -> None:  # type: ignore[no-untyped-def]
        captured.append(turn)

    asyncio.run(channel.start(handler))

    assert len(captured) == 1
    assert captured[0].text == "你可以看到我吗"  # type: ignore[attr-defined]
    assert spoken == []


def test_voice_channel_does_not_clarify_ability_question(monkeypatch) -> None:
    """Questions like '你能做什么' should reach the agent, not be blocked."""
    captured: list[object] = []
    spoken: list[str] = []

    async def fake_start(handler) -> None:  # type: ignore[no-untyped-def]
        await handler("你能做什么", {"source": "mic"})

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    channel = VoiceChannel(
        ChannelContext(
            name="voice",
            deployment_id="test",
            spec=ChannelSpec(
                type="voice",
                settings={"asr": {"model": "ep-asr"}, "tts": {"enabled": False}},
            ),
        )
    )
    monkeypatch.setattr(channel.loop, "start", fake_start)
    monkeypatch.setattr(channel.loop, "speak", fake_speak)

    async def handler(turn) -> None:  # type: ignore[no-untyped-def]
        captured.append(turn)

    asyncio.run(channel.start(handler))

    assert len(captured) == 1
    assert captured[0].text == "你能做什么"  # type: ignore[attr-defined]
    assert spoken == []


# ── ASR nonsense filter ──────────────────────────────────────────────


def test_is_nonsense_asr_detects_repeated_characters() -> None:
    """Text with 3+ consecutive identical characters is ASR noise."""
    assert _is_nonsense_asr("请请请") is True
    assert _is_nonsense_asr("啊啊啊") is True
    assert _is_nonsense_asr("嗯嗯嗯嗯") is True
    assert _is_nonsense_asr("测试请请请一下") is True


def test_is_nonsense_asr_allows_normal_speech() -> None:
    """Normal speech without 3+ repeated chars should pass."""
    assert _is_nonsense_asr("你好") is False
    assert _is_nonsense_asr("你能看到我吗") is False
    assert _is_nonsense_asr("请请一下") is False  # 2 repeats = valid edge case
    assert _is_nonsense_asr("") is False
    assert _is_nonsense_asr("跟着我") is False


def test_voice_channel_drops_nonsense_asr(monkeypatch) -> None:
    """Nonsense ASR output like '请请请' should be silently dropped."""
    captured: list[object] = []
    spoken: list[str] = []

    async def fake_start(handler) -> None:  # type: ignore[no-untyped-def]
        await handler("请请请", {"source": "mic"})

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    channel = VoiceChannel(
        ChannelContext(
            name="voice",
            deployment_id="test",
            spec=ChannelSpec(
                type="voice",
                settings={"asr": {"model": "ep-asr"}, "tts": {"enabled": False}},
            ),
        )
    )
    monkeypatch.setattr(channel.loop, "start", fake_start)
    monkeypatch.setattr(channel.loop, "speak", fake_speak)

    async def handler(turn) -> None:  # type: ignore[no-untyped-def]
        captured.append(turn)

    asyncio.run(channel.start(handler))

    assert captured == []
    assert spoken == []
