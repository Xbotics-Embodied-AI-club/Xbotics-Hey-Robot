from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from hey_robot.config import PolicySpec
from hey_robot.policies.runtime import (
    MockPolicyAdapter,
    PolicyHealth,
    PolicyIOSchema,
    PolicyRuntime,
    PolicyRuntimeInput,
    PolicyRuntimeOutput,
    _accelerator_from_device,
    build_policy_runtime,
)


class TestAcceleratorFromDevice:
    def test_cuda(self) -> None:
        assert _accelerator_from_device("cuda:0") == "cuda"
        assert _accelerator_from_device("CUDA") == "cuda"

    def test_mps(self) -> None:
        assert _accelerator_from_device("mps") == "mps"

    def test_cpu(self) -> None:
        assert _accelerator_from_device("cpu") == "cpu"

    def test_unknown_returns_lowered(self) -> None:
        assert _accelerator_from_device("TPU") == "tpu"

    def test_empty_returns_none(self) -> None:
        assert _accelerator_from_device("") is None


class TestPolicyIOSchema:
    def test_defaults(self) -> None:
        schema = PolicyIOSchema(policy_id="p1", policy_type="lerobot", robot_id="r1")
        assert schema.control_frequency_hz == 20.0
        assert schema.device == "cpu"
        assert schema.accelerator is None
        assert schema.stateful is True
        assert schema.supports_interrupt is True

    def test_frozen(self) -> None:
        schema = PolicyIOSchema(policy_id="p1", policy_type="lerobot", robot_id="r1")
        with pytest.raises(FrozenInstanceError):
            schema.device = "cuda"  # type: ignore[misc]


class TestPolicyHealth:
    def test_creation(self) -> None:
        health = PolicyHealth(
            policy_id="p1", loaded=True, device="cpu", error="timeout"
        )
        assert health.policy_id == "p1"
        assert health.loaded is True
        assert health.error == "timeout"

    def test_defaults(self) -> None:
        health = PolicyHealth(policy_id="p1", loaded=False, device="cpu")
        assert health.error is None
        assert health.metrics == {}


class TestPolicyRuntimeInput:
    def test_frozen(self) -> None:
        from hey_robot.protocol import Envelope, RobotObservation, SkillIntent

        obs = RobotObservation(
            envelope=Envelope(), frame_id=1, images=[], proprioception=[]
        )
        intent = SkillIntent(envelope=Envelope(), skill_id="s1")
        inp = PolicyRuntimeInput(observation=obs, intent=intent)
        with pytest.raises(FrozenInstanceError):
            inp.observation = obs  # type: ignore[misc]


class TestPolicyRuntimeOutput:
    def test_default_metadata(self) -> None:
        from hey_robot.protocol import Envelope, RobotAction

        action = RobotAction(envelope=Envelope(), values=[])
        out = PolicyRuntimeOutput(action=action)
        assert out.metadata == {}


class TestMockPolicyAdapter:
    def test_warmup_sets_loaded(self) -> None:
        spec = PolicySpec(
            type="mock", robot_id="r1", freq_hz=10, model_path="", device="cpu"
        )
        adapter = MockPolicyAdapter("p1", spec)
        assert adapter.loaded is False
        asyncio.run(adapter.warmup())
        assert adapter.loaded is True

    def test_predict_returns_list(self) -> None:
        from hey_robot.protocol import Envelope, RobotObservation, SkillIntent

        spec = PolicySpec(
            type="mock", robot_id="r1", freq_hz=10, model_path="", device="cpu"
        )
        adapter = MockPolicyAdapter("p1", spec)
        obs = RobotObservation(
            envelope=Envelope(), frame_id=1, images=[], proprioception=[]
        )
        intent = SkillIntent(envelope=Envelope(), skill_id="s1")
        result = asyncio.run(adapter.predict(None, obs, intent))
        assert result == [0.0]

    def test_close_unsets_loaded(self) -> None:
        spec = PolicySpec(
            type="mock", robot_id="r1", freq_hz=10, model_path="", device="cpu"
        )
        adapter = MockPolicyAdapter("p1", spec)
        asyncio.run(adapter.warmup())
        assert adapter.loaded is True
        asyncio.run(adapter.close())
        assert adapter.loaded is False

    def test_health(self) -> None:
        spec = PolicySpec(
            type="mock", robot_id="r1", freq_hz=10, model_path="", device="cuda:0"
        )
        adapter = MockPolicyAdapter("p1", spec)
        health = adapter.health()
        assert health.policy_id == "p1"
        assert health.loaded is False
        assert health.device == "cuda:0"

    def test_schema(self) -> None:
        spec = PolicySpec(
            type="mock",
            robot_id="r1",
            freq_hz=30,
            model_path="",
            device="cpu",
            settings={"key": "val"},
        )
        adapter = MockPolicyAdapter("p1", spec)
        schema = adapter.schema()
        assert schema.policy_type == "mock"
        assert schema.robot_id == "r1"
        assert schema.control_frequency_hz == 30.0
        assert schema.device == "cpu"
        assert "modalities" in schema.observation_schema


class TestPolicyRuntime:
    def test_control_period_sec(self) -> None:
        from hey_robot.perception.codecs.simple import SimpleVectorCodec

        spec = PolicySpec(
            type="mock", robot_id="r1", freq_hz=10, model_path="", device="cpu"
        )
        adapter = MockPolicyAdapter("p1", spec)
        runtime = PolicyRuntime(
            policy_id="p1", spec=spec, codec=SimpleVectorCodec(), adapter=adapter
        )
        assert runtime.control_period_sec == 0.1

    def test_warmup_delegates(self) -> None:
        from hey_robot.perception.codecs.simple import SimpleVectorCodec

        spec = PolicySpec(
            type="mock", robot_id="r1", freq_hz=10, model_path="", device="cpu"
        )
        adapter = MockPolicyAdapter("p1", spec)
        runtime = PolicyRuntime(
            policy_id="p1", spec=spec, codec=SimpleVectorCodec(), adapter=adapter
        )
        asyncio.run(runtime.warmup())
        assert adapter.loaded is True

    def test_close_delegates(self) -> None:
        from hey_robot.perception.codecs.simple import SimpleVectorCodec

        spec = PolicySpec(
            type="mock", robot_id="r1", freq_hz=10, model_path="", device="cpu"
        )
        adapter = MockPolicyAdapter("p1", spec)
        runtime = PolicyRuntime(
            policy_id="p1", spec=spec, codec=SimpleVectorCodec(), adapter=adapter
        )
        asyncio.run(runtime.warmup())
        asyncio.run(runtime.close())
        assert adapter.loaded is False

    def test_health_delegates(self) -> None:
        from hey_robot.perception.codecs.simple import SimpleVectorCodec

        spec = PolicySpec(
            type="mock", robot_id="r1", freq_hz=10, model_path="", device="cpu"
        )
        adapter = MockPolicyAdapter("p1", spec)
        runtime = PolicyRuntime(
            policy_id="p1", spec=spec, codec=SimpleVectorCodec(), adapter=adapter
        )
        health = runtime.health()
        assert health.policy_id == "p1"

    def test_schema_delegates(self) -> None:
        from hey_robot.perception.codecs.simple import SimpleVectorCodec

        spec = PolicySpec(
            type="mock", robot_id="r1", freq_hz=10, model_path="", device="cpu"
        )
        adapter = MockPolicyAdapter("p1", spec)
        runtime = PolicyRuntime(
            policy_id="p1", spec=spec, codec=SimpleVectorCodec(), adapter=adapter
        )
        schema = runtime.schema()
        assert schema.policy_type == "mock"


class TestBuildPolicyRuntime:
    def test_unsupported_type_raises(self) -> None:
        from hey_robot.config import DeploymentConfig
        from hey_robot.media import LocalMediaStore, MediaResolver
        from hey_robot.perception.codecs.simple import SimpleVectorCodec

        spec = PolicySpec(
            type="unsupported_type",
            robot_id="r1",
            freq_hz=10,
            model_path="",
            device="cpu",
        )
        with pytest.raises(ValueError, match="unsupported policy type"):
            build_policy_runtime(
                "p1",
                spec,
                config=DeploymentConfig.from_dict({}),
                codec=SimpleVectorCodec(),
                media_resolver=MediaResolver(LocalMediaStore()),
            )
