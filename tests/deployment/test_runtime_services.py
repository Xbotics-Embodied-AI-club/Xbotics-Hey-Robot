from __future__ import annotations

import pytest

from hey_robot.app import DeploymentRunner
from hey_robot.channels import ChannelContext, WebChannel
from hey_robot.config import ChannelSpec, DeploymentConfig
from hey_robot.events import EventKind, RuntimeEvent
from hey_robot.media import LocalMediaStore
from hey_robot.perception import CodecRegistry
from hey_robot.protocol import Envelope, RobotAction, RobotObservation, SkillIntent
from hey_robot.robots import RobotManager, RobotRuntime, RobotSafetyError
from hey_robot.skills import RobotSkillAction


def test_runtime_event_roundtrip_and_filter() -> None:
    event = RuntimeEvent.make(
        EventKind.AGENT_TURN_START,
        source="agent",
        trace_id="tr1",
        agent_id="main",
        robot_id="mock0",
        payload={"text_len": 4},
    )
    restored = RuntimeEvent(**event.to_dict())

    assert restored.kind == "agent.turn.start"
    assert restored.payload == {"text_len": 4}


def test_web_channel_payload_to_turn_without_server() -> None:
    channel = WebChannel(
        ChannelContext(
            name="web",
            deployment_id="test",
            spec=ChannelSpec(type="web"),
        )
    )

    turn = channel._payload_to_turn(
        {"text": "pick", "sender_id": "u1", "chat_id": "c1"}
    )

    assert turn.text == "pick"
    assert turn.envelope.channel == "web"
    assert turn.envelope.sender_id == "u1"
    assert turn.envelope.robot_id is None


def test_runner_builds_local_services() -> None:
    config = DeploymentConfig.from_yaml("configs/mock.test.yaml")
    runner = DeploymentRunner(config)

    assert [service.name for service in runner.services] == [
        "robot",
        "skill-controller",
        "task-supervisor",
        "agent:main",
        "gateway",
    ]


def test_runner_builds_services_when_configured(tmp_path) -> None:
    config = DeploymentConfig.from_dict(
        {
            "deployment": {"id": "d1"},
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "robots": {"mock0": {"type": "mock"}},
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {"mode": "direct"},
                }
            },
            "channels": {"web": {"type": "web", "enabled": True}},
        }
    )

    runner = DeploymentRunner(config, episode_dir=tmp_path / "episodes")

    service_names = [service.name for service in runner.services]
    assert "robot" in service_names
    assert "task-supervisor" in service_names
    assert "agent:main" in service_names
    assert "gateway" in service_names


def test_codec_registry_converts_simple_action() -> None:
    codec = CodecRegistry().get("mock")
    envelope = Envelope(robot_id="mock0")
    observation = RobotObservation(envelope=envelope, frame_id=1, proprioception=[1.0])
    skill = SkillIntent(envelope=envelope, skill_id="cmd1", objective="move")

    policy_input = codec.observation_to_policy_input(observation, skill)
    action = codec.policy_output_to_action([0.1, 0.2], observation, skill)

    assert policy_input["task"] == "move"
    assert action.skill_id == "cmd1"
    assert action.values == [0.1, 0.2]


def test_robot_manager_supports_mock_and_rejects_unknown() -> None:
    config = DeploymentConfig.from_dict(
        {
            "robots": {
                "mock0": {"type": "mock"},
            }
        }
    )
    manager = RobotManager(config)

    assert manager.require("mock0").robot_id == "mock0"


async def test_robot_runtime_wraps_driver_capabilities_health_and_observation(
    tmp_path,
) -> None:
    config = DeploymentConfig.from_dict({"robots": {"mock0": {"type": "mock"}}})
    driver = RobotManager(config).require("mock0")
    runtime = RobotRuntime(driver, LocalMediaStore(tmp_path))

    snapshot = await runtime.start()
    observation = await runtime.observe()
    intent = SkillIntent(
        envelope=Envelope(robot_id="mock0"), skill_id="cmd1", objective="move forward"
    )
    status = await runtime.apply_action(
        RobotSkillAction(
            "move_base", {"direction": "forward", "distance_cm": 10.0}
        ).to_robot_action(intent)
    )
    health = await runtime.health()

    assert snapshot.capabilities.driver_type == "mock"
    assert snapshot.capabilities.metadata["runtime"] == "mock_xlerobot"
    assert snapshot.capabilities.metadata["robot_family"] == "xlerobot"
    assert snapshot.capabilities.metadata["environment"] == "mock"
    assert snapshot.capabilities.metadata["embodiment_profile"] == "xlerobot_mock"
    assert snapshot.health.online is True
    assert len(observation.images) == 3
    assert observation.raw["driver"] == "mock"
    assert observation.raw["robot_family"] == "xlerobot"
    assert observation.raw["embodiment_profile"] == "xlerobot_mock"
    assert {image.camera for image in observation.images} == {
        "front",
        "left_wrist",
        "right_wrist",
    }
    assert status.skill_id == "cmd1"
    assert status.success is True
    assert health.state == "skill_completed"


async def test_robot_runtime_rejects_non_skill_actions(tmp_path) -> None:
    config = DeploymentConfig.from_dict({"robots": {"mock0": {"type": "mock"}}})
    runtime = RobotRuntime(
        RobotManager(config).require("mock0"), LocalMediaStore(tmp_path)
    )
    await runtime.start()

    status = await runtime.apply_action(
        RobotAction(
            envelope=Envelope(robot_id="mock0"), skill_id="cmd1", values=[0.1, 0.2]
        )
    )

    assert status.success is False
    assert status.metrics["last_skill_result"]["failure_mode"] == "invalid_action"


async def test_robot_runtime_safety_blocks_estop(tmp_path) -> None:
    config = DeploymentConfig.from_dict(
        {"robots": {"mock0": {"type": "mock", "safety": {"estop": True}}}}
    )
    runtime = RobotRuntime(
        RobotManager(config).require("mock0"), LocalMediaStore(tmp_path)
    )
    await runtime.start()

    intent = SkillIntent(
        envelope=Envelope(robot_id="mock0"), skill_id="cmd1", objective="stop"
    )
    with pytest.raises(RobotSafetyError, match="estop"):
        await runtime.apply_action(
            RobotSkillAction("base_stop").to_robot_action(intent)
        )
