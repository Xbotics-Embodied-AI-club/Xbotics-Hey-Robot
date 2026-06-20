from __future__ import annotations

import asyncio
import json
import sys
import time
import types
from unittest.mock import patch

import pytest

from hey_robot.config import DeploymentConfig
from hey_robot.episode import RobotEpisodeStateStore
from hey_robot.events.bus import BusEventPublisher
from hey_robot.policies.runtime import PolicyRuntimeOutput
from hey_robot.protocol import (
    Envelope,
    RobotAction,
    RobotObservation,
    RobotStatus,
    SkillEvent,
    SkillIntent,
)
from hey_robot.protocol.messages import from_payload
from hey_robot.skills import RobotSkillAction
from hey_robot.skills.base import (
    BaseSkill,
    SkillResult as PluginSkillResult,
    SkillSpec,
)
from hey_robot.skills.controller import SkillControllerService


class FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))


class FakeRuntime:
    @property
    def control_period_sec(self) -> float:
        return 0.001

    def __init__(self) -> None:
        self.predicted_skill_ids: list[str] = []
        self.predicted = asyncio.Event()

    async def predict(self, payload) -> PolicyRuntimeOutput:
        self.predicted_skill_ids.append(payload.intent.skill_id)
        self.predicted.set()
        if payload.intent.name:
            return PolicyRuntimeOutput(
                action=RobotSkillAction(
                    payload.intent.name, dict(payload.intent.arguments)
                ).to_robot_action(payload.intent)
            )
        return PolicyRuntimeOutput(
            action=RobotAction(
                envelope=payload.intent.envelope,
                values=[0.0],
                skill_id=payload.intent.skill_id,
            )
        )

    async def close(self) -> None:
        return None


def _service(tmp_path) -> SkillControllerService:
    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "robots": {"xlerobot": {"type": "xlerobot"}},
            "policies": {
                "embodied_skills": {
                    "type": "skill",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "settings": {"codec": "skill"},
                }
            },
        }
    )
    service = SkillControllerService(config)
    service.bus = FakeBus()  # type: ignore[assignment]
    service.events = BusEventPublisher(service.bus, service.topics)  # type: ignore[arg-type]
    return service


def _service_with_vla_capability(
    tmp_path, *, settings: dict | None = None, locomotion_settings: dict | None = None
) -> SkillControllerService:
    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "robots": {"xlerobot": {"type": "xlerobot"}},
            "policies": {
                "embodied_skills": {
                    "type": "skill",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "settings": {"codec": "skill"},
                }
            },
            "capability_services": {
                "foundation_locomotion": {
                    "type": "mock_vla_service",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "skill_names": ["foundation_locomotion_run"],
                    "resources": ["base", "camera"],
                    "timeout_sec": 20,
                    "settings": locomotion_settings or {},
                },
                "arm_vla": {
                    "type": "mock_vla_service",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "skill_names": ["vla_manipulation"],
                    "resources": ["arm", "gripper", "camera"],
                    "timeout_sec": 30,
                    "settings": settings or {},
                },
            },
        }
    )
    service = SkillControllerService(config)
    service.bus = FakeBus()  # type: ignore[assignment]
    service.events = BusEventPublisher(service.bus, service.topics)  # type: ignore[arg-type]
    return service


def _mock_xlerobot_service(tmp_path) -> SkillControllerService:
    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "robots": {"mock0": {"type": "mock", "body": "xlerobot"}},
            "policies": {
                "embodied_skills": {
                    "type": "skill",
                    "enabled": True,
                    "robot_id": "mock0",
                    "settings": {"codec": "skill", "body": "xlerobot"},
                }
            },
        }
    )
    service = SkillControllerService(config)
    service.bus = FakeBus()  # type: ignore[assignment]
    service.events = BusEventPublisher(service.bus, service.topics)  # type: ignore[arg-type]
    return service


def test_skill_controller_uses_mock_xlerobot_embodiment_for_contracts(tmp_path) -> None:
    service = _mock_xlerobot_service(tmp_path)
    contract = service.plugin_skill_catalog.resolve(
        "set_gripper", robot_type="xlerobot"
    )

    assert contract.required_resources == ("gripper",)


def test_move_base_resolves_to_primitive_contract(tmp_path) -> None:
    service = _service(tmp_path)
    contract = service.plugin_skill_catalog.resolve("move_base")

    assert contract.level == "primitive"
    assert contract.required_resources == ("base",)


def test_turn_base_resolves_to_primitive_contract(tmp_path) -> None:
    service = _service(tmp_path)
    contract = service.plugin_skill_catalog.resolve("turn_base")

    assert contract.level == "primitive"
    assert contract.required_resources == ("base",)


def test_skill_controller_rejects_direct_hidden_implementation_skill(
    tmp_path,
) -> None:
    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "skills": {
                "mode": "production",
                "enabled": ["human_follow"],
            },
            "robots": {"xlerobot": {"type": "xlerobot"}},
            "policies": {
                "embodied_skills": {
                    "type": "skill",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "settings": {"codec": "skill"},
                }
            },
        }
    )
    service = SkillControllerService(config)
    service.bus = FakeBus()  # type: ignore[assignment]
    service.events = BusEventPublisher(service.bus, service.topics)  # type: ignore[arg-type]
    intent = SkillIntent(
        envelope=Envelope(trace_id="tr1", robot_id="xlerobot"),
        skill_id="skill1",
        name="move_base",
        arguments={"direction": "forward", "distance_cm": 20},
        objective="move forward",
    )

    asyncio.run(service._on_skill_intent(service.topics.skill_intent, intent.__dict__))

    state = service.states["embodied_skills"]
    assert state.active_runs == {}
    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]
    assert results[0]["status"] == "failed"
    assert results[0]["failure_mode"] == "unknown_skill"
    assert "move_base" in results[0]["summary"]


def test_skill_controller_routes_mock_inspect_scene_to_robot_runtime(
    tmp_path,
) -> None:
    service = _mock_xlerobot_service(tmp_path)
    state = service.states["embodied_skills"]
    state.runtime = FakeRuntime()
    state.latest_observation = RobotObservation(
        envelope=Envelope(robot_id="mock0"),
        frame_id=1,
        images=[],
        proprioception=[],
    )
    intent = SkillIntent(
        envelope=Envelope(trace_id="tr1", robot_id="mock0"),
        skill_id="skill1",
        name="inspect_scene",
        arguments={"question": "look for marker"},
        objective="inspect the scene",
    )

    async def run_once() -> None:
        await service._on_skill_intent(service.topics.skill_intent, intent.__dict__)
        await service._skill_loop_step_for_test("embodied_skills", state)

    asyncio.run(run_once())

    actions = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.robot_action
    ]  # type: ignore[attr-defined]
    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]

    assert len(actions) == 1
    assert actions[0]["metadata"]["skill"]["name"] == "inspect_scene"
    assert actions[0]["skill_id"] == "skill1"
    assert results == []


def test_skill_controller_accepts_contract_and_completes_from_status(tmp_path) -> None:
    service = _service(tmp_path)
    envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
    intent = SkillIntent(
        envelope=envelope,
        skill_id="skill1",
        name="set_gripper",
        arguments={"action": "open"},
        objective="open gripper",
    )

    asyncio.run(service._on_skill_intent(service.topics.skill_intent, intent.__dict__))
    asyncio.run(
        service._on_status(
            service.topics.robot_status,
            RobotStatus(
                envelope=envelope,
                frame_id=7,
                state="skill_completed",
                skill_id="skill1",
                success=True,
                metrics={"last_skill_result": {"message": "opened"}},
            ).__dict__,
        )
    )

    events = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_event
    ]  # type: ignore[attr-defined]
    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]

    assert events[0]["phase"] == "accepted"
    assert events[1]["phase"] == "completed"
    assert events[1]["frame_id"] == 7
    assert events[1]["progress"] == 1.0
    assert events[0]["metadata"]["contract"]["required_resources"] == ["gripper"]
    assert results[0]["status"] == "completed"
    assert results[0]["summary"] == "opened"
    assert results[0]["metadata"]["contract"]["name"] == "set_gripper"
    assert results[0]["metadata"]["skill"] == "set_gripper"


def test_skill_controller_executes_pure_plugin_skill_via_skill_runtime(
    tmp_path, monkeypatch
) -> None:
    module_name = "tests.fake_plugin_skill_module"
    module = types.ModuleType(module_name)

    class EchoSkill(BaseSkill):
        spec = SkillSpec(
            name="plugin_echo",
            description="Return a plugin-generated summary.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            required_resources=("memory",),
            safety_level="observe",
            timeout_sec=2.0,
            agent_visible=True,
        )

        async def execute(self, ctx, arguments):
            del ctx
            return PluginSkillResult(
                success=True,
                summary=f"plugin:{arguments['text']}",
            )

    def register_skills(registry):
        registry.register(EchoSkill())

    setattr(module, "register_skills", register_skills)
    monkeypatch.setitem(sys.modules, module_name, module)

    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "skills": {
                "modules": [module_name],
                "enabled": ["plugin_echo"],
            },
            "robots": {"xlerobot": {"type": "xlerobot"}},
            "policies": {
                "embodied_skills": {
                    "type": "skill",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "settings": {"codec": "skill"},
                }
            },
        }
    )
    service = SkillControllerService(config)
    service.bus = FakeBus()  # type: ignore[assignment]
    service.events = BusEventPublisher(service.bus, service.topics)  # type: ignore[arg-type]
    state = service.states["embodied_skills"]
    intent = SkillIntent(
        envelope=Envelope(trace_id="tr1", robot_id="xlerobot"),
        skill_id="skill1",
        name="plugin_echo",
        arguments={"text": "hello"},
        objective="echo text",
    )

    async def run() -> None:
        await service._on_skill_intent(service.topics.skill_intent, intent.__dict__)
        for _ in range(20):
            await service._skill_loop_step_for_test("embodied_skills", state)
            if not state.active_runs:
                break

    asyncio.run(run())

    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]
    actions = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.robot_action
    ]  # type: ignore[attr-defined]

    assert actions == []
    assert results[-1]["status"] == "completed"
    assert results[-1]["summary"] == "plugin:hello"
    assert state.active_runs == {}


def test_skill_controller_executes_human_follow_as_plugin_skill(
    tmp_path, monkeypatch
) -> None:
    import numpy as np

    from hey_robot.protocol import ImageRef
    from hey_robot.skills.base import SkillResult as PluginSkillResult
    from hey_robot.skills.builtin.navigation import HumanFollowSkill

    async def keep_observation(_ctx, observation):
        return observation

    monkeypatch.setattr(
        "hey_robot.skills.builtin.navigation.load_detector",
        lambda _path=None: None,
    )
    monkeypatch.setattr(
        "hey_robot.skills.builtin.navigation.detect_people",
        lambda _image: [
            types.SimpleNamespace(
                bbox=(60, 10, 70, 30),
                confidence=1.0,
                center=(65, 20),
                area=200,
            )
        ],
    )
    monkeypatch.setattr(
        "hey_robot.skills.builtin.navigation._refresh_observation",
        keep_observation,
    )

    service = _service(tmp_path)
    state = service.states["embodied_skills"]
    state.runtime = FakeRuntime()
    state.latest_observation = RobotObservation(
        envelope=Envelope(robot_id="xlerobot"),
        frame_id=1,
        images=[
            ImageRef(uri="media://local/images/xlerobot/frame.jpg", camera="front")
        ],
    )
    service.media_resolver.resolve_images = (  # type: ignore[method-assign]
        lambda _refs: [np.zeros((100, 100, 3), dtype=np.uint8)]
    )
    intent = SkillIntent(
        envelope=Envelope(trace_id="tr1", robot_id="xlerobot"),
        skill_id="hf1",
        name="human_follow",
        arguments={"max_steps": 1, "target_height_ratio": 0.3},
        objective="follow the person",
    )

    recorded_calls: list[tuple[str, dict]] = []

    async def fake_invoke_robot_skill(
        _policy_id,
        _state,
        _run,
        name,
        arguments,
    ) -> dict:
        recorded_calls.append((name, dict(arguments)))
        return {"success": True, "message": f"{name} completed"}

    original_execute = HumanFollowSkill.execute

    async def wrapped_execute(self, ctx, arguments):
        result = await original_execute(self, ctx, arguments)
        assert recorded_calls
        return PluginSkillResult(
            success=result.success,
            summary=result.summary,
            status=result.status,
            data=result.data,
            failure_mode=result.failure_mode,
            error=result.error,
        )

    monkeypatch.setattr(service, "_invoke_robot_skill", fake_invoke_robot_skill)
    monkeypatch.setattr(HumanFollowSkill, "execute", wrapped_execute)

    async def run() -> None:
        await service._on_skill_intent(service.topics.skill_intent, intent.__dict__)
        for _ in range(20):
            await service._skill_loop_step_for_test("embodied_skills", state)
            if not state.active_runs:
                break

    asyncio.run(run())

    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]

    assert HumanFollowSkill.spec.name == "human_follow"
    assert results[-1]["status"] == "completed"
    assert results[-1]["metadata"]["implementation_kind"] == "plugin"
    assert any(name == "base_velocity_step" for name, _arguments in recorded_calls)


def test_skill_controller_terminal_event_updates_episode_state(tmp_path) -> None:
    service = _service(tmp_path)
    episode_id = "ep1"
    envelope = Envelope(
        trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="xlerobot"
    )
    intent = SkillIntent(
        envelope=envelope,
        skill_id="skill1",
        name="inspect_scene",
        objective="inspect scene",
    )
    states = RobotEpisodeStateStore(tmp_path / "episodes")
    states.ensure(episode_id, agent_id="main", robot_id="xlerobot")

    asyncio.run(service._on_skill_intent(service.topics.skill_intent, intent.__dict__))
    accepted = next(
        payload
        for topic, payload in service.bus.published  # type: ignore[attr-defined]
        if topic == service.topics.skill_event  # type: ignore[attr-defined]
    )
    states.apply_skill_event(from_payload(SkillEvent, accepted))
    assert states.load(episode_id).active_skill_phase == "accepted"  # type: ignore[union-attr]

    asyncio.run(
        service._on_status(
            service.topics.robot_status,
            RobotStatus(
                envelope=envelope,
                frame_id=12,
                state="idle",
                skill_id="skill1",
                success=True,
                metrics={"last_skill_result": {"message": "perception refreshed"}},
            ).__dict__,
        )
    )

    completed = next(
        payload
        for topic, payload in service.bus.published  # type: ignore[attr-defined]
        if topic == service.topics.skill_event and payload["phase"] == "completed"  # type: ignore[attr-defined]
    )
    state = states.apply_skill_event(from_payload(SkillEvent, completed))

    assert state is not None
    assert state.active_skill_id == "skill1"
    assert state.active_skill_phase == "completed"
    assert state.last_observation_frame == 12
    assert state.recovery_required is False


def test_skill_controller_blocks_motion_when_robot_is_degraded(tmp_path) -> None:
    service = _service(tmp_path)
    envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
    state = service.states["embodied_skills"]
    state.latest_status = RobotStatus(envelope=envelope, state="degraded", metrics={})
    intent = SkillIntent(
        envelope=envelope,
        skill_id="skill1",
        name="move_base",
        arguments={"direction": "forward", "distance_cm": 5},
        objective="move forward",
    )

    asyncio.run(service._on_skill_intent(service.topics.skill_intent, intent.__dict__))

    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]
    assert results[0]["status"] == "failed"
    assert results[0]["failure_mode"] == "precondition_failed"
    assert "degraded" in results[0]["summary"]


def test_skill_controller_blocks_motion_when_battery_low(tmp_path) -> None:
    service = _service(tmp_path)
    envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
    state = service.states["embodied_skills"]
    state.latest_status = RobotStatus(
        envelope=envelope,
        state="idle",
        metrics={"battery": {"status": "low", "voltage": 10.0}},
    )
    intent = SkillIntent(
        envelope=envelope,
        skill_id="skill1",
        name="move_base",
        arguments={"direction": "forward", "distance_cm": 5},
        objective="move forward",
    )

    asyncio.run(service._on_skill_intent(service.topics.skill_intent, intent.__dict__))

    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]
    assert results[0]["status"] == "failed"
    assert results[0]["failure_mode"] == "precondition_failed"
    assert "battery low" in results[0]["summary"]


def test_skill_controller_rejects_second_skill_while_resource_busy(tmp_path) -> None:
    service = _service(tmp_path)
    envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
    first = SkillIntent(
        envelope=envelope,
        skill_id="skill1",
        name="set_gripper",
        arguments={"action": "open"},
        objective="open gripper",
    )
    second = SkillIntent(
        envelope=envelope,
        skill_id="skill2",
        name="set_gripper",
        arguments={"action": "close"},
        objective="close gripper",
    )

    asyncio.run(service._on_skill_intent(service.topics.skill_intent, first.__dict__))
    asyncio.run(service._on_skill_intent(service.topics.skill_intent, second.__dict__))

    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]
    assert results[0]["skill_id"] == "skill2"
    assert results[0]["failure_mode"] == "resource_busy"


def test_skill_controller_accepts_non_conflicting_resource_skills(tmp_path) -> None:
    service = _service(tmp_path)
    envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
    camera = SkillIntent(
        envelope=envelope, skill_id="skill1", name="inspect_scene", objective="capture"
    )
    gripper = SkillIntent(
        envelope=envelope,
        skill_id="skill2",
        name="set_gripper",
        arguments={"action": "open"},
        objective="open gripper",
    )

    asyncio.run(service._on_skill_intent(service.topics.skill_intent, camera.__dict__))
    asyncio.run(service._on_skill_intent(service.topics.skill_intent, gripper.__dict__))

    events = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_event
    ]  # type: ignore[attr-defined]
    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]

    assert [event["skill_id"] for event in events] == ["skill1", "skill2"]
    assert [event["phase"] for event in events] == ["accepted", "accepted"]
    assert results == []
    assert set(service.states["embodied_skills"].active_runs) == {"skill1", "skill2"}
    snapshot = json.loads(
        (tmp_path / "runtime" / "skill_scheduler" / "xlerobot.json").read_text(
            encoding="utf-8"
        )
    )
    assert snapshot["resource_leases"] == {"camera": "skill1", "gripper": "skill2"}
    assert snapshot["last_decision"]["phase"] == "accepted"


def test_skill_controller_global_robot_resource_conflicts_with_specific_resource(
    tmp_path,
) -> None:
    service = _service(tmp_path)
    envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
    first = SkillIntent(
        envelope=envelope,
        skill_id="skill1",
        name="set_gripper",
        arguments={"action": "open"},
        objective="open",
    )
    second = SkillIntent(
        envelope=envelope,
        skill_id="skill2",
        name="set_gripper",
        arguments={"action": "close"},
        objective="close",
    )

    asyncio.run(service._on_skill_intent(service.topics.skill_intent, first.__dict__))
    asyncio.run(service._on_skill_intent(service.topics.skill_intent, second.__dict__))

    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]
    assert results[0]["skill_id"] == "skill2"
    assert results[0]["failure_mode"] == "resource_busy"
    assert "skill1" in results[0]["summary"]
    snapshot = json.loads(
        (tmp_path / "runtime" / "skill_scheduler" / "xlerobot.json").read_text(
            encoding="utf-8"
        )
    )
    assert snapshot["last_decision"]["reason"] == "resource_busy"
    assert snapshot["last_decision"]["conflicting_skill_id"] == "skill1"


def test_skill_controller_allows_parallel_dual_arm_runs_when_targets_differ(
    tmp_path,
) -> None:
    service = _service(tmp_path)
    envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
    left = SkillIntent(
        envelope=envelope,
        skill_id="left1",
        name="set_gripper",
        arguments={"action": "open", "arm": "left"},
        objective="open left gripper",
    )
    right = SkillIntent(
        envelope=envelope,
        skill_id="right1",
        name="set_gripper",
        arguments={"action": "open", "arm": "right"},
        objective="open right gripper",
    )

    async def accept() -> None:
        await service._on_skill_intent(service.topics.skill_intent, left.__dict__)
        await service._on_skill_intent(service.topics.skill_intent, right.__dict__)

    asyncio.run(accept())

    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]
    snapshot = json.loads(
        (tmp_path / "runtime" / "skill_scheduler" / "xlerobot.json").read_text(
            encoding="utf-8"
        )
    )

    assert results == []
    assert set(service.states["embodied_skills"].active_runs) == {"left1", "right1"}
    assert snapshot["resource_leases"] == {
        "left_gripper": "left1",
        "right_gripper": "right1",
    }


def test_skill_controller_rejects_parallel_dual_arm_runs_when_same_target_conflicts(
    tmp_path,
) -> None:
    service = _service(tmp_path)
    envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
    first = SkillIntent(
        envelope=envelope,
        skill_id="left1",
        name="set_gripper",
        arguments={"action": "open", "arm": "left"},
        objective="open left gripper",
    )
    second = SkillIntent(
        envelope=envelope,
        skill_id="left2",
        name="set_gripper",
        arguments={"action": "close", "arm": "left"},
        objective="close left gripper",
    )

    async def accept() -> None:
        await service._on_skill_intent(service.topics.skill_intent, first.__dict__)
        await service._on_skill_intent(service.topics.skill_intent, second.__dict__)

    asyncio.run(accept())

    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]
    snapshot = json.loads(
        (tmp_path / "runtime" / "skill_scheduler" / "xlerobot.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(results) == 1
    assert results[0]["failure_mode"] == "resource_busy"
    assert snapshot["last_decision"]["conflicting_resources"] == ["left_gripper"]


def test_skill_controller_interrupts_all_active_runs_before_emergency_stop(
    tmp_path,
) -> None:
    service = _service(tmp_path)
    envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
    camera = SkillIntent(
        envelope=envelope, skill_id="skill1", name="inspect_scene", objective="capture"
    )
    gripper = SkillIntent(
        envelope=envelope,
        skill_id="skill2",
        name="set_gripper",
        arguments={"action": "open"},
        objective="open gripper",
    )
    interrupt = SkillIntent(
        envelope=envelope,
        skill_id="skill3",
        name="stop_motion",
        objective="stop now",
        interrupt=True,
    )

    asyncio.run(service._on_skill_intent(service.topics.skill_intent, camera.__dict__))
    asyncio.run(service._on_skill_intent(service.topics.skill_intent, gripper.__dict__))
    asyncio.run(
        service._on_skill_intent(service.topics.skill_intent, interrupt.__dict__)
    )

    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]
    events = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_event
    ]  # type: ignore[attr-defined]

    assert {
        result["skill_id"] for result in results if result["status"] == "interrupted"
    } == {"skill1", "skill2"}
    assert events[-1]["skill_id"] == "skill3"
    assert events[-1]["phase"] == "accepted"
    assert set(service.states["embodied_skills"].active_runs) == {"skill3"}


def test_skill_loop_advances_all_non_conflicting_active_runs(tmp_path) -> None:
    service = _service(tmp_path)
    state = service.states["embodied_skills"]
    runtime = FakeRuntime()
    state.runtime = runtime
    state.latest_observation = RobotObservation(
        envelope=Envelope(robot_id="xlerobot"), frame_id=1
    )
    envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
    camera = SkillIntent(
        envelope=envelope, skill_id="skill1", name="inspect_scene", objective="capture"
    )
    gripper = SkillIntent(
        envelope=envelope,
        skill_id="skill2",
        name="set_gripper",
        arguments={"action": "open"},
        objective="open gripper",
    )

    async def run_once() -> None:
        await service._on_skill_intent(service.topics.skill_intent, camera.__dict__)
        await service._on_skill_intent(service.topics.skill_intent, gripper.__dict__)
        loop_task = asyncio.create_task(service._skill_loop("embodied_skills", state))
        while len(runtime.predicted_skill_ids) < 2:
            runtime.predicted.clear()
            await asyncio.wait_for(runtime.predicted.wait(), timeout=1.0)
        service._stop.set()
        await loop_task

    asyncio.run(run_once())

    actions = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.robot_action
    ]  # type: ignore[attr-defined]
    assert runtime.predicted_skill_ids == ["skill1", "skill2"]
    assert [action["skill_id"] for action in actions] == ["skill1", "skill2"]


def test_skill_controller_times_out_run_and_records_scheduler_failure(tmp_path) -> None:
    service = _service(tmp_path)
    state = service.states["embodied_skills"]
    state.runtime = FakeRuntime()
    state.latest_observation = RobotObservation(
        envelope=Envelope(robot_id="xlerobot"), frame_id=3
    )
    envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
    intent = SkillIntent(
        envelope=envelope,
        skill_id="skill1",
        name="set_gripper",
        arguments={"action": "open"},
        objective="open gripper",
    )

    async def run_timeout() -> None:
        timeout_seen = asyncio.Event()
        original_publish = service.bus.publish  # type: ignore[attr-defined]

        async def publish_and_signal(topic: str, payload: dict) -> None:
            await original_publish(topic, payload)
            if (
                topic == service.topics.skill_result
                and payload.get("failure_mode") == "timeout"
            ):
                timeout_seen.set()

        service.bus.publish = publish_and_signal  # type: ignore[method-assign]
        await service._on_skill_intent(service.topics.skill_intent, intent.__dict__)
        state.active_runs["skill1"].accepted_at = time.time() - 999
        loop_task = asyncio.create_task(service._skill_loop("embodied_skills", state))
        await asyncio.wait_for(timeout_seen.wait(), timeout=1.0)
        service._stop.set()
        await loop_task

    asyncio.run(run_timeout())

    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]
    snapshot = json.loads(
        (tmp_path / "runtime" / "skill_scheduler" / "xlerobot.json").read_text(
            encoding="utf-8"
        )
    )

    assert results[-1]["status"] == "failed"
    assert results[-1]["failure_mode"] == "timeout"
    assert "timed out" in results[-1]["summary"]
    assert snapshot["last_decision"]["reason"] == "timeout"
    assert snapshot["active_runs"] == []


def test_skill_controller_estimates_timeout_from_turn_base_duration(tmp_path) -> None:
    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "robots": {
                "xlerobot": {
                    "type": "xlerobot",
                    "default_angular_speed": 0.45,
                    "motion_time_scale": 2.0,
                }
            },
            "policies": {
                "embodied_skills": {
                    "type": "skill",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "settings": {"codec": "skill"},
                }
            },
        }
    )
    service = SkillControllerService(config)
    service.bus = FakeBus()  # type: ignore[assignment]
    service.events = BusEventPublisher(service.bus, service.topics)  # type: ignore[arg-type]
    intent = SkillIntent(
        envelope=Envelope(trace_id="tr1", robot_id="xlerobot"),
        skill_id="skill1",
        name="turn_base",
        arguments={"direction": "right", "angle_deg": 180},
        objective="turn around",
    )

    async def accept() -> None:
        await service._on_skill_intent(service.topics.skill_intent, intent.__dict__)

    asyncio.run(accept())

    run = service.states["embodied_skills"].active_runs["skill1"]
    expected = (3.141592653589793 / 0.45 * 2.0) + 3.0
    assert run.timeout_sec == pytest.approx(expected)
    assert run.timeout_sec > run.contract.timeout_sec


def test_skill_controller_respects_explicit_intent_timeout_over_motion_estimate(
    tmp_path,
) -> None:
    service = _service(tmp_path)
    intent = SkillIntent(
        envelope=Envelope(trace_id="tr1", robot_id="xlerobot"),
        skill_id="skill1",
        name="turn_base",
        arguments={"direction": "right", "angle_deg": 180},
        objective="turn around",
        timeout_sec=6.0,
    )

    async def accept() -> None:
        await service._on_skill_intent(service.topics.skill_intent, intent.__dict__)

    asyncio.run(accept())

    run = service.states["embodied_skills"].active_runs["skill1"]
    assert run.timeout_sec == 6.0
    assert run.timeout_override_sec is None


def test_skill_controller_rejects_move_base_without_distance(
    tmp_path,
) -> None:
    service = _service(tmp_path)
    envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
    service.states["embodied_skills"].latest_status = RobotStatus(
        envelope=envelope,
        state="idle",
        metrics={"battery": {"status": "normal", "voltage": 12.0}},
    )
    intent = SkillIntent(
        envelope=envelope,
        skill_id="skill1",
        name="move_base",
        arguments={"direction": "forward"},
        objective="move forward a bit",
    )

    async def accept() -> None:
        await service._on_skill_intent(service.topics.skill_intent, intent.__dict__)

    asyncio.run(accept())

    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]
    assert results[-1]["failure_mode"] == "invalid_arguments"
    assert "distance_cm" in results[-1]["summary"]


def test_skill_controller_rejects_turn_base_without_angle(
    tmp_path,
) -> None:
    service = _service(tmp_path)
    envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
    service.states["embodied_skills"].latest_status = RobotStatus(
        envelope=envelope,
        state="idle",
        metrics={"battery": {"status": "normal", "voltage": 12.0}},
    )
    intent = SkillIntent(
        envelope=envelope,
        skill_id="skill1",
        name="turn_base",
        arguments={"direction": "left"},
        objective="turn left a bit",
    )

    async def accept() -> None:
        await service._on_skill_intent(service.topics.skill_intent, intent.__dict__)

    asyncio.run(accept())

    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]
    assert results[-1]["failure_mode"] == "invalid_arguments"
    assert "angle_deg" in results[-1]["summary"]


def test_skill_controller_times_out_without_latest_observation(tmp_path) -> None:
    service = _service(tmp_path)
    state = service.states["embodied_skills"]
    state.runtime = FakeRuntime()
    state.latest_observation = None
    envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
    intent = SkillIntent(
        envelope=envelope, skill_id="skill1", name="inspect_scene", objective="look"
    )

    async def run_timeout() -> None:
        await service._on_skill_intent(service.topics.skill_intent, intent.__dict__)
        state.active_runs["skill1"].accepted_at = time.time() - 999
        await service._skill_loop_step_for_test("embodied_skills", state)

    asyncio.run(run_timeout())

    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]

    assert results[-1]["status"] == "failed"
    assert results[-1]["failure_mode"] == "timeout"
    assert state.active_runs == {}


def test_skill_controller_precondition_and_failure_helpers_cover_remaining_paths(
    tmp_path,
) -> None:
    service = _service(tmp_path)
    motion = service.plugin_skill_catalog.resolve("move_base", robot_type="xlerobot")
    observe = service.plugin_skill_catalog.resolve(
        "inspect_scene", robot_type="xlerobot"
    )
    emergency = service.plugin_skill_catalog.resolve(
        "stop_motion", robot_type="xlerobot"
    )

    degraded = RobotStatus(
        envelope=Envelope(robot_id="xlerobot"), state="degraded", metrics={}
    )
    critical = RobotStatus(
        envelope=Envelope(robot_id="xlerobot"),
        state="idle",
        metrics={"battery": {"status": "critical", "voltage": 9.0}},
    )

    assert service._precondition_block(observe, degraded) is None
    assert service._precondition_block(emergency, degraded) is None
    assert "battery critical" in (service._precondition_block(motion, critical) or "")

    assert (
        service._failure_mode(
            RobotStatus(envelope=Envelope(), error="safety blocked by bumper")
        )
        == "safety_blocked"
    )
    assert (
        service._failure_mode(
            RobotStatus(envelope=Envelope(), error="battery voltage too low")
        )
        == "battery_critical"
    )
    assert (
        service._failure_mode(
            RobotStatus(envelope=Envelope(), error="controller timeout")
        )
        == "timeout"
    )
    assert (
        service._failure_mode(RobotStatus(envelope=Envelope(), error="unknown issue"))
        == "execution_failed"
    )


# Regression tests.


class TestInterruptActiveResilience:
    """Fix: _interrupt_active resolves the emergency stop contract BEFORE
    popping active runs, so a catalog miss preserves existing state."""

    def test_missing_emergency_stop_preserves_active_runs(self, tmp_path) -> None:
        service = _service(tmp_path)
        envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
        gripper = SkillIntent(
            envelope=envelope,
            skill_id="skill1",
            name="set_gripper",
            arguments={"action": "open"},
            objective="open",
        )

        asyncio.run(
            service._on_skill_intent(service.topics.skill_intent, gripper.__dict__)
        )
        assert "skill1" in service.states["embodied_skills"].active_runs

        with patch.object(
            service.plugin_skill_catalog,
            "resolve",
            side_effect=KeyError("no such skill"),
        ):
            interrupt = SkillIntent(
                envelope=envelope,
                skill_id="skill_int",
                name="stop_motion",
                objective="stop",
                interrupt=True,
            )
            with pytest.raises(KeyError):
                asyncio.run(
                    service._on_skill_intent(
                        service.topics.skill_intent, interrupt.__dict__
                    )
                )

        # Fix: active run preserved 鈥?contract resolution happens before popping
        assert "skill1" in service.states["embodied_skills"].active_runs
        assert len(service.states["embodied_skills"].active_runs) == 1


class TestObserveRunDuplicateCompletion:
    """Fix: _skill_loop_step re-fetches runs from the live dict on each
    iteration, so an observe run popped by _on_status is never processed
    from a stale snapshot 鈥?no duplicate completion events."""

    def test_observe_run_not_duplicate_completed_when_status_interleaves(
        self, tmp_path
    ) -> None:
        service = _service(tmp_path)
        state = service.states["embodied_skills"]
        envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
        # Base and camera resources can run concurrently.
        motion = SkillIntent(
            envelope=envelope,
            skill_id="motion1",
            name="move_base",
            arguments={"direction": "forward", "distance_cm": 10},
            objective="move forward",
        )
        observe = SkillIntent(
            envelope=envelope, skill_id="obs1", name="inspect_scene", objective="look"
        )

        async def interleaved() -> None:
            state.runtime = FakeRuntime()
            await service._on_skill_intent(service.topics.skill_intent, motion.__dict__)
            await service._on_skill_intent(
                service.topics.skill_intent, observe.__dict__
            )
            # Yield during base execution so the camera status can complete concurrently.
            predict_barrier = asyncio.Event()
            status_done = asyncio.Event()

            class YieldingRuntime:
                @property
                def control_period_sec(self) -> float:
                    return 0.001

                async def predict(self, payload):
                    if payload.intent.skill_id == "motion1":
                        predict_barrier.set()
                        await asyncio.wait_for(status_done.wait(), timeout=2.0)
                    return PolicyRuntimeOutput(
                        action=RobotAction(
                            envelope=envelope,
                            values=[0.0],
                            skill_id=payload.intent.skill_id,
                        )
                    )

                async def close(self) -> None:
                    return None

            state.runtime = YieldingRuntime()
            state.latest_observation = RobotObservation(envelope=envelope, frame_id=2)

            async def send_status() -> None:
                await predict_barrier.wait()
                await service._on_status(
                    service.topics.robot_status,
                    RobotStatus(
                        envelope=envelope,
                        frame_id=3,
                        state="idle",
                        skill_id="obs1",
                        success=True,
                    ).__dict__,
                )
                status_done.set()

            await asyncio.gather(
                service._skill_loop_step_for_test("embodied_skills", state),
                send_status(),
            )

        asyncio.run(interleaved())

        # Fix: observe run completed exactly once by _on_status, not again by
        # a stale loop snapshot 鈥?the loop re-fetches runs from the live dict.
        obs_events = [
            p
            for t, p in service.bus.published  # type: ignore[attr-defined]
            if t == service.topics.skill_event
            and p.get("skill_id") == "obs1"  # type: ignore[attr-defined]
            and p.get("phase") == "completed"
        ]
        obs_results = [
            p
            for t, p in service.bus.published  # type: ignore[attr-defined]
            if t == service.topics.skill_result
            and p.get("skill_id") == "obs1"  # type: ignore[attr-defined]
            and p.get("status") == "completed"
        ]
        assert len(obs_events) == 1, (
            f"expected exactly 1 completion, got {len(obs_events)}"
        )
        assert len(obs_results) == 1, (
            f"expected exactly 1 completion, got {len(obs_results)}"
        )


def test_skill_loop_waits_for_robot_status_before_completing_run(tmp_path) -> None:
    service = _service(tmp_path)
    state = service.states["embodied_skills"]
    state.runtime = FakeRuntime()
    state.latest_observation = RobotObservation(
        envelope=Envelope(robot_id="xlerobot"), frame_id=2
    )
    envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
    intent = SkillIntent(
        envelope=envelope,
        skill_id="move1",
        name="move_base",
        arguments={"direction": "forward", "distance_cm": 10},
        objective="move forward",
    )

    async def run() -> None:
        await service._on_skill_intent(service.topics.skill_intent, intent.__dict__)
        await service._skill_loop_step_for_test("embodied_skills", state)

    asyncio.run(run())

    results = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_result
    ]  # type: ignore[attr-defined]
    scheduler_events = [
        payload
        for topic, payload in service.bus.published  # type: ignore[attr-defined]
        if topic == service.topics.runtime_event  # type: ignore[attr-defined]
    ]

    assert "move1" in state.active_runs
    assert results == []
    assert not any(
        payload.get("kind") == "skill_scheduler.state"
        and payload.get("payload", {}).get("last_decision", {}).get("reason")
        == "completed_without_pending_action"
        for payload in scheduler_events
    )


class TestStaleSnapshotAfterInterrupt:
    """Bug: _skill_loop_step iterates a snapshot of runs. When predict() yields
    during normal execution, _interrupt_active can process the same run
    (publish "interrupted" + pop from active_runs). The loop then resumes and
    publishes robot_action for the already-interrupted skill without checking
    the terminal flag 鈥?the robot executes a cancelled action."""

    def test_loop_publishes_action_for_already_interrupted_skill(
        self, tmp_path
    ) -> None:
        service = _service(tmp_path)
        state = service.states["embodied_skills"]
        state.latest_observation = RobotObservation(
            envelope=Envelope(robot_id="xlerobot"), frame_id=1
        )
        envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
        intent = SkillIntent(
            envelope=envelope,
            skill_id="skill1",
            name="set_gripper",
            arguments={"action": "open"},
            objective="open",
        )

        async def race() -> None:
            predict_barrier = asyncio.Event()
            interrupt_done = asyncio.Event()

            class YieldingRuntime:
                @property
                def control_period_sec(self) -> float:
                    return 0.001

                async def predict(self, payload):
                    predict_barrier.set()
                    await asyncio.wait_for(interrupt_done.wait(), timeout=2.0)
                    return PolicyRuntimeOutput(
                        action=RobotAction(
                            envelope=envelope,
                            values=[0.0],
                            skill_id=payload.intent.skill_id,
                        )
                    )

                async def close(self) -> None:
                    return None

            state.runtime = YieldingRuntime()
            await service._on_skill_intent(service.topics.skill_intent, intent.__dict__)

            async def send_interrupt() -> None:
                await predict_barrier.wait()
                interrupt = SkillIntent(
                    envelope=envelope,
                    skill_id="skill_int",
                    name="stop_motion",
                    objective="stop now",
                    interrupt=True,
                )
                await service._on_skill_intent(
                    service.topics.skill_intent, interrupt.__dict__
                )
                interrupt_done.set()

            await asyncio.gather(
                service._skill_loop_step_for_test("embodied_skills", state),
                send_interrupt(),
            )

        asyncio.run(race())

        # skill1 was interrupted mid-execution, but the loop still published a
        # robot_action for it after the interrupt completed.
        robot_actions = [
            p
            for t, p in service.bus.published  # type: ignore[attr-defined]
            if t == service.topics.robot_action and p.get("skill_id") == "skill1"  # type: ignore[attr-defined]
        ]
        interrupt_results = [
            p
            for t, p in service.bus.published  # type: ignore[attr-defined]
            if t == service.topics.skill_result
            and p.get("skill_id") == "skill1"  # type: ignore[attr-defined]
            and p.get("failure_mode") == "interrupted"
        ]
        # Fix: terminal check after predict() skips the already-interrupted skill
        assert len(interrupt_results) == 1, (
            f"expected 1 interrupt result, got {len(interrupt_results)}"
        )
        assert len(robot_actions) == 0, (
            f"robot_action should NOT be published after interrupt, got {len(robot_actions)}"
        )


class TestOnSkillIntentExceptionScope:
    """Fix: _on_skill_intent catches all exceptions, publishes failure events
    instead of letting internal errors propagate unhandled."""

    def test_acceptance_decision_exception_is_published_as_failure(
        self, tmp_path
    ) -> None:
        service = _service(tmp_path)
        envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
        intent = SkillIntent(
            envelope=envelope,
            skill_id="skill1",
            name="set_gripper",
            arguments={"action": "open"},
            objective="open",
        )

        with patch.object(
            service.contracts,
            "acceptance_decision",
            side_effect=RuntimeError("readiness provider offline"),
        ):
            asyncio.run(
                service._on_skill_intent(service.topics.skill_intent, intent.__dict__)
            )

        results = [
            p
            for t, p in service.bus.published  # type: ignore[attr-defined]
            if t == service.topics.skill_result and p.get("skill_id") == "skill1"  # type: ignore[attr-defined]
        ]
        assert len(results) == 1
        assert results[0]["status"] == "failed"
        assert results[0]["failure_mode"] == "internal_error"
        assert "readiness provider offline" in results[0]["summary"]

    def test_expand_exception_is_published_as_failure(self, tmp_path) -> None:
        service = _service(tmp_path)
        envelope = Envelope(trace_id="tr1", robot_id="xlerobot")
        intent = SkillIntent(
            envelope=envelope,
            skill_id="skill1",
            name="set_gripper",
            arguments={"action": "open"},
            objective="open",
        )

        with patch.object(
            service,
            "_execution_plan",
            side_effect=ValueError("invalid execution plan"),
        ):
            asyncio.run(
                service._on_skill_intent(service.topics.skill_intent, intent.__dict__)
            )

        results = [
            p
            for t, p in service.bus.published  # type: ignore[attr-defined]
            if t == service.topics.skill_result and p.get("skill_id") == "skill1"  # type: ignore[attr-defined]
        ]
        assert len(results) == 1
        assert results[0]["status"] == "failed"
        assert results[0]["failure_mode"] == "internal_error"
        assert "invalid execution plan" in results[0]["summary"]
