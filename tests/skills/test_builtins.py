from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hey_robot.config import DeploymentConfig
from hey_robot.protocol import Envelope, RobotObservation
from hey_robot.robot_runtime.clients import RobotActionResult
from hey_robot.skills import (
    ResourceManager,
    SkillCommand,
    SkillContext,
    SkillRunner,
    load_skill_registry,
    registry_from_config,
)


@dataclass
class Sink:
    events: list = field(default_factory=list)

    async def emit(self, event) -> None:
        self.events.append(event)


class Robot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any], str]] = []

    async def observe(self, robot_id: str) -> RobotObservation:
        return RobotObservation(Envelope(robot_id=robot_id), frame_id=12, task="desk")

    async def execute(
        self,
        robot_id,
        action,
        arguments,
        *,
        run_id,
        expected_frame_id=None,
    ):
        del expected_frame_id
        self.calls.append((robot_id, action, arguments, run_id))
        return RobotActionResult(True, f"{action} done", frame_id=9)


class Models:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.requests: list[dict[str, Any]] = []

    async def infer(
        self,
        capability,
        request,
        *,
        run_id,
        robot_id,
        timeout_sec=None,
    ):
        from hey_robot.foundation.clients.models import ModelInferenceResult

        self.requests.append(
            {
                "capability": capability,
                "request": request,
                "run_id": run_id,
                "robot_id": robot_id,
                "timeout_sec": timeout_sec,
            }
        )
        return ModelInferenceResult(True, "policy action", data=self.data)


def _command(name: str, arguments: dict[str, Any]) -> SkillCommand:
    return SkillCommand(
        envelope=Envelope(robot_id="mock0"),
        run_id="run-1",
        task_id="task-1",
        robot_id="mock0",
        name=name,
        arguments=arguments,
    )


def _runner(
    robot: Robot,
    sink: Sink,
    *,
    models: Models | None = None,
    implementations: dict[str, str] | None = None,
) -> SkillRunner:
    registry = load_skill_registry(
        ("hey_robot.skills.builtins",), implementations=implementations
    )
    return SkillRunner(
        registry,
        resources=ResourceManager(),
        events=sink,
        context_factory=lambda command: SkillContext(
            run_id=command.run_id,
            task_id=command.task_id,
            robot_id=command.robot_id,
            robot=robot,
            models=models,
        ),
    )


async def test_native_builtin_registry_loads_core_skill_names() -> None:
    registry = load_skill_registry(("hey_robot.skills.builtins",))

    names = {skill.name for skill in registry.list()}

    assert {
        "inspect_scene",
        "move_base",
        "turn_base",
        "stop_motion",
        "set_gripper",
        "set_arm_pose",
        "move_arm_joints",
    }.issubset(names)


async def test_native_perception_skill_uses_robot_observation() -> None:
    robot = Robot()
    sink = Sink()
    result = await _runner(robot, sink).execute(_command("inspect_scene", {}))

    assert result.success is True
    assert result.summary == "desk"
    assert result.data["frame_id"] == 12
    assert result.evidence_ids == ("observation:run-1",)


async def test_native_base_and_manipulation_skills_call_robot_actions() -> None:
    robot = Robot()
    sink = Sink()
    runner = _runner(robot, sink)

    await runner.execute(_command("move_base", {"direction": "forward"}))
    await runner.execute(
        SkillCommand(
            envelope=Envelope(robot_id="mock0"),
            run_id="run-2",
            task_id="task-1",
            robot_id="mock0",
            name="set_gripper",
            arguments={"action": "open"},
        )
    )

    assert robot.calls == [
        (
            "mock0",
            "move_base",
            {"direction": "forward", "distance_cm": 20.0},
            "run-1",
        ),
        ("mock0", "set_gripper", {"action": "open"}, "run-2"),
    ]


async def test_native_vla_manipulate_uses_model_router_and_robot_client() -> None:
    robot = Robot()
    models = Models(
        {"action": {"name": "set_gripper", "arguments": {"action": "close"}}}
    )
    sink = Sink()

    result = await _runner(robot, sink, models=models).execute(
        _command("manipulate", {"task_prompt": "close gripper"})
    )

    assert result.success is True
    assert result.data["requires_reobservation"] is True
    assert models.requests[0]["capability"] == "manipulate"
    assert models.requests[0]["request"]["observation"]["frame_id"] == 12
    assert robot.calls == [("mock0", "set_gripper", {"action": "close"}, "run-1")]


async def test_native_tabletop_implementation_selection() -> None:
    robot = Robot()
    models = Models(
        {"action": {"name": "set_gripper", "arguments": {"action": "close"}}}
    )
    sink = Sink()

    result = await _runner(
        robot,
        sink,
        models=models,
        implementations={"pick": "vla"},
    ).execute(_command("pick", {"object": "cup"}))

    assert result.success is True
    assert models.requests[0]["request"]["task_prompt"] == "grasp cup"
    assert robot.calls == [("mock0", "set_gripper", {"action": "close"}, "run-1")]


def test_native_registry_from_config_uses_skill_implementations() -> None:
    config = DeploymentConfig.from_dict(
        {
            "skills": {
                "modules": ["hey_robot.skills.builtins"],
                "tools": ["pick"],
                "implementations": {"pick": "vla"},
            }
        }
    )

    registry = registry_from_config(config)

    assert registry.get("pick").required_models == ("manipulate",)
