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


class SequencedModels(Models):
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        super().__init__({})
        self._outputs = iter(outputs)

    async def infer(
        self,
        capability: str,
        request: dict[str, Any],
        *,
        run_id: str,
        robot_id: str,
        timeout_sec: float | None = None,
    ):
        result = await super().infer(
            capability,
            request,
            run_id=run_id,
            robot_id=robot_id,
            timeout_sec=timeout_sec,
        )
        return type(result)(
            success=result.success,
            summary=result.summary,
            data=next(self._outputs),
            failure_mode=result.failure_mode,
            error=result.error,
        )


class FreshRobot(Robot):
    def __init__(self) -> None:
        super().__init__()
        self.frame_id = 12

    async def observe(self, robot_id: str) -> RobotObservation:
        return RobotObservation(
            Envelope(robot_id=robot_id), frame_id=self.frame_id, task="desk"
        )

    async def execute(
        self,
        robot_id: str,
        action: str,
        arguments: dict[str, Any],
        *,
        run_id: str,
        expected_frame_id: int | None = None,
    ):
        self.frame_id += 1
        result = await super().execute(
            robot_id,
            action,
            arguments,
            run_id=run_id,
            expected_frame_id=expected_frame_id,
        )
        return RobotActionResult(
            result.success,
            result.summary,
            frame_id=self.frame_id,
            data=result.data,
        )


class DockRobot(Robot):
    def __init__(self) -> None:
        super().__init__()
        self.held_object: str | None = None

    async def execute(
        self,
        robot_id: str,
        action: str,
        arguments: dict[str, Any],
        *,
        run_id: str,
        expected_frame_id: int | None = None,
    ):
        del expected_frame_id
        self.calls.append((robot_id, action, arguments, run_id))
        if action == "sim_locate_object":
            data = {
                "operation_success": True,
                "samples": [[0.2, 0.1, 0.2]],
                "grasp_axis": [0.0, 0.0, 1.0],
            }
        elif action == "arm_solve_position_ik":
            data = {"operation_success": True, "joint_positions": [0.0] * 5}
        elif action == "set_gripper":
            if arguments.get("action") == "close":
                self.held_object = "wand"
            elif arguments.get("action") == "open":
                self.held_object = None
            data = {
                "held_object": self.held_object,
                "welds": {"wand": self.held_object == "wand"},
            }
        elif action == "sim_get_object_state":
            data = {
                "held_object": self.held_object,
                "welds": {"wand": self.held_object == "wand"},
                "dock_target": [0.04, 0.133, 0.72],
            }
        else:
            data = {}
        return RobotActionResult(True, f"{action} done", data=data)


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


async def test_native_vla_manipulate_reobserves_between_bounded_steps() -> None:
    robot = FreshRobot()
    models = SequencedModels(
        [
            {"action": {"name": "set_gripper", "arguments": {"action": "close"}}},
            {
                "action": {"name": "set_gripper", "arguments": {"action": "open"}},
                "done": True,
            },
        ]
    )
    sink = Sink()

    result = await _runner(robot, sink, models=models).execute(
        _command("manipulate", {"task_prompt": "exercise gripper", "max_steps": 2})
    )

    assert result.success is True
    assert result.data["termination_reason"] == "vla_done"
    assert len(result.data["steps"]) == 2
    assert [
        request["request"]["observation"]["frame_id"] for request in models.requests
    ] == [
        12,
        13,
    ]
    assert [event.phase for event in sink.events] == [
        "accepted",
        "running",
        "progress",
        "progress",
        "completed",
    ]


async def test_native_vln_navigation_runs_bounded_observe_plan_act_loop() -> None:
    robot = FreshRobot()
    models = SequencedModels(
        [
            {"vln": {"mode": "pixel_goal", "pixel_goal": [240, 320]}},
            {"vln": {"mode": "stop", "stop": True}},
        ]
    )
    sink = Sink()

    result = await _runner(robot, sink, models=models).execute(
        _command("navigate_to", {"target": "desk", "max_steps": 2})
    )

    assert result.success is True
    assert result.data["termination_reason"] == "model_stop"
    assert [call[1] for call in robot.calls] == ["move_base", "stop_motion"]
    assert [request["request"]["reset_policy"] for request in models.requests] == [
        True,
        False,
    ]
    assert [
        request["request"]["observation"]["frame_id"] for request in models.requests
    ] == [
        12,
        13,
    ]


async def test_native_dock_skills_use_robot_client_primitives() -> None:
    robot = DockRobot()
    sink = Sink()
    runner = _runner(robot, sink)

    picked = await runner.execute(_command("pick_wand_from_dock", {}))
    placed = await runner.execute(
        SkillCommand(
            envelope=Envelope(robot_id="mock0"),
            run_id="run-dock-place",
            task_id="task-1",
            robot_id="mock0",
            name="place_wand_to_dock",
            arguments={},
        )
    )

    assert picked.success is True
    assert placed.success is True
    assert robot.held_object is None
    assert "sim_locate_object" in [call[1] for call in robot.calls]
    assert "sim_get_object_state" in [call[1] for call in robot.calls]


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
