from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from hey_robot.config import DeploymentConfig
from hey_robot.protocol import Envelope, ImageRef, RobotObservation
from hey_robot.skill_os.base import BaseSkill, SkillResult, SkillSpec
from hey_robot.skill_os.context import SkillContext
from hey_robot.skill_os.registry import load_skill_registry, registry_from_config
from hey_robot.skill_os.runtime import SkillRuntime


def test_registry_loads_builtin_module_and_defaults_to_agent_visible_surface() -> None:
    registry = load_skill_registry()

    assert "inspect_scene" in registry.names()
    assert "reset_posture" in registry.names()
    assert "move_base" not in registry.names()


class _RobotAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def move_base(self, **arguments) -> dict:
        self.calls.append(("move_base", dict(arguments)))
        return {"ok": True, "primitive": "move_base"}

    async def turn_base(self, **arguments) -> dict:
        self.calls.append(("turn_base", dict(arguments)))
        return {"ok": True, "primitive": "turn_base"}

    async def stop_motion(self, **arguments) -> dict:
        self.calls.append(("stop_motion", dict(arguments)))
        return {"ok": True, "primitive": "stop_motion"}


def test_runtime_runs_plugin_backed_builtin_skill() -> None:
    robot = _RobotAPI()
    registry = load_skill_registry(enabled=("move_base",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "move_base",
            {"direction": "forward", "distance_cm": 10},
            context_factory=lambda invoke: SkillContext(robot=robot, invoke=invoke),
        )
    )

    assert result.success is True
    assert robot.calls == [("move_base", {"direction": "forward", "distance_cm": 10})]


def test_runtime_returns_failed_result_for_invalid_arguments() -> None:
    robot = _RobotAPI()
    registry = load_skill_registry(enabled=("move_base",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "move_base",
            {"direction": "forward"},
            context_factory=lambda invoke: SkillContext(robot=robot, invoke=invoke),
        )
    )

    assert result.success is False
    assert result.status == "failed"
    assert result.failure_mode == "invalid_arguments"
    assert "distance_cm" in (result.summary or "")
    assert robot.calls == []


def test_registry_from_config_loads_custom_module_and_filters_enabled_surface(
    monkeypatch,
) -> None:
    module_name = "tests.fake_registry_plugin"
    module = types.ModuleType(module_name)

    class VisibleSkill(BaseSkill):
        spec = SkillSpec(
            name="visible_plugin_skill",
            description="Visible test plugin.",
            agent_visible=True,
        )

        async def execute(self, ctx, arguments):
            del ctx, arguments
            return SkillResult(success=True, summary="visible")

    class HiddenSkill(BaseSkill):
        spec = SkillSpec(
            name="hidden_plugin_skill",
            description="Hidden test plugin.",
            agent_visible=False,
        )

        async def execute(self, ctx, arguments):
            del ctx, arguments
            return SkillResult(success=True, summary="hidden")

    def register_skills(registry) -> None:
        registry.register(VisibleSkill())
        registry.register(HiddenSkill())

    setattr(module, "register_skills", register_skills)
    monkeypatch.setitem(sys.modules, module_name, module)

    config = DeploymentConfig.from_dict(
        {
            "skills": {
                "modules": [module_name],
                "enabled": ["visible_plugin_skill"],
            }
        }
    )

    registry = registry_from_config(config)

    assert registry.names() == ("visible_plugin_skill",)
    assert registry.names(enabled_only=False) == (
        "visible_plugin_skill",
        "hidden_plugin_skill",
    )


def test_runtime_uses_context_factory() -> None:
    class EchoRobot:
        def __init__(self, label: str) -> None:
            self.label = label

        async def move_base(self, **arguments) -> str:
            return f"{self.label}:{arguments['distance_cm']}"

    class EchoMoveSkill(BaseSkill):
        spec = SkillSpec(
            name="echo_move",
            description="Echo context-specific robot result.",
            input_schema={
                "type": "object",
                "properties": {"distance_cm": {"type": "number"}},
                "required": ["distance_cm"],
            },
            agent_visible=True,
        )

        async def execute(self, ctx, arguments):
            summary = await ctx.robot.move_base(**arguments)
            return SkillResult(success=True, summary=summary)

    registry = load_skill_registry(enabled=())
    registry.register(EchoMoveSkill())
    registry = registry.configure(enabled=("echo_move",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "echo_move",
            {"distance_cm": 12},
            context_factory=lambda invoke: SkillContext(
                robot=EchoRobot("override"),
                invoke=invoke,
            ),
        )
    )

    assert result.success is True
    assert result.summary == "override:12"


def test_runtime_wraps_plugin_exception_as_internal_error() -> None:
    class BrokenSkill(BaseSkill):
        spec = SkillSpec(
            name="broken_skill",
            description="Raise during execution.",
            agent_visible=True,
        )

        async def execute(self, ctx, arguments):
            del ctx, arguments
            raise RuntimeError("plugin exploded")

    registry = load_skill_registry(enabled=())
    registry.register(BrokenSkill())
    registry = registry.configure(enabled=("broken_skill",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "broken_skill",
            context_factory=lambda invoke: SkillContext(invoke=invoke),
        )
    )

    assert result.success is False
    assert result.failure_mode == "internal_error"
    assert result.error == "plugin exploded"


def test_runtime_executes_vla_manipulation_skill() -> None:
    class ModelServiceAPI:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call(self, name: str, arguments: dict):
            self.calls.append((name, dict(arguments)))
            return SimpleNamespace(
                success=True,
                summary="object picked",
                status="completed",
                failure_mode=None,
                error=None,
                metrics={"vla": {"task_done": True}},
            )

    class FakeRobot:
        async def move_arm(self, **_arguments):
            return {"success": True}

        async def move_gripper(self, **_arguments):
            return {"success": True}

        async def stop_arm(self, **_arguments):
            return {"success": True}

        async def stop_motion(self, **_arguments):
            return {"success": True}

    model_services = ModelServiceAPI()
    registry = load_skill_registry(enabled=("vla_manipulation",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "vla_manipulation",
            {"task_prompt": "Pick up the red cup.", "max_steps": 1},
            context_factory=lambda invoke: SkillContext(
                model_services=model_services,
                invoke=invoke,
                robot=FakeRobot(),
            ),
        )
    )

    assert result.success is True
    assert model_services.calls == [
        (
            "vla_manipulation",
            {
                "skill_name": "vla_manipulation",
                "task_prompt": "Pick up the red cup.",
                "vla_step": 0,
                "policy_session_id": None,
            },
        )
    ]


def test_pick_object_routes_to_required_vla_model_service() -> None:
    class ModelServiceAPI:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call(self, name: str, arguments: dict):
            self.calls.append((name, dict(arguments)))
            return SimpleNamespace(
                success=True,
                summary="object picked",
                status="completed",
                failure_mode=None,
                error=None,
                metrics={"vla": {"task_done": True}},
            )

    class FakeRobot:
        async def stop_motion(self, **_arguments):
            return {"success": True}

    model_services = ModelServiceAPI()
    registry = load_skill_registry(enabled=("pick_object",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "pick_object",
            {"task_prompt": "Pick up the red cup.", "max_steps": 1},
            context_factory=lambda invoke: SkillContext(
                model_services=model_services,
                invoke=invoke,
                robot=FakeRobot(),
                skill_id="pick-1",
            ),
        )
    )

    assert result.success is True
    assert model_services.calls[0] == (
        "vla_manipulation",
        {
            "skill_name": "pick_object",
            "task_prompt": "Pick up the red cup.",
            "vla_step": 0,
            "policy_session_id": "pick-1",
        },
    )


def test_vla_max_steps_exhausted_fails() -> None:
    class ModelServiceAPI:
        async def call(self, _name: str, _arguments: dict):
            return SimpleNamespace(
                success=True,
                summary="still running",
                status="completed",
                failure_mode=None,
                error=None,
                metrics={"vla": {"task_done": False}},
            )

    class FakeRobot:
        async def stop_motion(self, **_arguments):
            return {"success": True}

    registry = load_skill_registry(enabled=("vla_manipulation",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "vla_manipulation",
            {"task_prompt": "Pick up the red cup.", "max_steps": 1},
            context_factory=lambda invoke: SkillContext(
                model_services=ModelServiceAPI(),
                invoke=invoke,
                robot=FakeRobot(),
            ),
        )
    )

    assert result.success is False
    assert result.failure_mode == "vla_max_steps_exhausted"


def test_vla_adapter_consumes_typed_action_chunk_policy_result() -> None:
    from hey_robot.skill_os.builtins.manipulation_adapter import (
        vla_output_to_primitives,
    )

    primitives = vla_output_to_primitives(
        {
            "policy_result": {
                "kind": "action_chunk",
                "actions": [
                    {
                        "joints": {"shoulder_pan": 0.25, "elbow_flex": 0.5},
                        "gripper": 0.3,
                    }
                ],
            }
        }
    )

    assert [item.primitive for item in primitives] == [
        "move_arm_joints",
        "set_gripper",
    ]
    assert primitives[0].arguments == {
        "joints": {"shoulder_pan": 0.25, "elbow_flex": 0.5},
        "mode": "absolute",
    }
    assert primitives[1].arguments == {"opening_pct": 30.0}


def test_vla_adapter_consumes_full_action_chunk_horizon() -> None:
    from hey_robot.skill_os.builtins.manipulation_adapter import (
        vla_output_to_primitives,
    )

    primitives = vla_output_to_primitives(
        {
            "policy_result": {
                "kind": "action_chunk",
                "actions": [
                    {"joints": {"shoulder_pan": 0.1}, "gripper": 1.0},
                    {"joints": {"shoulder_pan": 0.2}, "gripper": 0.2},
                ],
            }
        }
    )

    assert [item.primitive for item in primitives] == [
        "move_arm_joints",
        "set_gripper",
        "move_arm_joints",
        "set_gripper",
    ]
    assert primitives[2].arguments == {
        "joints": {"shoulder_pan": 0.2},
        "mode": "absolute",
    }
    assert "chunk action 2" in primitives[2].reason


def test_vla_skill_injects_observation_and_consumes_typed_policy_result() -> None:
    class ModelServiceAPI:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call(self, name: str, arguments: dict):
            self.calls.append((name, dict(arguments)))
            return SimpleNamespace(
                success=True,
                summary="action chunk produced",
                status="completed",
                failure_mode=None,
                error=None,
                metrics={
                    "policy_result": {
                        "kind": "action_chunk",
                        "action_space": "xlerobot_single_arm_joint",
                        "embodiment": "xlerobot",
                        "horizon": 2,
                        "dt": 0.033,
                        "done": True,
                        "actions": [
                            {"joints": {"shoulder_pan": 0.1}, "gripper": 1.0},
                            {"joints": {"shoulder_pan": 0.2}, "gripper": 0.2},
                        ],
                    },
                    "vla": {"backend_mode": "action_chunk_policy"},
                },
            )

    class FakeRobot:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def move_arm_joints(self, **arguments):
            self.calls.append(("move_arm_joints", dict(arguments)))
            return {"success": True}

        async def set_gripper(self, **arguments):
            self.calls.append(("set_gripper", dict(arguments)))
            return {"success": True}

    observation = RobotObservation(
        envelope=Envelope(robot_id="xlerobot"),
        frame_id=7,
        images=[
            ImageRef(
                uri="media://local/images/xlerobot/wrist/frame.jpg",
                camera="wrist",
            )
        ],
        proprioception=[0.1, 0.2],
    )
    model_services = ModelServiceAPI()
    robot = FakeRobot()
    registry = load_skill_registry(enabled=("pick_object",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "pick_object",
            {"task_prompt": "Pick up the red cup.", "max_steps": 1},
            context_factory=lambda invoke: SkillContext(
                model_services=model_services,
                invoke=invoke,
                robot=robot,
                skill_id="pick-typed",
                observation=observation,
                current_observation=lambda: observation,
            ),
        )
    )

    assert result.success is True
    assert model_services.calls[0][0] == "vla_manipulation"
    sent = model_services.calls[0][1]
    assert sent["observation"]["frame_id"] == 7
    assert sent["observation"]["images"][0]["camera"] == "wrist"
    assert sent["observation"]["proprioception"] == [0.1, 0.2]
    assert sent["policy_session_id"] == "pick-typed"
    assert robot.calls == [
        ("move_arm_joints", {"joints": {"shoulder_pan": 0.1}, "mode": "absolute"}),
        ("set_gripper", {"opening_pct": 100.0}),
        ("move_arm_joints", {"joints": {"shoulder_pan": 0.2}, "mode": "absolute"}),
        ("set_gripper", {"opening_pct": 20.0}),
    ]


def test_runtime_executes_navigate_to_skill_through_capability() -> None:
    class ModelServiceAPI:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call(self, name: str, arguments: dict):
            self.calls.append((name, dict(arguments)))
            return SimpleNamespace(
                success=True,
                summary="VLN planner produced pixel_goal",
                status="completed",
                failure_mode=None,
                error=None,
                metrics={"vln": {"mode": "pixel_goal", "pixel_goal": [240, 320]}},
            )

    model_services = ModelServiceAPI()
    registry = load_skill_registry(enabled=("navigate_to",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "navigate_to",
            {"target": "desk", "execute_primitives": False},
            context_factory=lambda invoke: SkillContext(
                model_services=model_services,
                invoke=invoke,
            ),
        )
    )

    assert result.success is True
    assert result.data == {"vln": {"mode": "pixel_goal", "pixel_goal": [240, 320]}}
    assert model_services.calls == [
        ("navigate_to", {"target": "desk", "reset_policy": True})
    ]


def test_navigate_to_skill_respects_execute_primitives_false() -> None:
    class ModelServiceAPI:
        async def call(self, name: str, arguments: dict):
            del name, arguments
            return SimpleNamespace(
                success=True,
                summary="VLN planner produced pixel_goal",
                status="completed",
                failure_mode=None,
                error=None,
                metrics={"vln": {"mode": "pixel_goal", "pixel_goal": [240, 320]}},
            )

    robot = _RobotAPI()
    registry = load_skill_registry(enabled=("navigate_to",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "navigate_to",
            {"target": "desk", "execute_primitives": False},
            context_factory=lambda invoke: SkillContext(
                robot=robot,
                model_services=ModelServiceAPI(),
                invoke=invoke,
            ),
        )
    )

    assert result.success is True
    assert robot.calls == []


def test_navigate_to_skill_executes_center_pixel_as_forward_step() -> None:
    class ModelServiceAPI:
        async def call(self, name: str, arguments: dict):
            del name, arguments
            return SimpleNamespace(
                success=True,
                summary="VLN planner produced pixel_goal",
                status="completed",
                failure_mode=None,
                error=None,
                metrics={"vln": {"mode": "pixel_goal", "pixel_goal": [240, 320]}},
            )

    robot = _RobotAPI()
    registry = load_skill_registry(enabled=("navigate_to",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "navigate_to",
            {"target": "desk", "execute_primitives": True},
            context_factory=lambda invoke: SkillContext(
                robot=robot,
                model_services=ModelServiceAPI(),
                invoke=invoke,
            ),
        )
    )

    assert result.success is True
    assert robot.calls == [("move_base", {"direction": "forward", "distance_cm": 15.0})]
    assert result.data["steps"][0]["primitive"] == "move_base"
    assert result.data["steps"][0]["success"] is True
    assert result.data["steps"][0]["primitive_result"] == {
        "ok": True,
        "primitive": "move_base",
    }


def test_navigate_to_skill_executes_off_center_pixel_as_turn() -> None:
    class ModelServiceAPI:
        async def call(self, name: str, arguments: dict):
            del name, arguments
            return SimpleNamespace(
                success=True,
                summary="VLN planner produced pixel_goal",
                status="completed",
                failure_mode=None,
                error=None,
                metrics={"vln": {"mode": "pixel_goal", "pixel_goal": [240, 32]}},
            )

    robot = _RobotAPI()
    registry = load_skill_registry(enabled=("navigate_to",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "navigate_to",
            {"target": "desk", "execute_primitives": True},
            context_factory=lambda invoke: SkillContext(
                robot=robot,
                model_services=ModelServiceAPI(),
                invoke=invoke,
            ),
        )
    )

    assert result.success is True
    assert robot.calls[0][0] == "turn_base"
    assert robot.calls[0][1]["direction"] == "left"
    assert result.data["steps"][0]["primitive"] == "turn_base"


def test_approach_object_skill_executes_stop_from_vln_planner() -> None:
    class ModelServiceAPI:
        async def call(self, name: str, arguments: dict):
            del name, arguments
            return SimpleNamespace(
                success=True,
                summary="VLN planner requested stop",
                status="completed",
                failure_mode=None,
                error=None,
                metrics={"vln": {"mode": "stop", "stop": True}},
            )

    robot = _RobotAPI()
    registry = load_skill_registry(enabled=("approach_object",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "approach_object",
            {"target": "cup", "execute_primitives": True},
            context_factory=lambda invoke: SkillContext(
                robot=robot,
                model_services=ModelServiceAPI(),
                invoke=invoke,
            ),
        )
    )

    assert result.success is True
    assert robot.calls == [("stop_motion", {})]
    assert result.data["steps"][0]["primitive"] == "stop_motion"


def test_navigate_to_skill_does_not_execute_primitive_when_capability_fails() -> None:
    class ModelServiceAPI:
        async def call(self, name: str, arguments: dict):
            del name, arguments
            return SimpleNamespace(
                success=False,
                summary="VLN planner failed",
                status="failed",
                failure_mode="vln_no_valid_goal",
                error="no valid goal",
                metrics={"vln": {"mode": "none"}},
            )

    robot = _RobotAPI()
    registry = load_skill_registry(enabled=("navigate_to",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "navigate_to",
            {"target": "desk", "execute_primitives": True},
            context_factory=lambda invoke: SkillContext(
                robot=robot,
                model_services=ModelServiceAPI(),
                invoke=invoke,
            ),
        )
    )

    assert result.success is False
    assert result.failure_mode == "vln_no_valid_goal"
    assert robot.calls == []


def test_navigate_to_skill_requires_robot_for_primitive_execution() -> None:
    class ModelServiceAPI:
        async def call(self, name: str, arguments: dict):
            del name, arguments
            return SimpleNamespace(
                success=True,
                summary="VLN planner produced pixel_goal",
                status="completed",
                failure_mode=None,
                error=None,
                metrics={"vln": {"mode": "pixel_goal", "pixel_goal": [240, 320]}},
            )

    registry = load_skill_registry(enabled=("navigate_to",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "navigate_to",
            {"target": "desk", "execute_primitives": True},
            context_factory=lambda invoke: SkillContext(
                model_services=ModelServiceAPI(),
                invoke=invoke,
            ),
        )
    )

    assert result.success is False
    assert result.failure_mode == "robot_runtime_unavailable"


def test_navigate_to_skill_records_multistep_progress_and_refreshes_observation() -> (
    None
):
    class ModelServiceAPI:
        def __init__(self) -> None:
            self.calls = 0

        async def call(self, name: str, arguments: dict):
            del name, arguments
            self.calls += 1
            metrics = (
                {"vln": {"mode": "pixel_goal", "pixel_goal": [240, 32]}}
                if self.calls == 1
                else {"vln": {"mode": "stop", "stop": True}}
            )
            return SimpleNamespace(
                success=True,
                summary="VLN planner step",
                status="completed",
                failure_mode=None,
                error=None,
                metrics=metrics,
            )

    robot = _RobotAPI()
    model_services = ModelServiceAPI()
    invocations: list[tuple[str, dict]] = []
    progress_events: list[dict] = []
    registry = load_skill_registry(enabled=("navigate_to",))
    runtime = SkillRuntime(registry)

    async def invoke(name: str, arguments: dict | None = None):
        invocations.append((name, dict(arguments or {})))

    async def progress(**kwargs):
        progress_events.append(dict(kwargs))

    result = __import__("asyncio").run(
        runtime.execute(
            "navigate_to",
            {"target": "desk", "execute_primitives": True, "max_steps": 2},
            context_factory=lambda _invoke: SkillContext(
                robot=robot,
                model_services=model_services,
                invoke=invoke,
                progress=progress,
            ),
        )
    )

    assert result.success is True
    assert [call[0] for call in robot.calls] == ["turn_base", "stop_motion"]
    assert invocations == [("inspect_scene", {"camera": "front"})]
    assert [step["primitive"] for step in result.data["steps"]] == [
        "turn_base",
        "stop_motion",
    ]
    assert [event["step"] for event in progress_events] == [
        "planning",
        "executed",
        "planning",
        "executed",
    ]
    assert progress_events[0]["metadata"]["ux"]["primitive"] == "turn_base"


def test_navigate_to_skill_handles_look_down_secondary_observation() -> None:
    class ModelServiceAPI:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call(self, name: str, arguments: dict):
            self.calls.append((name, dict(arguments)))
            metrics = (
                {
                    "vln": {
                        "mode": "look_down_required",
                        "requires_secondary_observation": True,
                    }
                }
                if len(self.calls) == 1
                else {"vln": {"mode": "pixel_goal", "pixel_goal": [240, 320]}}
            )
            return SimpleNamespace(
                success=True,
                summary="VLN planner step",
                status="completed",
                failure_mode=None,
                error=None,
                metrics=metrics,
            )

    robot = _RobotAPI()
    model_services = ModelServiceAPI()
    invocations: list[tuple[str, dict]] = []
    progress_events: list[dict] = []
    registry = load_skill_registry(enabled=("navigate_to",))
    runtime = SkillRuntime(registry)

    async def invoke(name: str, arguments: dict | None = None):
        invocations.append((name, dict(arguments or {})))

    async def progress(**kwargs):
        progress_events.append(dict(kwargs))

    result = __import__("asyncio").run(
        runtime.execute(
            "navigate_to",
            {
                "target": "desk",
                "camera": "front",
                "execute_primitives": True,
                "max_steps": 2,
            },
            context_factory=lambda _invoke: SkillContext(
                robot=robot,
                model_services=model_services,
                invoke=invoke,
                progress=progress,
            ),
        )
    )

    assert result.success is True
    assert model_services.calls[0][1]["reset_policy"] is True
    assert model_services.calls[1][1]["reset_policy"] is False
    assert model_services.calls[1][1]["look_down"] is True
    assert invocations == [("inspect_scene", {"camera": "front", "look_down": True})]
    assert robot.calls == [("move_base", {"direction": "forward", "distance_cm": 15.0})]
    assert "secondary_observation" in [event["step"] for event in progress_events]


def test_navigate_to_skill_fails_when_look_down_requested_without_remaining_step() -> (
    None
):
    class ModelServiceAPI:
        async def call(self, name: str, arguments: dict):
            del name, arguments
            return SimpleNamespace(
                success=True,
                summary="VLN planner requested look-down",
                status="completed",
                failure_mode=None,
                error=None,
                metrics={
                    "vln": {
                        "mode": "look_down_required",
                        "requires_secondary_observation": True,
                    }
                },
            )

    registry = load_skill_registry(enabled=("navigate_to",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "navigate_to",
            {"target": "desk", "execute_primitives": True, "max_steps": 1},
            context_factory=lambda invoke: SkillContext(
                robot=_RobotAPI(),
                model_services=ModelServiceAPI(),
                invoke=invoke,
            ),
        )
    )

    assert result.success is False
    assert result.failure_mode == "vln_secondary_observation_required"


def test_navigate_to_skill_returns_structured_failure_when_primitive_fails() -> None:
    class ModelServiceAPI:
        async def call(self, name: str, arguments: dict):
            del name, arguments
            return SimpleNamespace(
                success=True,
                summary="VLN planner produced pixel_goal",
                status="completed",
                failure_mode=None,
                error=None,
                metrics={"vln": {"mode": "pixel_goal", "pixel_goal": [240, 320]}},
            )

    class FailingRobot:
        async def move_base(self, **arguments):
            del arguments
            raise RuntimeError("base driver rejected command")

        async def turn_base(self, **arguments):
            del arguments

        async def stop_motion(self, **arguments):
            del arguments

    registry = load_skill_registry(enabled=("navigate_to",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "navigate_to",
            {"target": "desk", "execute_primitives": True},
            context_factory=lambda invoke: SkillContext(
                robot=FailingRobot(),
                model_services=ModelServiceAPI(),
                invoke=invoke,
            ),
        )
    )

    assert result.success is False
    assert result.failure_mode == "primitive_execution_failed"
    assert result.error == "base driver rejected command"
    assert result.data["steps"][0]["primitive"] == "move_base"
    assert result.data["steps"][0]["success"] is False


def test_navigate_to_skill_injects_latest_observation_image() -> None:
    class ModelServiceAPI:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call(self, name: str, arguments: dict):
            self.calls.append((name, dict(arguments)))
            return SimpleNamespace(
                success=True,
                summary="VLN planner produced pixel_goal",
                status="completed",
                failure_mode=None,
                error=None,
                metrics={"vln": {"mode": "pixel_goal", "pixel_goal": [240, 320]}},
            )

    observation = RobotObservation(
        envelope=Envelope(robot_id="xlerobot"),
        frame_id=42,
        images=[
            ImageRef(
                uri="media://local/images/xlerobot/wrist/frame.jpg", camera="wrist"
            ),
            ImageRef(
                uri="media://local/images/xlerobot/front/frame.jpg", camera="front"
            ),
        ],
    )
    model_services = ModelServiceAPI()
    registry = load_skill_registry(enabled=("navigate_to",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "navigate_to",
            {
                "target": "desk",
                "camera": "front",
                "execute_primitives": False,
                "observation": {
                    "frame_id": 42,
                    "images": [
                        {
                            "uri": "media://local/images/xlerobot/front/frame.jpg",
                            "camera": "front",
                            "width": None,
                            "height": None,
                            "timestamp": None,
                            "content_type": None,
                            "size_bytes": None,
                            "sha256": None,
                            "metadata": {},
                        }
                    ],
                },
            },
            context_factory=lambda invoke: SkillContext(
                model_services=model_services,
                observation=observation,
                current_observation=lambda: observation,
                invoke=invoke,
            ),
        )
    )

    assert result.success is True
    sent = model_services.calls[0][1]
    assert sent["observation"]["frame_id"] == 42
    assert sent["observation"]["images"] == [
        {
            "uri": "media://local/images/xlerobot/front/frame.jpg",
            "camera": "front",
            "width": None,
            "height": None,
            "timestamp": None,
            "content_type": None,
            "size_bytes": None,
            "sha256": None,
            "metadata": {},
        }
    ]


def test_navigate_to_skill_auto_injects_current_observation_image() -> None:
    class ModelServiceAPI:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call(self, name: str, arguments: dict):
            self.calls.append((name, dict(arguments)))
            return SimpleNamespace(
                success=True,
                summary="VLN planner produced pixel_goal",
                status="completed",
                failure_mode=None,
                error=None,
                metrics={"vln": {"mode": "pixel_goal", "pixel_goal": [240, 320]}},
            )

    observation = RobotObservation(
        envelope=Envelope(robot_id="xlerobot"),
        frame_id=43,
        images=[
            ImageRef(
                uri="media://local/images/xlerobot/wrist/frame.jpg", camera="wrist"
            ),
            ImageRef(
                uri="media://local/images/xlerobot/front/frame.jpg", camera="front"
            ),
        ],
        proprioception=[0.3, 0.4],
    )
    model_services = ModelServiceAPI()
    registry = load_skill_registry(enabled=("navigate_to",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "navigate_to",
            {"target": "desk", "camera": "front", "execute_primitives": False},
            context_factory=lambda invoke: SkillContext(
                model_services=model_services,
                observation=observation,
                current_observation=lambda: observation,
                invoke=invoke,
            ),
        )
    )

    assert result.success is True
    sent = model_services.calls[0][1]
    assert sent["observation"]["frame_id"] == 43
    assert sent["observation"]["images"] == [
        {
            "uri": "media://local/images/xlerobot/front/frame.jpg",
            "camera": "front",
            "width": None,
            "height": None,
            "timestamp": None,
            "content_type": None,
            "size_bytes": None,
            "sha256": None,
            "metadata": {},
        }
    ]
    assert sent["observation"]["proprioception"] == [0.3, 0.4]


def test_navigate_to_skill_keeps_explicit_image_path() -> None:
    class ModelServiceAPI:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call(self, name: str, arguments: dict):
            self.calls.append((name, dict(arguments)))
            return SimpleNamespace(
                success=True,
                summary="VLN planner produced pixel_goal",
                status="completed",
                failure_mode=None,
                error=None,
                metrics={},
            )

    observation = RobotObservation(
        envelope=Envelope(robot_id="xlerobot"),
        frame_id=42,
        images=[
            ImageRef(
                uri="media://local/images/xlerobot/front/frame.jpg", camera="front"
            )
        ],
    )
    model_services = ModelServiceAPI()
    registry = load_skill_registry(enabled=("navigate_to",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "navigate_to",
            {
                "target": "desk",
                "image_path": "D:/tmp/front.png",
                "execute_primitives": False,
            },
            context_factory=lambda invoke: SkillContext(
                model_services=model_services,
                observation=observation,
                invoke=invoke,
            ),
        )
    )

    assert result.success is True
    assert model_services.calls == [
        (
            "navigate_to",
            {
                "target": "desk",
                "image_path": "D:/tmp/front.png",
                "reset_policy": True,
            },
        )
    ]


def test_registry_rejects_duplicate_provides() -> None:
    registry = load_skill_registry(enabled=())
    duplicate = registry.get("inspect_scene", enabled_only=False).skill

    assert duplicate is not None
    with pytest.raises(ValueError, match="duplicate skill: inspect_scene"):
        registry.register(duplicate)


def test_robot_skill_catalog_exposes_capability_semantics() -> None:
    catalog = load_skill_registry().robot_skill_catalog()

    turn_base = catalog.get("turn_base")
    inspect_scene = catalog.get("inspect_scene")

    assert turn_base.capability_type == "base_turn"
    assert turn_base.evidence_outputs == ("base_turn_action_result",)
    assert inspect_scene.capability_type == "scene_observation"
    assert "base_turn_action_result" in inspect_scene.cannot_satisfy
    assert catalog.get("detect_marker").evidence_outputs == ("marker_detection_result",)

    navigate_to = catalog.get("navigate_to")
    approach_object = catalog.get("approach_object")

    assert navigate_to.capability_type == "semantic_navigation"
    assert navigate_to.evidence_outputs[0] == "vln_planner_result"
    assert navigate_to.required_model_service == "navigate_to"
    assert approach_object.capability_type == "object_approach"
    assert approach_object.evidence_outputs[0] == "vln_planner_result"
    assert approach_object.required_model_service == "approach_object"
