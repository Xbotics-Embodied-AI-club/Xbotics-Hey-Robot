from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hey_robot.protocol import Envelope
from hey_robot.robot_api import RobotActionResult
from hey_robot.skills import (
    ResourceManager,
    SkillCommand,
    SkillContext,
    SkillRegistry,
    SkillRunner,
)
from hey_robot.skills.builtins.safety import RESET_POSTURE, STOP_MOTION, register


@dataclass
class Sink:
    events: list = field(default_factory=list)

    async def emit(self, event) -> None:
        self.events.append(event)


class Robot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any], str]] = []

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
        return RobotActionResult(True, f"{action} done")


def _command(name: str, arguments: dict[str, Any]) -> SkillCommand:
    return SkillCommand(
        envelope=Envelope(robot_id="mock0"),
        run_id="run-1",
        task_id="task-1",
        robot_id="mock0",
        name=name,
        arguments=arguments,
    )


async def test_native_safety_skills_call_robot_client() -> None:
    robot = Robot()
    registry = SkillRegistry()
    register(registry)
    sink = Sink()
    runner = SkillRunner(
        registry,
        resources=ResourceManager(),
        events=sink,
        context_factory=lambda command: SkillContext(
            run_id=command.run_id,
            task_id=command.task_id,
            robot_id=command.robot_id,
            robot=robot,
        ),
    )

    result = await runner.execute(_command("stop_motion", {}))

    assert result.success is True
    assert robot.calls == [("mock0", "stop_motion", {"emergency": False}, "run-1")]
    assert sink.events[-1].phase == "completed"
    assert STOP_MOTION.required_actions == ("stop_motion",)
    assert RESET_POSTURE.required_actions == ("reset_posture",)
