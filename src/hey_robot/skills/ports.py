from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from hey_robot.capability.runtime import CapabilityExecutionResult

RobotInvoker = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
CapabilityInvoker = Callable[
    [str, dict[str, Any]],
    Awaitable[CapabilityExecutionResult],
]


class RobotActionPort:
    def __init__(self, invoke: RobotInvoker) -> None:
        self._invoke = invoke

    async def run(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await self._invoke(name, dict(arguments or {}))

    async def move_base(self, **arguments: Any) -> dict[str, Any]:
        return await self.run("move_base", arguments)

    async def turn_base(self, **arguments: Any) -> dict[str, Any]:
        return await self.run("turn_base", arguments)

    async def base_velocity_step(self, **arguments: Any) -> dict[str, Any]:
        return await self.run("base_velocity_step", arguments)

    async def stop_motion(self, **arguments: Any) -> dict[str, Any]:
        return await self.run("stop_motion", arguments)

    async def set_arm_pose(self, **arguments: Any) -> dict[str, Any]:
        return await self.run("set_arm_pose", arguments)

    async def move_arm_joints(self, **arguments: Any) -> dict[str, Any]:
        return await self.run("move_arm_joints", arguments)

    async def set_gripper(self, **arguments: Any) -> dict[str, Any]:
        return await self.run("set_gripper", arguments)

    async def reset_posture(self, **arguments: Any) -> dict[str, Any]:
        return await self.run("reset_posture", arguments)


class PerceptionPort:
    def __init__(self, robot: RobotActionPort) -> None:
        self._robot = robot

    async def run(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await self._robot.run(name, arguments)

    async def inspect_scene(self, **arguments: Any) -> dict[str, Any]:
        return await self.run("inspect_scene", arguments)

    async def look_around(self, **arguments: Any) -> dict[str, Any]:
        return await self.run("look_around", arguments)

    async def detect_marker(self, **arguments: Any) -> dict[str, Any]:
        return await self.run("detect_marker", arguments)

    async def human_follow(self, **arguments: Any) -> dict[str, Any]:
        return await self.run("human_follow", arguments)


class CapabilityPort:
    def __init__(self, invoke: CapabilityInvoker) -> None:
        self._invoke = invoke

    async def call(
        self, name: str, arguments: dict[str, Any]
    ) -> CapabilityExecutionResult:
        return await self._invoke(name, dict(arguments))
