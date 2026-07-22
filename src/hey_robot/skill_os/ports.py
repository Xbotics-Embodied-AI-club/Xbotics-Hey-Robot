from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from hey_robot.foundation.clients import ServiceInvocationResult

RobotInvoker = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
NativeActionInvoker = Callable[
    [list[float], int, list[float] | None], Awaitable[dict[str, Any]]
]
ModelServiceInvoker = Callable[
    [str, dict[str, Any]],
    Awaitable[ServiceInvocationResult],
]


class RobotActionPort:
    def __init__(
        self,
        invoke: RobotInvoker,
        native_action_invoke: NativeActionInvoker | None = None,
    ) -> None:
        self._invoke = invoke
        self._native_action_invoke = native_action_invoke

    async def run(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await self._invoke(name, dict(arguments or {}))

    async def apply_policy_action(
        self,
        values: list[float],
        *,
        expected_frame_id: int,
        raw_values: list[float] | None = None,
    ) -> dict[str, Any]:
        if self._native_action_invoke is None:
            raise RuntimeError("native policy actions are unavailable for this runtime")
        return await self._native_action_invoke(
            list(values), int(expected_frame_id), raw_values
        )

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

    async def arm_get_state(self, **arguments: Any) -> dict[str, Any]:
        return await self.run("arm_get_state", arguments)

    async def arm_solve_position_ik(self, **arguments: Any) -> dict[str, Any]:
        return await self.run("arm_solve_position_ik", arguments)

    async def sim_locate_object(self, **arguments: Any) -> dict[str, Any]:
        return await self.run("sim_locate_object", arguments)

    async def sim_get_object_state(self, **arguments: Any) -> dict[str, Any]:
        return await self.run("sim_get_object_state", arguments)


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


class ModelServicePort:
    def __init__(self, invoke: ModelServiceInvoker) -> None:
        self._invoke = invoke

    async def call(
        self, name: str, arguments: dict[str, Any]
    ) -> ServiceInvocationResult:
        return await self._invoke(name, dict(arguments))
