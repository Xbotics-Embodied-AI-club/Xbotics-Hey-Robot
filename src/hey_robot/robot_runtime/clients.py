"""Robot client boundary used by native skill implementations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from hey_robot.protocol import Envelope, RobotObservation, RobotSkillAction, SkillIntent


@dataclass(frozen=True)
class RobotActionSpec:
    name: str
    parameters: dict[str, Any]
    resources: tuple[str, ...] = ()
    motion: bool = False


@dataclass(frozen=True)
class RobotClientCapabilities:
    robot_id: str
    actions: tuple[RobotActionSpec, ...] = ()
    cameras: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotActionResult:
    success: bool
    summary: str
    status: str = "completed"
    failure_mode: str | None = None
    error: str | None = None
    frame_id: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


class RobotClient(Protocol):
    async def capabilities(self, robot_id: str) -> RobotClientCapabilities: ...

    async def observe(self, robot_id: str) -> RobotObservation: ...

    async def execute(
        self,
        robot_id: str,
        action: str,
        arguments: dict[str, Any],
        *,
        run_id: str,
        expected_frame_id: int | None = None,
    ) -> RobotActionResult: ...

    async def stop(self, robot_id: str, *, reason: str) -> None: ...


class LocalRobotClient:
    """Adapt an in-process RobotRuntime to the native Skill RobotClient boundary."""

    def __init__(self, runtimes: dict[str, Any]) -> None:
        self._runtimes = runtimes

    async def capabilities(self, robot_id: str) -> RobotClientCapabilities:
        runtime = self._runtime(robot_id)
        capabilities = await runtime.capabilities()
        actions = tuple(
            RobotActionSpec(name, {}, motion=True)
            for name in capabilities.metadata.get("supported_skills", ())
            if isinstance(name, str)
        )
        return RobotClientCapabilities(
            robot_id=robot_id,
            actions=actions,
            cameras=tuple(capabilities.cameras),
            metadata=dict(capabilities.metadata),
        )

    async def observe(self, robot_id: str) -> RobotObservation:
        return await self._runtime(robot_id).observe()

    async def execute(
        self,
        robot_id: str,
        action: str,
        arguments: dict[str, Any],
        *,
        run_id: str,
        expected_frame_id: int | None = None,
    ) -> RobotActionResult:
        runtime = self._runtime(robot_id)
        robot_action = RobotSkillAction(action, dict(arguments)).to_robot_action(
            SkillIntent(
                envelope=Envelope(robot_id=robot_id),
                skill_id=run_id,
                task_id=run_id,
                intent_kind="observation" if action == "inspect_scene" else "skill",
                name=action,
                arguments=dict(arguments),
                objective=f"execute {action}",
            )
        )
        if expected_frame_id is not None:
            robot_action = replace(
                robot_action,
                metadata={
                    **dict(robot_action.metadata),
                    "expected_frame_id": expected_frame_id,
                },
            )
        status = await runtime.apply_action(robot_action)
        last_result = status.metrics.get("last_skill_result")
        data = dict(last_result) if isinstance(last_result, dict) else {}
        success = status.success is not False and not status.error
        if isinstance(last_result, dict) and "success" in last_result:
            success = bool(last_result.get("success"))
        summary = str(
            data.get("summary")
            or data.get("message")
            or status.error
            or f"{action} completed"
        )
        return RobotActionResult(
            success,
            summary,
            status="completed" if success else "failed",
            failure_mode=data.get("failure_mode")
            if isinstance(data.get("failure_mode"), str)
            else None,
            error=status.error,
            frame_id=status.frame_id,
            data=data,
        )

    async def stop(self, robot_id: str, *, reason: str) -> None:
        await self.execute(robot_id, "stop_motion", {"reason": reason}, run_id="stop")

    def _runtime(self, robot_id: str) -> Any:
        try:
            return self._runtimes[robot_id]
        except KeyError as exc:
            raise KeyError(f"unknown robot runtime: {robot_id}") from exc
