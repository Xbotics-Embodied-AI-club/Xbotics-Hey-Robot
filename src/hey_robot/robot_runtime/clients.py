"""Robot client boundary used by native skill implementations."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any

from hey_robot.protocol import (
    Envelope,
    RobotAction,
    RobotObservation,
    RobotSkillAction,
    SkillIntent,
)
from hey_robot.robot_api import (
    RobotActionResult,
    RobotActionSpec,
    RobotClientCapabilities,
)
from hey_robot.robot_runtime.runtime import RobotRuntime


class LocalRobotClient:
    """Adapt an in-process RobotRuntime to the native Skill RobotClient boundary."""

    def __init__(self, runtimes: dict[str, RobotRuntime]) -> None:
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

    async def observe(
        self,
        robot_id: str,
        *,
        after_frame_id: int | None = None,
        timeout_sec: float | None = None,
    ) -> RobotObservation:
        runtime = self._runtime(robot_id)
        deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(
                    f"fresh observation timed out after frame {after_frame_id}"
                )
            observation = await asyncio.wait_for(runtime.observe(), timeout=remaining)
            if after_frame_id is None or observation.frame_id > after_frame_id:
                return observation
            await asyncio.sleep(min(0.01, remaining or 0.01))

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
        intent = SkillIntent(
            envelope=Envelope(robot_id=robot_id),
            skill_id=run_id,
            task_id=run_id,
            intent_kind="observation" if action == "inspect_scene" else "skill",
            name=action,
            arguments=dict(arguments),
            objective=f"execute {action}",
        )
        if action == "embodiment_native_action":
            values = arguments.get("values")
            if not isinstance(values, list):
                raise ValueError("embodiment_native_action requires list values")
            robot_action = RobotAction(
                envelope=intent.envelope,
                values=[float(value) for value in values],
                skill_id=run_id,
                task_id=run_id,
                metadata={
                    "action_type": "embodiment_native",
                    "action_space": arguments.get("action_space"),
                    "embodiment": arguments.get("embodiment"),
                    "raw_action": list(arguments.get("raw_values") or values),
                    "action_clipped": list(arguments.get("raw_values") or values)
                    != values,
                },
            )
        else:
            robot_action = RobotSkillAction(action, dict(arguments)).to_robot_action(
                intent
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
        observation: RobotObservation | None = None
        observation_error: str | None = None
        try:
            candidate = await runtime.observe()
            if status.frame_id is None or candidate.frame_id >= status.frame_id:
                observation = candidate
            else:
                observation_error = (
                    "stale post-action observation: "
                    f"frame={candidate.frame_id} status_frame={status.frame_id}"
                )
        except Exception as exc:
            observation_error = str(exc)
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
            observation=observation,
            observation_error=observation_error,
        )

    async def stop(self, robot_id: str, *, reason: str) -> None:
        await self.execute(robot_id, "stop_motion", {"reason": reason}, run_id="stop")

    async def emergency_stop(self, robot_id: str, *, reason: str) -> None:
        await self._runtime(robot_id).emergency_stop(reason=reason)

    def _runtime(self, robot_id: str) -> RobotRuntime:
        try:
            return self._runtimes[robot_id]
        except KeyError as exc:
            raise KeyError(f"unknown robot runtime: {robot_id}") from exc
