from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import SkillResult


@dataclass(frozen=True)
class VLNOptionRequest:
    capability: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class VLNOptionResult:
    success: bool
    summary: str
    termination_reason: str
    planner_history: tuple[dict[str, Any], ...] = ()
    steps: tuple[dict[str, Any], ...] = ()
    command: dict[str, Any] | None = None
    failure_mode: str | None = None
    error: str | None = None

    def to_skill_result(self) -> SkillResult:
        return SkillResult(
            self.success,
            self.summary,
            "completed" if self.success else "failed",
            data={
                "vln": self.planner_history[-1] if self.planner_history else {},
                "vln_history": list(self.planner_history),
                "steps": list(self.steps),
                "termination_reason": self.termination_reason,
                "command": self.command,
            },
            failure_mode=self.failure_mode,
            error=self.error,
        )


@dataclass
class _ExecutionTrace:
    planners: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)

    def result(
        self,
        success: bool,
        summary: str,
        termination_reason: str,
        *,
        command: dict[str, Any] | None = None,
        failure_mode: str | None = None,
        error: str | None = None,
    ) -> VLNOptionResult:
        return VLNOptionResult(
            success,
            summary,
            termination_reason,
            planner_history=tuple(self.planners),
            steps=tuple(self.steps),
            command=command,
            failure_mode=failure_mode,
            error=error,
        )


class VLNOptionRunner:
    """Run one bounded observe-plan-act VLN option inside a native Skill."""

    async def run(
        self, ctx: SkillContext, request: VLNOptionRequest
    ) -> VLNOptionResult:
        arguments = dict(request.arguments)
        if ctx.models is None:
            return VLNOptionResult(
                False,
                "VLN model router 不可用。",
                "model_unavailable",
                failure_mode="model_service_unavailable",
                error="model router is unavailable",
            )
        execute_primitives = bool(arguments.get("execute_primitives", True))
        if execute_primitives and ctx.robot is None:
            return VLNOptionResult(
                False,
                "执行 VLN primitive 需要 RobotClient。",
                "robot_unavailable",
                failure_mode="robot_client_unavailable",
                error="robot client is unavailable",
            )

        max_steps = max(1, int(arguments.get("max_steps", 30)))
        fresh_timeout = float(arguments.get("fresh_observation_timeout_sec", 2.0))
        observation = await ctx.observe(timeout_sec=fresh_timeout)
        trace = _ExecutionTrace()
        look_down_requested = False

        for step_index in range(max_steps):
            ctx.raise_if_cancelled()
            payload = _vln_payload(
                arguments,
                observation,
                reset_policy=step_index == 0,
                policy_session_id=ctx.run_id,
                look_down=look_down_requested,
            )
            inference = await ctx.models.infer(
                request.capability,
                payload,
                run_id=ctx.run_id,
                robot_id=ctx.robot_id,
                timeout_sec=arguments.get("model_timeout_sec"),
            )
            ctx.raise_if_cancelled()
            planner = _planner_data(inference.data)
            trace.planners.append(planner)
            if not inference.success:
                return trace.result(
                    False,
                    inference.summary,
                    "model_failed",
                    failure_mode=inference.failure_mode or "model_failed",
                    error=inference.error,
                )
            if _environment_done(planner):
                return trace.result(True, inference.summary, "environment_done")

            if _requires_secondary_observation(planner):
                if step_index + 1 >= max_steps:
                    return trace.result(
                        False,
                        "VLN 请求 secondary observation，但没有剩余 planning step。",
                        "secondary_observation_required",
                        failure_mode="vln_secondary_observation_required",
                    )
                look_down_requested = True
                await ctx.progress(
                    (step_index + 1) / max_steps,
                    "VLN 请求 secondary observation，准备重新观察。",
                )
                stale = await self._fresh_observation(
                    ctx,
                    trace,
                    observation,
                    fresh_timeout=fresh_timeout,
                    message="VLN secondary observation 超时。",
                )
                if isinstance(stale, VLNOptionResult):
                    return stale
                observation = stale
                continue

            try:
                command = planner_to_action(planner)
            except ValueError as exc:
                return trace.result(
                    False,
                    "VLN planner 未返回可执行 primitive。",
                    "no_valid_goal",
                    failure_mode="vln_no_valid_goal",
                    error=str(exc),
                )
            if not execute_primitives:
                return trace.result(
                    True, inference.summary, "plan_only", command=command
                )

            robot = ctx.robot
            assert robot is not None
            action = await robot.execute(
                ctx.robot_id,
                command["name"],
                command["arguments"],
                run_id=ctx.run_id,
                expected_frame_id=observation.frame_id,
            )
            trace.steps.append(
                {
                    "step_index": step_index,
                    "primitive": command["name"],
                    "arguments": command["arguments"],
                    "reason": command["reason"],
                    "success": action.success,
                    "summary": action.summary,
                    "frame_id": action.frame_id,
                    "data": dict(action.data),
                }
            )
            if not action.success:
                return trace.result(
                    False,
                    action.summary,
                    "action_failed",
                    failure_mode=action.failure_mode or "action_failed",
                    error=action.error,
                )
            if _environment_done(action.data):
                return trace.result(True, action.summary, "environment_done")
            await ctx.progress(
                (step_index + 1) / max_steps,
                f"VLN 已执行 {command['name']} ({step_index + 1}/{max_steps})。",
            )
            if command["name"] == "stop_motion":
                return trace.result(True, "VLN planner 已确认到达目标。", "model_done")
            if step_index + 1 < max_steps:
                fresh = await self._fresh_observation(
                    ctx,
                    trace,
                    observation,
                    fresh_timeout=fresh_timeout,
                    message="VLN action 后未获得 fresh observation。",
                )
                if isinstance(fresh, VLNOptionResult):
                    return fresh
                observation = fresh

        return trace.result(
            False,
            f"VLN 已达到 max_steps={max_steps}，尚未确认到达目标。",
            "max_steps",
            failure_mode="budget_exhausted",
        )

    @staticmethod
    async def _fresh_observation(
        ctx: SkillContext,
        trace: _ExecutionTrace,
        previous: Any,
        *,
        fresh_timeout: float,
        message: str,
    ) -> Any | VLNOptionResult:
        try:
            return await ctx.observe(
                after_frame_id=previous.frame_id,
                timeout_sec=fresh_timeout,
            )
        except TimeoutError:
            return trace.result(
                False,
                message,
                "observation_stale",
                failure_mode="observation_stale",
            )


def _vln_payload(
    arguments: dict[str, Any],
    observation: Any,
    *,
    reset_policy: bool,
    policy_session_id: str,
    look_down: bool,
) -> dict[str, Any]:
    internal = {
        "execute_primitives",
        "max_steps",
        "model_timeout_sec",
        "fresh_observation_timeout_sec",
    }
    payload = {key: value for key, value in arguments.items() if key not in internal}
    payload["observation"] = {
        "frame_id": observation.frame_id,
        "timestamp": observation.envelope.timestamp,
        "images": [asdict(image) for image in observation.images],
        "proprioception": list(observation.proprioception),
        "raw": dict(observation.raw),
    }
    payload["policy_session_id"] = policy_session_id
    payload["reset_policy"] = reset_policy
    if look_down:
        payload["look_down"] = True
    return payload


def _planner_data(data: dict[str, Any]) -> dict[str, Any]:
    for key in ("vln", "planner"):
        planner = data.get(key)
        if isinstance(planner, dict):
            return dict(planner)
    return dict(data)


def _requires_secondary_observation(planner: dict[str, Any]) -> bool:
    return bool(planner.get("requires_secondary_observation")) or (
        planner.get("mode") == "look_down_required"
    )


def _environment_done(data: dict[str, Any]) -> bool:
    if bool(data.get("environment_done")):
        return True
    environment = data.get("environment")
    return isinstance(environment, dict) and bool(environment.get("done"))


def planner_to_action(planner: dict[str, Any]) -> dict[str, Any]:
    if bool(planner.get("stop")) or planner.get("mode") == "stop":
        return {"name": "stop_motion", "arguments": {}, "reason": "planner_stop"}
    action_code = planner.get("action_code")
    if isinstance(action_code, int | float):
        code = int(action_code)
        if code == 1:
            return {
                "name": "move_base",
                "arguments": {
                    "direction": "forward",
                    "distance_cm": float(planner.get("forward_distance_cm") or 25.0),
                },
                "reason": "planner_discrete_forward",
            }
        if code in {2, 3}:
            discrete_heading = float(planner.get("heading_deg") or 0.0)
            return {
                "name": "turn_base",
                "arguments": {
                    "direction": "left" if code == 2 else "right",
                    "angle_deg": abs(discrete_heading) or 15.0,
                },
                "reason": "planner_discrete_turn",
            }
    heading = planner.get("heading_deg")
    if isinstance(heading, int | float):
        if abs(heading) < 10.0:
            return {
                "name": "move_base",
                "arguments": {"direction": "forward", "distance_cm": 15.0},
                "reason": "heading_centered",
            }
        return {
            "name": "turn_base",
            "arguments": {
                "direction": "right" if heading > 0 else "left",
                "angle_deg": min(abs(heading), 30.0),
            },
            "reason": "planner_heading",
        }
    pixel_goal = planner.get("pixel_goal")
    if isinstance(pixel_goal, list | tuple) and len(pixel_goal) >= 2:
        width = max(1.0, float(planner.get("image_width") or 640.0))
        offset = float(pixel_goal[1]) - width / 2.0
        if abs(offset) <= width * 0.125:
            return {
                "name": "move_base",
                "arguments": {"direction": "forward", "distance_cm": 15.0},
                "reason": "pixel_centered",
            }
        angle = 10.0 + 20.0 * min(abs(offset) / (width / 2.0), 1.0)
        return {
            "name": "turn_base",
            "arguments": {
                "direction": "right" if offset > 0 else "left",
                "angle_deg": angle,
            },
            "reason": "pixel_off_center",
        }
    raise ValueError("planner output lacks stop, heading_deg, or pixel_goal")
