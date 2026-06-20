from __future__ import annotations

import threading
from typing import Any

from hey_robot.robots.classic import (
    BaseVelocityStepPrimitive,
    ClassicPrimitiveBackend,
    ClassicSkillExecutor,
    MoveArmJointsPrimitive,
    MoveBasePrimitive,
    PerceptionPrimitive,
    ResetPosturePrimitive,
    SetArmPosePrimitive,
    SetGripperPrimitive,
    StopMotionPrimitive,
    TurnBasePrimitive,
)
from hey_robot.robots.classic.primitives import SUPPORTED_CLASSIC_PRIMITIVES
from hey_robot.robots.xlerobot.client import XLeRobotClient
from hey_robot.skills import (
    RobotSkillAction,
    RobotSkillResult,
)


class _XLeRobotBackend(ClassicPrimitiveBackend[dict[str, Any]]):
    def __init__(self, client: XLeRobotClient) -> None:
        self.client = client
        self._base_lock = threading.Lock()
        self._velocity_watchdog: threading.Timer | None = None
        self._velocity_generation = 0

    def _cancel_velocity_watchdog(self) -> None:
        self._velocity_generation += 1
        if self._velocity_watchdog is not None:
            self._velocity_watchdog.cancel()
            self._velocity_watchdog = None

    def _schedule_velocity_stop(self, duration_ms: int) -> None:
        self._cancel_velocity_watchdog()
        generation = self._velocity_generation

        def stop_if_stale() -> None:
            with self._base_lock:
                if generation != self._velocity_generation:
                    return
                self._velocity_watchdog = None
                self.client.base_stop()

        timer = threading.Timer(max(1, duration_ms) / 1000.0, stop_if_stale)
        timer.daemon = True
        self._velocity_watchdog = timer
        timer.start()

    def close(self) -> None:
        with self._base_lock:
            self._cancel_velocity_watchdog()

    def stop_base_only(self) -> dict[str, Any]:
        with self._base_lock:
            self._cancel_velocity_watchdog()
            return self.client.base_stop()

    def on_stop_motion(
        self, primitive: StopMotionPrimitive, *, skill_name: str
    ) -> dict[str, Any]:
        _ = skill_name
        with self._base_lock:
            self._cancel_velocity_watchdog()
            base_result = (
                self.client.base_emergency_stop()
                if primitive.emergency
                else self.client.base_stop()
            )
        arm_result = self.client.arm_stop()
        return {
            "success": bool(base_result.get("success", True))
            and bool(arm_result.get("success", True)),
            "message": "motion stopped",
            "base": base_result,
            "arm": arm_result,
        }

    def on_move_base(
        self, primitive: MoveBasePrimitive, *, skill_name: str
    ) -> dict[str, Any]:
        _ = skill_name
        with self._base_lock:
            self._cancel_velocity_watchdog()
            return (
                self.client.move_backward_cm(primitive.distance_cm)
                if primitive.direction == "backward"
                else self.client.move_forward_cm(primitive.distance_cm)
            )

    def on_turn_base(
        self, primitive: TurnBasePrimitive, *, skill_name: str
    ) -> dict[str, Any]:
        _ = skill_name
        with self._base_lock:
            self._cancel_velocity_watchdog()
            return (
                self.client.turn_right_deg(primitive.angle_deg)
                if primitive.direction == "right"
                else self.client.turn_left_deg(primitive.angle_deg)
            )

    def on_base_velocity_step(
        self, primitive: BaseVelocityStepPrimitive, *, skill_name: str
    ) -> dict[str, Any]:
        _ = skill_name
        with self._base_lock:
            response = self.client.set_velocity(
                primitive.vx,
                primitive.wz,
                vy=primitive.vy,
            )
            if bool(response.get("success", False)):
                self._schedule_velocity_stop(primitive.duration_ms)
            else:
                self._cancel_velocity_watchdog()
                self.client.base_stop()
        return {
            "success": bool(response.get("success", False)),
            "message": "base velocity applied"
            if response.get("success", False)
            else str(response.get("message") or "base velocity failed"),
            "velocity": response,
            "watchdog_ms": primitive.duration_ms,
        }

    def on_set_arm_pose(
        self, primitive: SetArmPosePrimitive, *, skill_name: str
    ) -> dict[str, Any]:
        _ = skill_name
        return self.client.move_named_pose(primitive.pose_name, arm_name=primitive.arm)

    def on_move_arm_joints(
        self, primitive: MoveArmJointsPrimitive, *, skill_name: str
    ) -> dict[str, Any]:
        _ = skill_name
        return (
            self.client.set_joints_delta(primitive.joints, arm_name=primitive.arm)
            if primitive.delta_mode
            else self.client.set_joints(primitive.joints, arm_name=primitive.arm)
        )

    def on_set_gripper(
        self, primitive: SetGripperPrimitive, *, skill_name: str
    ) -> dict[str, Any]:
        _ = skill_name
        if primitive.action == "open":
            return self.client.open_gripper(arm_name=primitive.arm)
        if primitive.action == "close":
            return self.client.close_gripper(arm_name=primitive.arm)
        return self.client.set_gripper_opening_pct(
            float(primitive.opening_pct or 0.0),
            arm_name=primitive.arm,
        )

    def on_reset_posture(
        self, primitive: ResetPosturePrimitive, *, skill_name: str
    ) -> dict[str, Any]:
        _ = skill_name
        with self._base_lock:
            self._cancel_velocity_watchdog()
            base_result = self.client.base_stop()
        pose_result = self.client.move_named_pose("rest", arm_name=primitive.arm)
        gripper_result = self.client.open_gripper(arm_name=primitive.arm)
        return {
            "success": all(
                bool(result.get("success", True))
                for result in (base_result, pose_result, gripper_result)
            ),
            "message": "robot reset posture requested",
            "base": base_result,
            "arm": pose_result,
            "gripper": gripper_result,
        }

    def on_perception(
        self, primitive: PerceptionPrimitive, *, skill_name: str
    ) -> dict[str, Any]:
        _ = skill_name
        return {
            "success": False,
            "message": f"{primitive.skill_name} must be handled by RobotRuntime perception pipeline",
            "skill": primitive.skill_name,
            "arguments": dict(primitive.arguments or {}),
            "failure_mode": "wrong_execution_boundary",
        }


class XLeRobotSkillExecutor:
    supported_skills = SUPPORTED_CLASSIC_PRIMITIVES

    def __init__(self, client: XLeRobotClient) -> None:
        self.client = client
        self._backend = _XLeRobotBackend(client)
        self._executor = ClassicSkillExecutor(self._backend)

    def close(self) -> None:
        self._backend.close()

    def stop_base_only(self) -> dict[str, Any]:
        return self._backend.stop_base_only()

    def execute(self, action: RobotSkillAction) -> RobotSkillResult:
        try:
            response = self._executor.execute(action)
        except Exception as exc:
            return RobotSkillResult(
                success=False,
                message=f"{type(exc).__name__}: {exc}",
                data={"skill": action.name, "error": str(exc)},
            )
        return RobotSkillResult.from_response(response)
