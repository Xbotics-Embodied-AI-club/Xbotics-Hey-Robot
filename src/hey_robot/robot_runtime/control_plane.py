from __future__ import annotations

import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from hey_robot.protocol import RobotAction, RobotSkillAction, RobotStatus, SkillIntent


@dataclass(frozen=True)
class ControlPlaneDecision:
    allowed: bool
    reason: str | None = None
    failure_mode: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionBufferEntry:
    action_id: str
    skill_id: str
    submitted_at: float
    deadline_at: float | None
    action_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RobotControlPlane:
    """Runtime-side control boundary for typed policy outputs and primitives."""

    def __init__(self, *, max_buffer_size: int = 32) -> None:
        self.max_buffer_size = max(1, int(max_buffer_size))
        self.action_buffer: deque[ActionBufferEntry] = deque(
            maxlen=self.max_buffer_size
        )
        self.current_action_id: str | None = None
        self.last_watchdog: dict[str, Any] | None = None
        self.preemptions: list[dict[str, Any]] = []

    async def apply_action(
        self,
        action: RobotAction,
        *,
        apply_fn: Callable[[RobotAction], Awaitable[RobotStatus]],
        stop_fn: Callable[[RobotAction], Awaitable[RobotStatus]] | None = None,
    ) -> RobotStatus:
        decision = self.evaluate(action)
        if not decision.allowed:
            return RobotStatus(
                envelope=action.envelope,
                skill_id=action.skill_id,
                success=False,
                error=decision.reason,
                metrics={
                    "last_skill_result": {
                        "success": False,
                        "message": decision.reason,
                        "failure_mode": decision.failure_mode,
                    },
                    "control_plane": decision.metadata,
                },
            )

        if bool(action.metadata.get("preempt", False)) and stop_fn is not None:
            status = await stop_fn(action)
            self.preemptions.append(
                {
                    "action_id": action.action_id,
                    "skill_id": action.skill_id,
                    "success": status.success,
                    "error": status.error,
                }
            )

        entry = self._buffer_entry(action)
        self.action_buffer.append(entry)
        self.current_action_id = action.action_id
        started_at = time.time()
        status = await apply_fn(action)
        self.last_watchdog = {
            "action_id": action.action_id,
            "skill_id": action.skill_id,
            "duration_sec": round(time.time() - started_at, 6),
            "deadline_at": entry.deadline_at,
            "deadline_missed": entry.deadline_at is not None
            and time.time() > entry.deadline_at,
            "success": status.success,
        }
        if self.current_action_id == action.action_id:
            self.current_action_id = None
        return RobotStatus(
            envelope=status.envelope,
            frame_id=status.frame_id,
            state=status.state,
            task=status.task,
            skill_id=status.skill_id,
            success=status.success,
            error=status.error,
            metrics={
                **dict(status.metrics),
                "control_plane": self.snapshot(),
            },
        )

    async def apply_policy_result(
        self,
        policy_result: dict[str, Any],
        *,
        intent: SkillIntent,
        apply_fn: Callable[[RobotAction], Awaitable[RobotStatus]],
        stop_fn: Callable[[RobotAction], Awaitable[RobotStatus]] | None = None,
    ) -> RobotStatus:
        actions = self.map_policy_result(policy_result, intent=intent)
        if not actions:
            return RobotStatus(
                envelope=intent.envelope,
                skill_id=intent.skill_id,
                success=False,
                error="policy result did not produce runtime actions",
                metrics={
                    "last_skill_result": {
                        "success": False,
                        "message": "policy result did not produce runtime actions",
                        "failure_mode": "empty_policy_action",
                    }
                },
            )
        status: RobotStatus | None = None
        for action in actions:
            status = await self.apply_action(
                action,
                apply_fn=apply_fn,
                stop_fn=stop_fn,
            )
            if status.success is False:
                return status
        assert status is not None
        return status

    def map_policy_result(
        self, policy_result: dict[str, Any], *, intent: SkillIntent
    ) -> list[RobotAction]:
        kind = str(policy_result.get("kind") or "")
        if kind == "action_chunk":
            return self._map_action_chunk(policy_result, intent=intent)
        if kind == "local_goal":
            return self._map_local_goal(policy_result, intent=intent)
        if kind == "whole_body_reference":
            return [self._whole_body_reference_action(policy_result, intent=intent)]
        return []

    def evaluate(self, action: RobotAction) -> ControlPlaneDecision:
        deadline_at = _deadline_at(action)
        now = time.time()
        if deadline_at is not None and now > deadline_at:
            return ControlPlaneDecision(
                allowed=False,
                reason="control action deadline expired before execution",
                failure_mode="control_deadline_expired",
                metadata={
                    "action_id": action.action_id,
                    "deadline_at": deadline_at,
                    "now": now,
                },
            )
        return ControlPlaneDecision(
            allowed=True,
            metadata={
                "action_id": action.action_id,
                "deadline_at": deadline_at,
                "action_type": str(action.metadata.get("action_type") or "raw"),
            },
        )

    async def stop_motion(
        self,
        action: RobotAction,
        *,
        apply_fn: Callable[[RobotAction], Awaitable[RobotStatus]],
    ) -> RobotStatus:
        stop_action = RobotSkillAction(
            "stop_motion",
            {"emergency": bool(action.metadata.get("emergency", False))},
        ).to_robot_action(_intent_like(action))
        return await apply_fn(stop_action)

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_action_id": self.current_action_id,
            "buffer_size": len(self.action_buffer),
            "max_buffer_size": self.max_buffer_size,
            "last_watchdog": self.last_watchdog,
            "preemptions": list(self.preemptions[-5:]),
            "buffer": [
                {
                    "action_id": item.action_id,
                    "skill_id": item.skill_id,
                    "submitted_at": item.submitted_at,
                    "deadline_at": item.deadline_at,
                    "action_type": item.action_type,
                    "metadata": dict(item.metadata),
                }
                for item in self.action_buffer
            ],
        }

    def _buffer_entry(self, action: RobotAction) -> ActionBufferEntry:
        return ActionBufferEntry(
            action_id=action.action_id,
            skill_id=action.skill_id,
            submitted_at=time.time(),
            deadline_at=_deadline_at(action),
            action_type=str(action.metadata.get("action_type") or "raw"),
            metadata={
                "kind": action.metadata.get("kind"),
                "policy_session_id": action.metadata.get("policy_session_id"),
            },
        )

    def _map_action_chunk(
        self, policy_result: dict[str, Any], *, intent: SkillIntent
    ) -> list[RobotAction]:
        raw_actions = policy_result.get("actions")
        if not isinstance(raw_actions, list):
            return []
        actions: list[RobotAction] = []
        dt = policy_result.get("dt")
        for index, item in enumerate(raw_actions):
            if not isinstance(item, dict):
                continue
            joint_angles = dict(
                item.get("joints")
                or item.get("joint_angles")
                or item.get("single_arm")
                or {}
            )
            metadata = _policy_metadata(policy_result, action_index=index)
            if dt is not None:
                metadata["deadline_sec"] = float(dt) * float(index + 1)
            if joint_angles:
                actions.append(
                    _skill_robot_action(
                        "move_arm_joints",
                        {"joints": joint_angles, "mode": "absolute"},
                        intent=intent,
                        metadata=metadata,
                    )
                )
            gripper = item.get("gripper")
            if gripper is None:
                gripper = item.get("gripper_action")
            if gripper is not None:
                opening_pct = max(0.0, min(100.0, float(gripper) * 100.0))
                actions.append(
                    _skill_robot_action(
                        "set_gripper",
                        {"opening_pct": opening_pct},
                        intent=intent,
                        metadata=metadata,
                    )
                )
            if bool(item.get("done", False)):
                actions.append(
                    _skill_robot_action(
                        "stop_motion",
                        {},
                        intent=intent,
                        metadata=metadata,
                    )
                )
        return actions

    def _map_local_goal(
        self, policy_result: dict[str, Any], *, intent: SkillIntent
    ) -> list[RobotAction]:
        local_goal = policy_result.get("local_goal")
        if not isinstance(local_goal, dict):
            local_goal = policy_result
        mode = str(local_goal.get("mode") or "")
        metadata = _policy_metadata(policy_result)
        if mode == "stop" or bool(local_goal.get("stop", False)):
            return [
                _skill_robot_action(
                    "stop_motion",
                    {},
                    intent=intent,
                    metadata=metadata,
                )
            ]
        heading = local_goal.get("heading_deg")
        if isinstance(heading, (int, float)):
            heading_value = float(heading)
            if abs(heading_value) < 5.0:
                return [
                    _skill_robot_action(
                        "move_base",
                        {"direction": "forward", "distance_cm": 15.0},
                        intent=intent,
                        metadata=metadata,
                    )
                ]
            return [
                _skill_robot_action(
                    "turn_base",
                    {
                        "direction": "right" if heading_value > 0 else "left",
                        "angle_deg": min(abs(heading_value), 15.0),
                    },
                    intent=intent,
                    metadata=metadata,
                )
            ]
        pixel = local_goal.get("pixel_goal")
        if isinstance(pixel, (list, tuple)) and len(pixel) >= 2:
            image_width = float(local_goal.get("image_width") or 640.0)
            x = float(pixel[1])
            center_x = image_width / 2.0
            half_band = image_width * 0.25 / 2.0
            offset = x - center_x
            if abs(offset) <= half_band:
                return [
                    _skill_robot_action(
                        "move_base",
                        {"direction": "forward", "distance_cm": 15.0},
                        intent=intent,
                        metadata=metadata,
                    )
                ]
            turn = 5.0 + (15.0 - 5.0) * min(abs(offset) / max(center_x, 1.0), 1.0)
            return [
                _skill_robot_action(
                    "turn_base",
                    {"direction": "right" if offset > 0 else "left", "angle_deg": turn},
                    intent=intent,
                    metadata=metadata,
                )
            ]
        return []

    def _whole_body_reference_action(
        self, policy_result: dict[str, Any], *, intent: SkillIntent
    ) -> RobotAction:
        values = policy_result.get("values")
        numeric_values = (
            [float(item) for item in values if isinstance(item, (int, float))]
            if isinstance(values, list)
            else []
        )
        return RobotAction(
            envelope=intent.envelope,
            skill_id=intent.skill_id,
            values=numeric_values,
            metadata={
                **_policy_metadata(policy_result),
                "action_type": "whole_body_reference",
                "whole_body_reference": dict(
                    policy_result.get("whole_body_reference") or {}
                ),
            },
        )


def _deadline_at(action: RobotAction) -> float | None:
    metadata = dict(action.metadata)
    if metadata.get("deadline_at") is not None:
        return float(metadata["deadline_at"])
    if metadata.get("deadline_sec") is not None:
        return action.timestamp + float(metadata["deadline_sec"])
    if metadata.get("dt") is not None and metadata.get("horizon") is not None:
        return action.timestamp + float(metadata["dt"]) * float(metadata["horizon"])
    return None


def _intent_like(action: RobotAction) -> SkillIntent:
    return SkillIntent(
        envelope=action.envelope,
        skill_id=action.skill_id,
        name="stop_motion",
        objective="preempt active control output",
    )


def _policy_metadata(
    policy_result: dict[str, Any], *, action_index: int | None = None
) -> dict[str, Any]:
    metadata = {
        "kind": policy_result.get("kind"),
        "policy_session_id": policy_result.get("policy_session_id"),
        "action_space": policy_result.get("action_space"),
        "embodiment": policy_result.get("embodiment"),
        "horizon": policy_result.get("horizon"),
        "dt": policy_result.get("dt"),
        "confidence": policy_result.get("confidence"),
    }
    if action_index is not None:
        metadata["action_index"] = action_index
    return {key: value for key, value in metadata.items() if value is not None}


def _skill_robot_action(
    name: str,
    arguments: dict[str, Any],
    *,
    intent: SkillIntent,
    metadata: dict[str, Any],
) -> RobotAction:
    action = RobotSkillAction(name, arguments).to_robot_action(intent)
    return replace(action, metadata={**dict(action.metadata), **metadata})
