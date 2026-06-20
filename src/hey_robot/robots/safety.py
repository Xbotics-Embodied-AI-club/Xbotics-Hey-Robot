from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hey_robot.protocol import RobotAction
from hey_robot.robots.base import RobotCapabilities, RobotHealth


class RobotSafetyError(RuntimeError):
    pass


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str | None = None
    metadata: dict[str, Any] | None = None


class RobotSafetySupervisor:
    """Deterministic runtime safety gate before actions reach a driver."""

    def evaluate_action(
        self,
        action: RobotAction,
        *,
        capabilities: RobotCapabilities,
        health: RobotHealth,
    ) -> SafetyDecision:
        if not health.online:
            return SafetyDecision(False, f"robot {health.robot_id} is offline")
        active_flags = [
            key
            for key in (
                "emergency_stop",
                "estop",
                "safety_stop",
                "collision_detected",
                "protective_stop",
            )
            if bool(health.metrics.get(key))
        ]
        if active_flags:
            return SafetyDecision(
                False, f"active safety flags: {','.join(active_flags)}"
            )
        battery = health.metrics.get("battery")
        if (
            isinstance(battery, dict)
            and battery.get("status") == "critical"
            and not _is_stop_action(action)
        ):
            voltage = battery.get("voltage")
            suffix = f" voltage={voltage}" if voltage is not None else ""
            return SafetyDecision(
                False, f"battery critical{suffix}", metadata={"battery": battery}
            )
        if capabilities.action_dimensions is not None and len(action.values) != int(
            capabilities.action_dimensions
        ):
            return SafetyDecision(
                False,
                f"action dimension mismatch: expected {capabilities.action_dimensions}, got {len(action.values)}",
            )
        max_abs = _safety_setting(capabilities, "max_abs_action")
        if max_abs is not None:
            limit = float(max_abs)
            if any(abs(float(value)) > limit for value in action.values):
                return SafetyDecision(False, f"action exceeds max_abs_action={limit}")
        bounds = _safety_setting(capabilities, "action_bounds")
        if isinstance(bounds, list) and len(bounds) == 2:
            lower = float(bounds[0])
            upper = float(bounds[1])
            if any(
                float(value) < lower or float(value) > upper for value in action.values
            ):
                return SafetyDecision(
                    False, f"action outside bounds [{lower}, {upper}]"
                )
        return SafetyDecision(True)


def _safety_setting(capabilities: RobotCapabilities, key: str) -> Any:
    safety = capabilities.metadata.get("safety")
    if isinstance(safety, dict) and key in safety:
        return safety[key]
    return capabilities.metadata.get(key)


def _is_stop_action(action: RobotAction) -> bool:
    skill = action.metadata.get("skill")
    if not isinstance(skill, dict):
        return False
    return str(skill.get("name") or "").lower() == "stop_motion"
