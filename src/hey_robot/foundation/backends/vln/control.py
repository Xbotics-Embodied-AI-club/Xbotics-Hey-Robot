from __future__ import annotations

import math
from typing import Any

from hey_robot.foundation.backends.vln.models import VLNPlannerResult


def build_base_action_chunk(
    result: VLNPlannerResult, settings: dict[str, Any]
) -> dict[str, Any]:
    """Convert native InternVLA actions into calibrated base control steps."""
    linear_speed = float(settings["base_linear_speed"])
    angular_speed = float(settings["base_angular_speed"])
    max_chunk_steps = int(settings["max_action_chunk_steps"])
    forward_distance_m = float(settings["discrete_forward_cm"]) / 100.0
    turn_angle_rad = math.radians(float(settings["discrete_turn_deg"]))
    stage = str(result.policy_stage or "system2")
    forward_duration_ms = _duration_ms(forward_distance_m, linear_speed)
    turn_duration_ms = _duration_ms(turn_angle_rad, angular_speed)

    stop = bool(result.stop or result.mode == "stop")
    stop_after_actions = False
    codes = list(result.action_sequence or ())
    if not codes and result.action_code is not None:
        codes = [result.action_code]

    actions: list[dict[str, Any]] = []
    for code in codes[:max_chunk_steps]:
        if code == 0:
            if actions:
                stop_after_actions = True
                stop = False
            else:
                stop = True
            break
        if code == 1:
            actions.append(
                _velocity_step(
                    linear_speed,
                    0.0,
                    forward_duration_ms,
                    f"{stage}_forward",
                )
            )
        elif code == 2:
            actions.append(
                _velocity_step(0.0, angular_speed, turn_duration_ms, f"{stage}_left")
            )
        elif code == 3:
            actions.append(
                _velocity_step(0.0, -angular_speed, turn_duration_ms, f"{stage}_right")
            )

    if not actions and not stop and not result.requires_secondary_observation:
        raise ValueError("dual-system VLN result did not contain a native action chunk")

    return {
        "kind": "base_velocity_chunk",
        "actions": actions,
        "stop": stop,
        "stop_after_actions": stop_after_actions,
        "replan_after_actions": len(actions),
    }


def _duration_ms(distance: float, speed: float) -> int:
    if speed <= 0.0:
        raise ValueError("VLN base control speed must be positive")
    duration_ms = round(1000.0 * distance / speed)
    if not 1 <= duration_ms <= 1000:
        raise ValueError(
            "VLN native action exceeds the 1000ms base velocity safety window"
        )
    return duration_ms


def _velocity_step(
    vx: float, wz: float, duration_ms: int, source: str
) -> dict[str, Any]:
    return {
        "kind": "base_velocity_step",
        "vx": vx,
        "vy": 0.0,
        "wz": wz,
        "duration_ms": duration_ms,
        "source": source,
    }
