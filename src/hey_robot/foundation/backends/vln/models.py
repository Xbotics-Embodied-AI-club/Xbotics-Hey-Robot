from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hey_robot.foundation.clients.models import PolicyStepResult


class VLNPlanningError(RuntimeError):
    def __init__(self, failure_mode: str, message: str) -> None:
        super().__init__(message)
        self.failure_mode = failure_mode


@dataclass(frozen=True)
class VLNPlannerInput:
    rgb: np.ndarray
    depth: np.ndarray | None
    pose: tuple[float, float, float]
    instruction: str
    intrinsic: np.ndarray
    look_down: bool = False
    image_source: str | None = None


@dataclass(frozen=True)
class VLNPlannerResult:
    mode: str
    pixel_goal: list[int] | None = None
    waypoint: list[float] | None = None
    heading_deg: float | None = None
    stop: bool = False
    confidence: float | None = None
    reason: str | None = None
    raw_output: str | None = None
    image_source: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    output_latent: Any | None = None
    requires_secondary_observation: bool = False
    policy_session_id: str | None = None

    def to_metrics(
        self, *, backend: str, camera: str, control_mode: str
    ) -> dict[str, Any]:
        local_goal = {
            "mode": self.mode,
            "pixel_goal": self.pixel_goal,
            "waypoint": self.waypoint,
            "heading_deg": self.heading_deg,
            "stop": self.stop,
            "confidence": self.confidence,
            "frame_id": None,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "requires_secondary_observation": self.requires_secondary_observation,
        }
        policy_result = PolicyStepResult(
            kind="local_goal",
            local_goal=local_goal,
            done=self.stop,
            confidence=self.confidence,
            valid=True,
            raw={
                "raw_output": self.raw_output,
                "image_source": self.image_source,
                "output_latent": json_public_value(self.output_latent),
            },
        ).to_metrics()
        return {
            "backend": backend,
            "control_mode": control_mode,
            "camera": camera,
            "mode": self.mode,
            "pixel_goal": self.pixel_goal,
            "waypoint": self.waypoint,
            "heading_deg": self.heading_deg,
            "stop": self.stop,
            "confidence": self.confidence,
            "reason": self.reason,
            "raw_output": self.raw_output,
            "image_source": self.image_source,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "output_latent": json_public_value(self.output_latent),
            "requires_secondary_observation": self.requires_secondary_observation,
            "policy_session_id": self.policy_session_id,
            "local_goal": local_goal,
            "policy_result": policy_result,
        }


def json_public_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): json_public_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_public_value(item) for item in value]
    if hasattr(value, "detach") and callable(value.detach):
        try:
            return value.detach().cpu().numpy().tolist()
        except Exception:
            return str(value)
    return str(value)
