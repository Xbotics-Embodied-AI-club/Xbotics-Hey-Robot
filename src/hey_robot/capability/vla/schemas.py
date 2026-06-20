from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_VALID_ACTION_MODES = frozenset(
    {
        "absolute_joint_position_rad",
        "delta_joint_position_rad",
        "normalized_joint_position",
    }
)


@dataclass(frozen=True)
class VLAConfig:
    """Configuration for a VLA policy-driven control loop."""

    policy_runtime: str = "fake"
    policy_endpoint: str = ""
    policy_type: str = "act"
    policy_model_path: str = ""
    task_prompt: str = "Pick up the object."
    camera_names: tuple[str, ...] = ("front", "right_wrist")
    arm: str = "right"
    fps: int = 25
    action_horizon: int = 16
    execution_time_sec: float = 10.0
    action_mode: str = "absolute_joint_position_rad"
    camera_key_map: dict[str, str] = field(default_factory=dict)
    state_key_map: dict[str, str] = field(default_factory=dict)
    language_key: str = "language"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.arm not in {"left", "right"}:
            raise ValueError(f"arm must be 'left' or 'right', got {self.arm!r}")
        if self.action_mode not in _VALID_ACTION_MODES:
            raise ValueError(
                f"action_mode must be one of {sorted(_VALID_ACTION_MODES)}, "
                f"got {self.action_mode!r}"
            )


@dataclass(frozen=True)
class VLARequest:
    """A single VLA execution request."""

    config: VLAConfig
    skill_id: str | None = None
    episode_id: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VLAResult:
    """Outcome of a VLA execution."""

    success: bool
    summary: str
    status: str = "completed"
    failure_mode: str | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
