from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EmbodimentProfile:
    name: str
    robot_family: str
    environment: str
    camera_layout: dict[str, Any] = field(default_factory=dict)
    pose_library: tuple[str, ...] = ()
    named_poses: dict[str, dict[str, float]] = field(default_factory=dict)
    actuator_layout: dict[str, tuple[int, int]] = field(default_factory=dict)
    gripper_range: tuple[float, float] | None = None
    readiness_resources: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def default_camera(self) -> str | None:
        camera = self.camera_layout.get("default_camera")
        return str(camera) if camera else None

    def named_pose(self, name: str) -> dict[str, float] | None:
        pose = self.named_poses.get(name)
        if pose is None:
            return None
        return {str(joint): float(value) for joint, value in pose.items()}

    def joint_actuator_pair(self, joint_name: str) -> tuple[int, int] | None:
        pair = self.actuator_layout.get(joint_name)
        if pair is None:
            return None
        return int(pair[0]), int(pair[1])

    @property
    def gripper_closed_value(self) -> float:
        if self.gripper_range is None:
            return 0.0
        return float(self.gripper_range[0])

    @property
    def gripper_open_value(self) -> float:
        if self.gripper_range is None:
            return 100.0
        return float(self.gripper_range[1])
