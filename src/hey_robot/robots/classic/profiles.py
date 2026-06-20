from __future__ import annotations

from dataclasses import dataclass

from hey_robot.robots.embodiments import EmbodimentProfile


@dataclass(frozen=True)
class ClassicEmbodimentProfile:
    name: str
    robot_family: str
    environment: str
    named_poses: dict[str, dict[str, float]]
    actuator_layout: dict[str, tuple[int, int]]
    gripper_range: tuple[float, float]

    @classmethod
    def from_embodiment(
        cls, embodiment: EmbodimentProfile | None
    ) -> ClassicEmbodimentProfile | None:
        if embodiment is None:
            return None
        return cls(
            name=embodiment.name,
            robot_family=embodiment.robot_family,
            environment=embodiment.environment,
            named_poses={
                name: embodiment.named_pose(name) or {}
                for name in embodiment.pose_library or embodiment.named_poses.keys()
            },
            actuator_layout={
                str(joint): (int(pair[0]), int(pair[1]))
                for joint, pair in embodiment.actuator_layout.items()
            },
            gripper_range=(
                embodiment.gripper_closed_value,
                embodiment.gripper_open_value,
            ),
        )

    def named_pose(self, pose_name: str) -> dict[str, float] | None:
        pose = self.named_poses.get(pose_name)
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
        return float(self.gripper_range[0])

    @property
    def gripper_open_value(self) -> float:
        return float(self.gripper_range[1])
