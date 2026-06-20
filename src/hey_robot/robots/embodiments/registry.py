from __future__ import annotations

from hey_robot.config import RobotSpec
from hey_robot.robots.embodiments.base import EmbodimentProfile

DEFAULT_EMBODIMENT_PROFILES: dict[str, EmbodimentProfile] = {
    "xlerobot_real": EmbodimentProfile(
        name="xlerobot_real",
        robot_family="xlerobot",
        environment="real",
        camera_layout={"default_camera": "front", "owner": "robot_driver"},
        pose_library=(
            "home",
            "pregrasp",
            "pregrasp_table",
            "place",
        ),
        named_poses={
            "home": {
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": 80.0,
            },
            "pregrasp": {
                "shoulder_pan": 0.0,
                "shoulder_lift": 35.0,
                "elbow_flex": -50.0,
                "wrist_flex": 20.0,
                "wrist_roll": 0.0,
                "gripper": 100.0,
            },
            "pregrasp_table": {
                "shoulder_pan": 0.0,
                "shoulder_lift": 42.0,
                "elbow_flex": -62.0,
                "wrist_flex": 24.0,
                "wrist_roll": 0.0,
                "gripper": 100.0,
            },
            "place": {
                "shoulder_pan": 10.0,
                "shoulder_lift": 25.0,
                "elbow_flex": -35.0,
                "wrist_flex": 10.0,
                "wrist_roll": 0.0,
                "gripper": 70.0,
            },
        },
        readiness_resources=("base", "arm", "gripper", "camera"),
        metadata={"driver_kind": "native"},
    ),
    "xlerobot_sim": EmbodimentProfile(
        name="xlerobot_sim",
        robot_family="xlerobot",
        environment="sim",
        camera_layout={
            "default_camera": "front",
            "owner": "simulation",
            "cameras": ("front", "left_wrist", "right_wrist"),
        },
        pose_library=(
            "home",
            "pregrasp",
            "pregrasp_table",
            "place",
        ),
        named_poses={
            "home": {
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.8,
                "elbow_flex": 0.7,
                "wrist_flex": -0.6,
                "wrist_roll": 0.0,
                "gripper": 0.08,
            },
            "pregrasp": {
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.7,
                "elbow_flex": 0.9,
                "wrist_flex": -0.5,
                "wrist_roll": 0.0,
                "gripper": 0.08,
            },
            "pregrasp_table": {
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.9,
                "elbow_flex": 1.05,
                "wrist_flex": -0.75,
                "wrist_roll": 0.0,
                "gripper": 0.08,
            },
            "place": {
                "shoulder_pan": 0.17,
                "shoulder_lift": 0.65,
                "elbow_flex": 0.6,
                "wrist_flex": -0.45,
                "wrist_roll": 0.0,
                "gripper": 0.07,
            },
        },
        actuator_layout={
            "shoulder_pan": (9, 3),
            "shoulder_lift": (10, 4),
            "elbow_flex": (11, 5),
            "wrist_flex": (12, 6),
            "wrist_roll": (13, 7),
            "gripper": (14, 8),
        },
        gripper_range=(0.0, 1.2),
        readiness_resources=(
            "base",
            "arm",
            "gripper",
            "camera",
            "front_camera",
            "left_wrist_camera",
            "right_wrist_camera",
        ),
        metadata={"driver_kind": "mujoco"},
    ),
    "xlerobot_mock": EmbodimentProfile(
        name="xlerobot_mock",
        robot_family="xlerobot",
        environment="mock",
        camera_layout={"default_camera": "front", "owner": "mock"},
        pose_library=(
            "home",
            "pregrasp",
            "pregrasp_table",
            "place",
        ),
        named_poses={
            "home": {
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": 80.0,
            },
            "pregrasp": {
                "shoulder_pan": 0.0,
                "shoulder_lift": 35.0,
                "elbow_flex": -50.0,
                "wrist_flex": 20.0,
                "wrist_roll": 0.0,
                "gripper": 100.0,
            },
            "pregrasp_table": {
                "shoulder_pan": 0.0,
                "shoulder_lift": 42.0,
                "elbow_flex": -62.0,
                "wrist_flex": 24.0,
                "wrist_roll": 0.0,
                "gripper": 100.0,
            },
            "place": {
                "shoulder_pan": 10.0,
                "shoulder_lift": 25.0,
                "elbow_flex": -35.0,
                "wrist_flex": 10.0,
                "wrist_roll": 0.0,
                "gripper": 70.0,
            },
        },
        gripper_range=(0.0, 100.0),
        readiness_resources=("base", "arm", "gripper", "camera"),
        metadata={"driver_kind": "mock"},
    ),
    "so101_real": EmbodimentProfile(
        name="so101_real",
        robot_family="so101",
        environment="real",
        camera_layout={"default_camera": "front", "owner": "robot_driver"},
        pose_library=("home",),
        readiness_resources=("arm", "gripper", "camera"),
        metadata={"driver_kind": "native"},
    ),
    "lekiwi_real": EmbodimentProfile(
        name="lekiwi_real",
        robot_family="lekiwi",
        environment="real",
        camera_layout={"default_camera": "front", "owner": "robot_driver"},
        pose_library=(),
        readiness_resources=("base", "camera"),
        metadata={"driver_kind": "native"},
    ),
}


def resolve_embodiment_profile_name(spec: RobotSpec) -> str:
    if spec.embodiment_profile:
        return str(spec.embodiment_profile)
    return f"{spec.robot_family}_{spec.robot_environment}"


def get_embodiment_profile(spec: RobotSpec) -> EmbodimentProfile:
    name = resolve_embodiment_profile_name(spec)
    profile = DEFAULT_EMBODIMENT_PROFILES.get(name)
    if profile is not None:
        return profile
    return EmbodimentProfile(
        name=name,
        robot_family=spec.robot_family,
        environment=spec.robot_environment,
        camera_layout={},
        pose_library=(),
        named_poses={},
        readiness_resources=(),
        metadata={"driver_kind": spec.driver_kind},
    )
