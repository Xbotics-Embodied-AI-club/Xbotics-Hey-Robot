from __future__ import annotations

from dataclasses import dataclass

from hey_robot.skills import RobotSkillAction


@dataclass(frozen=True)
class StopMotionPrimitive:
    emergency: bool = False


@dataclass(frozen=True)
class MoveBasePrimitive:
    direction: str
    distance_cm: float


@dataclass(frozen=True)
class TurnBasePrimitive:
    direction: str
    angle_deg: float


@dataclass(frozen=True)
class BaseVelocityStepPrimitive:
    vx: float
    vy: float
    wz: float
    duration_ms: int


@dataclass(frozen=True)
class SetArmPosePrimitive:
    pose_name: str
    arm: str | None = None


@dataclass(frozen=True)
class MoveArmJointsPrimitive:
    joints: dict[str, float]
    delta_mode: bool = False
    arm: str | None = None


@dataclass(frozen=True)
class SetGripperPrimitive:
    action: str | None = None
    opening_pct: float | None = None
    arm: str | None = None


@dataclass(frozen=True)
class ResetPosturePrimitive:
    arm: str | None = None


@dataclass(frozen=True)
class PerceptionPrimitive:
    skill_name: str
    camera: str | None = None
    arguments: dict[str, object] | None = None


ClassicPrimitive = (
    StopMotionPrimitive
    | MoveBasePrimitive
    | TurnBasePrimitive
    | BaseVelocityStepPrimitive
    | SetArmPosePrimitive
    | MoveArmJointsPrimitive
    | SetGripperPrimitive
    | ResetPosturePrimitive
    | PerceptionPrimitive
)


PERCEPTION_SKILLS = {
    "inspect_scene",
    "look_around",
    "detect_marker",
    "human_follow",
}

SUPPORTED_CLASSIC_PRIMITIVES = (
    "stop_motion",
    "move_base",
    "turn_base",
    "base_velocity_step",
    "set_arm_pose",
    "move_arm_joints",
    "set_gripper",
    "reset_posture",
    *sorted(PERCEPTION_SKILLS),
)


def decode_classic_primitive(action: RobotSkillAction) -> ClassicPrimitive:
    name = action.name
    args = dict(action.arguments)

    if name == "stop_motion":
        return StopMotionPrimitive(emergency=bool(args.get("emergency", False)))
    if name == "move_base":
        return MoveBasePrimitive(
            direction=str(args.get("direction", "forward")).lower(),
            distance_cm=float(args.get("distance_cm", 0.0)),
        )
    if name == "turn_base":
        return TurnBasePrimitive(
            direction=str(args.get("direction", "left")).lower(),
            angle_deg=float(args.get("angle_deg", 0.0)),
        )
    if name == "base_velocity_step":
        duration_ms = int(args.get("duration_ms", 250))
        return BaseVelocityStepPrimitive(
            vx=float(args.get("vx", 0.0)),
            vy=float(args.get("vy", 0.0)),
            wz=float(args.get("wz", 0.0)),
            duration_ms=max(1, min(duration_ms, 1000)),
        )
    if name == "set_arm_pose":
        return SetArmPosePrimitive(
            pose_name=str(args["pose_name"]),
            arm=_optional_target(args, "arm"),
        )
    if name == "move_arm_joints":
        joints = args.get("joints")
        if not isinstance(joints, dict):
            raise ValueError("move_arm_joints requires arguments.joints")
        return MoveArmJointsPrimitive(
            joints={str(key): float(value) for key, value in joints.items()},
            delta_mode=str(args.get("mode", "absolute")).lower() == "delta",
            arm=_optional_target(args, "arm"),
        )
    if name == "set_gripper":
        action_name = str(args.get("action", "")).strip().lower() or None
        if action_name in {"open", "close"}:
            return SetGripperPrimitive(
                action=action_name,
                arm=_optional_target(args, "arm"),
            )
        return SetGripperPrimitive(
            opening_pct=float(args["opening_pct"]),
            arm=_optional_target(args, "arm"),
        )
    if name == "reset_posture":
        return ResetPosturePrimitive(arm=_optional_target(args, "arm"))
    if name in PERCEPTION_SKILLS:
        return PerceptionPrimitive(
            skill_name=name,
            camera=_optional_target(args, "camera"),
            arguments=args,
        )
    raise ValueError(f"unsupported classic primitive: {name}")


def _optional_target(arguments: dict[str, object], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    target = str(value).strip()
    return target or None
