"""Native skill definitions introduced during the Skill OS migration."""

from hey_robot.skills.builtins.dock import PICK_WAND_FROM_DOCK, PLACE_WAND_TO_DOCK
from hey_robot.skills.builtins.manipulation import (
    MOVE_ARM_JOINTS,
    SET_ARM_POSE,
    SET_GRIPPER,
)
from hey_robot.skills.builtins.navigation import (
    APPROACH_OBJECT,
    BASE_VELOCITY_STEP,
    MOVE_BASE,
    NAVIGATE_TO,
    TURN_BASE,
)
from hey_robot.skills.builtins.perception import (
    DETECT_MARKER,
    INSPECT_SCENE,
    LOOK_AROUND,
)
from hey_robot.skills.builtins.safety import RESET_POSTURE, STOP_MOTION
from hey_robot.skills.builtins.tabletop import PICK_PARAMETERS, PLACE_PARAMETERS
from hey_robot.skills.builtins.vla import MANIPULATE
from hey_robot.skills.registry import SkillRegistry


def register(
    registry: SkillRegistry, *, implementations: dict[str, str] | None = None
) -> None:
    from hey_robot.skills.builtins import (
        dock,
        manipulation,
        navigation,
        perception,
        safety,
        tabletop,
        vla,
    )

    perception.register(registry)
    navigation.register(registry)
    safety.register(registry)
    manipulation.register(registry)
    dock.register(registry)
    vla.register(registry)
    tabletop.register(registry, implementations=implementations)


__all__ = [
    "APPROACH_OBJECT",
    "BASE_VELOCITY_STEP",
    "DETECT_MARKER",
    "INSPECT_SCENE",
    "LOOK_AROUND",
    "MANIPULATE",
    "MOVE_ARM_JOINTS",
    "MOVE_BASE",
    "NAVIGATE_TO",
    "PICK_PARAMETERS",
    "PICK_WAND_FROM_DOCK",
    "PLACE_PARAMETERS",
    "PLACE_WAND_TO_DOCK",
    "RESET_POSTURE",
    "SET_ARM_POSE",
    "SET_GRIPPER",
    "STOP_MOTION",
    "TURN_BASE",
    "register",
]
