from __future__ import annotations

from hey_robot.skill_os.builtins.capability import VLAManipulationSkill
from hey_robot.skill_os.builtins.manipulation import (
    MoveArmJointsSkill,
    SetArmPoseSkill,
    SetGripperSkill,
)
from hey_robot.skill_os.builtins.navigation import (
    ApproachObjectSkill,
    BaseVelocityStepSkill,
    HumanFollowSkill,
    MoveBaseSkill,
    NavigateToSkill,
    TurnBaseSkill,
)
from hey_robot.skill_os.builtins.perception import (
    DetectMarkerSkill,
    InspectSceneSkill,
    LookAroundSkill,
)
from hey_robot.skill_os.builtins.safety import ResetPostureSkill, StopMotionSkill
from hey_robot.skill_os.registry import SkillRegistry


def register_skills(registry: SkillRegistry) -> None:
    for skill in (
        InspectSceneSkill(),
        LookAroundSkill(),
        DetectMarkerSkill(),
        MoveBaseSkill(),
        TurnBaseSkill(),
        NavigateToSkill(),
        ApproachObjectSkill(),
        BaseVelocityStepSkill(),
        HumanFollowSkill(),
        StopMotionSkill(),
        ResetPostureSkill(),
        SetArmPoseSkill(),
        MoveArmJointsSkill(),
        SetGripperSkill(),
        VLAManipulationSkill(),
    ):
        registry.register(skill)
