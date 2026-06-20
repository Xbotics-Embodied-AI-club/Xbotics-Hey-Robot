from __future__ import annotations

from hey_robot.skills.builtin.capability import VLAManipulationSkill
from hey_robot.skills.builtin.manipulation import (
    MoveArmJointsSkill,
    SetArmPoseSkill,
    SetGripperSkill,
)
from hey_robot.skills.builtin.navigation import (
    BaseVelocityStepSkill,
    HumanFollowSkill,
    MoveBaseSkill,
    TurnBaseSkill,
)
from hey_robot.skills.builtin.perception import (
    DetectMarkerSkill,
    InspectSceneSkill,
    LookAroundSkill,
)
from hey_robot.skills.builtin.safety import ResetPostureSkill, StopMotionSkill
from hey_robot.skills.registry import SkillRegistry


def register_skills(registry: SkillRegistry) -> None:
    for skill in (
        InspectSceneSkill(),
        LookAroundSkill(),
        DetectMarkerSkill(),
        MoveBaseSkill(),
        TurnBaseSkill(),
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
