from __future__ import annotations

from hey_robot.protocol import (
    RobotSkillCatalog,
    SkillContractDecision,
    SkillContractRuntime as ProtocolSkillContractRuntime,
)


class SkillContractRuntime(ProtocolSkillContractRuntime):
    def __init__(self, catalog: RobotSkillCatalog | None = None) -> None:
        if catalog is None:
            from hey_robot.skill_os.registry import load_skill_registry

            catalog = load_skill_registry().robot_skill_catalog()
        super().__init__(catalog)


__all__ = ["SkillContractDecision", "SkillContractRuntime"]
