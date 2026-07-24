"""Native Skill 到 runtime contract 的投影。

这个模块只做数据投影：Agent 和 Skill Runner 继续使用 native ``Skill``，
Robot Runtime gate 继续使用稳定的 ``SkillContractCatalog``。
"""

from __future__ import annotations

from hey_robot.config import DeploymentConfig
from hey_robot.contracts import SkillContract, SkillContractCatalog
from hey_robot.skills.loader import registry_from_config
from hey_robot.skills.models import Skill


def skill_contract_from_native(skill: Skill) -> SkillContract:
    """把 native Skill 描述转换为 Robot Runtime contract。"""

    return SkillContract(
        name=skill.name,
        description=skill.description,
        level="semantic",
        agent_visible=True,
        input_schema=dict(skill.parameters),
        supported_robots=skill.supported_robots,
        required_model_service=(
            skill.required_models[0] if len(skill.required_models) == 1 else None
        ),
        driver_primitives=skill.required_actions,
        required_resources=skill.resources,
        dependencies=skill.dependencies,
        timeout_sec=skill.timeout_sec,
    )


def skill_contract_catalog_from_config(
    config: DeploymentConfig,
    *,
    selected_only: bool = False,
) -> SkillContractCatalog:
    """从部署配置加载 native Skill，并生成 runtime contract catalog。"""

    registry = registry_from_config(config)
    skills = (
        registry.select(config.skills.tool_names) if selected_only else registry.list()
    )
    return SkillContractCatalog(
        tuple(skill_contract_from_native(skill) for skill in skills)
    )
