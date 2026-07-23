"""Compatibility adapters from native skills to legacy deployment catalogs."""

from __future__ import annotations

from typing import Any

from hey_robot.skill_os.base import SkillSpec
from hey_robot.skill_os.registry import SkillRegistry as LegacySkillRegistry
from hey_robot.skills.loader import registry_from_config
from hey_robot.skills.models import Skill


def legacy_registry_from_native_config(config: Any) -> LegacySkillRegistry:
    native = registry_from_config(config)
    registry = LegacySkillRegistry()
    for skill in native.list():
        registry.register_spec(skill_spec_from_native(skill))
    skills = getattr(config, "skills")
    return registry.configure(enabled=tuple(getattr(skills, "tool_names", ()) or ()))


def skill_spec_from_native(skill: Skill) -> SkillSpec:
    return SkillSpec(
        name=skill.name,
        description=skill.description,
        input_schema=skill.parameters,
        required_resources=skill.resources,
        driver_primitives=skill.required_actions,
        required_model_service=skill.required_models[0]
        if skill.required_models
        else None,
        supported_robots=skill.supported_robots,
        timeout_sec=skill.timeout_sec,
    )
