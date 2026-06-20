from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from hey_robot.skills.actions import RobotSkillAction
from hey_robot.skills.base import BaseSkill, SkillCatalog, SkillSpec
from hey_robot.skills.catalog import RobotSkillCatalog, RobotSkillSpec


@dataclass(frozen=True)
class RegisteredSkill:
    name: str
    spec: SkillSpec
    skill: BaseSkill | None = None


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, RegisteredSkill] = {}
        self.enabled_names: tuple[str, ...] = ()

    def register_spec(self, spec: SkillSpec) -> None:
        self._ensure_unique(spec.name)
        self._skills[spec.name] = RegisteredSkill(name=spec.name, spec=spec)

    def register(self, skill: BaseSkill) -> None:
        self._ensure_unique(skill.spec.name)
        self._skills[skill.spec.name] = RegisteredSkill(
            name=skill.spec.name,
            spec=skill.spec,
            skill=skill,
        )

    def _ensure_unique(self, name: str) -> None:
        if name in self._skills:
            raise ValueError(f"duplicate skill: {name}")

    def register_module(self, module_name: str) -> None:
        module = importlib.import_module(module_name)
        register_fn = getattr(module, "register_skills", None)
        if callable(register_fn):
            register_fn(self)
            return
        skills = getattr(module, "SKILLS", None)
        if isinstance(skills, (list, tuple)):
            for skill in skills:
                if isinstance(skill, BaseSkill):
                    self.register(skill)
                elif isinstance(skill, SkillSpec):
                    self.register_spec(skill)
            return
        raise ValueError(
            f"skill module {module_name!r} does not expose register_skills"
        )

    def configure(
        self,
        *,
        enabled: tuple[str, ...] = (),
    ) -> SkillRegistry:
        configured = SkillRegistry()
        configured._skills = dict(self._skills)
        if enabled:
            enabled_names = tuple(
                name for name in enabled if name in configured._skills
            )
        else:
            enabled_names = tuple(
                name
                for name, item in configured._skills.items()
                if item.spec.agent_visible
            )
        configured.enabled_names = enabled_names
        return configured

    def get(self, name: str, *, enabled_only: bool = True) -> RegisteredSkill:
        if enabled_only and self.enabled_names and name not in self.enabled_names:
            raise KeyError(f"skill {name!r} is not enabled")
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"unknown skill: {name}") from exc

    def names(self, *, enabled_only: bool = True) -> tuple[str, ...]:
        if enabled_only and self.enabled_names:
            return self.enabled_names
        return tuple(self._skills.keys())

    def catalog(
        self,
        *,
        enabled_only: bool = True,
        semantic_only: bool = False,
    ) -> SkillCatalog:
        specs: list[SkillSpec] = []
        for name in self.names(enabled_only=enabled_only):
            spec = self._skills[name].spec
            if semantic_only and not spec.agent_visible:
                continue
            specs.append(spec)
        return SkillCatalog(tuple(specs))

    def robot_skill_catalog(self) -> RobotSkillCatalog:
        return RobotSkillCatalog(
            tuple(
                _contract_to_robot_spec(self._skills[name].spec)
                for name in self.names(enabled_only=False)
            )
        )

    def action_for(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> RobotSkillAction:
        return RobotSkillAction(name, dict(arguments or {}))


def load_skill_registry(
    *,
    modules: tuple[str, ...] = ("hey_robot.skills.builtin",),
    enabled: tuple[str, ...] = (),
) -> SkillRegistry:
    registry = SkillRegistry()
    for module_name in modules:
        registry.register_module(module_name)
    return registry.configure(
        enabled=enabled,
    )


def registry_from_config(config: Any | None) -> SkillRegistry:
    skills_config = getattr(config, "skills", None)
    if skills_config is None:
        return load_skill_registry()
    return load_skill_registry(
        modules=tuple(
            getattr(skills_config, "modules", ()) or ("hey_robot.skills.builtin",)
        ),
        enabled=tuple(getattr(skills_config, "enabled", ()) or ()),
    )


def _contract_to_robot_spec(spec: SkillSpec) -> RobotSkillSpec:
    level = "semantic" if spec.agent_visible else "primitive"
    return RobotSkillSpec(
        name=spec.name,
        description=spec.description,
        level=level,
        agent_visible=spec.agent_visible,
        category=spec.category,
        input_schema=dict(spec.input_schema),
        supported_robots=tuple(spec.supported_robots),
        external_capability=spec.external_capability,
        driver_primitives=tuple(spec.driver_primitives),
        safety_level=spec.safety_level,
        required_resources=tuple(spec.required_resources),
        preconditions=tuple(spec.preconditions),
        success_criteria=tuple(spec.success_criteria),
        failure_modes=tuple(spec.failure_modes),
        recovery_hints=tuple(spec.recovery_hints),
        timeout_sec=float(spec.timeout_sec),
        interruptible=bool(spec.interruptible),
        feedback_mode=spec.feedback_mode,
        refresh_observation=bool(spec.refresh_observation),
    )
