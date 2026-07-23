"""Load native skill modules into a SkillRegistry."""

from __future__ import annotations

import importlib
import inspect

from hey_robot.skills.registry import SkillRegistry


def registry_from_config(config: object) -> SkillRegistry:
    skills = getattr(config, "skills")
    return load_skill_registry(
        getattr(skills, "modules", ("hey_robot.skills.builtins",)),
        implementations=dict(getattr(skills, "implementations", {}) or {}),
    )


def load_skill_registry(
    modules: tuple[str, ...] | list[str],
    *,
    implementations: dict[str, str] | None = None,
) -> SkillRegistry:
    registry = SkillRegistry()
    for module_name in modules:
        module = importlib.import_module(module_name)
        register = getattr(module, "register", None)
        if register is None:
            register = getattr(module, "register_skills", None)
        if register is None:
            raise ValueError(f"skill module {module_name!r} has no register function")
        if "implementations" in inspect.signature(register).parameters:
            register(registry, implementations=implementations or {})
        else:
            register(registry)
    return registry
