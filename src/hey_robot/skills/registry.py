"""The single registry for executable skills and agent tool surfaces."""

from __future__ import annotations

from collections.abc import Sequence

from hey_robot.skills.models import Skill


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"duplicate skill: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"unknown skill: {name}") from exc

    def list(self) -> tuple[Skill, ...]:
        return tuple(self._skills.values())

    def select(self, names: Sequence[str]) -> tuple[Skill, ...]:
        return tuple(self.get(name) for name in names)
