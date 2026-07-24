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

    def resolve_dependencies(self, names: Sequence[str]) -> tuple[Skill, ...]:
        resolved: list[Skill] = []
        visited: set[str] = set()
        visiting: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                cycle = " -> ".join((*visiting[visiting.index(name) :], name))
                raise ValueError(f"skill dependency cycle: {cycle}")
            try:
                skill = self.get(name)
            except KeyError as exc:
                owner = visiting[-1] if visiting else "skill surface"
                raise ValueError(
                    f"skill {owner!r} depends on unknown skill {name!r}"
                ) from exc
            visiting.append(name)
            for dependency in skill.dependencies:
                visit(dependency)
            visiting.pop()
            visited.add(name)
            resolved.append(skill)

        for name in names:
            visit(name)
        return tuple(resolved)
