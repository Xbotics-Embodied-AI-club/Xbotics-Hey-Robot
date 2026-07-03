from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from hey_robot.contracts import SkillContract


@dataclass(frozen=True)
class SkillSpec(SkillContract):
    output_schema: dict[str, Any] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "level",
            "semantic" if self.agent_visible else "primitive",
        )


class SkillCatalog:
    def __init__(self, specs: list[SkillSpec] | tuple[SkillSpec, ...]) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def get(self, name: str) -> SkillSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unknown skill: {name}") from exc

    def list(self) -> tuple[SkillSpec, ...]:
        return tuple(self._specs.values())

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs.keys())

    def subset(self, names: Sequence[str]) -> SkillCatalog:
        return SkillCatalog(tuple(self.get(name) for name in names))

    def semantic_skills(self) -> SkillCatalog:
        return SkillCatalog(
            tuple(spec for spec in self._specs.values() if spec.agent_visible)
        )


@dataclass(frozen=True)
class SkillResult:
    success: bool
    summary: str
    status: str = "completed"
    failure_mode: str | None = None
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


class BaseSkill:
    spec: SkillSpec

    async def execute(
        self,
        ctx: SkillContext,
        arguments: dict[str, Any],
    ) -> SkillResult:
        raise NotImplementedError


from hey_robot.skill_os.context import SkillContext
