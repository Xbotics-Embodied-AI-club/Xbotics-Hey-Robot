from __future__ import annotations

from dataclasses import dataclass

import pytest

from hey_robot.cognition.tools import (
    HarnessTool,
    HarnessToolCall,
    ToolSpec,
)
from hey_robot.cognition.tools.models import PhysicalToolCall
from hey_robot.cognition.tools.registry import ToolDependencies, ToolRegistry
from hey_robot.protocol import ToolOutcome
from hey_robot.skills.models import Skill, SkillResult


async def _unused_skill(_context, _arguments) -> SkillResult:
    return SkillResult(True, "unused", "completed")


@dataclass(frozen=True)
class _Catalog:
    skills: tuple[Skill, ...]

    def list(self) -> tuple[Skill, ...]:
        return self.skills

    def get(self, name: str) -> Skill:
        return next(skill for skill in self.skills if skill.name == name)


def _skill() -> Skill:
    return Skill(
        name="pick",
        description="Pick a named object.",
        parameters={
            "type": "object",
            "properties": {"object": {"type": "string"}},
            "required": ["object"],
            "additionalProperties": False,
        },
        handler=_unused_skill,
    )


def test_skill_tool_schema_is_projected_from_skill() -> None:
    skill = _skill()
    registry = ToolRegistry(ToolDependencies((skill,)))

    definition = next(
        item for item in registry.definitions if item["function"]["name"] == "pick"
    )

    assert definition["function"]["description"] == skill.description
    assert definition["function"]["parameters"] == skill.parameters
    proposal = registry.prepare("pick", {"object": "mug"})
    assert proposal == PhysicalToolCall("pick", {"object": "mug"})


@pytest.mark.asyncio
async def test_registry_accepts_harness_and_skill_tools() -> None:
    received: list[dict[str, object]] = []

    async def remember(arguments: dict[str, object]) -> ToolOutcome:
        received.append(arguments)
        return ToolOutcome("completed", "remembered", data={"stored": True})

    memory = HarnessTool(
        ToolSpec(
            "remember",
            "Store one value in bounded in-memory state.",
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        ),
        remember,
    )
    registry = ToolRegistry(ToolDependencies((_skill(),), (memory,)))

    call = registry.prepare("remember", {"value": "blue cup"})

    assert isinstance(call, HarnessToolCall)
    assert call.arguments == {"value": "blue cup"}
    assert (await call.execute()).data == {"stored": True}
    assert received == [{"value": "blue cup"}]


def test_harness_tool_rejects_unknown_arguments_before_handler() -> None:
    called = False

    def handler(_arguments: dict[str, object]) -> ToolOutcome:
        nonlocal called
        called = True
        return ToolOutcome("completed", "done")

    tool = HarnessTool(
        ToolSpec(
            "bounded",
            "A bounded test tool.",
            {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
        handler,
    )
    registry = ToolRegistry(ToolDependencies((), (tool,)))

    with pytest.raises(ValueError, match="unexpected arguments"):
        registry.prepare("bounded", {"unsafe": True})

    assert called is False
