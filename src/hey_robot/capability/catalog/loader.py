from __future__ import annotations

from typing import TYPE_CHECKING

from hey_robot.capability.catalog.models import (
    CapabilityManifest,
    RobotSkillCapability,
    ToolCapability,
)
from hey_robot.skills.base import SkillCatalog
from hey_robot.skills.catalog import RobotSkillCatalog
from hey_robot.skills.registry import SkillRegistry

if TYPE_CHECKING:
    from hey_robot.agents.tools.registry import ToolRegistry


class CapabilityLoader:
    """Build the current agent capability inventory from runtime components."""

    def __init__(
        self,
        *,
        tools: ToolRegistry | None = None,
        robot_skills: SkillRegistry | SkillCatalog | RobotSkillCatalog | None = None,
    ) -> None:
        self.tools = tools
        self.robot_skills = robot_skills

    def build(
        self,
        *,
        robot_type: str | None = None,
    ) -> CapabilityManifest:
        return CapabilityManifest(
            tools=self._tools(),
            robot_skill_actions=self._robot_skills(robot_type),
            robot_type=robot_type,
        )

    def _tools(self) -> tuple[ToolCapability, ...]:
        if self.tools is None:
            return ()
        capabilities = []
        for item in self.tools.list_tools():
            annotations = _mapping(item.get("annotations"))
            capabilities.append(
                ToolCapability(
                    name=str(item.get("name") or ""),
                    source=str(annotations.get("source") or "local"),
                    description=str(item.get("description") or ""),
                    input_schema=dict(item.get("inputSchema") or {}),
                    safety_level=str(annotations.get("safetyLevel") or "normal"),
                    read_only=bool(annotations.get("readOnlyHint", False)),
                    destructive=bool(annotations.get("destructiveHint", False)),
                )
            )
        return tuple(capabilities)

    def _robot_skills(self, robot_type: str | None) -> tuple[RobotSkillCapability, ...]:
        if self.robot_skills is None:
            return ()
        if isinstance(self.robot_skills, SkillRegistry):
            return self._runtime_catalog_skills(
                self.robot_skills.catalog(enabled_only=False)
            )
        if isinstance(self.robot_skills, RobotSkillCatalog):
            return tuple(
                RobotSkillCapability(
                    name=item.name,
                    description=item.description,
                    input_schema=item.input_schema,
                    safety_level=item.safety_level,
                    required_resources=item.required_resources,
                    preconditions=item.preconditions,
                    success_criteria=item.success_criteria,
                    failure_modes=item.failure_modes,
                    recovery_hints=item.recovery_hints,
                    timeout_sec=item.timeout_sec,
                    interruptible=item.interruptible,
                    feedback_mode=item.feedback_mode,
                    refresh_observation=_refresh_observation(item),
                )
                for item in self.robot_skills.list(robot_type=robot_type)
            )
        return self._runtime_catalog_skills(self.robot_skills)

    def _runtime_catalog_skills(
        self,
        catalog: SkillCatalog,
    ) -> tuple[RobotSkillCapability, ...]:
        return tuple(
            RobotSkillCapability(
                name=item.name,
                description=item.description,
                input_schema=item.input_schema,
                safety_level=item.safety_level,
                required_resources=item.required_resources,
                preconditions=item.preconditions,
                success_criteria=item.success_criteria,
                failure_modes=item.failure_modes,
                recovery_hints=item.recovery_hints,
                timeout_sec=item.timeout_sec,
                interruptible=item.interruptible,
                feedback_mode=item.feedback_mode,
                refresh_observation=_refresh_observation(item),
            )
            for item in catalog.list()
        )


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _refresh_observation(item: object) -> bool:
    refresh = getattr(item, "refresh_observation", None)
    if isinstance(refresh, bool):
        return refresh
    return True
