from __future__ import annotations

from typing import Any, Protocol, cast

from hey_robot.contracts import SkillContractCatalog
from hey_robot.foundation.catalog.models import (
    RobotSkillSurface,
    SkillSurfaceManifest,
    ToolSurface,
)


class ToolRegistryLike(Protocol):
    def list_tools(self) -> list[dict[str, Any]]: ...


class RuntimeSkillCatalogLike(Protocol):
    def list(self) -> tuple[Any, ...] | list[Any]: ...


class RobotSkillRegistryLike(Protocol):
    def robot_skill_catalog(self) -> SkillContractCatalog: ...


class SkillSurfaceLoader:
    """Build the current agent tool and skill surface from runtime components."""

    def __init__(
        self,
        *,
        tools: ToolRegistryLike | None = None,
        robot_skills: (
            RobotSkillRegistryLike
            | RuntimeSkillCatalogLike
            | SkillContractCatalog
            | None
        ) = None,
    ) -> None:
        self.tools = tools
        self.robot_skills = robot_skills

    def build(
        self,
        *,
        robot_type: str | None = None,
    ) -> SkillSurfaceManifest:
        return SkillSurfaceManifest(
            tools=self._tools(),
            robot_skills=self._robot_skills(robot_type),
            robot_type=robot_type,
        )

    def _tools(self) -> tuple[ToolSurface, ...]:
        if self.tools is None:
            return ()
        surfaces = []
        for item in self.tools.list_tools():
            annotations = _mapping(item.get("annotations"))
            surfaces.append(
                ToolSurface(
                    name=str(item.get("name") or ""),
                    source=str(annotations.get("source") or "local"),
                    description=str(item.get("description") or ""),
                    input_schema=dict(item.get("inputSchema") or {}),
                    safety_level=str(annotations.get("safetyLevel") or "normal"),
                    read_only=bool(annotations.get("readOnlyHint", False)),
                    destructive=bool(annotations.get("destructiveHint", False)),
                )
            )
        return tuple(surfaces)

    def _robot_skills(self, robot_type: str | None) -> tuple[RobotSkillSurface, ...]:
        if self.robot_skills is None:
            return ()
        if isinstance(self.robot_skills, SkillContractCatalog):
            return tuple(
                RobotSkillSurface(
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
        runtime_catalog = getattr(self.robot_skills, "catalog", None)
        if callable(runtime_catalog):
            try:
                catalog = runtime_catalog(enabled_only=True)
            except TypeError:
                catalog = runtime_catalog()
            return self._runtime_catalog_skills(catalog)
        robot_skill_catalog = getattr(self.robot_skills, "robot_skill_catalog", None)
        if callable(robot_skill_catalog):
            catalog = robot_skill_catalog()
            return tuple(
                RobotSkillSurface(
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
                for item in catalog.list(robot_type=robot_type)
            )
        return self._runtime_catalog_skills(
            cast(RuntimeSkillCatalogLike, self.robot_skills)
        )

    def _runtime_catalog_skills(
        self,
        catalog: RuntimeSkillCatalogLike,
    ) -> tuple[RobotSkillSurface, ...]:
        return tuple(
            RobotSkillSurface(
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
