from __future__ import annotations

from typing import Any, Protocol

from hey_robot.foundation.catalog.models import (
    RobotSkillSurface,
    SkillSurfaceManifest,
    ToolSurface,
)


class ToolRegistryLike(Protocol):
    def list_tools(self) -> list[dict[str, Any]]: ...


class RuntimeSkillCatalogLike(Protocol):
    def list(self) -> tuple[Any, ...] | list[Any]: ...


class SkillSurfaceLoader:
    """根据运行时组件构建当前 Agent 的工具和 Skill 接口。"""

    def __init__(
        self,
        *,
        tools: ToolRegistryLike | None = None,
        robot_skills: RuntimeSkillCatalogLike | tuple[Any, ...] | None = None,
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
        del robot_type
        items = (
            self.robot_skills
            if isinstance(self.robot_skills, tuple)
            else tuple(self.robot_skills.list())
        )
        return self._runtime_catalog_skills(items)

    def _runtime_catalog_skills(
        self,
        items: tuple[Any, ...] | list[Any],
    ) -> tuple[RobotSkillSurface, ...]:
        return tuple(
            RobotSkillSurface(
                name=item.name,
                description=item.description,
                input_schema=dict(
                    getattr(item, "input_schema", getattr(item, "parameters", {}))
                ),
                safety_level=str(getattr(item, "safety_level", "normal")),
                required_resources=tuple(
                    getattr(item, "required_resources", getattr(item, "resources", ()))
                ),
                preconditions=tuple(getattr(item, "preconditions", ())),
                success_criteria=tuple(getattr(item, "success_criteria", ())),
                failure_modes=tuple(getattr(item, "failure_modes", ())),
                recovery_hints=tuple(getattr(item, "recovery_hints", ())),
                timeout_sec=item.timeout_sec,
                interruptible=bool(getattr(item, "interruptible", True)),
                feedback_mode=str(getattr(item, "feedback_mode", "status")),
                refresh_observation=_refresh_observation(item),
            )
            for item in items
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
