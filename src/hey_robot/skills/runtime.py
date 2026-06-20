from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hey_robot.protocol import RobotStatus
from hey_robot.skills.base import SkillResult
from hey_robot.skills.catalog import RobotSkillSpec
from hey_robot.skills.context import SkillContext
from hey_robot.skills.contracts import SkillContractDecision, SkillContractRuntime
from hey_robot.skills.registry import SkillRegistry

SkillInvoke = Callable[[str, dict[str, Any] | None], Any]
SkillContextFactory = Callable[[SkillInvoke], SkillContext]


class SkillRuntime:
    """The single execution boundary for top-level and nested skills."""

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry
        self.contracts = SkillContractRuntime(registry.robot_skill_catalog())

    def validate(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        enabled_only: bool = True,
        status: RobotStatus | None = None,
        robot_type: str | None = None,
    ) -> tuple[RobotSkillSpec, SkillContractDecision]:
        registered = self.registry.get(name, enabled_only=enabled_only)
        if registered.skill is None:
            raise KeyError(f"skill {name!r} is not backed by a plugin implementation")
        contract = self.contracts.resolve(name, robot_type=robot_type)
        decision = self.contracts.acceptance_decision(
            contract,
            status=status,
            arguments=dict(arguments or {}),
        )
        return contract, decision

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        context_factory: SkillContextFactory,
        enabled_only: bool = True,
        status: RobotStatus | None = None,
        robot_type: str | None = None,
    ) -> SkillResult:
        resolved_arguments = dict(arguments or {})
        _, decision = self.validate(
            name,
            resolved_arguments,
            enabled_only=enabled_only,
            status=status,
            robot_type=robot_type,
        )
        if not decision.allowed:
            return SkillResult(
                success=False,
                summary=decision.reason,
                status="failed",
                failure_mode=decision.failure_mode,
                error=decision.reason,
                data=dict(decision.metadata),
            )

        async def invoke(
            child_name: str,
            child_arguments: dict[str, Any] | None = None,
        ) -> SkillResult:
            return await self.execute(
                child_name,
                child_arguments,
                context_factory=context_factory,
                enabled_only=False,
                status=status,
                robot_type=robot_type,
            )

        try:
            registered = self.registry.get(name, enabled_only=enabled_only)
            assert registered.skill is not None
            return await registered.skill.execute(
                context_factory(invoke),
                resolved_arguments,
            )
        except Exception as exc:
            return SkillResult(
                success=False,
                summary=str(exc),
                status="failed",
                failure_mode="internal_error",
                error=str(exc),
            )
