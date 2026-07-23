"""Explicit migration boundary for the pre-refactor Skill OS types."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from hey_robot.bus.factory import create_bus_client
from hey_robot.bus.types import MessageBus
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import (
    ActionProposal,
    ShortOperationCommand,
    SkillControl,
    SkillResult as LegacyProtocolSkillResult,
    Topics,
)
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.skill_os.base import (
    BaseSkill,
    SkillResult as LegacySkillResult,
)
from hey_robot.skill_os.context import SkillContext as LegacySkillContext
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import (
    Skill,
    SkillCancel,
    SkillCommand,
    SkillEvent,
    SkillResult,
)
from hey_robot.skills.registry import SkillRegistry

LegacyContextFactory = Callable[[SkillContext], LegacySkillContext]


def adapt_legacy_skill(
    legacy_skill: BaseSkill,
    *,
    context_factory: LegacyContextFactory,
) -> Skill:
    """Adapt one old Skill OS implementation at the worker composition root."""

    spec = legacy_skill.spec

    async def handler(
        context: SkillContext, arguments: dict[str, object]
    ) -> SkillResult:
        result = await legacy_skill.execute(context_factory(context), arguments)
        return from_legacy_result(result)

    required_models = (
        (spec.required_model_service,) if spec.required_model_service else ()
    )
    return Skill(
        name=spec.name,
        description=spec.description,
        parameters=spec.input_schema,
        handler=handler,
        resources=spec.required_resources,
        timeout_sec=spec.timeout_sec,
        supported_robots=spec.supported_robots,
        required_actions=spec.driver_primitives,
        required_models=required_models,
    )


def adapt_legacy_skills(
    legacy_skills: Iterable[BaseSkill],
    *,
    context_factory: LegacyContextFactory,
) -> SkillRegistry:
    """Build a new registry from executable legacy skills during migration."""

    registry = SkillRegistry()
    for legacy_skill in legacy_skills:
        registry.register(
            adapt_legacy_skill(legacy_skill, context_factory=context_factory)
        )
    return registry


def from_legacy_result(value: LegacySkillResult) -> SkillResult:
    status = value.status if value.status in {"failed", "cancelled"} else "completed"
    if not value.success and status == "completed":
        status = "failed"
    return SkillResult(
        success=value.success,
        summary=value.summary,
        status=status,
        data=dict(value.data),
        failure_mode=value.failure_mode,
        error=value.error,
    )


def to_legacy_result(event: SkillEvent) -> LegacyProtocolSkillResult:
    if event.result is None:
        raise ValueError("terminal SkillEvent must include result")
    status = "completed" if event.result.status == "completed" else "failed"
    if event.result.status == "cancelled":
        status = "interrupted"
    return LegacyProtocolSkillResult(
        envelope=event.envelope,
        skill_id=event.run_id,
        name=event.name,
        status=status,  # type: ignore[arg-type]
        success=event.result.success,
        steps_executed=0,
        progress=event.progress or 1.0,
        summary=event.result.summary,
        failure_mode=event.result.failure_mode,
        frame_id=event.frame_id,
        error=event.result.error,
        observations=list(event.result.observations),
        evidence=(),
        metadata=dict(event.result.data),
    )


@dataclass(frozen=True)
class _LegacyRun:
    command: SkillCommand
    intent_kind: str


class LegacySkillWorkerBridge:
    """Adapt the existing controller wire protocol during the worker migration."""

    def __init__(self, bus: MessageBus, *, topics: Topics | None = None) -> None:
        self._bus = bus
        self._topics = topics or Topics()
        self._runs: dict[str, _LegacyRun] = {}
        self._sequences: dict[str, int] = {}
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await self._bus.subscribe([self._topics.skill_command], self._on_command)
        await self._bus.subscribe([self._topics.skill_cancel], self._on_cancel)
        await self._bus.subscribe([self._topics.skill_result], self._on_result)
        self._started = True

    async def close(self) -> None:
        if not self._started:
            return
        await self._bus.unsubscribe(
            [
                self._topics.skill_command,
                self._topics.skill_cancel,
                self._topics.skill_result,
            ]
        )
        self._started = False

    async def _on_command(self, _topic: str, payload: dict[str, Any]) -> None:
        command = from_payload(SkillCommand, payload)
        if command.run_id in self._runs:
            return
        intent_kind = "observation" if command.name == "inspect_scene" else "skill"
        self._runs[command.run_id] = _LegacyRun(command, intent_kind)
        await self._emit(command, "accepted")
        await self._bus.publish(
            self._topics.short_operation_command,
            to_payload(
                ShortOperationCommand(
                    envelope=command.envelope,
                    operation_id=command.run_id,
                    proposal=ActionProposal(
                        intent_kind,
                        command.name,
                        _objective(command),
                        command.arguments,
                    ),
                    timeout_sec=_timeout(command),
                )
            ),
        )
        await self._emit(command, "running")

    async def _on_cancel(self, _topic: str, payload: dict[str, Any]) -> None:
        cancel = from_payload(SkillCancel, payload)
        run = self._runs.get(cancel.run_id)
        task_id = run.command.task_id if run is not None else None
        envelope = run.command.envelope if run is not None else cancel.envelope
        await self._bus.publish(
            self._topics.skill_control,
            to_payload(
                SkillControl(
                    envelope=envelope,
                    control_id=f"cancel_{cancel.run_id}",
                    action="interrupt",
                    target_skill_id=cancel.run_id,
                    task_id=task_id,
                    reason=cancel.reason,
                )
            ),
        )

    async def _on_result(self, _topic: str, payload: dict[str, Any]) -> None:
        legacy = from_payload(LegacyProtocolSkillResult, payload)
        run = self._runs.get(legacy.skill_id)
        if run is None:
            return
        success = legacy.status == "completed" and legacy.success is True
        status = "completed" if success else "failed"
        if legacy.status == "interrupted":
            status = "cancelled"
        result = SkillResult(
            success=success,
            summary=legacy.summary or legacy.error or "skill finished",
            status=status,
            data=dict(legacy.metadata),
            evidence_ids=tuple(item.evidence_id for item in legacy.evidence),
            observations=tuple(legacy.observations),
            failure_mode=legacy.failure_mode,
            error=legacy.error,
        )
        await self._emit(
            run.command,
            result.status,
            result=result,
            frame_id=legacy.frame_id,
        )
        self._runs.pop(legacy.skill_id, None)

    async def _emit(
        self,
        command: SkillCommand,
        phase: str,
        *,
        result: SkillResult | None = None,
        frame_id: int | None = None,
    ) -> None:
        sequence = self._sequences.get(command.run_id, 0) + 1
        self._sequences[command.run_id] = sequence
        await self._bus.publish(
            self._topics.skill_run_event,
            to_payload(
                SkillEvent(
                    envelope=command.envelope,
                    run_id=command.run_id,
                    sequence=sequence,
                    name=command.name,
                    phase=phase,  # type: ignore[arg-type]
                    timestamp=time.time(),
                    frame_id=frame_id,
                    result=result,
                )
            ),
        )


def _objective(command: SkillCommand) -> str:
    value = command.arguments.get("objective") or command.arguments.get("question")
    return (
        value.strip()
        if isinstance(value, str) and value.strip()
        else f"execute {command.name}"
    )


def _timeout(command: SkillCommand) -> float:
    if command.deadline_at is None:
        return 60.0
    return max(0.001, command.deadline_at - time.time())


class LegacySkillWorkerBridgeService:
    """Deploy the migration bridge beside the existing SkillControllerService."""

    def __init__(self, config: DeploymentConfig) -> None:
        self._bus = create_bus_client(config.deployment.bus, role="skill_controller")
        self._bridge = LegacySkillWorkerBridge(self._bus)

    async def start(self) -> None:
        await self._bus.connect()
        await self._bridge.start()
        await asyncio.Event().wait()

    async def stop(self) -> None:
        await self._bridge.close()
        await self._bus.close()
