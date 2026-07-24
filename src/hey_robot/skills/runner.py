"""Single execution boundary for top-level and nested skills."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol

from hey_robot.skills.context import SkillCancelledError, SkillContext
from hey_robot.skills.models import Skill, SkillCommand, SkillEvent, SkillResult
from hey_robot.skills.registry import SkillRegistry
from hey_robot.skills.resources import ResourceManager


class SkillEventSink(Protocol):
    async def emit(self, event: SkillEvent) -> None: ...


ContextFactory = Callable[[SkillCommand], SkillContext]


class SkillRunner:
    def __init__(
        self,
        registry: SkillRegistry,
        *,
        resources: ResourceManager,
        events: SkillEventSink,
        context_factory: ContextFactory | None = None,
    ) -> None:
        self._registry = registry
        self._resources = resources
        self._events = events
        self._context_factory = context_factory or _default_context
        self._cancelled: set[str] = set()
        self._sequences: dict[str, int] = {}

    def cancel(self, run_id: str) -> None:
        self._cancelled.add(run_id)

    async def execute(self, command: SkillCommand) -> SkillResult:
        try:
            skill = self._registry.get(command.name)
            arguments = validate_arguments(skill.parameters, command.arguments)
        except (KeyError, ValueError) as exc:
            result = SkillResult(
                False,
                f"{command.name} rejected",
                "failed",
                failure_mode="invalid_request",
                error=str(exc),
            )
            try:
                await self._emit(command, command.name, "accepted")
                await self._emit(command, command.name, "failed", result=result)
            finally:
                self._clear_run_state(command.run_id)
            return result

        await self._emit(command, skill.name, "accepted")
        await self._emit(command, skill.name, "running")
        try:
            result = await self._execute_skill(command, skill, arguments)
        except asyncio.CancelledError:
            result = SkillResult(
                False,
                "skill cancelled",
                "cancelled",
                failure_mode="cancelled",
            )
        except SkillCancelledError as exc:
            result = SkillResult(
                False,
                "skill cancelled",
                "cancelled",
                failure_mode="cancelled",
                error=str(exc),
            )
        except TimeoutError:
            result = SkillResult(
                False,
                f"{skill.name} timed out",
                "failed",
                failure_mode="timeout",
            )
        except Exception as exc:
            result = SkillResult(
                False,
                f"{skill.name} failed",
                "failed",
                failure_mode="internal_error",
                error=str(exc),
            )
        result = _normalize_result(result)
        phase = result.status
        try:
            await self._emit(command, skill.name, phase, result=result)
        finally:
            self._clear_run_state(command.run_id)
        return result

    def _clear_run_state(self, run_id: str) -> None:
        self._cancelled.discard(run_id)
        self._sequences.pop(run_id, None)

    async def _execute_skill(
        self,
        command: SkillCommand,
        skill: Skill,
        arguments: dict[str, Any],
    ) -> SkillResult:
        context = self._context_factory(command)

        async def invoke(name: str, child_arguments: dict[str, Any]) -> SkillResult:
            if name not in skill.dependencies:
                raise RuntimeError(
                    f"skill {skill.name!r} called undeclared dependency {name!r}"
                )
            child = self._registry.get(name)
            resolved = validate_arguments(child.parameters, child_arguments)
            return await self._execute_skill(command, child, resolved)

        async def progress(value: float, summary: str | None) -> None:
            await self._emit(
                command, skill.name, "progress", progress=value, summary=summary
            )

        context._invoke = invoke
        context._progress = progress
        context._cancelled = lambda: command.run_id in self._cancelled
        timeout_sec = _effective_timeout(skill, command.deadline_at)
        async with self._resources.acquire(
            command.robot_id,
            skill.resources,
            owner=command.run_id,
        ):
            context.raise_if_cancelled()
            return await asyncio.wait_for(
                skill.handler(context, arguments), timeout_sec
            )

    async def _emit(
        self,
        command: SkillCommand,
        name: str,
        phase: str,
        *,
        progress: float | None = None,
        summary: str | None = None,
        result: SkillResult | None = None,
    ) -> None:
        sequence = self._sequences.get(command.run_id, 0) + 1
        self._sequences[command.run_id] = sequence
        await self._events.emit(
            SkillEvent(
                envelope=command.envelope,
                run_id=command.run_id,
                sequence=sequence,
                name=name,
                phase=phase,  # type: ignore[arg-type]
                timestamp=time.time(),
                progress=progress,
                summary=summary,
                result=result,
            )
        )


def validate_arguments(
    schema: dict[str, Any], arguments: dict[str, Any]
) -> dict[str, Any]:
    if schema.get("type") not in {None, "object"}:
        raise ValueError("skill parameters must be an object schema")
    resolved = dict(arguments)
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return resolved
    if schema.get("additionalProperties") is False:
        unknown = set(resolved) - set(properties)
        if unknown:
            raise ValueError(f"unexpected arguments: {', '.join(sorted(unknown))}")
    for name, definition in properties.items():
        if (
            name not in resolved
            and isinstance(definition, dict)
            and "default" in definition
        ):
            resolved[name] = definition["default"]
    required = schema.get("required", ())
    for name in required if isinstance(required, list | tuple) else ():
        if name not in resolved:
            raise ValueError(f"missing required argument: {name}")
    for name, value in resolved.items():
        definition = properties.get(name)
        if isinstance(definition, dict):
            _validate_value(name, value, definition)
    return resolved


def _validate_value(name: str, value: Any, definition: dict[str, Any]) -> None:
    expected = definition.get("type")
    valid = {
        "string": isinstance(value, str),
        "number": isinstance(value, int | float) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
    }
    if expected in valid and not valid[expected]:
        raise ValueError(f"argument {name} must be a {expected}")
    if "enum" in definition and value not in definition["enum"]:
        raise ValueError(f"argument {name} must be one of {definition['enum']}")
    min_length = definition.get("minLength")
    if (
        isinstance(value, str)
        and isinstance(min_length, int)
        and len(value) < min_length
    ):
        raise ValueError(f"argument {name} must have length >= {min_length}")
    if isinstance(value, int | float) and not isinstance(value, bool):
        minimum = definition.get("minimum")
        maximum = definition.get("maximum")
        if isinstance(minimum, int | float) and value < minimum:
            raise ValueError(f"argument {name} must be >= {minimum}")
        if isinstance(maximum, int | float) and value > maximum:
            raise ValueError(f"argument {name} must be <= {maximum}")


def _effective_timeout(skill: Skill, deadline_at: float | None) -> float:
    if deadline_at is None:
        return skill.timeout_sec
    return max(0.001, min(skill.timeout_sec, deadline_at - time.time()))


def _normalize_result(result: SkillResult) -> SkillResult:
    if result.success and result.status != "completed":
        return replace(result, status="completed")
    if not result.success and result.status == "completed":
        return replace(result, status="failed")
    return result


def _default_context(command: SkillCommand) -> SkillContext:
    return SkillContext(
        run_id=command.run_id,
        task_id=command.task_id,
        robot_id=command.robot_id,
    )
