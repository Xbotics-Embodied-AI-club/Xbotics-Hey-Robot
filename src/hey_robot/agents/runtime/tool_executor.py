from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass
from typing import Any

from hey_robot.agents.runtime.audit import ToolAuditLogger, ToolAuditRecord
from hey_robot.agents.runtime.hooks import ToolHook, ToolHookContext
from hey_robot.agents.runtime.permissions import PermissionManager
from hey_robot.agents.tools.registry import ToolRegistry
from hey_robot.capability.catalog.resolver import CapabilityResolver


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_call_id: str
    tool: str
    arguments: dict[str, Any]
    result: str
    success: bool
    permission_behavior: str
    permission_reason: str
    capability_behavior: str = "allow"
    capability_reason: str = "not evaluated"
    capability_rule: str = "none"
    capability_source: str = ""
    capability_safety_level: str = ""
    error: str | None = None


class ToolExecutor:
    """Production-oriented execution boundary around runtime tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        permission_manager: PermissionManager | None = None,
        hooks: list[ToolHook] | None = None,
        audit_logger: ToolAuditLogger | None = None,
        capability_resolver: CapabilityResolver | None = None,
        default_timeout_sec: float = 60.0,
    ) -> None:
        self.registry = registry
        self.permission_manager = permission_manager or PermissionManager()
        self.hooks = list(hooks or [])
        self.audit_logger = audit_logger
        self.capability_resolver = capability_resolver or CapabilityResolver(registry)
        self.default_timeout_sec = float(default_timeout_sec)

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] | None = None,
        task: str | None = None,
        task_step: str | None = None,
    ) -> ToolExecutionResult:
        started = time.time()
        tool_call_id = str(uuid.uuid4())
        args = dict(arguments or {})
        resolution = self.capability_resolver.resolve(name, context=context)
        if not resolution.allowed:
            result = ToolExecutionResult(
                tool_call_id=tool_call_id,
                tool=name,
                arguments=args,
                result=f"Capability {resolution.behavior}: {resolution.reason}",
                success=False,
                permission_behavior=resolution.behavior,
                permission_reason=resolution.reason,
                capability_behavior=resolution.behavior,
                capability_reason=resolution.reason,
                capability_rule=resolution.rule,
                capability_source=resolution.source,
                capability_safety_level=resolution.safety_level,
                error=resolution.reason,
            )
            self._audit(result, started, task=task, task_step=task_step)
            return result
        assert resolution.tool is not None
        tool = resolution.tool
        args = _coerce_json_schema_arguments(args, tool.input_schema)
        validation_error = _validate_json_schema(args, tool.input_schema)
        if validation_error:
            result = ToolExecutionResult(
                tool_call_id=tool_call_id,
                tool=name,
                arguments=args,
                result=f"ValidationError: {validation_error}",
                success=False,
                permission_behavior="deny",
                permission_reason="input validation failed",
                capability_behavior=resolution.behavior,
                capability_reason=resolution.reason,
                capability_rule=resolution.rule,
                capability_source=resolution.source,
                capability_safety_level=resolution.safety_level,
                error=validation_error,
            )
            self._audit(result, started, task=task, task_step=task_step)
            return result

        decision = self.permission_manager.check(tool, args)
        if decision.updated_input is not None:
            args = decision.updated_input
        if decision.behavior != "allow":
            result = ToolExecutionResult(
                tool_call_id=tool_call_id,
                tool=name,
                arguments=args,
                result=f"Permission {decision.behavior}: {decision.reason}",
                success=False,
                permission_behavior=decision.behavior,
                permission_reason=decision.reason,
                capability_behavior=resolution.behavior,
                capability_reason=resolution.reason,
                capability_rule=resolution.rule,
                capability_source=resolution.source,
                capability_safety_level=resolution.safety_level,
                error=decision.reason,
            )
            self._audit(result, started, task=task, task_step=task_step)
            return result

        hook_context = ToolHookContext(
            tool_call_id=tool_call_id, tool=tool, arguments=args
        )
        try:
            for hook in self.hooks:
                await hook.before_tool(hook_context)
            timeout = (
                tool.timeout_sec
                if tool.timeout_sec is not None
                else self.default_timeout_sec
            )
            tool_call = self.registry.call_tool(name, args, context=context)
            if timeout <= 0:
                result_text = await tool_call
            else:
                result_text = await asyncio.wait_for(tool_call, timeout=timeout)
            for hook in self.hooks:
                await hook.after_tool(hook_context, result_text)
            result = ToolExecutionResult(
                tool_call_id=tool_call_id,
                tool=name,
                arguments=args,
                result=result_text,
                success=True,
                permission_behavior=decision.behavior,
                permission_reason=decision.reason,
                capability_behavior=resolution.behavior,
                capability_reason=resolution.reason,
                capability_rule=resolution.rule,
                capability_source=resolution.source,
                capability_safety_level=resolution.safety_level,
            )
        except Exception as exc:
            for hook in self.hooks:
                with contextlib.suppress(Exception):
                    await hook.on_tool_error(hook_context, exc)
            result = ToolExecutionResult(
                tool_call_id=tool_call_id,
                tool=name,
                arguments=args,
                result=f"{type(exc).__name__}: {exc}",
                success=False,
                permission_behavior=decision.behavior,
                permission_reason=decision.reason,
                capability_behavior=resolution.behavior,
                capability_reason=resolution.reason,
                capability_rule=resolution.rule,
                capability_source=resolution.source,
                capability_safety_level=resolution.safety_level,
                error=f"{type(exc).__name__}: {exc}",
            )

        self._audit(result, started, task=task, task_step=task_step)
        return result

    def _audit(
        self,
        result: ToolExecutionResult,
        started: float,
        *,
        task: str | None,
        task_step: str | None,
    ) -> None:
        if self.audit_logger is None:
            return
        ended = time.time()
        self.audit_logger.write(
            ToolAuditRecord(
                tool_call_id=result.tool_call_id,
                tool=result.tool,
                arguments=result.arguments,
                status="success" if result.success else "failed",
                started_at=started,
                ended_at=ended,
                duration_sec=ended - started,
                permission_behavior=result.permission_behavior,
                permission_reason=result.permission_reason,
                capability_behavior=result.capability_behavior,
                capability_reason=result.capability_reason,
                capability_rule=result.capability_rule,
                capability_source=result.capability_source,
                capability_safety_level=result.capability_safety_level,
                result_preview=result.result[:1000],
                error=result.error,
                task=task,
                task_step=task_step,
            )
        )


def _resolve_type(raw_type: Any) -> Any:
    """Extract the non-null type from a JSON Schema type field."""
    if isinstance(raw_type, list):
        return next((x for x in raw_type if x != "null"), None)
    return raw_type


def _validate_json_schema(
    arguments: dict[str, Any], schema: dict[str, Any]
) -> str | None:
    required = schema.get("required") or []
    for key in required:
        if key not in arguments:
            return f"missing required argument: {key}"
    properties = schema.get("properties") or {}
    for key, value in arguments.items():
        raw_type = properties.get(key, {}).get("type")
        expected = _resolve_type(raw_type)
        if value is None and isinstance(raw_type, list) and "null" in raw_type:
            continue
        if expected and not _matches_json_type(value, expected):
            return f"argument {key} expected {expected}, got {type(value).__name__}"
    return None


def _coerce_json_schema_arguments(
    arguments: dict[str, Any], schema: dict[str, Any]
) -> dict[str, Any]:
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        return arguments
    coerced = dict(arguments)
    for key, value in list(coerced.items()):
        expected = (
            properties.get(key, {}).get("type")
            if isinstance(properties.get(key), dict)
            else None
        )
        resolved = _resolve_type(expected)
        if resolved == "boolean" and isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1", "是", "对"}:
                coerced[key] = True
            elif normalized in {"false", "no", "0", "否", "不"}:
                coerced[key] = False
        elif resolved in ("integer", "number") and isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                continue
            try:
                coerced[key] = (
                    int(normalized) if resolved == "integer" else float(normalized)
                )
            except ValueError:
                continue
    return coerced


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return True
