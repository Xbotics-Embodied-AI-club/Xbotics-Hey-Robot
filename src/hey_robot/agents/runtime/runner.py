from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from hey_robot.agents.runtime.agent_run import AgentRunRecorder
from hey_robot.agents.runtime.audit import ToolAuditLogger
from hey_robot.agents.runtime.grounding import (
    is_perception_evidence_record,
    needs_perception_grounding,
)
from hey_robot.agents.runtime.hooks import (
    AgentRuntimeHook,
    AgentRuntimeHookContext,
    ToolHook,
)
from hey_robot.agents.runtime.message_protocol import (
    validate_provider_messages,
    validate_provider_response,
)
from hey_robot.agents.runtime.message_window import (
    MessageWindowPolicy,
    apply_message_window,
)
from hey_robot.agents.runtime.permissions import PermissionManager, PermissionMode
from hey_robot.agents.runtime.prompts import (
    AgentPromptTemplates,
    build_system_prompt,
    build_turn_prompt,
    load_agent_prompt_templates,
)
from hey_robot.agents.runtime.registry import ToolResultPolicy
from hey_robot.agents.runtime.response_policy import (
    decide_response,
    looks_like_internal_agent_protocol,
    looks_like_unexecuted_tool_protocol,
)
from hey_robot.agents.runtime.state import AgentState
from hey_robot.agents.runtime.tool_executor import ToolExecutor
from hey_robot.agents.tools.registry import ToolRegistry
from hey_robot.capability.catalog.resolver import CapabilityResolver
from hey_robot.providers import (
    ReasoningImage,
    ReasoningMessage,
    ReasoningProvider,
    ReasoningResponse,
)
from hey_robot.user_reply import (
    looks_like_internal_user_reply,
    present_tool_result_for_user,
)

AsyncTool = Callable[..., Awaitable[str] | str]


@dataclass
class AgentRuntimeInput:
    task: str
    images: list[np.ndarray]
    robot_state: str
    robot_status: dict[str, Any] | None = None
    memory_context: str | None = None
    autonomy_context: str | None = None
    last_feedback: str | None = None
    next_hint: str | None = None
    skill_in_progress: bool = False
    recovery_context: str | None = None
    allowed_tools: set[str] | None = None


@dataclass
class AgentRunSpec:
    messages: list[ReasoningMessage]
    task: str
    robot_state: str
    robot_status: dict[str, Any] | None = None
    allowed_tools: set[str] | None = None
    max_iterations: int | None = None
    provider_timeout_sec: float | None = None
    message_window: MessageWindowPolicy | None = None
    payload: AgentRuntimeInput | None = None


@dataclass
class AgentRunResult:
    final_content: str | None
    messages: list[ReasoningMessage]
    tools_used: list[str]
    stop_reason: str
    runtime_result: AgentRuntimeResult
    error: str | None = None


@dataclass
class AgentRuntimeResult:
    tool: str
    args: dict[str, Any]
    result: str
    reason: str = ""
    tool_call_id: str = ""
    task_finished: bool = False
    stop_reason: str = "completed"
    tool_success: bool | None = None


@dataclass
class _ExecutedToolCall:
    provider_tool_call_id: str
    tool_call_id: str
    tool: str
    args: dict[str, Any]
    result: str
    success: bool


class AgentRuntime:
    """Tool-using high-level robot model runtime."""

    def __init__(
        self,
        provider: ReasoningProvider,
        *,
        max_iterations: int = 8,
        tool_registry: ToolRegistry | None = None,
        permission_mode: PermissionMode = "guarded",
        audit_logger: ToolAuditLogger | None = None,
        agent_run_recorder: AgentRunRecorder | None = None,
        hooks: list[ToolHook] | None = None,
        tool_executor: ToolExecutor | None = None,
        capability_resolver: CapabilityResolver | None = None,
        prompt_templates: AgentPromptTemplates | None = None,
        provider_timeout_sec: float = 300.0,
        runtime_hooks: list[AgentRuntimeHook] | None = None,
        message_window_policy: MessageWindowPolicy | None = None,
    ) -> None:
        self.provider = provider
        self.max_iterations = max(1, int(max_iterations))
        self.tools = tool_registry or ToolRegistry()
        self.state = AgentState()
        self.prompt_templates = prompt_templates or load_agent_prompt_templates()
        self.tool_executor = tool_executor or ToolExecutor(
            self.tools,
            permission_manager=PermissionManager(permission_mode),
            hooks=hooks,
            audit_logger=audit_logger,
            capability_resolver=capability_resolver,
        )
        self.agent_run_recorder = agent_run_recorder
        self.memory: Any | None = None
        self.provider_timeout_sec = max(5.0, float(provider_timeout_sec))
        self.runtime_hooks = list(runtime_hooks or [])
        self.message_window_policy = message_window_policy or MessageWindowPolicy()

    def register_tool(
        self,
        name: str,
        func: AsyncTool,
        *,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        read_only: bool | None = None,
        destructive: bool = False,
        safety_level: str = "normal",
        timeout_sec: float | None = None,
        exclusive: bool = False,
        resources: list[str] | tuple[str, ...] | None = None,
        result_policy: ToolResultPolicy = "continue_reasoning",
    ) -> None:
        self.tools.register_simple(
            name,
            func,
            description=description,
            input_schema=input_schema,
            read_only=read_only,
            destructive=destructive,
            safety_level=safety_level,
            timeout_sec=timeout_sec,
            exclusive=exclusive,
            resources=tuple(resources)
            if resources is not None and not isinstance(resources, tuple)
            else resources,
            result_policy=result_policy,
        )

    async def step(self, payload: AgentRuntimeInput) -> AgentRuntimeResult:
        self.state.task = payload.task
        run_result = await self.run(
            AgentRunSpec(
                messages=self._initial_messages(payload),
                task=payload.task,
                robot_state=payload.robot_state,
                robot_status=payload.robot_status,
                allowed_tools=payload.allowed_tools,
                payload=payload,
            )
        )
        return run_result.runtime_result

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        payload = spec.payload or AgentRuntimeInput(
            task=spec.task,
            images=[],
            robot_state=spec.robot_state,
            robot_status=spec.robot_status,
            allowed_tools=spec.allowed_tools,
        )
        messages = list(spec.messages)
        last_tool_result: AgentRuntimeResult | None = None
        perception_guard_attempted = False
        tools_used: list[str] = []
        max_iterations = spec.max_iterations or self.max_iterations
        provider_timeout_sec = max(
            5.0, float(spec.provider_timeout_sec or self.provider_timeout_sec)
        )

        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            hook_context = AgentRuntimeHookContext(
                iteration=iterations, messages=messages
            )
            await self._emit_runtime_hook("before_iteration", hook_context)
            try:
                messages_for_model = apply_message_window(
                    messages, spec.message_window or self.message_window_policy
                )
                validate_provider_messages(messages_for_model)
                response = await asyncio.wait_for(
                    self._request_provider(
                        messages_for_model, allowed_tools=spec.allowed_tools
                    ),
                    timeout=provider_timeout_sec,
                )
            except TimeoutError:
                response = ReasoningResponse(
                    content="LLM provider call timed out",
                    finish_reason="error",
                    error_kind="timeout",
                )
            except ValueError as exc:
                result = AgentRuntimeResult(
                    tool="wait",
                    args={"reason": str(exc)},
                    result=str(exc),
                    reason="invalid_message_protocol",
                    stop_reason="invalid_message_protocol",
                    tool_success=False,
                )
                return AgentRunResult(
                    final_content=None,
                    messages=messages,
                    tools_used=tools_used,
                    stop_reason=result.stop_reason,
                    runtime_result=result,
                    error=result.result,
                )
            hook_context.response = response
            await self._emit_runtime_hook("after_model_response", hook_context)
            try:
                validate_provider_response(response)
            except ValueError as exc:
                result = AgentRuntimeResult(
                    tool="wait",
                    args={"reason": str(exc)},
                    result=str(exc),
                    reason="invalid_message_protocol",
                    stop_reason="invalid_message_protocol",
                    tool_success=False,
                )
                hook_context.final_result = result
                await self._emit_runtime_hook("after_iteration", hook_context)
                return AgentRunResult(
                    final_content=None,
                    messages=messages,
                    tools_used=tools_used,
                    stop_reason=result.stop_reason,
                    runtime_result=result,
                    error=result.result,
                )
            decision = decide_response(response)
            if decision.action == "provider_error":
                result = AgentRuntimeResult(
                    tool="wait",
                    args={"reason": decision.content},
                    result=decision.content,
                    reason=decision.reason,
                    stop_reason="provider_error",
                )
                hook_context.final_result = result
                await self._emit_runtime_hook("after_iteration", hook_context)
                return AgentRunResult(
                    final_content=None,
                    messages=messages,
                    tools_used=tools_used,
                    stop_reason=result.stop_reason,
                    runtime_result=result,
                    error=result.result,
                )
            if decision.action != "execute_tools":
                content = decision.content
                if decision.action == "text":
                    if looks_like_internal_agent_protocol(content):
                        guidance = self._internal_protocol_retry_guidance(
                            payload, last_tool_result
                        )
                        if iterations < max_iterations:
                            messages.append(
                                ReasoningMessage(role="user", content=guidance)
                            )
                            continue
                        result = AgentRuntimeResult(
                            tool="wait",
                            args={
                                "reason": "provider returned internal agent protocol as final response"
                            },
                            result="provider returned internal agent protocol as final response",
                            reason="internal_protocol_response",
                            stop_reason="internal_protocol_response",
                            tool_success=False,
                        )
                        return await self._finish_run(
                            result, messages, tools_used, hook_context
                        )
                    if self._looks_like_unexecuted_tool_protocol(content):
                        if (
                            last_tool_result is not None
                            and last_tool_result.tool_success
                        ):
                            fallback_reply = self._final_answer_from_tool_result(
                                last_tool_result
                            )
                            if fallback_reply:
                                result = self._text_response_result(
                                    tool="final_response",
                                    args={},
                                    result=fallback_reply,
                                    reason="invalid_tool_protocol_after_successful_tool",
                                    task_finished=False,
                                    stop_reason="text_response",
                                    tool_success=True,
                                )
                                return await self._finish_run(
                                    result, messages, tools_used, hook_context
                                )
                        result = self._text_response_result(
                            tool="wait",
                            args={
                                "reason": "provider returned textual tool protocol instead of structured tool call"
                            },
                            result="provider returned textual tool protocol instead of structured tool call",
                            reason="invalid_tool_protocol",
                            stop_reason="invalid_tool_protocol",
                            tool_success=False,
                        )
                        return await self._finish_run(
                            result, messages, tools_used, hook_context
                        )
                    if (
                        not perception_guard_attempted
                        and self._needs_perception_before_final(payload, content)
                        and (
                            payload.allowed_tools is None
                            or "request_perception" in payload.allowed_tools
                        )
                    ):
                        perception_guard_attempted = True
                        guarded = await self._execute_grounding_perception(
                            payload, messages
                        )
                        if guarded is not None:
                            last_tool_result = guarded
                            continue
                    result = self._text_response_result(
                        tool="final_response",
                        args={},
                        result=content,
                        reason="text_response",
                        stop_reason="text_response",
                    )
                    return await self._finish_run(
                        result, messages, tools_used, hook_context
                    )
                result = AgentRuntimeResult(
                    tool="wait",
                    args={"reason": "no tool call or text from provider"},
                    result="no tool call or text from provider",
                    reason="empty_response",
                    stop_reason="empty_response",
                )
                return await self._finish_run(
                    result, messages, tools_used, hook_context
                )

            reason = response.reasoning_content or response.content or ""

            messages.append(
                ReasoningMessage(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=list(response.tool_calls),
                )
            )

            await self._emit_runtime_hook("before_execute_tools", hook_context)
            executed_calls = await self._execute_tool_calls(response, payload)
            hook_context.tool_results = list(executed_calls)
            await self._emit_runtime_hook("after_tool_results", hook_context)
            for executed in executed_calls:
                tool = executed.tool
                args = executed.args
                tool_result_text = executed.result
                success = executed.success
                tools_used.append(tool)

                self.state.add_tool_call(tool, args, tool_result_text, success=success)
                last_tool_result = AgentRuntimeResult(
                    tool_call_id=executed.tool_call_id,
                    tool=tool,
                    args=args,
                    result=tool_result_text,
                    reason=reason,
                    stop_reason="tool_result",
                    tool_success=success,
                )
                messages.append(
                    ReasoningMessage(
                        role="tool",
                        content=tool_result_text,
                        tool_call_id=executed.provider_tool_call_id,
                        tool_name=tool,
                    )
                )
                continuation_guidance = self._post_tool_guidance(
                    tool=tool,
                    args=args,
                    result=tool_result_text,
                    success=success,
                    payload=payload,
                )
                if continuation_guidance:
                    messages.append(
                        ReasoningMessage(role="user", content=continuation_guidance)
                    )
                if self.agent_run_recorder is not None:
                    self.agent_run_recorder.record_decision(
                        task=payload.task,
                        robot_state=payload.robot_state,
                        decision={
                            "tool": tool,
                            "args": args,
                            "reason": reason,
                            "provider": self.provider.get_default_model(),
                        },
                        result={
                            "tool_call_id": executed.tool_call_id,
                            "tool": tool,
                            "args": args,
                            "success": success,
                            "result": tool_result_text,
                        },
                    )

                self._record_tool_memory(tool, args, tool_result_text, success)
                try:
                    tool_spec = self.tools.get_tool(tool)
                except ValueError:
                    tool_spec = None
                if tool_spec is not None and tool_spec.result_policy == "return_direct":
                    runtime_result = self._text_response_result(
                        tool_call_id=executed.tool_call_id,
                        tool=tool,
                        args=args,
                        result=tool_result_text,
                        reason=reason,
                        stop_reason="text_response",
                        tool_success=success,
                    )
                    return await self._finish_run(
                        runtime_result, messages, tools_used, hook_context
                    )

        if last_tool_result is not None and last_tool_result.tool_success:
            synthesized_reply = await self._synthesize_final_answer_after_tool(
                messages, last_tool_result
            )
            if synthesized_reply:
                if self._looks_like_unexecuted_tool_protocol(synthesized_reply):
                    synthesized_reply = (
                        self._final_answer_from_tool_result(last_tool_result) or ""
                    )
                if looks_like_internal_agent_protocol(
                    synthesized_reply
                ) or looks_like_internal_user_reply(synthesized_reply):
                    synthesized_reply = (
                        self._final_answer_from_tool_result(last_tool_result) or ""
                    )
                if looks_like_internal_agent_protocol(
                    synthesized_reply
                ) or looks_like_internal_user_reply(synthesized_reply):
                    synthesized_reply = ""
                if not synthesized_reply:
                    runtime_result = AgentRuntimeResult(
                        tool_call_id=last_tool_result.tool_call_id,
                        tool=last_tool_result.tool,
                        args=last_tool_result.args,
                        result=last_tool_result.result,
                        reason="invalid_tool_protocol_after_successful_tool",
                        task_finished=False,
                        stop_reason="max_iterations_after_tool_result",
                        tool_success=True,
                    )
                    return await self._finish_run(
                        runtime_result, messages, tools_used, hook_context
                    )
                runtime_result = self._text_response_result(
                    tool="final_response",
                    args={},
                    result=synthesized_reply,
                    reason="post_tool_final_answer",
                    task_finished=False,
                    stop_reason="text_response",
                    tool_success=True,
                )
                return await self._finish_run(
                    runtime_result, messages, tools_used, hook_context
                )
            fallback_reply = self._final_answer_from_tool_result(last_tool_result)
            if fallback_reply and looks_like_internal_agent_protocol(fallback_reply):
                fallback_reply = ""
            if fallback_reply:
                runtime_result = self._text_response_result(
                    tool="final_response",
                    args={},
                    result=fallback_reply,
                    reason="required_final_answer_from_tool_result",
                    task_finished=False,
                    stop_reason="text_response",
                    tool_success=True,
                )
                return await self._finish_run(
                    runtime_result, messages, tools_used, hook_context
                )
            runtime_result = AgentRuntimeResult(
                tool_call_id=last_tool_result.tool_call_id,
                tool=last_tool_result.tool,
                args=last_tool_result.args,
                result=last_tool_result.result,
                reason="max_iterations_after_tool_result",
                task_finished=False,
                stop_reason="max_iterations_after_tool_result",
                tool_success=True,
            )
            return await self._finish_run(
                runtime_result, messages, tools_used, hook_context
            )
        if last_tool_result is not None and not last_tool_result.tool_success:
            fallback_reply = self._fallback_reply_from_failed_tool(last_tool_result)
            if fallback_reply:
                runtime_result = self._text_response_result(
                    tool="final_response",
                    args={},
                    result=fallback_reply,
                    reason="failed_tool_fallback",
                    task_finished=False,
                    stop_reason="text_response",
                    tool_success=False,
                )
                return await self._finish_run(
                    runtime_result, messages, tools_used, hook_context
                )
        runtime_result = AgentRuntimeResult(
            tool="wait",
            args={"reason": f"max_iterations ({max_iterations}) exhausted"},
            result="max_iterations exhausted",
            reason="max_iterations",
            stop_reason="max_iterations",
        )
        return await self._finish_run(
            runtime_result, messages, tools_used, hook_context
        )

    async def _finish_run(
        self,
        result: AgentRuntimeResult,
        messages: list[ReasoningMessage],
        tools_used: list[str],
        hook_context: AgentRuntimeHookContext,
    ) -> AgentRunResult:
        hook_context.final_result = result
        await self._emit_runtime_hook("after_iteration", hook_context)
        final_content = (
            result.result
            if result.stop_reason == "text_response" and result.tool == "final_response"
            else None
        )
        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=list(tools_used),
            stop_reason=result.stop_reason,
            runtime_result=result,
            error=result.result
            if result.stop_reason
            in {"provider_error", "empty_response", "invalid_message_protocol"}
            else None,
        )

    async def _emit_runtime_hook(
        self, method_name: str, context: AgentRuntimeHookContext
    ) -> None:
        for hook in self.runtime_hooks:
            method = getattr(hook, method_name)
            await method(context)

    async def _synthesize_final_answer_after_tool(
        self,
        messages: list[ReasoningMessage],
        result: AgentRuntimeResult,
    ) -> str | None:
        prompt = self._final_answer_prompt_for_tool_result(result)
        if not prompt:
            return None
        try:
            response = await asyncio.wait_for(
                self.provider.chat(
                    messages=[*messages, ReasoningMessage(role="user", content=prompt)],
                    tools=None,
                    tool_choice="auto",
                ),
                timeout=self.provider_timeout_sec,
            )
        except TimeoutError:
            return None
        if response.finish_reason == "error":
            return None
        content = (response.content or "").strip()
        return content or None

    def _fallback_reply_from_failed_tool(
        self, result: AgentRuntimeResult
    ) -> str | None:
        if result.tool == "request_capability":
            error = (result.result or result.reason or "").strip()
            lowered = error.lower()
            if "consecutivemotionblocked" in lowered:
                return "我需要先重新观察一下前方，再决定能不能继续移动。"
            if "camerastale" in lowered or "cameraunsafe" in lowered:
                return "当前视觉信息不够新或不够可靠，我需要先重新观察，再执行移动。"
            if error:
                return f"这个动作没有成功提交：{error}"
        return None

    def _internal_protocol_retry_guidance(
        self,
        payload: AgentRuntimeInput,
        last_tool_result: AgentRuntimeResult | None,
    ) -> str:
        lines = [
            "上一条 assistant 回复是内部运行协议，不是面向用户的回复。",
            "不要重复 execution feedback 标题、Task continuation 段落、Skill trace 段落或原始状态字段。",
            f"原始任务：{payload.task}",
        ]
        if last_tool_result is not None:
            lines.append(
                f"最近工具：{last_tool_result.tool} success={last_tool_result.tool_success}"
            )
        lines.extend(
            [
                "如果任务还没有完成，继续调用下一个有用工具。",
                "只有在任务目标确实完成，或已经没有安全的下一步时，才给用户最终答复。",
                "最终用户答复必须使用简体中文纯文本，不要使用 Markdown 标题、列表、代码块或表格。",
            ]
        )
        return "\n".join(lines)

    def _final_answer_prompt_for_tool_result(
        self, result: AgentRuntimeResult
    ) -> str | None:
        if result.tool != "request_capability":
            return None
        return (
            "机器人能力已经执行完成，上方已有工具结果。\n"
            "现在生成面向用户的最终回复。\n"
            "必须使用简体中文纯文本，不要使用 Markdown 标题、列表、代码块或表格。\n"
            "不要重复原始 execution feedback 标题、字段列表或内部状态标签。\n"
            "简要说明实际执行结果；如有不确定性要直接说明；只有在确实有用时才问一个简短的下一步问题。"
        )

    def _final_answer_from_tool_result(self, result: AgentRuntimeResult) -> str | None:
        presented = present_tool_result_for_user(
            tool=result.tool,
            args=result.args,
            result=result.result,
            success=result.tool_success,
        )
        if presented:
            return presented
        try:
            tool = self.tools.get_tool(result.tool)
        except ValueError:
            return None
        if tool.result_policy != "require_final_answer":
            if tool.read_only:
                text = (result.result or "").strip()
                return text or None
            return None
        payload = _json_object(result.result)
        if payload is None:
            text = (result.result or "").strip()
            return text or None
        if str(payload.get("tool") or result.tool) == "request_perception":
            return _perception_reply_from_payload(payload)
        text = str(payload.get("result") or payload.get("summary") or "").strip()
        return text or None

    def _needs_perception_before_final(
        self, payload: AgentRuntimeInput, content: str
    ) -> bool:
        if not needs_perception_grounding(payload.task, content):
            return False
        return not any(
            is_perception_evidence_record(
                record.name, record.arguments, success=record.success
            )
            for record in self.state.tool_calls
        )

    async def _execute_grounding_perception(
        self,
        payload: AgentRuntimeInput,
        messages: list[ReasoningMessage],
    ) -> AgentRuntimeResult | None:
        try:
            self.tools.get_tool("request_perception")
        except ValueError:
            result = (
                "PerceptionUnavailable: perception evidence is required before answering this visual question, "
                "but no perception tool is registered."
            )
            messages.append(
                ReasoningMessage(
                    role="user",
                    content=(
                        f"{result}\nSay that fresh visual perception is unavailable instead of describing the scene."
                    ),
                )
            )
            return AgentRuntimeResult(
                tool="request_perception",
                args={},
                result=result,
                reason="perception_tool_unavailable",
                stop_reason="tool_unavailable",
                tool_success=False,
            )
        execution = await self.tool_executor.execute(
            "request_perception",
            {"question": payload.task, "freshness": "fresh"},
            context={"robot_status": payload.robot_status},
            task=payload.task,
        )
        self.state.add_tool_call(
            "request_perception",
            execution.arguments,
            execution.result,
            success=execution.success,
        )
        messages.append(
            ReasoningMessage(
                role="user",
                content=(
                    "Perception evidence from request_perception:\n"
                    f"{execution.result}\n\n"
                    "Answer the user's visual question using only this evidence. "
                    "If evidence_status is not ok, say what perception failed instead of guessing."
                ),
            )
        )
        return AgentRuntimeResult(
            tool_call_id=execution.tool_call_id,
            tool="request_perception",
            args=execution.arguments,
            result=execution.result,
            reason="perception_grounding_required",
            stop_reason="tool_result",
            tool_success=execution.success,
        )

    def _initial_messages(self, payload: AgentRuntimeInput) -> list[ReasoningMessage]:
        prompt = build_turn_prompt(
            templates=self.prompt_templates,
            task=payload.task,
            robot_state=payload.robot_state,
            memory_context=payload.memory_context,
            autonomy_context=payload.autonomy_context,
            last_feedback=payload.last_feedback,
            next_hint=payload.next_hint,
            skill_in_progress=payload.skill_in_progress,
            recovery_context=payload.recovery_context,
            loop_warning=self.state.loop_warning_context(),
        )
        images = [ReasoningImage(data=image) for image in payload.images]
        return [
            ReasoningMessage(
                role="system", content=build_system_prompt(self.prompt_templates)
            ),
            ReasoningMessage(role="user", content=prompt, images=images),
        ]

    async def _request_provider(
        self,
        messages: list[ReasoningMessage],
        *,
        allowed_tools: set[str] | None = None,
    ) -> ReasoningResponse:
        chat_with_retry = getattr(self.provider, "chat_with_retry", None)
        tools = self.tools.list_tools()
        if allowed_tools is not None:
            tools = [tool for tool in tools if tool["name"] in allowed_tools]
        kwargs = {
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        if callable(chat_with_retry):
            return cast(ReasoningResponse, await chat_with_retry(**kwargs))
        return await self.provider.chat(
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

    async def _execute_tool_calls(
        self,
        response: ReasoningResponse,
        payload: AgentRuntimeInput,
    ) -> list[_ExecutedToolCall]:
        results: list[_ExecutedToolCall] = []
        for batch in self._tool_call_batches(response.tool_calls):
            if len(batch) > 1:
                batch_results = await asyncio.gather(
                    *(
                        self._execute_one_tool_call(tool_call, payload)
                        for tool_call in batch
                    )
                )
            else:
                batch_results = [await self._execute_one_tool_call(batch[0], payload)]
            results.extend(batch_results)
        return results

    async def _execute_one_tool_call(
        self, tool_call, payload: AgentRuntimeInput
    ) -> _ExecutedToolCall:
        tool = tool_call.name
        args = dict(tool_call.arguments)
        if payload.allowed_tools is not None and tool not in payload.allowed_tools:
            result = (
                f"ToolUnavailable: {tool} is not available in this execution context"
            )
            return _ExecutedToolCall(
                provider_tool_call_id=tool_call.id,
                tool_call_id="",
                tool=tool,
                args=args,
                result=result,
                success=False,
            )
        try:
            execution = await self.tool_executor.execute(
                tool,
                args,
                context={"robot_status": payload.robot_status},
                task=payload.task,
            )
            result = execution.result
            success = execution.success
            tool_call_id = execution.tool_call_id
        except Exception as exc:
            result = f"{type(exc).__name__}: {exc}"
            success = False
            tool_call_id = ""

        return _ExecutedToolCall(
            provider_tool_call_id=tool_call.id,
            tool_call_id=tool_call_id,
            tool=tool,
            args=args,
            result=result,
            success=success,
        )

    def _tool_call_batches(self, tool_calls) -> list[list[Any]]:
        batches: list[list[Any]] = []
        current: list[Any] = []
        for tool_call in tool_calls:
            try:
                tool = self.tools.get_tool(tool_call.name)
                can_batch = tool.concurrency_safe
            except ValueError:
                can_batch = False
            if can_batch:
                current.append(tool_call)
                continue
            if current:
                batches.append(current)
                current = []
            batches.append([tool_call])
        if current:
            batches.append(current)
        return batches

    def _record_tool_memory(
        self, tool: str, args: dict[str, Any], result: str, success: bool
    ) -> None:
        memory = getattr(self, "memory", None)
        if memory is None:
            return
        try:
            memory.record_tool_result(
                tool,
                args,
                result,
                success,
                context_summary=self.state.last_observation_summary or "",
            )
        except Exception:
            return

    def _post_tool_guidance(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        result: str,
        success: bool,
        payload: AgentRuntimeInput,
    ) -> str | None:
        if tool != "request_capability":
            return None
        capability = str(args.get("capability") or "").strip()
        objective = str(args.get("objective") or "").strip()
        safety_level = _resolve_capability_safety_level(capability)
        if success:
            self.state.last_capability_safety_level = safety_level
            self.state.last_capability_name = capability or None
            lines = [
                "Task continuation guidance:",
                f"- original_task: {payload.task}",
            ]
            if objective:
                lines.append(f"- latest_completed_step: {objective}")
            if capability:
                lines.append(f"- latest_capability: {capability}")
            if safety_level in {"motion", "actuate"}:
                lines.extend(
                    [
                        "- perception_required: 刚完成移动或执行动作，现在先执行 inspect_scene 来确认结果，再决定下一步。",
                        "- one_motion_one_perception: 每次移动后都必须先执行 inspect_scene 或 request_perception，之后才允许再次移动。",
                    ]
                )
                if payload.next_hint:
                    lines.append(f"- next_hint: {payload.next_hint}")
                lines.append(
                    "- completion_rule: 只有在原始任务目标已经满足，或没有更安全的下一步时，"
                    "才给用户最终答复；最终答复必须使用简体中文纯文本，不使用 Markdown。"
                )
            else:
                lines.extend(
                    [
                        (
                            "- do_not_stop_early: 如果原始任务还没有完全完成，现在继续调用下一个有用工具。"
                        ),
                        (
                            "- completion_rule: 只有在原始任务目标已经满足，或没有更安全的下一步时，"
                            "才给用户最终答复；最终答复必须使用简体中文纯文本，不使用 Markdown。"
                        ),
                    ]
                )
                if payload.next_hint:
                    lines.append(f"- next_hint: {payload.next_hint}")
            return "\n".join(lines)
        lines = [
            "Task recovery guidance:",
            f"- original_task: {payload.task}",
        ]
        if objective:
            lines.append(f"- failed_step: {objective}")
        if capability:
            lines.append(f"- failed_capability: {capability}")
        if safety_level in {"motion", "actuate"}:
            lines.append(
                "- motion_failed: 重试或改用其他动作前，先检查相机和机器人状态。"
            )
        lines.extend(
            [
                (
                    "- do_not_retry_blindly: 不要立刻重复同一个能力，除非已经获得新证据或选择了不同的恢复步骤。"
                ),
                (
                    "- next_action_rule: 先检查最新工具结果、执行反馈、机器人状态或感知结果，再决定下一步。"
                ),
            ]
        )
        if payload.recovery_context:
            lines.append(
                "- recovery_context_active: 新的动作执行前，优先处理当前恢复状态。"
            )
        failure_text = result.strip()
        if failure_text:
            lines.append(f"- latest_failure_signal: {failure_text}")
        return "\n".join(lines)

    def _text_response_result(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        result: str,
        reason: str,
        stop_reason: str = "text_response",
        tool_call_id: str = "",
        task_finished: bool = False,
        tool_success: bool | None = None,
    ) -> AgentRuntimeResult:
        if self._looks_like_unexecuted_tool_protocol(result):
            return AgentRuntimeResult(
                tool="wait",
                args={
                    "reason": "provider returned textual tool protocol instead of structured tool call"
                },
                result="provider returned textual tool protocol instead of structured tool call",
                reason="invalid_tool_protocol",
                stop_reason="invalid_tool_protocol",
                tool_success=False,
            )
        return AgentRuntimeResult(
            tool_call_id=tool_call_id,
            tool=tool,
            args=args,
            result=result,
            reason=reason,
            task_finished=task_finished,
            stop_reason=stop_reason,
            tool_success=tool_success,
        )

    @staticmethod
    def _looks_like_unexecuted_tool_protocol(content: str) -> bool:
        return looks_like_unexecuted_tool_protocol(content)


def _resolve_capability_safety_level(capability: str) -> str | None:
    if not capability:
        return None
    try:
        from hey_robot.skills.registry import load_skill_registry

        contract = load_skill_registry().catalog(enabled_only=False).get(capability)
        return contract.safety_level
    except Exception:
        return None


def _json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _perception_reply_from_payload(payload: dict[str, Any]) -> str | None:
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        text = str(payload.get("result") or "").strip()
        return text or None
    status = str(evidence.get("status") or payload.get("evidence_status") or "")
    summary = str(evidence.get("summary") or payload.get("result") or "").strip()
    if status == "ok" and summary:
        return summary
    if summary:
        return summary
    if status in {"no_observation", "no_image", "stale", "caption_failed"}:
        return "我暂时没有拿到可用的视觉证据，不能可靠描述当前画面。"
    return None
