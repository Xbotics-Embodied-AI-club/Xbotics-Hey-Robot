"""Evidence-grounded completion review for sustained robot tasks."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from hey_robot.cognition.runtime.agent_task_store import AgentTask, AgentTaskStep
from hey_robot.providers import ReasoningMessage, ReasoningProvider

_ACCEPT_TOOL = "accept_task_completion"
_REJECT_TOOL = "reject_task_completion"
_VERDICT_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": _ACCEPT_TOOL,
            "description": "仅当证据直接支持整个任务目标和完成陈述时接受完成。",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": _REJECT_TOOL,
            "description": "证据不足、含糊或与目标不一致时拒绝，并说明下一步需要什么。",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass(frozen=True)
class CompletionVerdict:
    accepted: bool
    reason: str


class TaskCompletionVerifier:
    """Use a separate evidence-review pass before closing a physical task."""

    def __init__(self, provider: ReasoningProvider) -> None:
        self._provider = provider

    async def verify(
        self,
        task: AgentTask,
        recap: str,
        steps: tuple[AgentTaskStep, ...],
        evidence_ids: tuple[str, ...],
        *,
        timeout_sec: float = 30.0,
    ) -> CompletionVerdict:
        referenced = [
            step
            for step in steps
            if any(item in evidence_ids for item in step.evidence_ids)
        ]
        evidence = [
            {
                "sequence": step.sequence,
                "intent": step.proposal.intent_kind,
                "skill": step.proposal.skill_name,
                "step_objective": step.proposal.objective,
                "status": step.outcome.status,
                "summary": step.outcome.user_summary,
                "evidence_ids": step.evidence_ids,
            }
            for step in referenced
        ]
        messages = [
            ReasoningMessage(
                role="system",
                content=(
                    "你是机器人持续任务的完成证据审计器。只审查证据，不规划动作。"
                    "动作成功只证明该动作本身发生，不自动证明到达、进入、找到或完成世界目标。"
                    "观察中的‘前方有目标’不等于机器人已经到达或进入目标。"
                    "只有引用证据直接支持完整 objective 和 recap 时才能接受；"
                    "任何含糊、缺失或矛盾都必须拒绝，并指出下一步需要获取的证据。"
                    "必须且只能调用一个 verdict 工具，不要输出普通文本。"
                ),
            ),
            ReasoningMessage(
                role="user",
                content=json.dumps(
                    {
                        "objective": task.objective,
                        "proposed_recap": recap,
                        "referenced_evidence": evidence,
                    },
                    ensure_ascii=False,
                ),
            ),
        ]
        try:
            response = await asyncio.wait_for(
                self._provider.chat(messages=messages, tools=_VERDICT_TOOLS),
                timeout=max(0.001, timeout_sec),
            )
        except Exception as exc:
            return CompletionVerdict(False, f"完成证据审计失败：{exc}")
        if len(response.tool_calls) != 1:
            return CompletionVerdict(False, "完成证据审计没有返回唯一的结构化结论。")
        call = response.tool_calls[0]
        reason = str(call.arguments.get("reason") or "").strip()
        if call.name == _ACCEPT_TOOL:
            return CompletionVerdict(True, reason or "证据支持任务完成。")
        if call.name == _REJECT_TOOL:
            return CompletionVerdict(False, reason or "现有证据不足以确认任务完成。")
        return CompletionVerdict(False, "完成证据审计返回了未知结论。")
