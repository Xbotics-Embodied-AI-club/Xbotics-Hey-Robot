"""处理会话轮次和持续任务的单一 Agent 服务。"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from hey_robot.bus.factory import create_bus_client
from hey_robot.cognition.conversation_entities import EntityResolver
from hey_robot.cognition.robot_execution_gateway import RobotExecutionGateway
from hey_robot.cognition.runtime.agent_runner import (
    AgentRunner,
    AgentTurnRequest,
    AgentTurnResult,
)
from hey_robot.cognition.runtime.agent_task_store import (
    AgentTask,
    AgentTaskStore,
    TaskStatus,
)
from hey_robot.cognition.runtime.completion_verifier import TaskCompletionVerifier
from hey_robot.cognition.runtime.conversation_store import ConversationStore
from hey_robot.cognition.tools.robot import (
    CompleteTaskProposal,
    ControlTaskProposal,
    ToolDependencies,
    ToolRegistry,
)
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import (
    ActionProposal,
    ConversationResult,
    ConversationTurn,
    Envelope,
    RobotObservation,
    SkillResult,
    ToolOutcome,
    Topics,
)
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.providers import ReasoningMessage, ReasoningToolCall, build_provider
from hey_robot.skill_os.registry import registry_from_config
from hey_robot.templates.loader import TemplateStore

_CONVERSATION_TOOLS = frozenset(
    {
        "request_observation",
        "request_skill",
        "complete_task",
        "control_task",
    }
)
_MAX_STEPS_PER_SLICE = 8
_DEFAULT_HARD_MAX_CONTINUATIONS = 12


@dataclass(frozen=True)
class _StepOutcome:
    final_text: str | None = None


class AutonomousAgentService:
    """每个已配置 Agent 只拥有一个 Provider、Runner 和工具注册表。"""

    def __init__(self, config: DeploymentConfig, *, agent_id: str) -> None:
        self.config = config
        self.agent_id = agent_id
        self.topics = Topics()
        self.bus = create_bus_client(config.deployment.bus, role="robot-agent")

        root = Path(config.resources.runtime_dir) / config.deployment.id
        root.mkdir(parents=True, exist_ok=True)
        self.conversations = ConversationStore(root / "conversations.sqlite3")
        self.tasks = AgentTaskStore(root / "sustained_tasks.sqlite3")

        catalog = registry_from_config(config).catalog(semantic_only=False)
        agent_spec = config.agents.get(agent_id)
        configured_template_root = (
            agent_spec.settings.get("template_root") if agent_spec is not None else None
        )
        self.templates = TemplateStore(
            configured_template_root
            if isinstance(configured_template_root, str)
            and configured_template_root.strip()
            else None
        )
        self.entities = EntityResolver(
            config.agent_runtime.entity_catalog,
            aliases=config.agent_runtime.entity_aliases,
        )
        self.tools = ToolRegistry(ToolDependencies(catalog))
        provider = build_provider(config, agent_id, purpose="agent")
        self.runner = AgentRunner(provider, self.tools)
        self.completion_verifier = TaskCompletionVerifier(provider)
        self.execution = RobotExecutionGateway(
            self.bus,
            self.topics,
            catalog,
            self.conversations,
        )
        self._session_locks: dict[str, asyncio.Lock] = {}

    async def start(self) -> None:
        await self.bus.connect()
        await self.bus.subscribe([self.topics.conversation_turn], self._on_turn)
        await self.bus.subscribe([self.topics.skill_result], self._on_skill_result)
        await self.bus.subscribe(
            [self.topics.robot_observation], self._on_robot_observation
        )
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self.conversations.close()
        self.tasks.close()
        await self.bus.close()

    async def _on_turn(self, _topic: str, payload: dict) -> None:
        turn = from_payload(ConversationTurn, payload)
        if turn.envelope.agent_id and turn.envelope.agent_id != self.agent_id:
            return
        lock = self._session_locks.setdefault(turn.session_key, asyncio.Lock())
        async with lock:
            messages = self._conversation_context(turn)
            self.conversations.append(turn.session_key, "user", turn.text)
            text = await self._run_conversation_loop(
                messages,
                turn.envelope,
                turn.session_key,
                turn.interaction_id,
                turn.text,
            )
            self.conversations.append(turn.session_key, "assistant", text)
        await self.bus.publish(
            self.topics.conversation_result,
            to_payload(ConversationResult(turn.envelope, turn.interaction_id, text)),
        )

    def _conversation_context(self, turn: ConversationTurn) -> list[ReasoningMessage]:
        policy = self.templates.render(
            "agent/SYSTEM.md",
            agent_soul=self.templates.render("agent/SOUL.md"),
            task_context=self.tasks.projection(turn.session_key),
            entity_context=self.entities.context(turn.envelope.robot_id),
            tool_instructions=self.tools.instructions,
        )
        return [
            ReasoningMessage(role="system", content=policy),
            *self.conversations.recent(turn.session_key),
            ReasoningMessage(role="user", content=turn.text),
        ]

    async def _run_conversation_loop(
        self,
        messages: list[ReasoningMessage],
        envelope: Envelope,
        session_key: str,
        run_id: str,
        objective: str,
    ) -> str:
        """Run one unified task loop for ordinary turns and active tasks."""
        max_continuations = max(
            1,
            int(self.config.agent_runtime.hard_max_continuations),
        )
        slice_used = 0
        while True:
            active_task = self.tasks.active_task(session_key)
            if (
                active_task is not None
                and active_task.deadline_at is not None
                and time.time() >= active_task.deadline_at
            ):
                self.tasks.control_task(
                    active_task.task_id,
                    "blocked",
                    "任务已达到最长持续时间，需要人工确认后再继续。",
                )
                return "任务已暂停：达到最长持续时间，需要你确认后再继续。"
            deadline = _next_decision_deadline(active_task)
            outcome = await self._run_task_step(
                messages,
                envelope,
                session_key,
                run_id,
                deadline,
                active_task,
                objective,
            )
            if outcome.final_text is not None:
                return outcome.final_text
            slice_used += 1
            active_task = self.tasks.active_task(session_key)
            if active_task is None:
                return "这次请求已经处理。"
            if slice_used < _MAX_STEPS_PER_SLICE:
                continue
            if active_task.continuation_count >= max_continuations:
                self.tasks.control_task(
                    active_task.task_id,
                    "blocked",
                    "任务已达到持续执行切片预算，需要人工确认后再继续。",
                )
                return "任务已暂停：达到持续执行切片预算，需要你确认后再继续。"
            self.tasks.continue_task(active_task.task_id)
            messages.append(
                ReasoningMessage(
                    role="user",
                    content=_continuation_message(active_task),
                )
            )
            slice_used = 0

    async def _run_task_step(
        self,
        messages: list[ReasoningMessage],
        envelope: Envelope,
        session_key: str,
        run_id: str,
        deadline: float,
        active_task: AgentTask | None,
        objective: str,
    ) -> _StepOutcome:
        decision = await self.runner.run(
            AgentTurnRequest(tuple(messages), _CONVERSATION_TOOLS, deadline, run_id)
        )
        if decision.status == "returned":
            if active_task is not None:
                messages.extend(
                    (
                        ReasoningMessage(
                            role="assistant", content=decision.final_text or ""
                        ),
                        ReasoningMessage(
                            role="user", content=_continuation_message(active_task)
                        ),
                    )
                )
                return _StepOutcome()
            return _StepOutcome(decision.final_text or "")
        if decision.status == "failed":
            return _StepOutcome(_decision_failure_text(decision))
        proposal = decision.proposal
        if isinstance(proposal, ActionProposal):
            current_task = self.tasks.active_task(session_key)
            max_skills = max(1, int(self.config.agent_runtime.hard_max_skills))
            if current_task is not None and current_task.step_count >= max_skills:
                self.tasks.control_task(
                    current_task.task_id,
                    "blocked",
                    "任务已达到最大机器人步骤预算，需要人工确认后再继续。",
                )
                return _StepOutcome(
                    "任务已暂停：达到最大机器人步骤预算，需要你确认后再继续。"
                )
            if current_task is None:
                current_task = self.tasks.create_task(
                    session_key=session_key,
                    envelope=envelope,
                    objective=objective,
                    ui_summary=objective,
                    deadline_at=time.time()
                    + self.config.agent_runtime.hard_max_wall_time_sec,
                )
            outcome = await self.execution.execute(proposal, envelope, session_key)
            step = self.tasks.add_step(current_task.task_id, proposal, outcome)
            call = decision.tool_calls[0]
            messages.extend(
                (
                    ReasoningMessage(
                        role="assistant",
                        content="",
                        tool_calls=[
                            ReasoningToolCall(
                                call.tool_call_id, call.name, call.arguments
                            )
                        ],
                    ),
                    ReasoningMessage(
                        role="tool",
                        content=_tool_outcome_context(
                            proposal, outcome, step, current_task
                        ),
                        tool_call_id=call.tool_call_id,
                        tool_name=call.name,
                    ),
                )
            )
            if outcome.status in {"accepted", "waiting"}:
                if current_task is not None:
                    self.tasks.control_task(
                        current_task.task_id,
                        "blocked",
                        "机器人执行仍处于未决状态，没有返回最终结果。",
                    )
                return _StepOutcome("这次操作没有完成：机器人还没有返回最终执行结果。")
            if not outcome.retryable and outcome.status == "failed":
                if current_task is not None:
                    self.tasks.control_task(
                        current_task.task_id,
                        "blocked",
                        outcome.user_summary or "操作没有完成。",
                    )
                return _StepOutcome(_tool_outcome_text(outcome))
            return _StepOutcome()
        if isinstance(proposal, CompleteTaskProposal):
            outcome = await self._complete_task(proposal, session_key)
            self._append_nonphysical_tool_result(messages, decision, outcome)
            if outcome.status == "completed":
                return _StepOutcome(proposal.recap)
            return _StepOutcome()
        if isinstance(proposal, ControlTaskProposal):
            return _StepOutcome(self._control_task(proposal, session_key))
        return _StepOutcome("这次请求没有完成：工具没有产生有效的机器人提案。")

    async def _complete_task(
        self, proposal: CompleteTaskProposal, session_key: str
    ) -> ToolOutcome:
        task = self.tasks.active_task(session_key)
        if task is None:
            return ToolOutcome("failed", "当前没有进行中的持续任务。", retryable=True)
        check = self.tasks.check_completion(task.task_id, proposal.evidence_ids)
        if not check.accepted:
            return ToolOutcome(
                "failed",
                f"任务还不能确认完成：{check.reason}",
                retryable=True,
                operation_id=task.task_id,
            )
        verdict = await self.completion_verifier.verify(
            task,
            proposal.recap,
            self.tasks.recent_steps(task.task_id, limit=200),
            proposal.evidence_ids,
        )
        if not verdict.accepted:
            return ToolOutcome(
                "failed",
                f"任务还不能确认完成：{verdict.reason}",
                retryable=True,
                operation_id=task.task_id,
            )
        check = self.tasks.complete_task(
            task.task_id, recap=proposal.recap, evidence_ids=proposal.evidence_ids
        )
        if not check.accepted:
            return ToolOutcome(
                "failed",
                f"任务还不能确认完成：{check.reason}",
                retryable=True,
                operation_id=task.task_id,
            )
        return ToolOutcome(
            "completed",
            proposal.recap,
            {"task_id": task.task_id},
            operation_id=task.task_id,
        )

    def _control_task(self, proposal: ControlTaskProposal, session_key: str) -> str:
        task = self.tasks.active_task(session_key)
        if task is None and proposal.action != "emergency_stop":
            return "当前没有进行中的持续任务。"
        reason = (
            proposal.reason
            or {
                "cancel": "任务已取消。",
                "block": "任务已阻塞，需要人工确认。",
                "emergency_stop": "已请求紧急停止。",
            }[proposal.action]
        )
        if task is not None:
            status = cast(
                TaskStatus,
                {
                    "cancel": "cancelled",
                    "block": "blocked",
                    "emergency_stop": "cancelled",
                }[proposal.action],
            )
            self.tasks.control_task(task.task_id, status, reason)
        return reason

    @staticmethod
    def _append_nonphysical_tool_result(
        messages: list[ReasoningMessage],
        decision: AgentTurnResult,
        outcome: ToolOutcome,
    ) -> None:
        call = decision.tool_calls[0]
        messages.extend(
            (
                ReasoningMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ReasoningToolCall(call.tool_call_id, call.name, call.arguments)
                    ],
                ),
                ReasoningMessage(
                    role="tool",
                    content=(
                        f"tool_result status={outcome.status}; "
                        f"summary={outcome.user_summary or ''}; "
                        f"data={json.dumps(outcome.data, ensure_ascii=False)}"
                    ),
                    tool_call_id=call.tool_call_id,
                    tool_name=call.name,
                ),
            )
        )

    async def _on_skill_result(self, _topic: str, payload: dict) -> None:
        self.execution.accept_result(from_payload(SkillResult, payload))

    async def _on_robot_observation(self, _topic: str, payload: dict) -> None:
        self.entities.update(from_payload(RobotObservation, payload))


def _tool_outcome_text(outcome: ToolOutcome) -> str:
    if outcome.user_summary and outcome.user_summary.strip():
        return outcome.user_summary.strip()
    if outcome.status in {"accepted", "waiting"}:
        return "这次操作没有完成：机器人还没有返回最终执行结果。"
    if outcome.status == "failed":
        return "这次操作没有完成。"
    return "请求已经处理。"


def _decision_failure_text(decision: AgentTurnResult) -> str:
    if decision.failure and decision.failure.code == "MULTIPLE_ACTION_PROPOSALS":
        return "一次只能提交一个操作或一个任务，请重新说明。"
    detail = decision.failure.message if decision.failure else "模型决策失败"
    return f"这次请求没有完成：{detail}"


def _next_decision_deadline(task: AgentTask | None) -> float:
    timeout_sec = 120.0
    if task is not None and task.deadline_at is not None:
        timeout_sec = min(timeout_sec, max(0.001, task.deadline_at - time.time()))
    return time.monotonic() + timeout_sec


def _continuation_message(task: AgentTask) -> str:
    return (
        "继续当前 active task，不要把阶段性文本当作最终回复。\n\n"
        f"任务目标：{task.objective}\n\n"
        "根据已有证据继续观察或执行一个有界 Skill。只有证据直接支持完整目标时才能调用 "
        "complete_task；确实无法继续时调用 control_task。"
    )


def _tool_outcome_context(
    proposal: ActionProposal,
    outcome: ToolOutcome,
    step: object | None = None,
    task: AgentTask | None = None,
) -> str:
    """Provide the next deliberation with an authoritative, compact tool result."""
    summary = outcome.user_summary or "no user-visible summary"
    evidence = ""
    if step is not None:
        evidence_ids = getattr(step, "evidence_ids", ())
        if evidence_ids:
            evidence = "; evidence_ids=" + ",".join(str(item) for item in evidence_ids)
    context = (
        f"tool_result status={outcome.status}; skill={proposal.skill_name}; "
        f"intent={proposal.intent_kind}; summary={summary}{evidence}"
    )
    if task is not None:
        context += (
            f"\nactive_task id={task.task_id}; objective={task.objective}; "
            "继续根据真实工具结果推进；只有 complete_task 或 control_task 才能结束任务。"
        )
    if proposal.intent_kind == "observation":
        return (
            f"{context}\n"
            "这次观察只更新证据，不自动完成用户的原始动作请求。继续根据原始请求推进，"
            "必要时继续执行、调用 complete_task，或在缺少不可替代参数时询问用户。"
        )
    return context
