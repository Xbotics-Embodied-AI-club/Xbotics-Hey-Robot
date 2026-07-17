"""同时处理对话轮次和 Goal 审议的单一 Agent 服务。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

from hey_robot.bus.factory import create_bus_client
from hey_robot.cognition.autonomous.context_builder import build_context
from hey_robot.cognition.conversation_entities import EntityResolver
from hey_robot.cognition.conversation_execution import RobotExecutionAdapter
from hey_robot.cognition.conversation_goal import (
    GoalContractBuilder,
    GoalControlProposal,
    GoalProposal,
)
from hey_robot.cognition.policy.task_evaluator import TaskEvaluator
from hey_robot.cognition.runtime.agent_runner import (
    AgentRunner,
    AgentTurnRequest,
    AgentTurnResult,
)
from hey_robot.cognition.runtime.conversation_store import ConversationStore
from hey_robot.cognition.runtime.deliberation_store import DeliberationStore
from hey_robot.cognition.task.contract import TaskContract
from hey_robot.cognition.tools.robot import ToolDependencies, ToolRegistry
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import (
    ActionProposal,
    ConversationResult,
    ConversationTurn,
    DeliberationRequest,
    DeliberationResult,
    Envelope,
    FailurePayload,
    GoalEvent,
    RobotObservation,
    SkillResult,
    ToolOutcome,
    Topics,
)
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.providers import ReasoningMessage, build_provider
from hey_robot.skill_os.registry import registry_from_config
from hey_robot.templates.loader import TemplateStore

_CONVERSATION_TOOLS = frozenset(
    {"request_observation", "request_skill", "request_goal", "control_goal"}
)
_GOAL_TOOLS = frozenset({"request_observation", "request_skill"})


class RobotAgentService:
    """每个已配置 Agent 只拥有一个 Provider、Runner 和工具注册表。"""

    def __init__(self, config: DeploymentConfig, *, agent_id: str) -> None:
        self.config = config
        self.agent_id = agent_id
        self.topics = Topics()
        self.bus = create_bus_client(config.deployment.bus, role="robot-agent")

        root = Path(config.resources.runtime_dir) / config.deployment.id
        root.mkdir(parents=True, exist_ok=True)
        self.conversations = ConversationStore(root / "conversations.sqlite3")
        self.deliberations = DeliberationStore(root / "agent_deliberations.sqlite3")

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
        goal_builder = GoalContractBuilder(config.autonomy.entity_catalog)
        self.entities = EntityResolver(
            config.autonomy.entity_catalog,
            aliases=config.autonomy.entity_aliases,
        )
        self.tools = ToolRegistry(
            ToolDependencies(catalog, goal_kinds=goal_builder.goal_kinds)
        )
        provider = build_provider(config, agent_id, purpose="agent")
        self.runner = AgentRunner(provider, self.tools)
        self.execution = RobotExecutionAdapter(
            self.bus,
            self.topics,
            catalog,
            self.conversations,
            known_entities=config.autonomy.entity_catalog,
            goal_contract_builder=goal_builder,
            entity_resolver=self.entities,
        )
        self.evaluator = TaskEvaluator()
        self._session_locks: dict[str, asyncio.Lock] = {}

    async def start(self) -> None:
        await self.bus.connect()
        await self._recover_interrupted_deliberations()
        await self.bus.subscribe([self.topics.conversation_turn], self._on_turn)
        await self.bus.subscribe(
            [self.topics.agent_deliberation], self._on_deliberation
        )
        await self.bus.subscribe([self.topics.skill_result], self._on_skill_result)
        await self.bus.subscribe([self.topics.goal_event], self._on_goal_event)
        await self.bus.subscribe(
            [self.topics.robot_observation], self._on_robot_observation
        )
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self.conversations.close()
        self.deliberations.close()
        await self.bus.close()

    async def _recover_interrupted_deliberations(self) -> None:
        for request in self.deliberations.interrupt_incomplete():
            result = self._failed_deliberation(
                request,
                request_hash=_hash(to_payload(request)),
                stage="SUPERVISION",
                code="AGENT_PROCESS_INTERRUPTED",
                message="agent process restarted before deliberation became terminal",
            )
            self.deliberations.terminal(result, _hash(to_payload(result)))
            await self.bus.publish(
                self.topics.agent_deliberation_result, to_payload(result)
            )

    async def _on_turn(self, _topic: str, payload: dict) -> None:
        turn = from_payload(ConversationTurn, payload)
        if turn.envelope.agent_id and turn.envelope.agent_id != self.agent_id:
            return
        lock = self._session_locks.setdefault(turn.session_key, asyncio.Lock())
        async with lock:
            messages = self._conversation_context(turn)
            self.conversations.append(turn.session_key, "user", turn.text)
            decision = await self.runner.run(
                AgentTurnRequest(
                    tuple(messages),
                    _CONVERSATION_TOOLS,
                    time.monotonic() + 120.0,
                    turn.interaction_id,
                )
            )
            text = await self._resolve_conversation_decision(
                decision, turn.envelope, turn.session_key
            )
            self.conversations.append(turn.session_key, "assistant", text)
        await self.bus.publish(
            self.topics.conversation_result,
            to_payload(ConversationResult(turn.envelope, turn.interaction_id, text)),
        )

    def _conversation_context(self, turn: ConversationTurn) -> list[ReasoningMessage]:
        active_goal = self.conversations.active_goal(turn.session_key)
        policy = self.templates.render(
            "agent/SYSTEM.md",
            agent_soul=self.templates.render("agent/SOUL.md"),
            active_goal_context=_goal_projection(active_goal),
            entity_context=self.entities.context(turn.envelope.robot_id),
            tool_instructions=self.tools.instructions,
        )
        return [
            ReasoningMessage(role="system", content=policy),
            *self.conversations.recent(turn.session_key),
            ReasoningMessage(role="user", content=turn.text),
        ]

    async def _resolve_conversation_decision(
        self,
        decision: AgentTurnResult,
        envelope: Envelope,
        session_key: str,
    ) -> str:
        if decision.status == "returned":
            return decision.final_text or ""
        if decision.status == "failed":
            if (
                decision.failure
                and decision.failure.code == "MULTIPLE_ACTION_PROPOSALS"
            ):
                return "一次只能提交一个操作或一个任务，请重新说明。"
            detail = decision.failure.message if decision.failure else "模型决策失败"
            return f"这次请求没有完成：{detail}"
        proposal = decision.proposal
        if not isinstance(
            proposal, ActionProposal | GoalProposal | GoalControlProposal
        ):
            return "这次请求没有完成：工具没有产生有效的机器人提案。"
        outcome = await self.execution.execute(proposal, envelope, session_key)
        return _tool_outcome_text(outcome)

    async def _on_deliberation(self, _topic: str, payload: dict) -> None:
        request = from_payload(DeliberationRequest, payload)
        if request.envelope.agent_id and request.envelope.agent_id != self.agent_id:
            return
        request_hash = _hash(payload)
        old = self.deliberations.result(request.deliberation_id)
        if old is not None:
            await self.bus.publish(
                self.topics.agent_deliberation_result, to_payload(old)
            )
            return
        if not self.deliberations.schedule(
            deliberation_id=request.deliberation_id,
            request_hash=request_hash,
            goal_id=request.goal.goal_id,
            task_id=request.goal.task_id,
            request=request,
        ):
            return
        result = await self._run_goal_deliberation(request, request_hash)
        self.deliberations.terminal(result, _hash(to_payload(result)))
        await self.bus.publish(
            self.topics.agent_deliberation_result, to_payload(result)
        )

    async def _run_goal_deliberation(
        self, request: DeliberationRequest, request_hash: str
    ) -> DeliberationResult:
        self.deliberations.transition(request.deliberation_id, "SCHEDULED", "BUILDING")
        contract = TaskContract(
            request.goal.contract_id,
            request.goal.task_id,
            request.goal.goal_id,
            request.goal.objective,
            request.goal.success_criteria,
            1,
            request.goal.contract_hash,
        )
        evaluation = self.evaluator.evaluate(contract, request.evidence)
        if evaluation.outcome == "satisfied":
            return DeliberationResult(
                request.envelope,
                request.deliberation_id,
                request_hash,
                request.goal.goal_id,
                request.goal.task_id,
                "completed",
                evaluation=evaluation,
            )
        if not self.deliberations.transition(
            request.deliberation_id, "BUILDING", "BEFORE_MODEL_REQUEST"
        ):
            return self._failed_deliberation(
                request,
                request_hash,
                "PERSISTENCE",
                "PERSISTENCE_FAILED",
                "could not record model boundary",
            )
        context = build_context(
            request,
            evaluation_text=_evaluation_text(evaluation),
            templates=self.templates,
        )
        if context.failure is not None:
            return DeliberationResult(
                request.envelope,
                request.deliberation_id,
                request_hash,
                request.goal.goal_id,
                request.goal.task_id,
                "failed",
                failure=context.failure,
                evaluation=evaluation,
            )
        decision = await self.runner.run(
            AgentTurnRequest(
                context.messages,
                _GOAL_TOOLS,
                time.monotonic() + 120.0,
                request.deliberation_id,
            )
        )
        self.deliberations.transition(
            request.deliberation_id,
            "BEFORE_MODEL_REQUEST",
            "MODEL_RESPONSE_RECEIVED",
        )
        if decision.status == "action_proposed" and isinstance(
            decision.proposal, ActionProposal
        ):
            return DeliberationResult(
                request.envelope,
                request.deliberation_id,
                request_hash,
                request.goal.goal_id,
                request.goal.task_id,
                "action_proposed",
                proposal=decision.proposal,
                evaluation=evaluation,
            )
        if decision.status == "returned":
            failure = FailurePayload(
                "MODEL_PROTOCOL",
                "MODEL_STOPPED_BEFORE_GOAL",
                "RobotAgentService",
                "model returned text before contract was satisfied",
            )
        else:
            failure = decision.failure or FailurePayload(
                "TOOL_VALIDATION",
                "INVALID_TOOL_ARGUMENTS",
                "RobotAgentService",
                "goal tool did not produce an ActionProposal",
            )
        return DeliberationResult(
            request.envelope,
            request.deliberation_id,
            request_hash,
            request.goal.goal_id,
            request.goal.task_id,
            "failed",
            failure=failure,
            evaluation=evaluation,
        )

    async def _on_skill_result(self, _topic: str, payload: dict) -> None:
        self.execution.accept_result(from_payload(SkillResult, payload))

    async def _on_robot_observation(self, _topic: str, payload: dict) -> None:
        self.entities.update(from_payload(RobotObservation, payload))

    async def _on_goal_event(self, _topic: str, payload: dict) -> None:
        event = from_payload(GoalEvent, payload)
        linked = self.conversations.goal_link(event.goal_id)
        if linked is None:
            return
        self.conversations.update_goal_status(event.goal_id, event.status)
        session_key, fields = linked
        envelope = Envelope(
            channel=fields["channel"],
            chat_id=fields["chat_id"],
            sender_id=fields["sender_id"],
            user_id=fields["user_id"],
            agent_id=fields["agent_id"],
            robot_id=fields["robot_id"],
            episode_id=fields["episode_id"],
        )
        text = _goal_event_text(event)
        self.conversations.append(session_key, "assistant", text)
        await self.bus.publish(
            self.topics.conversation_result,
            to_payload(ConversationResult(envelope, f"goal:{event.event_id}", text)),
        )

    @staticmethod
    def _failed_deliberation(
        request: DeliberationRequest,
        request_hash: str,
        stage: str,
        code: str,
        message: str,
    ) -> DeliberationResult:
        return DeliberationResult(
            request.envelope,
            request.deliberation_id,
            request_hash,
            request.goal.goal_id,
            request.goal.task_id,
            "failed",
            failure=FailurePayload(stage, code, "RobotAgentService", message),
        )


def _evaluation_text(evaluation) -> str:
    if evaluation.outcome == "satisfied":
        return f"CONTRACT SATISFIED: {evaluation.reason}"
    if evaluation.missing_criteria_ids:
        return "INCONCLUSIVE - missing criteria: " + ", ".join(
            evaluation.missing_criteria_ids
        )
    return f"INCONCLUSIVE - {evaluation.reason}"


def _goal_projection(active_goal: dict[str, str] | None) -> str:
    if active_goal is None:
        return "当前会话没有进行中的长程任务。"
    return (
        "当前长程任务："
        f"id={active_goal['goal_id']}；objective={active_goal['objective']}；"
        f"status={active_goal['status']}。该信息仅为投影，Supervisor 事件才是权威状态。"
    )


def _goal_event_text(event: GoalEvent) -> str:
    if event.status == "completed" and event.evaluation is not None:
        return f"任务已经完成。验证依据：{event.evaluation.reason}。"
    if event.status in {"failed", "blocked", "needs_review"} and event.failure:
        return f"任务没有完成：{event.failure.message}。"
    if event.status == "cancelled":
        return "任务已取消。"
    return {
        "pending": "任务已接收，正在开始执行。",
        "active": "任务正在执行。",
        "waiting": "任务正在等待执行结果。",
        "waiting_condition": "任务需要你的确认或补充信息。",
        "needs_review": "任务需要人工检查。",
        "blocked": "任务当前被阻止。",
    }.get(event.status, "任务状态已更新。")


def _tool_outcome_text(outcome: ToolOutcome) -> str:
    if outcome.user_summary and outcome.user_summary.strip():
        return outcome.user_summary.strip()
    if outcome.status in {"accepted", "waiting"}:
        return "请求已接收，正在等待机器人执行结果。"
    if outcome.status == "failed":
        return "这次操作没有完成。"
    return "请求已经处理。"


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
