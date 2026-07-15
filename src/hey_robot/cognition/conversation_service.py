"""Conversation Agent service: session context -> model -> tool -> final reply."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Protocol

from hey_robot.bus.factory import create_bus_client
from hey_robot.cognition.runtime.conversation_runner import (
    ConversationToolRunner,
    ToolRegistry,
)
from hey_robot.cognition.runtime.conversation_store import ConversationStore
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import (
    ConversationResult,
    ConversationTurn,
    Envelope,
    GoalEvent,
    SkillResult,
    Topics,
)
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.providers import ReasoningMessage, build_provider


class ToolExecutor(Protocol):
    async def execute(
        self, proposal: Any, envelope: Envelope, session_key: str
    ) -> Any: ...

    def accept_result(self, result: SkillResult) -> None: ...


ExecutionFactory = Any


class ConversationAgentService:
    def __init__(
        self,
        config: DeploymentConfig,
        *,
        agent_id: str,
        tools: ToolRegistry,
        execution_factory: ExecutionFactory,
    ) -> None:
        self.config = config
        self.agent_id = agent_id
        self.topics = Topics()
        self.bus = create_bus_client(config.deployment.bus, role="conversation-agent")
        self.tools = tools
        self.runner = ConversationToolRunner(
            build_provider(config, agent_id, purpose="agent"), self.tools
        )
        self.store = ConversationStore(
            Path(config.resources.runtime_dir)
            / config.deployment.id
            / "conversations.sqlite3"
        )
        self.execution: ToolExecutor = execution_factory(
            self.bus, self.topics, self.store
        )
        self._session_locks: dict[str, asyncio.Lock] = {}

    async def start(self) -> None:
        await self.bus.connect()
        await self.bus.subscribe([self.topics.conversation_turn], self._on_turn)
        await self.bus.subscribe([self.topics.skill_result], self._on_skill_result)
        await self.bus.subscribe([self.topics.goal_event], self._on_goal_event)
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self.store.close()
        await self.bus.close()

    async def _on_turn(self, _topic: str, payload: dict) -> None:
        turn = from_payload(ConversationTurn, payload)
        if turn.envelope.agent_id and turn.envelope.agent_id != self.agent_id:
            return
        lock = self._session_locks.setdefault(turn.session_key, asyncio.Lock())
        async with lock:
            history = self.store.recent(turn.session_key)
            messages = [
                ReasoningMessage(
                    role="system",
                    content="Use tools for fresh robot observations or physical actions. Never claim a physical action completed unless the tool result says status=completed. Describe a scene only from tool user_summary; if observation has no trusted summary, say it could not be reliably identified. "
                    + self.tools.instructions,
                ),
                *history,
                ReasoningMessage(role="user", content=turn.text),
            ]
            self.store.append(turn.session_key, "user", turn.text)
            text = await self.runner.run(
                messages,
                lambda proposal: self.execution.execute(
                    proposal, turn.envelope, turn.session_key
                ),
            )
            self.store.append(turn.session_key, "assistant", text)
        await self.bus.publish(
            self.topics.conversation_result,
            to_payload(ConversationResult(turn.envelope, turn.interaction_id, text)),
        )

    async def _on_skill_result(self, _topic: str, payload: dict) -> None:
        self.execution.accept_result(from_payload(SkillResult, payload))

    async def _on_goal_event(self, _topic: str, payload: dict) -> None:
        event = from_payload(GoalEvent, payload)
        linked = self.store.goal_link(event.goal_id)
        if linked is None:
            return
        _session_key, fields = linked
        envelope = Envelope(
            channel=fields["channel"],
            chat_id=fields["chat_id"],
            sender_id=fields["sender_id"],
            user_id=fields["user_id"],
            agent_id=fields["agent_id"],
            robot_id=fields["robot_id"],
            episode_id=fields["episode_id"],
        )
        text = _goal_event_text(event.status)
        self.store.append(_session_key, "assistant", text)
        await self.bus.publish(
            self.topics.conversation_result,
            to_payload(ConversationResult(envelope, f"goal:{event.event_id}", text)),
        )


def _goal_event_text(status: str) -> str:
    return {
        "pending": "任务已接收，正在开始执行。",
        "active": "任务正在执行。",
        "waiting": "任务正在等待新的条件。",
        "waiting_condition": "任务需要你的确认或补充信息。",
        "completed": "任务已经完成。",
        "failed": "任务没有完成。",
        "cancelled": "任务已取消。",
        "needs_review": "任务需要人工检查后再继续。",
        "blocked": "任务当前被阻止。",
    }.get(status, "任务状态已更新。")
