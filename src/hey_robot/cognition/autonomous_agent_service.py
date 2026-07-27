"""Bus adapter and composition root for the interactive robot Agent."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from typing import Literal

from hey_robot.bus.factory import create_bus_client
from hey_robot.cognition.runtime.agent import (
    Agent,
    AgentCommand,
    AgentRunResult,
    ResumeTrigger,
)
from hey_robot.cognition.runtime.agent_context import AgentContextBuilder
from hey_robot.cognition.runtime.agent_runner import AgentRunner
from hey_robot.cognition.runtime.agent_task_store import AgentTask, AgentTaskStore
from hey_robot.cognition.runtime.conversation_store import ConversationStore
from hey_robot.cognition.runtime.task_coordinator import (
    AppliedSkillEvent,
    TaskCoordinator,
)
from hey_robot.cognition.tools.agent_response import AgentResponseTool
from hey_robot.cognition.tools.executor import AgentToolExecutor
from hey_robot.cognition.tools.models import AgentTool
from hey_robot.cognition.tools.registry import (
    ToolDependencies,
    ToolRegistry,
)
from hey_robot.config import DeploymentConfig
from hey_robot.model import create_model_client
from hey_robot.protocol import (
    AgentControl,
    ConversationResult,
    ConversationTurn,
    Envelope,
    Topics,
)
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.skills import registry_from_config
from hey_robot.skills.client import SkillClient
from hey_robot.skills.models import Skill, SkillEvent
from hey_robot.templates.loader import TemplateStore


class AutonomousAgentService:
    """Route protocol messages to one stateful Agent per session."""

    def __init__(
        self,
        config: DeploymentConfig,
        *,
        agent_id: str,
        skill_client: SkillClient | None = None,
        agent_skills: tuple[Skill, ...] | None = None,
        extra_tools: tuple[AgentTool, ...] = (),
    ) -> None:
        self.config = config
        self.agent_id = agent_id
        self.topics = Topics()
        self.bus = create_bus_client(config.deployment.bus, role="robot-agent")

        root = Path(config.resources.runtime_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.conversations = ConversationStore(root / "conversations.sqlite3")
        self.tasks = AgentTaskStore(root / "sustained_tasks.sqlite3")
        skills = (
            agent_skills
            if agent_skills is not None
            else registry_from_config(config).select(config.skills.tool_names)
        )
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
        self.tools = ToolRegistry(
            ToolDependencies(skills, (*extra_tools, AgentResponseTool()))
        )
        model_client = create_model_client(config, agent_id, purpose="agent")
        self.runner = AgentRunner(model_client, self.tools)
        if skill_client is None:
            raise ValueError("AutonomousAgentService requires native SkillClient")
        self.skill_client = skill_client
        self.task_coordinator = TaskCoordinator(self.tasks, skill_client)
        self.context_builder = AgentContextBuilder(
            self.templates,
            self.conversations,
            self.tasks,
        )
        self.tool_executor = AgentToolExecutor(
            config,
            self.tasks,
            self.task_coordinator,
            skill_client,
        )
        self._agents: dict[str, Agent] = {}
        self._skill_event_consumer: asyncio.Task[object] | None = None

    async def start(self) -> None:
        await self.bus.connect()
        await self.bus.subscribe([self.topics.conversation_turn], self._on_turn)
        await self.bus.subscribe([self.topics.agent_control], self._on_control)
        await self._recover_tasks()
        self._skill_event_consumer = asyncio.create_task(
            self._consume_skill_events(), name=f"agent:{self.agent_id}:skill-events"
        )
        await asyncio.Event().wait()

    async def _recover_tasks(self) -> None:
        reconciled = await self.task_coordinator.reconcile_active_run_results()
        for event, applied in reconciled:
            if applied.final_text is None:
                continue
            envelope = self.tasks.task_envelope(applied.step.task_id)
            if envelope is not None:
                await self._publish_result(envelope, event.run_id, applied.final_text)
        for task in self.tasks.resumable_tasks():
            reason = (
                "检测到重启前尚未完成审议的任务。为避免自动触发新的机器人动作，"
                "任务已暂停；请确认后继续。"
            )
            self.tasks.pause_task(task.task_id, reason)
            envelope = self.tasks.task_envelope(task.task_id)
            if envelope is not None:
                await self._publish_result(
                    envelope,
                    self.tasks.task_interaction_id(task.task_id)
                    or f"recovery_{task.task_id}",
                    reason,
                )

    async def stop(self) -> None:
        for agent in tuple(self._agents.values()):
            await agent.abort_inference()
        if self._skill_event_consumer is not None:
            self._skill_event_consumer.cancel()
            with suppress(asyncio.CancelledError):
                await self._skill_event_consumer
            self._skill_event_consumer = None
        self.conversations.close()
        self.tasks.close()
        await self.bus.close()

    async def _on_turn(self, _topic: str, payload: dict) -> None:
        turn = from_payload(ConversationTurn, payload)
        if turn.envelope.agent_id and turn.envelope.agent_id != self.agent_id:
            return

        async def publish_text_delta(delta: str) -> None:
            await self._publish_result(
                turn.envelope, turn.interaction_id, delta, final=False
            )

        command = AgentCommand(
            turn.session_key,
            turn.interaction_id,
            turn.envelope,
            turn.text,
        )
        agent = self._agent(turn.session_key)
        active_task = self.tasks.active_task(turn.session_key)
        try:
            has_active_run = bool(
                active_task and self.tasks.active_run_ids(active_task.task_id)
            )
            if turn.kind == "steer" or has_active_run:
                result = await agent.steer(command, on_text_delta=publish_text_delta)
            else:
                result = await agent.prompt(command, on_text_delta=publish_text_delta)
        except asyncio.CancelledError:
            return
        await self._publish_result(
            turn.envelope,
            turn.interaction_id,
            result.text,
            final=result.status != "waiting",
        )

    async def _on_control(self, _topic: str, payload: dict) -> None:
        command = from_payload(AgentControl, payload)
        if command.envelope.agent_id and command.envelope.agent_id != self.agent_id:
            return
        if command.action == "resume":
            text = await self._resume_task_from_control(command)
        else:
            text = await self.tool_executor.control(command)
        await self._publish_result(command.envelope, command.interaction_id, text)

    async def _resume_task_from_control(self, command: AgentControl) -> str:
        current = self.tasks.current_task(command.session_key)
        if (
            current is None
            or current.status != "paused"
            or self.tasks.active_run_ids(current.task_id)
        ):
            return await self.tool_executor.control(command)
        text = await self.tool_executor.control(command)
        task = self.tasks.active_task(command.session_key)
        if task is None:
            return text
        result = await self._agent(command.session_key).resume(
            ResumeTrigger(task.task_id, "user_resume")
        )
        return result.text or text

    async def _consume_skill_events(self) -> None:
        async for event in self.skill_client.events():
            await self._handle_skill_event(event)

    async def _handle_skill_event(self, event: SkillEvent) -> None:
        applied = self.task_coordinator.apply_result(event)
        if applied is None:
            return
        task = self.tasks.task(applied.step.task_id)
        if task is None:
            return
        envelope = self.tasks.task_envelope(task.task_id)
        if envelope is None:
            return
        if applied.final_text is not None:
            await self._publish_result(envelope, event.run_id, applied.final_text)
            return
        if applied.should_resume:
            await self._resume_applied(task, applied, event.run_id)

    async def _resume_applied(
        self, task: AgentTask, applied: AppliedSkillEvent, interaction_id: str
    ) -> None:
        result = await self._agent(task.session_key).resume(
            ResumeTrigger(task.task_id, "skill_terminal", applied.step.step_id)
        )
        envelope = self.tasks.task_envelope(task.task_id)
        if envelope is not None:
            routed_interaction_id = (
                self.tasks.task_interaction_id(task.task_id) or interaction_id
            )
            await self._publish_result(
                envelope,
                routed_interaction_id,
                result.text,
                final=getattr(result, "status", None) != "waiting",
            )

    async def _resume_task(
        self,
        task: AgentTask,
        *,
        source: Literal["skill_terminal", "startup_recovery"],
    ) -> AgentRunResult:
        trigger_source: Literal["skill_terminal", "startup_recovery"] = (
            "startup_recovery" if source == "startup_recovery" else "skill_terminal"
        )
        result = await self._agent(task.session_key).resume(
            ResumeTrigger(task.task_id, trigger_source)
        )
        envelope = self.tasks.task_envelope(task.task_id)
        if envelope is not None:
            routed_interaction_id = self.tasks.task_interaction_id(task.task_id)
            await self._publish_result(
                envelope,
                routed_interaction_id
                or (
                    f"resume_{task.task_id}_"
                    f"{self.tasks.resume_after_sequence(task.task_id)}"
                ),
                result.text,
                final=getattr(result, "status", None) != "waiting",
            )
        return result

    def _agent(self, session_key: str) -> Agent:
        agent = self._agents.get(session_key)
        if agent is None:
            agent = Agent(
                session_key=session_key,
                runner=self.runner,
                tools=self.tools,
                executor=self.tool_executor,
                context=self.context_builder,
                tasks=self.tasks,
                conversations=self.conversations,
            )
            self._agents[session_key] = agent
        return agent

    async def _publish_result(
        self,
        envelope: Envelope,
        interaction_id: str,
        text: str,
        *,
        final: bool = True,
    ) -> None:
        await self.bus.publish(
            self.topics.conversation_result,
            to_payload(ConversationResult(envelope, interaction_id, text, final=final)),
        )
