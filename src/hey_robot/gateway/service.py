from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from hey_robot.bus.factory import create_bus_client
from hey_robot.channels import (
    ChannelContext,
    ChannelManager,
    CLIChannel,
    FeishuChannel,
    VoiceChannel,
    WebChannel,
)
from hey_robot.cognition.autonomous.store import AutonomyStore
from hey_robot.config import DeploymentConfig
from hey_robot.episode import JsonlEpisodeStore, allocate_episode
from hey_robot.episode.scope import DEFAULT_EPISODE_DIMENSIONS
from hey_robot.events import EventKind, RuntimeEvent
from hey_robot.events.bus import BusEventPublisher
from hey_robot.events.store import RuntimeEventStore
from hey_robot.gateway.identity import ClaimedBinding, IdentityResolver, PendingBinding
from hey_robot.health import HealthReportService
from hey_robot.logging import HeyRobotLogger
from hey_robot.protocol import (
    AgentReply,
    Envelope,
    GoalCommand,
    RobotStatus,
    SkillEvent,
    SkillResult,
    Topics,
    UserTurn,
)
from hey_robot.protocol.messages import (
    GoalBudgets,
    SuccessCriterion,
    from_payload,
    to_payload,
)
from hey_robot.providers import ReasoningMessage, build_provider
from hey_robot.skill_os import SkillStore
from hey_robot.skill_os.command_store import SkillCommandStore

logger = HeyRobotLogger(name="gateway")
_BINDING_COMMAND = re.compile(
    r"^\s*(?:bind|绑定)\s+([A-Za-z0-9]{4,12})\s*$", re.IGNORECASE
)
_ROBOT_STATUS_PERSIST_INTERVAL_SEC = 5.0
_ROUTE_INTERACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "route_interaction",
        "description": "Classify the user request as a text conversation or a physical robot goal.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"type": "string", "enum": ["conversation", "goal"]},
                "response_text": {"type": ["string", "null"]},
                "objective": {"type": "string"},
                "success_criteria": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "criterion_id": {"type": "string"},
                            "criterion_type": {"type": "string"},
                            "subject_id": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object_id": {"type": "string"},
                            "max_age_sec": {"type": "number"},
                        },
                        "required": [
                            "criterion_id",
                            "criterion_type",
                            "subject_id",
                            "predicate",
                            "object_id",
                            "max_age_sec",
                        ],
                    },
                },
            },
            "required": ["kind"],
        },
    },
}


class GatewayService:
    """Channel gateway that normalizes inbound turns and forwards outbound replies."""

    def __init__(
        self, config: DeploymentConfig, *, episode_dir: str | Path | None = None
    ) -> None:
        self.config = config
        self.topics = Topics()
        self.episode_root = Path(episode_dir or config.resources.episodes_root)
        self.episodes = JsonlEpisodeStore(self.episode_root)
        self.channels = ChannelManager()
        self.bus = create_bus_client(config.deployment.bus, role="gateway")
        self.events = BusEventPublisher(self.bus, self.topics)
        self.event_store = RuntimeEventStore(
            Path(config.resources.runtime_dir) / "events",
            max_items=config.resources.events_max_items,
        )
        self.skill_store = SkillStore(
            Path(config.resources.runtime_dir) / "skills",
            max_items=config.resources.events_max_items,
        )
        autonomy_path = (
            Path(config.resources.runtime_dir)
            / config.deployment.id
            / "autonomy.sqlite3"
        )
        self.autonomy_store = AutonomyStore(autonomy_path)
        self.skill_receipts = SkillCommandStore(
            Path(config.resources.runtime_dir) / "skill_receipts.sqlite3"
        )
        self.latest_robot_status: dict[str, RobotStatus] = {}
        self.identity = IdentityResolver(
            config.identity,
            state_path=Path(config.resources.runtime_dir)
            / "identity"
            / "bindings.json",
        )
        self._ready = asyncio.Event()
        self._last_robot_status_persisted_at: dict[str, float] = {}
        self._presentation_providers: dict[str, Any] = {}
        self._register_channels()

    async def start(self) -> None:
        enabled_channels = (
            ",".join(sorted(name for name, _ in self.channels.items())) or "none"
        )
        logger.info(
            f"start gateway deployment=[{self.config.deployment.id}] "
            f"channels={enabled_channels} bus={self.config.deployment.bus.url}"
        )
        await self.bus.connect()
        logger.info("gateway connected to bus")
        event = RuntimeEvent.make(EventKind.GATEWAY_START, source="gateway")
        await self.events.publish(event)
        self.event_store.append(event)
        await self.bus.subscribe([self.topics.agent_reply], self._on_agent_reply)
        await self.bus.subscribe([self.topics.runtime_event], self._on_runtime_event)
        await self.bus.subscribe([self.topics.robot_status], self._on_robot_status)
        await self.bus.subscribe([self.topics.skill_event], self._on_skill_event)
        await self.bus.subscribe([self.topics.skill_result], self._on_skill_result)
        logger.info(
            f"gateway subscribed {self.topics.agent_reply}, {self.topics.runtime_event}, "
            f"{self.topics.robot_status}, {self.topics.skill_event}, {self.topics.skill_result}"
        )
        await self.channels.start_all(self._on_user_turn)
        self._log_channel_ready()
        event = RuntimeEvent.make(EventKind.GATEWAY_READY, source="gateway")
        await self.events.publish(event)
        self.event_store.append(event)
        logger.info("gateway ready")
        self._ready.set()
        await asyncio.Event().wait()

    async def stop(self) -> None:
        event = RuntimeEvent.make(EventKind.GATEWAY_SHUTDOWN, source="gateway")
        await self.events.publish(event)
        self.event_store.append(event)
        await self.channels.stop_all()
        await self.bus.close()

    async def _on_user_turn(self, turn: UserTurn) -> None:
        if await self._try_handle_identity_binding_turn(turn):
            return
        agent_id = self._agent_id(turn.envelope.agent_id)
        robot_id = self.config.default_robot_id(agent_id)
        identity = self.identity.resolve(turn.envelope)
        logger.debug(
            f"gateway received turn channel={turn.envelope.channel} trace={turn.envelope.trace_id} "
            f"agent={agent_id} robot={robot_id} text_len={len(turn.text)}"
        )
        envelope = turn.envelope.child(
            agent_id=agent_id, robot_id=robot_id, user_id=identity.user_id
        )
        allocation = allocate_episode(
            envelope,
            agent_id=agent_id,
            dimensions=self._episode_dimensions(envelope),
        )
        self.episodes.ensure(
            allocation.episode_id, allocation.scope, allocation.aliases
        )
        self.episodes.append_user_turn(
            allocation.episode_id,
            replace(turn, envelope=envelope.child(episode_id=allocation.episode_id)),
        )
        if await self._handle_goal_command(turn.text, envelope):
            return
        await self._reply_to_presentation_turn(envelope, turn.text)

    async def _reply_to_presentation_turn(self, envelope: Envelope, text: str) -> None:
        """Handle chat directly or create a validated GoalCommand from one model tool call."""
        agent_id = self._agent_id(envelope.agent_id)
        try:
            provider = self._presentation_providers.get(agent_id)
            if provider is None:
                provider = build_provider(self.config, agent_id, purpose="agent")
                self._presentation_providers[agent_id] = provider
            response = await asyncio.wait_for(
                provider.chat(
                    messages=[
                        ReasoningMessage(
                            role="system",
                            content=(
                                "Route every request by calling route_interaction exactly once. "
                                "Use kind=conversation with response_text for non-physical requests. "
                                "Use kind=goal for a request to physically observe or act with the robot, "
                                "and provide a concrete immutable success contract. "
                                "A question about what the robot sees or what is in its environment is a "
                                "physical observation request and must use kind=goal. "
                                f"Allowed entity IDs: {sorted(self.config.autonomy.entity_catalog)}. "
                                "Never claim that a physical task has completed. "
                                "For a scene observation, use exactly this success criterion: "
                                "criterion_id=scene_observed, criterion_type=evidence_present, "
                                "subject_id=robot:sim_robot, predicate=observed, object_id=scene, "
                                "max_age_sec=60. Every kind=goal result must contain at least one "
                                "complete success_criteria item."
                            ),
                        ),
                        ReasoningMessage(role="user", content=text),
                    ],
                    tools=[_ROUTE_INTERACTION_TOOL],
                ),
                timeout=60.0,
            )
            if response.tool_calls:
                if (
                    len(response.tool_calls) != 1
                    or response.tool_calls[0].name != "route_interaction"
                ):
                    reply = "I could not form a safe autonomous goal from that request."
                else:
                    reply = await self._route_model_interaction(
                        envelope, response.tool_calls[0].arguments
                    )
            else:
                reply = (
                    response.content or "I cannot provide a text response right now."
                )
        except Exception:
            logger.exception("presentation request failed")
            reply = "The text assistant is temporarily unavailable."
        await self._send_reply(AgentReply(envelope=envelope, text=reply))

    async def _route_model_interaction(
        self, envelope: Envelope, arguments: dict[str, Any]
    ) -> str:
        if arguments.get("kind") == "conversation":
            response_text = arguments.get("response_text")
            return (
                str(response_text).strip()
                or "I cannot provide a text response right now."
            )
        if arguments.get("kind") != "goal":
            return "I could not determine how to handle that request."

        try:
            objective = str(arguments["objective"]).strip()
            criteria = self._goal_criteria_from_arguments(arguments)
            if not objective or not criteria:
                raise ValueError("missing objective or criteria")
            command = GoalCommand(
                envelope,
                str(uuid.uuid4()),
                "create",
                objective=objective,
                success_criteria=criteria,
                budgets=GoalBudgets(),
            )
            # Decode once here to enforce protocol validation before publication.
            from_payload(GoalCommand, to_payload(command))
        except (KeyError, TypeError, ValueError):
            return "I need a clearer, verifiable success condition before creating that robot task."
        await self.bus.publish(self.topics.goal_command, to_payload(command))
        return f"Created autonomous goal: {objective}"

    @staticmethod
    def _goal_criteria_from_arguments(
        arguments: dict[str, Any],
    ) -> tuple[SuccessCriterion, ...]:
        raw_criteria = arguments.get("success_criteria")
        try:
            if not isinstance(raw_criteria, list):
                raise ValueError("criteria must be a list")
            criteria = tuple(
                SuccessCriterion(**item)
                for item in raw_criteria
                if isinstance(item, dict)
            )
            if criteria:
                return criteria
        except (TypeError, ValueError):
            pass
        return ()

    async def _handle_goal_command(self, text: str, envelope: Envelope) -> bool:
        """The gateway accepts only explicit, structured autonomous commands."""
        stripped = text.strip()
        if not stripped.startswith("/goal "):
            return False
        parts = stripped.split(maxsplit=2)
        if len(parts) < 2 or parts[1] not in {
            "create",
            "cancel",
            "reconcile",
            "emergency_stop",
        }:
            await self._send_reply(
                AgentReply(envelope=envelope, text="GOAL_CONTRACT_INVALID")
            )
            return True
        import json

        if parts[1] == "reconcile":
            if len(parts) != 3:
                await self._send_reply(
                    AgentReply(envelope=envelope, text="RECONCILE_CONTRACT_INVALID")
                )
                return True
            try:
                body = json.loads(parts[2])
                skill_id = str(body["skill_id"])
                robot_id = str(body.get("robot_id") or envelope.robot_id or "")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                await self._send_reply(
                    AgentReply(envelope=envelope, text="RECONCILE_CONTRACT_INVALID")
                )
                return True
            status = self.latest_robot_status.get(robot_id)
            if (
                not robot_id
                or status is None
                or status.state != "idle"
                or status.skill_id is not None
                or self.skill_receipts.is_active(skill_id)
            ):
                await self._send_reply(
                    AgentReply(envelope=envelope, text="RECONCILE_IDLE_PROOF_REQUIRED")
                )
                return True
            reconciled = self.autonomy_store.reconcile_unknown_action(
                reconcile_id=str(uuid.uuid4()),
                robot_id=robot_id,
                skill_id=skill_id,
                operator_id=envelope.user_id or envelope.sender_id or "anonymous",
                status_payload=to_payload(status),
            )
            await self._send_reply(
                AgentReply(
                    envelope=envelope,
                    text="RECONCILE_COMPLETED" if reconciled else "RECONCILE_REJECTED",
                )
            )
            return True
        if parts[1] == "cancel":
            if len(parts) != 3 or not parts[2].strip():
                await self._send_reply(
                    AgentReply(envelope=envelope, text="GOAL_CONTRACT_INVALID")
                )
                return True
            command = GoalCommand(
                envelope, str(uuid.uuid4()), "cancel", goal_id=parts[2].strip()
            )
        elif parts[1] == "emergency_stop":
            if len(parts) != 2:
                await self._send_reply(
                    AgentReply(envelope=envelope, text="GOAL_CONTRACT_INVALID")
                )
                return True
            command = GoalCommand(envelope, str(uuid.uuid4()), "emergency_stop")
        else:
            if len(parts) != 3:
                await self._send_reply(
                    AgentReply(envelope=envelope, text="GOAL_CONTRACT_REQUIRED")
                )
                return True
            try:
                body = json.loads(parts[2])
                criteria = tuple(
                    SuccessCriterion(**item) for item in body["success_criteria"]
                )
                budgets = GoalBudgets(**dict(body.get("budgets") or {}))
                command = GoalCommand(
                    envelope,
                    str(uuid.uuid4()),
                    "create",
                    objective=str(body["objective"]),
                    contract_template_id=body.get("contract_template_id"),
                    success_criteria=criteria,
                    budgets=budgets,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                await self._send_reply(
                    AgentReply(envelope=envelope, text="GOAL_CONTRACT_INVALID")
                )
                return True
        await self.bus.publish(self.topics.goal_command, to_payload(command))
        return True

    async def _send_reply(self, reply: AgentReply) -> None:
        await self.bus.publish(self.topics.agent_reply, to_payload(reply))

    async def _on_agent_reply(self, _topic: str, payload: dict) -> None:
        reply = self._materialize_reply(from_payload(AgentReply, payload))
        logger.debug(
            f"gateway received agent reply trace={reply.envelope.trace_id} "
            f"channel={reply.envelope.channel} text_len={len(reply.text)}"
        )
        if reply.envelope.episode_id:
            self.episodes.append_agent_reply(reply.envelope.episode_id, reply)
        await self.channels.send(reply)

    def _materialize_reply(self, reply: AgentReply) -> AgentReply:
        envelope = reply.envelope
        agent_id = self._agent_id(envelope.agent_id)
        robot_id = envelope.robot_id or self.config.default_robot_id(agent_id)
        resolved_envelope = envelope.child(
            agent_id=agent_id,
            robot_id=robot_id,
            deployment_id=envelope.deployment_id or self.config.deployment.id,
            user_id=self.identity.resolve(envelope).user_id,
        )
        if resolved_envelope.episode_id is None:
            allocation = allocate_episode(
                resolved_envelope,
                agent_id=agent_id,
                dimensions=self._episode_dimensions(resolved_envelope),
            )
            self.episodes.ensure(
                allocation.episode_id, allocation.scope, allocation.aliases
            )
            resolved_envelope = resolved_envelope.child(
                episode_id=allocation.episode_id
            )
        return replace(reply, envelope=resolved_envelope)

    async def _on_runtime_event(self, _topic: str, payload: dict) -> None:
        try:
            event = RuntimeEvent(**payload)
        except TypeError:
            return
        if event.source == "gateway":
            await self.channels.publish_event(event)
            return
        self.event_store.append(event)
        await self.channels.publish_event(event)

    async def _on_robot_status(self, _topic: str, payload: dict) -> None:
        status = from_payload(RobotStatus, payload)
        if status.envelope.robot_id:
            self.latest_robot_status[status.envelope.robot_id] = status
        event = RuntimeEvent.make(
            EventKind.ROBOT_STATUS,
            source="robot",
            trace_id=status.envelope.trace_id,
            episode_id=status.envelope.episode_id,
            agent_id=status.envelope.agent_id,
            robot_id=status.envelope.robot_id,
            channel=status.envelope.channel,
            payload={
                "frame_id": status.frame_id,
                "state": status.state,
                "task": status.task,
                "skill_id": status.skill_id,
                "success": status.success,
                "error": status.error,
                "metrics": _compact_status_metrics(status.metrics or {}),
            },
        )
        robot_key = status.envelope.robot_id or "default"
        now = time.monotonic()
        last_persisted = self._last_robot_status_persisted_at.get(robot_key, 0.0)
        if now - last_persisted >= _ROBOT_STATUS_PERSIST_INTERVAL_SEC:
            self.event_store.append(event)
            self._last_robot_status_persisted_at[robot_key] = now
        await self.channels.publish_event(event)

    async def _on_skill_event(self, _topic: str, payload: dict) -> None:
        event = from_payload(SkillEvent, payload)
        self.skill_store.append(event)
        ux_metadata = event.metadata.get("ux")
        ux_payload = dict(ux_metadata) if isinstance(ux_metadata, dict) else None
        await self.channels.publish_event(
            RuntimeEvent.make(
                "skill.lifecycle",
                source="skill",
                trace_id=event.envelope.trace_id,
                episode_id=event.envelope.episode_id,
                agent_id=event.envelope.agent_id,
                robot_id=event.envelope.robot_id,
                channel=event.envelope.channel,
                payload={
                    "skill_id": event.skill_id,
                    "name": event.name,
                    "phase": event.phase,
                    "step": event.step,
                    "progress": event.progress,
                    "steps_executed": event.steps_executed,
                    "summary": event.summary,
                    "error": event.error,
                    "ux": ux_payload,
                },
            )
        )

    async def _on_skill_result(self, _topic: str, payload: dict) -> None:
        from_payload(SkillResult, payload)

    async def _web_history(self, envelope: Envelope, limit: int) -> dict:
        agent_id = self._agent_id(envelope.agent_id)
        robot_id = self.config.default_robot_id(agent_id)
        scoped = envelope.child(
            agent_id=agent_id,
            robot_id=robot_id,
            user_id=self.identity.resolve(envelope).user_id,
        )
        allocation = allocate_episode(
            scoped, agent_id=agent_id, dimensions=self._episode_dimensions(scoped)
        )
        records = self.episodes.history(allocation.episode_id, limit=limit)
        return {
            "episode_id": allocation.episode_id,
            "agent_id": agent_id,
            "robot_id": robot_id,
            "user_id": scoped.user_id,
            "continuity": self._identity_continuity(scoped),
            "records": [
                {
                    "role": record.role,
                    "content": record.content,
                    "timestamp": record.timestamp,
                    "payload": record.payload,
                }
                for record in records
            ],
        }

    async def _web_cockpit(self, episode_id: str) -> dict[str, Any] | None:
        del episode_id
        goals = self.autonomy_store.goals_recent(limit=1)
        if not goals:
            return None
        return {
            "goal": _goal_payload(goals[0]),
            "health": HealthReportService(self.config).payload(
                robot_id=goals[0]["robot_id"]
            ),
        }

    async def _web_tasks_list(self, limit: int) -> dict[str, Any]:
        return {
            "goals": [
                _goal_payload(goal)
                for goal in self.autonomy_store.goals_recent(limit=limit)
            ]
        }

    async def _web_runtime_summary(self, limit: int) -> dict[str, Any]:
        goals = self.autonomy_store.goals_recent(limit)
        skills = self.skill_store.recent(limit=limit)
        events = self.event_store.recent(limit=limit)
        return {
            "goals": [_goal_payload(goal) for goal in goals],
            "robots": [
                {
                    "robot_id": status.envelope.robot_id,
                    "state": status.state,
                    "status": {
                        "frame_id": status.frame_id,
                        "success": status.success,
                        "error": status.error,
                        "battery_percentage": status.battery_percentage,
                    },
                    "updated_at": status.envelope.timestamp,
                }
                for status in self.latest_robot_status.values()
            ],
            "skills": list(skills),
            "events": [
                {
                    "kind": e.get("kind", ""),
                    "timestamp": e.get("timestamp"),
                    "summary": _runtime_event_summary(e),
                }
                if isinstance(e, dict)
                else e
                for e in sorted(
                    (events or []),
                    key=lambda item: (
                        item.get("timestamp", 0) if isinstance(item, dict) else 0
                    ),
                    reverse=True,
                )
            ],
            "stats": {
                "goal_count": len(goals),
                "robot_count": len(self.latest_robot_status),
                "skill_count": len(skills),
                "event_count": len(events or []),
            },
        }

    async def _web_episode_task(self, episode_id: str) -> dict[str, Any] | None:
        del episode_id
        goals = self.autonomy_store.goals_recent(limit=1)
        return (
            None
            if not goals
            else {
                "goal": _goal_payload(goals[0]),
                "actions": self.autonomy_store.actions_for_goal(goals[0]["goal_id"]),
            }
        )

    async def create_identity_binding(
        self, envelope: Envelope, ttl_sec: float = 600.0
    ) -> dict[str, Any]:
        binding = self.identity.create_binding(envelope, ttl_sec=ttl_sec)
        scoped = envelope.child(user_id=binding.user_id)
        return self._binding_payload(
            binding, status="pending", continuity=self._identity_continuity(scoped)
        )

    async def identity_binding_status(self, code: str) -> dict[str, Any]:
        binding = self.identity.binding_status(code)
        if binding is None:
            return {"code": code.strip().upper(), "status": "missing"}
        if isinstance(binding, ClaimedBinding):
            envelope = Envelope(
                channel=binding.target_channel,
                chat_id=binding.target_chat_id,
                sender_id=binding.target_sender_id,
                user_id=binding.user_id,
            )
            return self._binding_payload(
                binding,
                status="claimed",
                continuity=self._identity_continuity(envelope),
            )
        envelope = Envelope(
            channel=binding.source_channel,
            chat_id=binding.source_chat_id,
            sender_id=binding.source_sender_id,
            user_id=binding.user_id,
        )
        return self._binding_payload(
            binding, status="pending", continuity=self._identity_continuity(envelope)
        )

    def _agent_id(self, requested: str | None) -> str:
        if requested and requested in self.config.agents:
            return requested
        return self.config.default_agent_id()

    def _episode_dimensions(self, envelope: Envelope) -> list[str]:
        if self.config.identity.unified_user_episodes and envelope.user_id:
            return ["user", "robot"]
        return DEFAULT_EPISODE_DIMENSIONS

    def _register_channels(self) -> None:
        for name, spec in self.config.channels.items():
            if not spec.enabled:
                continue
            context = ChannelContext(
                name=name, spec=spec, deployment_id=self.config.deployment.id
            )
            if spec.type == "cli":
                self.channels.register(CLIChannel(context))
                continue
            if spec.type == "web":
                self.channels.register(
                    WebChannel(
                        context,
                        history_provider=self._web_history,
                        binding_provider=self.create_identity_binding,
                        binding_status_provider=self.identity_binding_status,
                        cockpit_provider=self._web_cockpit,
                        tasks_list_provider=self._web_tasks_list,
                        episode_task_provider=self._web_episode_task,
                        runtime_summary_provider=self._web_runtime_summary,
                    )
                )
                continue
            if spec.type == "voice":
                self.channels.register(VoiceChannel(context))
                continue
            if spec.type == "feishu":
                self.channels.register(FeishuChannel(context))
                continue
            raise ValueError(f"unsupported channel type: {spec.type}")

    def _log_channel_ready(self) -> None:
        for name, _channel in sorted(self.channels.items()):
            spec = self.config.channels[name]
            if spec.type == "web":
                host = spec.settings.get("host", "127.0.0.1")
                port = spec.settings.get("port", 8080)
                logger.info(f"gateway channel [{name}] web ready http://{host}:{port}")
            elif spec.type == "cli":
                prompt = spec.settings.get("prompt", "user> ")
                logger.info(f"gateway channel [{name}] cli ready prompt={prompt!r}")
            elif spec.type == "voice":
                input_device = spec.settings.get("recorder", {}).get("input_device")
                logger.info(
                    f"gateway channel [{name}] voice ready input_device={input_device!r}"
                )
            elif spec.type == "feishu":
                domain = spec.settings.get("domain", "feishu")
                logger.info(f"gateway channel [{name}] feishu ready domain={domain!r}")
            else:
                logger.info(f"gateway channel [{name}] ready type={spec.type}")

    async def _try_handle_identity_binding_turn(self, turn: UserTurn) -> bool:
        match = _BINDING_COMMAND.match(turn.text)
        if match is None:
            return False
        code = match.group(1).upper()
        binding = self.identity.claim_binding(code, turn.envelope)
        reply_text = (
            "绑定成功。这个飞书入口现在会和你当前的 Web 会话使用同一个内部用户身份。"
            if binding is not None
            else "绑定码无效或已过期，请回到 Web 端重新生成。"
        )
        await self.channels.send(
            AgentReply(
                envelope=self._binding_reply_envelope(turn.envelope),
                text=reply_text,
                metadata={
                    "identity_binding": True,
                    "binding_code": code,
                    "success": binding is not None,
                },
            )
        )
        return True

    def _binding_reply_envelope(self, envelope: Envelope) -> Envelope:
        agent_id = self._agent_id(envelope.agent_id)
        robot_id = self.config.default_robot_id(agent_id)
        return envelope.child(
            agent_id=agent_id,
            robot_id=robot_id,
            deployment_id=envelope.deployment_id or self.config.deployment.id,
            user_id=self.identity.resolve(envelope).user_id,
        )

    @staticmethod
    def _binding_payload(
        binding: PendingBinding | ClaimedBinding,
        *,
        status: str,
        continuity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": binding.code,
            "status": status,
            "user_id": binding.user_id,
            "source_channel": binding.source_channel,
            "source_sender_id": binding.source_sender_id,
            "source_chat_id": binding.source_chat_id,
            "created_at": binding.created_at,
            "expires_at": binding.expires_at,
        }
        if isinstance(binding, ClaimedBinding):
            payload.update(
                {
                    "target_channel": binding.target_channel,
                    "target_sender_id": binding.target_sender_id,
                    "target_chat_id": binding.target_chat_id,
                    "claimed_at": binding.claimed_at,
                }
            )
        if continuity:
            payload["continuity"] = continuity
        return payload

    def _identity_continuity(self, envelope: Envelope) -> dict[str, Any]:
        resolution = self.identity.resolve(envelope)
        user_id = resolution.user_id
        linked_channels = self.identity.known_channels(user_id or "")
        linked_targets = [
            {
                "channel": item.channel,
                "chat_id": item.chat_id,
                "sender_id": item.sender_id,
            }
            for item in self.identity.linked_channel_targets(user_id or "")
        ]
        return {
            "user_id": user_id,
            "matched_key": resolution.matched_key,
            "shared_episode_scope": bool(
                self.config.identity.unified_user_episodes and user_id
            ),
            "linked_channels": linked_channels,
            "linked_target_count": len(linked_targets),
            "linked_targets": linked_targets,
        }


def _goal_payload(goal: dict[str, Any]) -> dict[str, Any]:
    snapshot = dict(goal.get("snapshot") or {})
    return {
        "goal_id": goal["goal_id"],
        "task_id": snapshot.get("task_id"),
        "robot_id": goal["robot_id"],
        "objective": snapshot.get("objective"),
        "status": goal["status"],
        "termination_reason": goal.get("termination_reason"),
        "created_at": goal.get("created_at"),
    }


def _compact_status_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    compact = dict(metrics or {})
    last_skill_result = compact.get("last_skill_result")
    if isinstance(last_skill_result, dict):
        compact["last_skill_result"] = _compact_last_skill_result(last_skill_result)
    base_control = compact.get("base_control")
    if isinstance(base_control, dict):
        compact["base_control"] = _compact_base_control(base_control)
    return compact


def _robot_state_name(last_status: Any) -> str:
    if isinstance(last_status, dict):
        return str(last_status.get("state") or "unknown")
    if isinstance(last_status, str):
        return last_status or "unknown"
    return "unknown"


def _robot_status_summary(last_status: Any) -> dict[str, Any]:
    if not isinstance(last_status, dict):
        return {}
    metrics = last_status.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    return {
        "frame_id": last_status.get("frame_id"),
        "success": last_status.get("success"),
        "error": last_status.get("error"),
        "battery": metrics.get("battery"),
        "readiness": metrics.get("readiness"),
    }


def _runtime_event_summary(event: dict[str, Any]) -> str:
    payload = event.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    for value in (
        event.get("summary"),
        event.get("text"),
        payload.get("summary"),
        payload.get("text"),
        payload.get("error"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    if event.get("kind") == EventKind.ROBOT_STATUS.value:
        state = str(payload.get("state") or "unknown")
        frame_id = payload.get("frame_id")
        return f"state={state}" + (
            f", frame={frame_id}" if frame_id is not None else ""
        )
    return ""


def _compact_last_skill_result(value: dict[str, Any]) -> dict[str, Any]:
    compact = dict(value)
    if isinstance(compact.get("motion_trace"), dict):
        compact["motion_trace"] = _motion_trace_summary(compact["motion_trace"])
    last_motion_response = compact.get("last_motion_response")
    if isinstance(last_motion_response, dict) and isinstance(
        last_motion_response.get("control"), dict
    ):
        compact["last_motion_response"] = {
            **last_motion_response,
            "control": _control_summary(last_motion_response["control"]),
        }
    stop_response = compact.get("stop_response")
    if isinstance(stop_response, dict) and isinstance(
        stop_response.get("control"), dict
    ):
        compact["stop_response"] = {
            **stop_response,
            "control": _control_summary(stop_response["control"]),
        }
    return compact


def _compact_base_control(value: dict[str, Any]) -> dict[str, Any]:
    compact = dict(value)
    if isinstance(compact.get("last_motion_report"), dict):
        compact["last_motion_report"] = _motion_trace_summary(
            compact["last_motion_report"]
        )
    base = compact.get("base")
    if isinstance(base, dict):
        base_compact = dict(base)
        if isinstance(base_compact.get("last_motion_report"), dict):
            base_compact["last_motion_report"] = _motion_trace_summary(
                base_compact["last_motion_report"]
            )
        compact["base"] = base_compact
    return compact


def _motion_trace_summary(value: dict[str, Any]) -> dict[str, Any]:
    summary = {key: item for key, item in value.items() if key != "iterations"}
    iterations = value.get("iterations")
    if isinstance(iterations, list):
        summary["iteration_count"] = len(iterations)
        if iterations:
            first = iterations[0] if isinstance(iterations[0], dict) else {}
            last = iterations[-1] if isinstance(iterations[-1], dict) else {}
            summary["first_iteration"] = _iteration_summary(first)
            summary["last_iteration"] = _iteration_summary(last)
    return summary


def _iteration_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in ("index", "elapsed_sec", "success", "message")
        if key in value
    }


def _control_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in ("kind", "timestamp", "requested", "clamped", "success")
        if key in value
    }
