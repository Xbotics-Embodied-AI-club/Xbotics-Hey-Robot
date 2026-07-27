from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from hey_robot.bus.factory import create_bus_client
from hey_robot.channels import (
    ChannelContext,
    ChannelManager,
    CLIChannel,
    FeishuChannel,
    VoiceChannel,
    WebChannel,
)
from hey_robot.cognition.runtime.agent_task_store import (
    AgentTask,
    AgentTaskStore,
)
from hey_robot.config import DeploymentConfig
from hey_robot.episode import JsonlEpisodeStore, allocate_episode
from hey_robot.episode.scope import DEFAULT_EPISODE_DIMENSIONS
from hey_robot.events import EventKind, RuntimeEvent
from hey_robot.events.bus import BusEventPublisher
from hey_robot.events.store import RuntimeEventStore
from hey_robot.gateway.identity import ClaimedBinding, IdentityResolver, PendingBinding
from hey_robot.gateway.receipts import InteractionReceiptStore
from hey_robot.health import HealthReportService
from hey_robot.logging import HeyRobotLogger
from hey_robot.persistence import FileRunStore
from hey_robot.protocol import (
    AgentControl,
    AgentReply,
    ConversationResult,
    ConversationTurn,
    Envelope,
    RobotStatus,
    SkillEvent,
    Topics,
    UserTurn,
)
from hey_robot.protocol.messages import (
    from_payload,
    to_payload,
)

logger = HeyRobotLogger(name="gateway")
_BINDING_COMMAND = re.compile(
    r"^\s*(?:bind|绑定)\s+([A-Za-z0-9]{4,12})\s*$", re.IGNORECASE
)
_ROBOT_STATUS_PERSIST_INTERVAL_SEC = 5.0


class GatewayService:
    """负责渠道输入输出标准化，但不修改任务状态的 Gateway。"""

    def __init__(
        self, config: DeploymentConfig, *, episode_dir: str | Path | None = None
    ) -> None:
        unsupported_channels = sorted(
            {
                spec.type
                for spec in config.channels.values()
                if spec.enabled and spec.type not in {"cli", "web", "voice", "feishu"}
            }
        )
        if unsupported_channels:
            raise ValueError(
                f"unsupported channel type: {', '.join(unsupported_channels)}"
            )
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
        self.run_store = FileRunStore(Path(config.resources.runtime_dir) / "runs")
        task_path = Path(config.resources.runtime_dir) / "sustained_tasks.sqlite3"
        self.task_store = AgentTaskStore(task_path)
        self.interaction_receipts = InteractionReceiptStore(
            Path(config.resources.runtime_dir) / "interaction_receipts.sqlite3"
        )
        self.latest_robot_status: dict[str, RobotStatus] = {}
        self.identity = IdentityResolver(
            config.identity,
            state_path=Path(config.resources.runtime_dir)
            / "identity"
            / "bindings.json",
        )
        self._ready = asyncio.Event()
        self._stopped = False
        self._last_robot_status_persisted_at: dict[str, float] = {}
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
        await self.bus.subscribe(
            [self.topics.conversation_result], self._on_conversation_result
        )
        await self.bus.subscribe([self.topics.runtime_event], self._on_runtime_event)
        await self.bus.subscribe([self.topics.robot_status], self._on_robot_status)
        await self.bus.subscribe([self.topics.skill_event], self._on_skill_event)
        logger.info(
            f"gateway subscribed {self.topics.agent_reply}, {self.topics.conversation_result}, {self.topics.runtime_event}, "
            f"{self.topics.robot_status}, {self.topics.skill_event}"
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
        if self._stopped:
            return
        self._stopped = True
        event = RuntimeEvent.make(EventKind.GATEWAY_SHUTDOWN, source="gateway")
        await self.events.publish(event)
        self.event_store.append(event)
        await self.channels.stop_all()
        await self.bus.close()
        self.interaction_receipts.close()

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
        payload_hash = self._interaction_payload_hash(turn, envelope)
        interaction_id = self._interaction_id(envelope, payload_hash)
        if not self.interaction_receipts.claim(interaction_id, payload_hash):
            logger.info(f"Ignoring replayed interaction {interaction_id}")
            return
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
        session_key = self._session_key(envelope)
        if await self._handle_safety_command(
            turn.text, envelope, interaction_id, session_key
        ):
            self.interaction_receipts.complete(interaction_id, "safety_command")
            return
        kind: Literal["prompt", "steer"] = (
            "steer" if turn.intent in {"steer", "follow_up"} else "prompt"
        )
        await self.bus.publish(
            self.topics.conversation_turn,
            to_payload(
                ConversationTurn(
                    envelope.child(episode_id=allocation.episode_id),
                    session_key,
                    interaction_id,
                    turn.text,
                    kind,
                )
            ),
        )

    async def _on_conversation_result(self, _topic: str, payload: dict) -> None:
        result = from_payload(ConversationResult, payload)
        await self._send_reply(
            AgentReply(
                envelope=result.envelope,
                text=result.text,
                final=result.final,
                metadata={"interaction_id": result.interaction_id},
            )
        )

    def _session_key(self, envelope: Envelope) -> str:
        if self.config.identity.unified_user_episodes and envelope.user_id:
            principal = envelope.user_id
        else:
            principal = (
                f"{envelope.channel or 'unknown'}:"
                f"{envelope.chat_id or envelope.sender_id or 'anonymous'}"
            )
        return f"{self.config.deployment.id}:{envelope.agent_id or self._agent_id(None)}:{principal}"

    async def _handle_safety_command(
        self,
        text: str,
        envelope: Envelope,
        interaction_id: str,
        session_key: str,
    ) -> bool:
        """路由高优先级控制，无需等待 LLM 回复。"""
        normalized = " ".join(str(text or "").lower().split())
        compact = normalized.replace(" ", "")
        emergency = {
            "emergency stop",
            "emergencystop",
            "e-stop",
            "estop",
            "\u6025\u505c",
            "\u7d27\u6025\u505c\u6b62",
        }
        if compact in {item.replace(" ", "") for item in emergency}:
            await self.bus.publish(
                self.topics.agent_control,
                to_payload(
                    AgentControl(
                        envelope,
                        session_key,
                        interaction_id,
                        "emergency_stop",
                        "gateway emergency stop",
                    )
                ),
            )
            return True

        cancel = {
            "cancel current task",
            "cancel task",
            "stop current task",
            "\u53d6\u6d88\u5f53\u524d\u4efb\u52a1",
        }
        if compact in {item.replace(" ", "") for item in cancel}:
            await self.bus.publish(
                self.topics.agent_control,
                to_payload(
                    AgentControl(
                        envelope,
                        session_key,
                        interaction_id,
                        "cancel",
                        "user cancelled current task",
                    )
                ),
            )
            return True

        pause = {"pause", "pause task", "暂停", "暂停任务"}
        if compact in {item.replace(" ", "") for item in pause}:
            await self.bus.publish(
                self.topics.agent_control,
                to_payload(
                    AgentControl(
                        envelope,
                        session_key,
                        interaction_id,
                        "pause",
                        "user paused current task",
                    )
                ),
            )
            return True

        confirmations = {"confirm", "yes", "\u786e\u8ba4", "\u7ee7\u7eed"}
        if compact in confirmations:
            await self.bus.publish(
                self.topics.agent_control,
                to_payload(
                    AgentControl(
                        envelope,
                        session_key,
                        interaction_id,
                        "resume",
                        "user resumed current task",
                    )
                ),
            )
            return True

        query = {
            "status",
            "task status",
            "current progress",
            "\u5f53\u524d\u8fdb\u5ea6",
            "\u673a\u5668\u4eba\u72b6\u6001",
        }
        if compact in {item.replace(" ", "") for item in query}:
            return False
        return False

    @staticmethod
    def _interaction_id(envelope: Envelope, payload_hash: str) -> str:
        # transport message_id/turn_id 用于识别重试投递。本地输入如果没有该标识，
        # 会获得一个按 payload 作用域生成的 receipt，避免同一进程调用方中共享 Envelope
        # 的不同命令被误吞掉。
        source = envelope.message_id or envelope.turn_id
        if source is None:
            source = f"{envelope.trace_id}:{payload_hash}"
        raw = "|".join(
            (
                str(envelope.deployment_id or ""),
                str(envelope.channel or ""),
                str(envelope.account_id or ""),
                str(envelope.user_id or envelope.sender_id or ""),
                str(source or ""),
            )
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _interaction_payload_hash(turn: UserTurn, envelope: Envelope) -> str:
        payload = {
            "text": turn.text,
            "media": [item.uri for item in turn.media],
            "channel": envelope.channel,
            "user_id": envelope.user_id,
            "message_id": envelope.message_id,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    async def _send_reply(self, reply: AgentReply) -> None:
        await self.bus.publish(self.topics.agent_reply, to_payload(reply))

    async def _on_agent_reply(self, _topic: str, payload: dict) -> None:
        reply = self._materialize_reply(from_payload(AgentReply, payload))
        logger.debug(
            f"gateway received agent reply trace={reply.envelope.trace_id} "
            f"channel={reply.envelope.channel} text_len={len(reply.text)}"
        )
        if reply.final and reply.envelope.episode_id:
            self.episodes.append_agent_reply(reply.envelope.episode_id, reply)
        await self.channels.send(reply)
        interaction_id = reply.metadata.get("interaction_id")
        if reply.final and isinstance(interaction_id, str):
            self.interaction_receipts.complete(interaction_id, "conversation_turn")

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
        tasks = self.task_store.recent_tasks(limit=1)
        if not tasks:
            return None
        task = tasks[0]
        steps = self.task_store.recent_steps(task.task_id)
        return {
            "task": _task_payload(task),
            "steps": [_step_payload(step) for step in steps],
            "runs": [
                _run_payload(event)
                for step in steps
                if step.run_id is not None
                if (event := self.run_store.latest_event(step.run_id)) is not None
            ],
            "health": HealthReportService(self.config).payload(robot_id=task.robot_id),
        }

    async def _web_tasks_list(self, limit: int) -> dict[str, Any]:
        return {
            "tasks": [
                _task_payload(task) for task in self.task_store.recent_tasks(limit)
            ]
        }

    async def _web_runtime_summary(self, limit: int) -> dict[str, Any]:
        tasks = self.task_store.recent_tasks(limit)
        runs = self.run_store.recent(limit=limit)
        events = self.event_store.recent(limit=limit)
        return {
            "tasks": [_task_payload(task) for task in tasks],
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
            "skills": [_run_payload(event) for event in runs],
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
                "task_count": len(tasks),
                "robot_count": len(self.latest_robot_status),
                "skill_count": len(runs),
                "event_count": len(events or []),
            },
        }

    async def _web_episode_task(self, episode_id: str) -> dict[str, Any] | None:
        del episode_id
        tasks = self.task_store.recent_tasks(limit=1)
        if not tasks:
            return None
        task = tasks[0]
        steps = self.task_store.recent_steps(task.task_id)
        return {
            "task": _task_payload(task),
            "steps": [_step_payload(step) for step in steps],
            "runs": [
                _run_payload(event)
                for step in steps
                if step.run_id is not None
                if (event := self.run_store.latest_event(step.run_id)) is not None
            ],
        }

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
                        environment_completion_provider=self._web_environment_complete,
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

    def _web_environment_complete(self, task_id: str, reason: str) -> dict[str, Any]:
        task = self.task_store.task(task_id)
        if task is None:
            raise ValueError(f"unknown task: {task_id}")
        self.task_store.complete_from_environment(task_id, recap=reason)
        completed = self.task_store.task(task_id) or task
        return _task_payload(completed)

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


def _task_payload(task: AgentTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "session_key": task.session_key,
        "robot_id": task.robot_id,
        "objective": task.objective,
        "ui_summary": task.ui_summary,
        "status": task.status,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "step_count": task.step_count,
        "last_error": task.last_error,
        "final_recap": task.final_recap,
    }


def _step_payload(step: Any) -> dict[str, Any]:
    return {
        "step_id": step.step_id,
        "task_id": step.task_id,
        "sequence": step.sequence,
        "skill": step.proposal.name,
        "arguments": dict(step.proposal.arguments),
        "status": step.outcome.status,
        "summary": step.outcome.user_summary,
        "evidence_ids": list(step.evidence_ids),
        "started_at": step.started_at,
        "completed_at": step.completed_at,
        "run_id": step.run_id,
    }


def _run_payload(event: Any) -> dict[str, Any]:
    result = event.result
    return {
        "run_id": event.run_id,
        "sequence": event.sequence,
        "name": event.name,
        "skill": event.name,
        "envelope": to_payload(event.envelope),
        "phase": event.phase,
        "timestamp": event.timestamp,
        "progress": event.progress,
        "frame_id": event.frame_id,
        "summary": result.summary if result is not None else event.summary,
        "success": result.success if result is not None else None,
        "failure_mode": result.failure_mode if result is not None else None,
        "error": result.error if result is not None else None,
        "data": dict(result.data) if result is not None else {},
        "artifacts": [
            {
                "uri": artifact.uri,
                "artifact_type": artifact.artifact_type,
                "role": artifact.role,
            }
            for artifact in (result.artifacts if result is not None else ())
        ],
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
