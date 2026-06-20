from __future__ import annotations

import asyncio
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from hey_robot.agents.task_run import TaskRunStore
from hey_robot.bus.factory import create_bus_client
from hey_robot.channels import (
    ChannelContext,
    ChannelManager,
    CLIChannel,
    FeishuChannel,
    VoiceChannel,
    WebChannel,
)
from hey_robot.config import DeploymentConfig
from hey_robot.episode import (
    JsonlEpisodeStore,
    RobotEpisodeStateStore,
    allocate_episode,
)
from hey_robot.episode.scope import DEFAULT_EPISODE_DIMENSIONS
from hey_robot.events import EventKind, RuntimeEvent
from hey_robot.events.bus import BusEventPublisher
from hey_robot.events.store import RuntimeEventStore
from hey_robot.gateway.identity import ClaimedBinding, IdentityResolver, PendingBinding
from hey_robot.health import HealthReportService
from hey_robot.interaction import InteractionStateStore
from hey_robot.logging import HeyRobotLogger
from hey_robot.memory import SceneMemoryStore
from hey_robot.protocol import (
    AgentReply,
    Envelope,
    RobotStatus,
    SkillEvent,
    SkillResult,
    Topics,
    UserTurn,
)
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.skills import SkillStore
from hey_robot.tasks import TaskSessionQueryService

logger = HeyRobotLogger(name="gateway")
_BINDING_COMMAND = re.compile(
    r"^\s*(?:bind|绑定)\s+([A-Za-z0-9]{4,12})\s*$", re.IGNORECASE
)
_ROBOT_STATUS_PERSIST_INTERVAL_SEC = 5.0


class GatewayService:
    """Channel gateway that normalizes inbound turns and forwards outbound replies."""

    def __init__(
        self, config: DeploymentConfig, *, episode_dir: str | Path | None = None
    ) -> None:
        self.config = config
        self.topics = Topics()
        self.episode_root = Path(episode_dir or config.resources.episodes_root)
        self.episodes = JsonlEpisodeStore(self.episode_root)
        self.task_runs = TaskRunStore(self.episode_root)
        self.robot_states = RobotEpisodeStateStore(self.episode_root)
        self.channels = ChannelManager()
        self.bus = create_bus_client(config.deployment.bus)
        self.events = BusEventPublisher(self.bus, self.topics)
        self.event_store = RuntimeEventStore(
            Path(config.resources.runtime_dir) / "events",
            max_items=config.resources.events_max_items,
        )
        self.skill_store = SkillStore(
            Path(config.resources.runtime_dir) / "skills",
            max_items=config.resources.events_max_items,
        )
        self.scene_memory = SceneMemoryStore(
            Path(config.resources.runtime_dir) / "scene_memory",
            max_items=config.resources.events_max_items,
        )
        self.interaction_states = InteractionStateStore(
            Path(config.resources.runtime_dir) / "interaction"
        )
        self.task_views = TaskSessionQueryService(
            task_store=self.task_runs,
            scene_memory=self.scene_memory,
            skill_store=self.skill_store,
            interaction_store=self.interaction_states,
        )
        self.identity = IdentityResolver(
            config.identity,
            state_path=Path(config.resources.runtime_dir)
            / "identity"
            / "bindings.json",
        )
        self._ready = asyncio.Event()
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
            envelope, agent_id=agent_id, dimensions=self._episode_dimensions(envelope)
        )
        envelope = envelope.child(episode_id=allocation.episode_id)
        forwarded = UserTurn(
            envelope=envelope,
            text=turn.text,
            media=turn.media,
            intent=turn.intent,
            metadata=turn.metadata,
        )
        self.episodes.ensure(
            allocation.episode_id, allocation.scope, allocation.aliases
        )
        self.robot_states.ensure(
            allocation.episode_id, agent_id=agent_id, robot_id=robot_id
        )
        self.episodes.append_user_turn(allocation.episode_id, forwarded)
        active_task = self.task_runs.load_active(allocation.episode_id)
        self.interaction_states.record_turn(
            forwarded,
            active_task_id=active_task.task_id if active_task is not None else None,
            pending_confirmation=(
                dict(active_task.pending_confirmation)
                if active_task is not None
                and isinstance(active_task.pending_confirmation, dict)
                else None
            ),
            robot_busy=(
                active_task is not None
                and active_task.status not in {"completed", "failed", "cancelled"}
            ),
        )
        event = RuntimeEvent.make(
            EventKind.EPISODE_ALLOCATED,
            source="gateway",
            trace_id=envelope.trace_id,
            episode_id=allocation.episode_id,
            agent_id=agent_id,
            robot_id=robot_id,
            channel=envelope.channel,
            payload={"aliases": allocation.aliases, "user_id": envelope.user_id},
        )
        await self.events.publish(event)
        self.event_store.append(event)
        await self.bus.publish(self.topics.user_turn, to_payload(forwarded))
        logger.debug(
            f"gateway forwarded turn trace={forwarded.envelope.trace_id} "
            f"episode={allocation.episode_id} agent={agent_id} robot={robot_id}"
        )

    async def _on_agent_reply(self, _topic: str, payload: dict) -> None:
        reply = self._materialize_reply(from_payload(AgentReply, payload))
        logger.debug(
            f"gateway received agent reply trace={reply.envelope.trace_id} "
            f"channel={reply.envelope.channel} text_len={len(reply.text)}"
        )
        if reply.envelope.episode_id:
            self.episodes.append_agent_reply(reply.envelope.episode_id, reply)
            active_task = self.task_runs.load_active(reply.envelope.episode_id)
            self.interaction_states.set_pending_confirmation(
                reply.envelope.episode_id,
                (
                    dict(active_task.pending_confirmation)
                    if active_task is not None
                    and isinstance(active_task.pending_confirmation, dict)
                    else None
                ),
            )
            self.interaction_states.record_reply(reply)
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
            self.robot_states.ensure(
                allocation.episode_id, agent_id=agent_id, robot_id=robot_id
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
        self.robot_states.apply_skill_event(event)
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
        result = from_payload(SkillResult, payload)
        self.robot_states.apply_skill_result(result)

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
        records = await asyncio.to_thread(
            self.episodes.history, allocation.episode_id, limit=limit
        )
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
        view = self.task_views.view_for_episode(episode_id)
        if view is None:
            return None
        return {
            "episode_id": episode_id,
            "view": view.to_dict(),
            "health": HealthReportService(
                self.config,
                episode_dir=self.episode_root,
            ).payload(robot_id=view.robot_id),
        }

    async def _web_tasks_list(self, limit: int) -> dict[str, Any]:
        tasks = await asyncio.to_thread(self.task_runs.list_recent, limit=limit)
        return {
            "tasks": [
                {
                    "task_id": t.task_id,
                    "episode_id": t.episode_id,
                    "robot_id": t.robot_id,
                    "agent_id": t.agent_id,
                    "root_task": t.root_task,
                    "status": t.status,
                    "task_success": t.task_success,
                    "failure_reason": t.failure_reason,
                    "retry_count": t.retry_count,
                    "recovery_count": t.recovery_count,
                    "skill_ids": t.skill_ids,
                    "created_at": t.created_at,
                    "updated_at": t.updated_at,
                }
                for t in tasks
            ]
        }

    async def _web_runtime_summary(self, limit: int) -> dict[str, Any]:
        tasks, robot_states, skills, events = await asyncio.gather(
            asyncio.to_thread(self.task_runs.list_recent, limit=limit),
            asyncio.to_thread(self.robot_states.list_states),
            asyncio.to_thread(self.skill_store.recent, limit=limit),
            asyncio.to_thread(self.event_store.recent, limit=limit),
        )
        return {
            "tasks": [
                {
                    "task_id": t.task_id,
                    "episode_id": t.episode_id,
                    "robot_id": t.robot_id,
                    "root_task": t.root_task,
                    "status": t.status,
                    "task_success": t.task_success,
                    "failure_reason": t.failure_reason,
                    "updated_at": t.updated_at,
                }
                for t in tasks
            ],
            "robots": [
                {
                    "episode_id": rs.episode_id,
                    "robot_id": rs.robot_id,
                    "state": _robot_state_name(rs.last_status),
                    "status": _robot_status_summary(rs.last_status),
                    "active_task": rs.active_task,
                    "updated_at": rs.updated_at,
                }
                for rs in (robot_states or [])
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
                "task_count": len(tasks),
                "robot_count": len(robot_states or []),
                "skill_count": len(skills),
                "event_count": len(events or []),
            },
        }

    async def _web_episode_task(self, episode_id: str) -> dict[str, Any] | None:
        tasks = await asyncio.to_thread(self.task_runs.list_for_episode, episode_id)
        if not tasks:
            return None
        task = tasks[0]
        robot_state = await asyncio.to_thread(self.robot_states.load, episode_id)
        result: dict[str, Any] = {
            "episode_id": episode_id,
            "task": {
                "task_id": task.task_id,
                "episode_id": task.episode_id,
                "robot_id": task.robot_id,
                "agent_id": task.agent_id,
                "root_task": task.root_task,
                "status": task.status,
                "task_success": task.task_success,
                "failure_reason": task.failure_reason,
                "retry_count": task.retry_count,
                "recovery_count": task.recovery_count,
                "skill_ids": task.skill_ids,
                "created_at": task.created_at,
                "updated_at": task.updated_at,
                "attempts": [a.to_dict() for a in (task.attempts or [])],
            },
        }
        if robot_state is not None:
            result["robot"] = (
                robot_state.to_dict()
                if hasattr(robot_state, "to_dict")
                else robot_state
            )
        return result

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
