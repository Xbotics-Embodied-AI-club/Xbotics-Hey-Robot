from __future__ import annotations

import asyncio
import time
from typing import Any

from hey_robot.bus.factory import create_bus_client
from hey_robot.config import DeploymentConfig
from hey_robot.events import EventKind, RuntimeEvent, Severity
from hey_robot.events.bus import BusEventPublisher
from hey_robot.logging import HeyRobotLogger
from hey_robot.media import LocalMediaStore
from hey_robot.perception.frame_stream import encode_frame_packet
from hey_robot.protocol import RobotAction, RobotObservation, RobotStatus, Topics
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.robots.manager import RobotManager
from hey_robot.robots.runtime import RobotRuntime
from hey_robot.robots.safety import RobotSafetyError

logger = HeyRobotLogger(name="robot")


class RobotService:
    """Runs robot drivers and exposes them through robot action topics."""

    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.topics = Topics()
        self.manager = RobotManager(config)
        self.bus = create_bus_client(config.deployment.bus)
        self.events = BusEventPublisher(self.bus, self.topics)
        self.media_store = LocalMediaStore(
            config.resources.media_root, max_items=config.resources.media_max_items
        )
        self.runtimes = {
            driver.robot_id: RobotRuntime(
                driver,
                self.media_store,
                image_save_every_n=config.resources.media_image_save_every_n,
            )
            for driver in self.manager.all()
        }
        self.publish_hz = float(
            config.deployment.bus.options.get("robot_publish_hz", 10.0)
        )
        self.status_log_every_frames = int(
            config.deployment.bus.options.get("robot_status_log_every_frames", 30)
        )
        self._last_status_log_frame: dict[str, int] = {}
        self._stop = asyncio.Event()
        self.camera_stream_hz = float(
            config.deployment.bus.options.get("camera_stream_hz", 15.0)
        )
        self._base_streams: dict[str, dict[str, Any]] = {}

    def get(self, robot_id: str):
        """Return the raw driver for a robot id (used by SkillController for VLA I/O adapter injection)."""
        return self.manager.get(robot_id)

    async def start(self) -> None:
        logger.info(
            f"启动 robot service, deployment=[{self.config.deployment.id}] "
            f"robots={','.join(sorted(self.runtimes)) or 'none'}"
        )
        await self.bus.connect()
        await asyncio.gather(*(runtime.start() for runtime in self.runtimes.values()))
        for runtime in self.runtimes.values():
            health = await runtime.health()
            capabilities = await runtime.capabilities()
            logger.info(
                f"{runtime.robot_id} 就绪 state={health.state} "
                f"cameras={','.join(capabilities.cameras)} hz={capabilities.control_hz}"
            )
            await self.events.publish(
                RuntimeEvent.make(
                    EventKind.ROBOT_STARTED,
                    source="robot",
                    robot_id=runtime.robot_id,
                    payload={
                        "capabilities": capabilities.__dict__,
                        "health": health.__dict__,
                    },
                )
            )
        await self.bus.subscribe(
            [self.topics.robot_action],
            self._on_bus_message,
        )
        await self.bus.subscribe(
            [
                self.topics.for_robot(self.topics.base_velocity_stream, robot_id)
                for robot_id in self.runtimes
            ],
            self._on_base_velocity_stream,
        )
        logger.info(f"robot service 就绪, 已订阅 {self.topics.robot_action}")
        await asyncio.gather(
            self._publish_loop(), self._camera_stream_loop(), self._stop.wait()
        )

    async def stop(self) -> None:
        self._stop.set()
        await asyncio.gather(
            *(runtime.close() for runtime in self.runtimes.values()),
            return_exceptions=True,
        )
        await self.bus.close()

    async def _publish_loop(self) -> None:
        period = 1.0 / max(self.publish_hz, 0.1)
        while not self._stop.is_set():
            cycle_started = time.monotonic()
            for runtime in self.runtimes.values():
                observation = await runtime.observe()
                status = self._status_for_publish(await runtime.status())
                if self._should_publish_observation(observation):
                    await self.bus.publish(
                        self.topics.robot_observation, to_payload(observation)
                    )
                await self.bus.publish(self.topics.robot_status, to_payload(status))
                if self._should_log_status(runtime.robot_id, status):
                    logger.info(
                        f"{runtime.robot_id} 心跳 frame={status.frame_id} "
                        f"state={status.state} success={status.success}"
                    )
            remaining = period - (time.monotonic() - cycle_started)
            await asyncio.sleep(max(0.0, remaining))

    async def _camera_stream_loop(self) -> None:
        period = 1.0 / max(self.camera_stream_hz, 0.1)
        while not self._stop.is_set():
            cycle_started = time.monotonic()
            for robot_id, runtime in self.runtimes.items():
                stream_frames = getattr(runtime.driver, "stream_camera_frames", None)
                if not callable(stream_frames):
                    continue
                frames = await stream_frames(timeout_ms=max(20, int(period * 1000)))
                for camera, item in frames.items():
                    image = item.get("image")
                    frame_id = item.get("frame_id")
                    if image is None or frame_id is None:
                        continue
                    packet = await asyncio.to_thread(
                        encode_frame_packet,
                        image,
                        {
                            "robot_id": robot_id,
                            "camera": camera,
                            "frame_id": int(frame_id),
                            "captured_at": time.time(),
                        },
                    )
                    await self.bus.publish_raw(
                        self.topics.for_robot(self.topics.camera_frame, robot_id),
                        packet,
                    )
            remaining = period - (time.monotonic() - cycle_started)
            await asyncio.sleep(max(0.0, remaining))

    async def _on_base_velocity_stream(
        self, _topic: str, payload: dict[str, Any]
    ) -> None:
        robot_id = str(payload.get("robot_id") or "")
        session_id = str(payload.get("session_id") or "")
        runtime = self.runtimes.get(robot_id)
        if runtime is None or not session_id:
            return
        action = str(payload.get("action") or "velocity")
        if action == "open":
            self._base_streams[robot_id] = {"session_id": session_id, "sequence": -1}
            return
        active = self._base_streams.get(robot_id)
        if active is None or active.get("session_id") != session_id:
            return
        if action == "close":
            self._base_streams.pop(robot_id, None)
            stop = getattr(runtime.driver, "stop_base_stream", None)
            if callable(stop):
                await stop()
            return
        sequence = int(payload.get("sequence") or 0)
        if sequence <= int(active.get("sequence", -1)):
            return
        if float(payload.get("expires_at") or 0.0) < time.time():
            return
        active["sequence"] = sequence
        apply_velocity = getattr(runtime.driver, "apply_stream_velocity", None)
        if callable(apply_velocity):
            await apply_velocity(
                vx=float(payload.get("vx") or 0.0),
                vy=float(payload.get("vy") or 0.0),
                wz=float(payload.get("wz") or 0.0),
                watchdog_ms=int(payload.get("watchdog_ms") or 400),
            )

    async def _on_bus_message(self, topic: str, payload: dict) -> None:
        await self._on_action(topic, payload)

    async def _on_action(self, _topic: str, payload: dict) -> None:
        action = from_payload(RobotAction, payload)
        robot_id = action.envelope.robot_id
        if not robot_id:
            return
        action_metadata = dict(action.metadata or {})
        raw_skill_payload = action_metadata.get("skill")
        skill_payload = raw_skill_payload if isinstance(raw_skill_payload, dict) else {}
        await self.events.publish(
            RuntimeEvent.make(
                EventKind.ROBOT_SKILL_RECEIVED,
                source="robot",
                trace_id=action.envelope.trace_id,
                episode_id=action.envelope.episode_id,
                agent_id=action.envelope.agent_id,
                robot_id=robot_id,
                channel=action.envelope.channel,
                payload={
                    "summary": f"received {skill_payload.get('name') or action.skill_id or 'robot action'}",
                    "action_id": action.action_id,
                    "skill_id": action.skill_id or None,
                    "skill": skill_payload,
                },
            )
        )
        try:
            status = await self._runtime(robot_id).apply_action(action)
        except RobotSafetyError as exc:
            logger.warning(f"robot action 被安全策略阻止 robot={robot_id} error={exc}")
            status = RobotStatus(
                envelope=action.envelope,
                frame_id=None,
                state="failed",
                task=None,
                skill_id=action.skill_id or None,
                success=False,
                error=str(exc),
                metrics={"source": "robot.safety"},
            )
        status = self._status_for_publish(status, action.envelope)
        status_metrics = dict(status.metrics or {})
        logger.info(
            "robot_status_trace publishing_action_status "
            f"robot={robot_id} action_id={action.action_id} "
            f"action_skill_id={action.skill_id or None} status_skill_id={status.skill_id or None} "
            f"success={status.success} state={status.state} frame={status.frame_id} "
            f"trace={status.envelope.trace_id} episode={status.envelope.episode_id}"
        )
        await self.bus.publish(self.topics.robot_status, to_payload(status))
        await self.events.publish(
            RuntimeEvent.make(
                "robot.skill.executed",
                source="robot",
                severity=Severity.INFO if status.success else Severity.ERROR,
                trace_id=action.envelope.trace_id,
                episode_id=action.envelope.episode_id,
                agent_id=action.envelope.agent_id,
                robot_id=robot_id,
                channel=action.envelope.channel,
                payload={
                    "summary": f"{skill_payload.get('name') or action.skill_id or 'robot action'} "
                    f"{'completed' if status.success else 'failed'}",
                    "action_id": action.action_id,
                    "skill_id": status.skill_id,
                    "state": status.state,
                    "success": status.success,
                    "error": status.error,
                    "last_skill_result": status_metrics.get("last_skill_result"),
                    "base_control": status_metrics.get("base_control"),
                },
            )
        )
        observation = await self._runtime(robot_id).observe()
        observation = self._observation_for_publish(observation, action.envelope)
        if self._should_publish_observation(observation):
            await self.bus.publish(
                self.topics.robot_observation, to_payload(observation)
            )
        if self._should_log_status(robot_id, status):
            logger.info(
                f"{robot_id} 已执行 skill={action.skill_id!r} "
                f"dims={len(action.values)} state={status.state} success={status.success}"
            )

    def _runtime(self, robot_id: str) -> RobotRuntime:
        runtime = self.runtimes.get(robot_id)
        if runtime is None:
            raise KeyError(robot_id)
        return runtime

    def _status_for_publish(self, status: RobotStatus, envelope=None) -> RobotStatus:
        base_envelope = (
            status.envelope
            if envelope is None
            else status.envelope.child(
                trace_id=envelope.trace_id,
                episode_id=envelope.episode_id,
                agent_id=envelope.agent_id,
                channel=envelope.channel,
                account_id=envelope.account_id,
                chat_id=envelope.chat_id,
                chat_type=envelope.chat_type,
                sender_id=envelope.sender_id,
                robot_id=envelope.robot_id or status.envelope.robot_id,
                deployment_id=envelope.deployment_id or status.envelope.deployment_id,
            )
        )
        return RobotStatus(
            envelope=base_envelope,
            frame_id=status.frame_id,
            state=status.state,
            task=status.task,
            skill_id=status.skill_id,
            success=status.success,
            error=status.error,
            metrics=status.metrics,
        )

    @staticmethod
    def _observation_for_publish(
        observation: RobotObservation, envelope
    ) -> RobotObservation:
        return RobotObservation(
            envelope=observation.envelope.child(
                trace_id=envelope.trace_id,
                episode_id=envelope.episode_id,
                agent_id=envelope.agent_id,
                channel=envelope.channel,
                account_id=envelope.account_id,
                chat_id=envelope.chat_id,
                chat_type=envelope.chat_type,
                sender_id=envelope.sender_id,
                robot_id=envelope.robot_id or observation.envelope.robot_id,
                deployment_id=envelope.deployment_id
                or observation.envelope.deployment_id,
            ),
            frame_id=observation.frame_id,
            images=observation.images,
            artifacts=observation.artifacts,
            proprioception=observation.proprioception,
            task=observation.task,
            raw=observation.raw,
        )

    def _should_log_status(self, robot_id: str, status: RobotStatus) -> bool:
        frame_id = status.frame_id
        if frame_id is None:
            return False
        last = self._last_status_log_frame.get(robot_id)
        if last == frame_id:
            return False
        if (
            frame_id != 0
            and frame_id % max(self.status_log_every_frames, 1) != 0
            and status.state
            not in {
                "failed",
                "terminated",
                "skill_completed",
            }
        ):
            return False
        self._last_status_log_frame[robot_id] = frame_id
        return True

    @staticmethod
    def _should_publish_observation(observation: RobotObservation) -> bool:
        if observation.artifacts:
            return True
        observation_raw: dict[str, object] = dict(observation.raw or {})
        perception = observation_raw.get("perception")
        if not isinstance(perception, dict):
            return bool(observation.images)
        if "valid_image_count" in perception:
            return int(perception.get("valid_image_count") or 0) > 0
        return bool(observation.images)
