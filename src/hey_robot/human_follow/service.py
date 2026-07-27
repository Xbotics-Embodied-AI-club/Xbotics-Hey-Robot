from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from hey_robot.bus.factory import create_bus_client
from hey_robot.config import DeploymentConfig
from hey_robot.human_follow.perception import (
    HumanFollowRunner,
    load_detector,
)
from hey_robot.logging import HeyRobotLogger
from hey_robot.protocol import Topics
from hey_robot.robot_media.frame_stream import decode_frame_packet

logger = HeyRobotLogger(name="human_follow_service")


@dataclass
class _Session:
    robot_id: str
    skill_id: str
    session_id: str
    arguments: dict[str, Any]
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None


class HumanFollowService:
    """为 human_follow Skill 提供持久化 NATS 数据面的服务。"""

    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.bus = create_bus_client(config.deployment.bus, role="human_follow")
        self.topics = Topics()
        self._stop = asyncio.Event()
        self._frames: dict[str, tuple[dict[str, Any], Any]] = {}
        self._frame_events = {robot_id: asyncio.Event() for robot_id in config.robots}
        self._sessions: dict[str, _Session] = {}

    async def start(self) -> None:
        await self.bus.connect()
        await asyncio.to_thread(load_detector, "models/yolo26n.pt")
        await self.bus.subscribe_raw(
            [
                self.topics.for_robot(self.topics.camera_frame, robot_id)
                for robot_id in self.config.robots
            ],
            self._on_frame,
        )
        await self.bus.subscribe(
            [
                self.topics.for_robot(self.topics.human_follow_command, robot_id)
                for robot_id in self.config.robots
            ],
            self._on_command,
        )
        logger.info("human follow service ready; model preloaded")
        await self._stop.wait()

    async def stop(self) -> None:
        self._stop.set()
        for session in self._sessions.values():
            session.stop.set()
        tasks = [session.task for session in self._sessions.values() if session.task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.bus.close()

    async def _on_frame(self, _topic: str, payload: bytes) -> None:
        try:
            metadata, image = await asyncio.to_thread(decode_frame_packet, payload)
        except Exception as exc:
            logger.warning(f"invalid camera stream frame: {exc}")
            return
        robot_id = str(metadata.get("robot_id") or "")
        if robot_id not in self._frame_events:
            return
        self._frames[robot_id] = (metadata, image)
        self._frame_events[robot_id].set()

    async def _on_command(self, _topic: str, payload: dict[str, Any]) -> None:
        action = str(payload.get("action") or "start")
        session_id = str(payload.get("session_id") or "")
        robot_id = str(payload.get("robot_id") or "")
        if action == "stop":
            session = self._sessions.get(session_id)
            if session is not None:
                session.stop.set()
            return
        if not session_id or robot_id not in self._frame_events:
            return
        active = next(
            (s for s in self._sessions.values() if s.robot_id == robot_id), None
        )
        if active is not None:
            await self._publish_status(
                payload,
                kind="result",
                success=False,
                summary="human follow service is busy",
                failure_mode="service_busy",
            )
            return
        session = _Session(
            robot_id=robot_id,
            skill_id=str(payload.get("skill_id") or ""),
            session_id=session_id,
            arguments=dict(payload.get("arguments") or {}),
        )
        self._sessions[session_id] = session
        session.task = asyncio.create_task(
            self._run_session(session), name=f"human-follow:{robot_id}"
        )

    async def _run_session(self, session: _Session) -> None:
        args = session.arguments
        command_topic = self.topics.for_robot(
            self.topics.base_velocity_stream, session.robot_id
        )
        base = {
            "robot_id": session.robot_id,
            "skill_id": session.skill_id,
            "session_id": session.session_id,
        }
        sequence = 0
        last_progress_at = 0.0

        async def get_frame():
            nonlocal sequence
            event = self._frame_events[session.robot_id]
            try:
                await asyncio.wait_for(event.wait(), timeout=0.5)
            except TimeoutError:
                return None
            event.clear()
            return self._frames[session.robot_id]

        async def apply_velocity(vx, vy, wz):
            nonlocal sequence
            sequence += 1
            now = time.time()
            await self.bus.publish(
                command_topic,
                {
                    **base,
                    "action": "velocity",
                    "sequence": sequence,
                    "vx": vx,
                    "vy": vy,
                    "wz": wz,
                    "expires_at": now + 0.3,
                    "watchdog_ms": 400,
                },
            )

        async def emit_progress(**payload):
            nonlocal last_progress_at
            if time.monotonic() - last_progress_at < 0.5:
                return
            last_progress_at = time.monotonic()
            serialized = dict(payload)
            target = serialized.pop("target", None)
            if target is not None:
                serialized["target"] = {
                    "bbox": list(getattr(target, "bbox", [])),
                    "confidence": getattr(target, "confidence", None),
                    "area": getattr(target, "area", None),
                }
            detections = serialized.get("detections")
            if isinstance(detections, list):
                serialized["detections"] = len(detections)
            await self._publish_session(
                session,
                kind="progress",
                **serialized,
            )

        async def on_start():
            await self.bus.publish(command_topic, {**base, "action": "open"})
            await self._publish_session(session, kind="progress", phase="starting")

        async def on_stop():
            await self.bus.publish(command_topic, {**base, "action": "close"})

        runner = HumanFollowRunner(
            args,
            get_frame=get_frame,
            apply_velocity=apply_velocity,
            emit_progress=emit_progress,
            is_stopped=lambda: session.stop.is_set(),
            on_start=on_start,
            on_stop=on_stop,
        )

        result: dict[str, Any] = {}
        try:
            result = await runner.run()
        except asyncio.CancelledError:
            result = {"success": False, "summary": "human follow interrupted"}
            raise
        except Exception as exc:
            result = {
                "success": False,
                "summary": str(exc),
                "failure_mode": "internal_error",
                "error": str(exc),
            }
        finally:
            await self._publish_session(session, kind="result", **result)
            self._sessions.pop(session.session_id, None)

    async def _publish_session(self, session: _Session, **payload: Any) -> None:
        await self.bus.publish(
            self.topics.for_robot(self.topics.human_follow_status, session.robot_id),
            {
                "robot_id": session.robot_id,
                "skill_id": session.skill_id,
                "session_id": session.session_id,
                "timestamp": time.time(),
                **payload,
            },
        )

    async def _publish_status(self, command: dict[str, Any], **payload: Any) -> None:
        await self.bus.publish(
            self.topics.for_robot(
                self.topics.human_follow_status, str(command.get("robot_id") or "")
            ),
            {**command, **payload, "timestamp": time.time()},
        )
