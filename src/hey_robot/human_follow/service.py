from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from hey_robot.bus.factory import create_bus_client
from hey_robot.config import DeploymentConfig
from hey_robot.logging import HeyRobotLogger
from hey_robot.perception.frame_stream import decode_frame_packet
from hey_robot.perception.human_follow import (
    FollowController,
    TargetTracker,
    VelocityCommand,
    detect_people,
    load_detector,
)
from hey_robot.protocol import Topics

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
    """Persistent NATS data-plane service for the human_follow Skill."""

    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.bus = create_bus_client(config.deployment.bus)
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
        tracker = TargetTracker(
            max_age=int(args.get("max_tracking_age") or 30),
            min_iou=float(args.get("min_iou_threshold") or 0.3),
        )
        controller = FollowController(
            target_distance=float(args.get("target_distance_m") or 0.7),
            target_width_ratio=float(args.get("target_width_ratio") or 0.35),
            target_height_ratio=float(args.get("target_height_ratio") or 1.0),
            kp_linear=float(args.get("kp_linear") or 0.35),
            kp_angular=float(args.get("kp_angular") or 1.0),
            max_linear_speed=float(args.get("max_linear_speed") or 0.3),
            max_backward_speed=float(args.get("max_backward_speed") or 0.2),
            allow_backward=bool(args.get("allow_backward", True)),
            max_angular_speed=float(args.get("max_angular_speed") or 1.0),
            dead_zone_x=float(args.get("dead_zone_x") or 0.15),
            dead_zone_area=float(args.get("dead_zone_area") or 0.1),
        )
        duration_raw = args.get("duration_sec")
        max_steps = int(args.get("max_steps") or 0)
        if duration_raw is None and max_steps <= 0:
            duration_raw = 120.0
        duration = float(duration_raw) if duration_raw is not None else None
        deadline = time.monotonic() + duration if duration else None
        current = VelocityCommand(0.0, 0.0, 0.0)
        last_frame_id: int | None = None
        sequence = 0
        last_progress_at = 0.0
        command_topic = self.topics.for_robot(
            self.topics.base_velocity_stream, session.robot_id
        )
        base = {
            "robot_id": session.robot_id,
            "skill_id": session.skill_id,
            "session_id": session.session_id,
        }
        await self.bus.publish(command_topic, {**base, "action": "open"})
        await self._publish_session(session, kind="progress", phase="starting")
        result: dict[str, Any] = {
            "success": True,
            "summary": "human follow stopped",
        }
        try:
            while not session.stop.is_set():
                if deadline is not None and time.monotonic() >= deadline:
                    result["summary"] = "human follow completed"
                    break
                if max_steps > 0 and sequence >= max_steps:
                    result["summary"] = "human follow completed"
                    break
                event = self._frame_events[session.robot_id]
                try:
                    await asyncio.wait_for(event.wait(), timeout=0.5)
                except TimeoutError:
                    await self._publish_session(
                        session, kind="progress", phase="waiting_for_camera"
                    )
                    continue
                event.clear()
                metadata, image = self._frames[session.robot_id]
                frame_id = int(metadata.get("frame_id") or 0)
                if frame_id == last_frame_id:
                    continue
                last_frame_id = frame_id
                detections = await asyncio.to_thread(detect_people, image)
                target = tracker.update(detections)
                height, width = image.shape[:2]
                command = controller.compute_velocity(
                    target, frame_width=width, frame_height=height
                )
                phase = "following"
                if command is None:
                    if not controller.is_searching():
                        continue
                    command = controller.compute_search_velocity()
                    phase = "searching"
                if controller.is_target_lost():
                    result = {
                        "success": False,
                        "summary": "person lost during human follow",
                        "failure_mode": "person_lost",
                        "error": "person lost during human follow",
                    }
                    break
                current = controller.smooth_velocity(current, command, alpha=0.3)
                sequence += 1
                now = time.time()
                await self.bus.publish(
                    command_topic,
                    {
                        **base,
                        "action": "velocity",
                        "sequence": sequence,
                        "frame_id": frame_id,
                        "vx": current.vx,
                        "vy": current.vy,
                        "wz": current.vz,
                        "expires_at": now + 0.3,
                        "watchdog_ms": 400,
                    },
                )
                if time.monotonic() - last_progress_at >= 0.5:
                    await self._publish_session(
                        session,
                        kind="progress",
                        phase=phase,
                        frame_id=frame_id,
                        command={"vx": current.vx, "vy": current.vy, "wz": current.vz},
                        detections=len(detections),
                    )
                    last_progress_at = time.monotonic()
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
            await self.bus.publish(command_topic, {**base, "action": "close"})
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
