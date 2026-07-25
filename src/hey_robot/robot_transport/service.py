"""Bus adapter exposing robot runtimes to external producers and consumers."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from hey_robot.bus.factory import create_bus_client
from hey_robot.config import DeploymentConfig
from hey_robot.events import EventKind, RuntimeEvent
from hey_robot.events.bus import BusEventPublisher
from hey_robot.logging import HeyRobotLogger
from hey_robot.protocol import RobotObservation, RobotStatus, Topics
from hey_robot.protocol.messages import to_payload
from hey_robot.robot_api import BaseVelocityStreamDriver, RobotActionSpec
from hey_robot.robot_media import LocalMediaStore
from hey_robot.robot_media.frame_stream import encode_frame_packet
from hey_robot.robot_runtime.manager import RobotManager
from hey_robot.robot_runtime.runtime import RobotRuntime, SceneCaptioner

logger = HeyRobotLogger(name="robot")


class RobotService:
    """运行机器人驱动并发布观测与状态。"""

    def __init__(
        self,
        config: DeploymentConfig,
        *,
        action_specs: tuple[RobotActionSpec, ...] = (),
        scene_captioner_factory: Callable[[LocalMediaStore], SceneCaptioner]
        | None = None,
    ) -> None:
        self.config = config
        self.topics = Topics()
        self.manager = RobotManager(config, action_specs=action_specs)
        self.bus = create_bus_client(config.deployment.bus, role="robot")
        self.events = BusEventPublisher(self.bus, self.topics)
        self.media_store = LocalMediaStore(
            config.resources.media_root, max_items=config.resources.media_max_items
        )
        scene_captioner = (
            scene_captioner_factory(self.media_store)
            if scene_captioner_factory is not None
            else None
        )
        self.runtimes = {
            driver.robot_id: RobotRuntime(
                driver,
                self.media_store,
                scene_captioner=scene_captioner,
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
        """返回指定 robot_id 的原始驱动，供集成方访问兼容驱动接口。"""
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
                        "skill-surface": capabilities.__dict__,
                        "health": health.__dict__,
                    },
                )
            )
        await self.bus.subscribe(
            [
                self.topics.for_robot(self.topics.base_velocity_stream, robot_id)
                for robot_id in self.runtimes
            ],
            self._on_base_velocity_stream,
        )
        logger.info("robot service 就绪")
        await asyncio.gather(self._observation_loops(), self._stop.wait())

    async def stop(self) -> None:
        self._stop.set()
        await asyncio.gather(
            *(runtime.close() for runtime in self.runtimes.values()),
            return_exceptions=True,
        )
        await self.bus.close()

    async def _observation_loops(self) -> None:
        """Run one isolated observation producer per robot."""
        await asyncio.gather(
            *(
                self._observation_loop(robot_id, runtime)
                for robot_id, runtime in self.runtimes.items()
            )
        )

    async def _observation_loop(self, robot_id: str, runtime: RobotRuntime) -> None:
        period = 1.0 / max(self.publish_hz, self.camera_stream_hz, 0.1)
        while not self._stop.is_set():
            cycle_started = time.monotonic()
            try:
                snapshot = await runtime.refresh_observation(reason="observation_loop")
                status = self._status_for_publish(await runtime.status())

                # 发布机器人观测和状态。
                if self._should_publish_observation(snapshot.observation):
                    await self.bus.publish(
                        self.topics.robot_observation,
                        to_payload(snapshot.observation),
                    )
                await self.bus.publish(self.topics.robot_status, to_payload(status))
                if self._should_log_status(runtime.robot_id, status):
                    logger.info(
                        f"{runtime.robot_id} 心跳 frame={status.frame_id} "
                        f"state={status.state} success={status.success}"
                    )

                # 从同一次 driver observation 发布原始相机帧。
                driver_obs = snapshot.driver_observation
                if driver_obs is not None:
                    for asset in driver_obs.assets:
                        if asset.kind != "image" or asset.data is None:
                            continue
                        packet = await asyncio.to_thread(
                            encode_frame_packet,
                            asset.data,
                            {
                                "robot_id": robot_id,
                                "camera": asset.name or "unknown",
                                "frame_id": snapshot.observation.frame_id,
                                "captured_at": time.time(),
                            },
                        )
                        await self.bus.publish_raw(
                            self.topics.for_robot(self.topics.camera_frame, robot_id),
                            packet,
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(f"robot observation cycle failed: robot={robot_id}")
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
            if isinstance(runtime.driver, BaseVelocityStreamDriver):
                await runtime.driver.stop_base_stream()
            return
        sequence = int(payload.get("sequence") or 0)
        if sequence <= int(active.get("sequence", -1)):
            return
        if float(payload.get("expires_at") or 0.0) < time.time():
            return
        active["sequence"] = sequence
        if isinstance(runtime.driver, BaseVelocityStreamDriver):
            await runtime.driver.apply_stream_velocity(
                vx=float(payload.get("vx") or 0.0),
                vy=float(payload.get("vy") or 0.0),
                wz=float(payload.get("wz") or 0.0),
                watchdog_ms=int(payload.get("watchdog_ms") or 400),
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
            location_id=status.location_id,
            motion_state=status.motion_state,
            battery_percentage=status.battery_percentage,
            task=status.task,
            skill_id=status.skill_id,
            success=status.success,
            error=status.error,
            metrics=status.metrics,
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
