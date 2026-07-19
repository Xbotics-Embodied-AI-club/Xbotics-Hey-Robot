from __future__ import annotations

import math
import time
from typing import Literal

from hey_robot.protocol import Envelope, RobotAction, RobotStatus
from hey_robot.robot_runtime.base import (
    RobotCapabilities,
    RobotDriverContext,
    RobotHealth,
)
from hey_robot.robot_runtime.observations import DriverObservation, ObservationAsset
from hey_robot.robot_runtime.robocasa_remote.protocol import (
    RemoteEpisodeClient,
    RemoteObservation,
)


class RoboCasaRemoteDriver:
    """RobotDriver adapter for a single remote RoboCasa episode.

    The simulator and its Python dependencies stay in the standalone container.
    This driver intentionally transports only observations and normalized 12-D
    actions; it never translates them to an SO101/XLeRobot joint convention.
    """

    ACTION_DIMENSIONS = 12
    CAMERA_NAMES = ("camera1", "camera2", "camera3")

    def __init__(
        self, context: RobotDriverContext, client: RemoteEpisodeClient
    ) -> None:
        self.context = context
        self.client = client
        self.robot_id = context.robot_id
        settings = context.spec.settings
        self.task = str(settings.get("task", "CloseFridge"))
        self.seed = int(settings.get("seed", 1000))
        self.state: Literal[
            "created", "idle", "executing", "completed", "error", "closed"
        ] = "created"
        self.episode_id: str | None = None
        self.frame_id = 0
        self.done = False
        self.success: bool | None = None
        self.last_error: str | None = None
        self.last_reward: float | None = None
        self._last_observation: RemoteObservation | None = None

    async def start(self) -> None:
        health = await self.client.health()
        if not bool(health.get("online", False)):
            self.state = "error"
            self.last_error = str(health.get("error") or "RoboCasa runtime offline")
            return
        if not bool(health.get("loaded", False)):
            self.state = "error"
            self.last_error = str(
                health.get("error")
                or "RoboCasa runtime dependencies or assets unavailable"
            )
            return
        observation = await self.client.create_episode(task=self.task, seed=self.seed)
        self._accept_observation(observation)
        self.state = "completed" if observation.done else "idle"

    async def capabilities(self) -> RobotCapabilities:
        return RobotCapabilities(
            robot_id=self.robot_id,
            driver_type="robocasa_remote",
            action_dimensions=self.ACTION_DIMENSIONS,
            control_hz=float(self.context.spec.settings.get("control_hz", 20.0)),
            cameras=list(self.CAMERA_NAMES),
            observation_modalities=["image", "state", "status"],
            supports_reset=True,
            supports_interrupt=False,
            metadata={
                "body": "robocasa",
                "robot_family": self.context.spec.robot_family,
                "environment": self.context.spec.robot_environment,
                "embodiment_profile": self.context.embodiment.name
                if self.context.embodiment
                else None,
                "control": "normalized_action",
                "action_schema": {"dimensions": 12, "bounds": "environment"},
                "state_dimensions": 16,
                "runtime": "remote_simulator",
                "simulator_only": True,
            },
        )

    async def health(self) -> RobotHealth:
        online = self.state not in {"created", "closed", "error"}
        return RobotHealth(
            robot_id=self.robot_id,
            online=online,
            state=self.state,
            frame_id=self.frame_id,
            error=self.last_error,
            metrics={
                "episode_id": self.episode_id,
                "task": self.task,
                "seed": self.seed,
                "done": self.done,
                "success": self.success,
                "last_reward": self.last_reward,
                "simulator_only": True,
            },
        )

    async def observe(self) -> DriverObservation:
        observation = await self._require_observation(refresh=True)
        return self._driver_observation(observation)

    async def status(self) -> RobotStatus:
        status_state: Literal["idle", "executing", "error", "offline", "unknown"]
        if self.state == "executing":
            status_state = "executing"
        elif self.state == "error":
            status_state = "error"
        else:
            status_state = "idle"
        return RobotStatus(
            envelope=self._envelope(),
            frame_id=self.frame_id,
            state=status_state,
            task=self.task,
            success=self.success,
            error=self.last_error,
            metrics={
                "driver": "robocasa_remote",
                "episode_id": self.episode_id,
                "done": self.done,
                "last_reward": self.last_reward,
                "simulator_only": True,
            },
        )

    async def apply_action(self, action: RobotAction) -> RobotStatus:
        try:
            self._validate_action(action)
            if self.done:
                raise ValueError(
                    "episode is already done; reset before applying another action"
                )
            if self.episode_id is None:
                raise ValueError("RoboCasa episode has not been created")
            self.state = "executing"
            step = await self.client.step(
                episode_id=self.episode_id,
                action=[float(value) for value in action.values],
                expected_frame_id=self.frame_id,
            )
            self.last_reward = float(step.reward)
            self.done = bool(step.done)
            self.success = bool(step.success) if self.done else None
            self._accept_observation(step.observation)
            self.state = "idle" if not self.done else "completed"
            self.last_error = None
            status = await self.status()
            return RobotStatus(
                envelope=status.envelope,
                frame_id=status.frame_id,
                state=status.state,
                task=self.task,
                skill_id=action.skill_id,
                success=True,
                metrics={**status.metrics, "step": dict(step.metrics)},
            )
        except Exception as exc:
            self.state = "error"
            self.last_error = f"{type(exc).__name__}: {exc}"
            status = await self.status()
            return RobotStatus(
                envelope=status.envelope,
                frame_id=status.frame_id,
                state="error",
                task=self.task,
                skill_id=action.skill_id,
                success=False,
                error=self.last_error,
                metrics=status.metrics,
            )

    async def reset(self) -> RobotStatus:
        if self.episode_id is None:
            self.last_error = "RoboCasa episode has not been created"
            self.state = "error"
            return await self.status()
        try:
            observation = await self.client.reset(episode_id=self.episode_id)
            self._accept_observation(observation)
            self.done = False
            self.success = None
            self.last_reward = None
            self.last_error = None
            self.state = "idle"
        except Exception as exc:
            self.state = "error"
            self.last_error = f"{type(exc).__name__}: {exc}"
        return await self.status()

    async def close(self) -> None:
        try:
            if self.episode_id is not None:
                await self.client.close_episode(episode_id=self.episode_id)
                self.episode_id = None
        finally:
            await self.client.close()
            self.state = "closed"

    def _validate_action(self, action: RobotAction) -> None:
        if len(action.values) != self.ACTION_DIMENSIONS:
            raise ValueError(
                f"action dimension mismatch: expected {self.ACTION_DIMENSIONS}, got {len(action.values)}"
            )
        if not all(math.isfinite(float(value)) for value in action.values):
            raise ValueError("action contains a non-finite value")
        expected = action.metadata.get("expected_frame_id")
        if expected is not None and int(expected) != self.frame_id:
            raise ValueError(
                f"stale action: expected_frame_id={expected}, current_frame_id={self.frame_id}"
            )

    async def _require_observation(self, *, refresh: bool) -> RemoteObservation:
        if self.episode_id is None:
            raise RuntimeError("RoboCasa episode has not been created")
        observation = (
            await self.client.observe(episode_id=self.episode_id)
            if refresh
            else self._last_observation
        )
        if observation is None:
            raise RuntimeError("RoboCasa runtime returned no observation")
        self._accept_observation(observation)
        return observation

    def _accept_observation(self, observation: RemoteObservation) -> None:
        if self.episode_id is not None and observation.episode_id != self.episode_id:
            raise ValueError(
                "RoboCasa runtime returned an observation for another episode"
            )
        if self._last_observation and observation.frame_id < self.frame_id:
            raise ValueError("RoboCasa runtime returned a stale observation frame")
        if len(observation.state) != 16 or not all(
            math.isfinite(float(value)) for value in observation.state
        ):
            raise ValueError(
                "RoboCasa runtime must return exactly 16 finite state values"
            )
        cameras = {image.camera for image in observation.images}
        if cameras != set(self.CAMERA_NAMES):
            raise ValueError(
                f"RoboCasa runtime camera mismatch: expected {self.CAMERA_NAMES}, got {sorted(cameras)}"
            )
        self.episode_id = observation.episode_id
        self.frame_id = int(observation.frame_id)
        self.done = bool(observation.done)
        if self.done:
            self.success = bool(observation.success)
        self._last_observation = observation

    def _driver_observation(self, observation: RemoteObservation) -> DriverObservation:
        return DriverObservation(
            envelope=self._envelope(),
            frame_id=observation.frame_id,
            assets=[
                ObservationAsset(
                    kind="image",
                    role="camera",
                    name=image.camera,
                    data=image.data,
                    content_type=image.content_type,
                    metadata={"width": image.width, "height": image.height},
                )
                for image in observation.images
            ],
            proprioception=[float(value) for value in observation.state],
            task=observation.task or self.task,
            metadata={
                "driver": "robocasa_remote",
                "episode_id": observation.episode_id,
                "done": observation.done,
                "success": observation.success,
                **dict(observation.metadata),
            },
            timestamp=time.time(),
        )

    def _envelope(self) -> Envelope:
        return Envelope(robot_id=self.robot_id)
