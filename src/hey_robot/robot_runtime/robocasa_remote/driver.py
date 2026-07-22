from __future__ import annotations

import io
import math
import time
import uuid
from typing import Literal

import grpc
import numpy as np
from PIL import Image

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
        self,
        context: RobotDriverContext,
        client: RemoteEpisodeClient,
        *,
        control_client: RemoteEpisodeClient | None = None,
    ) -> None:
        self.context = context
        self.client = client
        self.control_client = control_client
        self.robot_id = context.robot_id
        self.task: str | None = None
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
        self.state = "idle"

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
                "done": self.done,
                "success": self.success,
                "last_reward": self.last_reward,
                "simulator_only": True,
            },
        )

    async def observe(self) -> DriverObservation:
        try:
            observation = await self._require_observation(refresh=True)
        except grpc.aio.AioRpcError as exc:
            waiting_for_trial = "trial_unavailable" in exc.details()
            waiting_for_prepare = exc.code() is grpc.StatusCode.DEADLINE_EXCEEDED
            if not (waiting_for_trial or waiting_for_prepare):
                raise
            self.last_error = (
                "waiting for evaluator to begin or prepare a RoboCasa trial"
            )
            return DriverObservation(
                envelope=self._envelope(),
                frame_id=self.frame_id,
                assets=[],
                proprioception=[],
                task=None,
                metadata={"driver": "robocasa_remote", "trial_unavailable": True},
                timestamp=time.time(),
            )
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
            success=None,
            error=self.last_error,
            metrics={
                "driver": "robocasa_remote",
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
                action=[float(value) for value in action.values],
                expected_frame_id=self.frame_id,
                raw_action=[
                    float(value)
                    for value in action.metadata.get("raw_action", action.values)
                ],
                action_clipped=bool(action.metadata.get("action_clipped", False)),
            )
            self.last_reward = float(step.reward)
            self.done = bool(step.done)
            # Official simulator success belongs exclusively to evaluator
            # ReadTruth and is deliberately unavailable on the data plane.
            self.success = None
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
        if self.control_client is None:
            self.last_error = "RoboCasa reset control plane is unavailable"
            self.state = "error"
            return await self.status()
        settings = self.context.spec.settings
        try:
            if self.episode_id is not None:
                await self.control_client.end_trial(reason="robot_reset")
                # The next observation intentionally belongs to a new trial.
                # Clear the old causal identity only after EndTrial succeeds.
                self.episode_id = None
                self._last_observation = None
            observation = await self.control_client.begin_trial(
                trial_id=f"runtime-reset-{uuid.uuid4().hex}",
                task=str(settings.get("default_task") or "CloseFridge"),
                seed=int(settings.get("default_seed") or 1000),
                split=str(settings.get("task_split") or "target"),
                registries=tuple(settings.get("object_registries") or ("lightwheel",)),
            )
            self._accept_observation(observation)
            self.done = False
            self.success = None
            self.last_error = None
            self.state = "idle"
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.state = "error"
        return await self.status()

    async def close(self) -> None:
        await self.client.close()
        if self.control_client is not None and self.control_client is not self.client:
            await self.control_client.close()
        self.episode_id = None
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
        observation = await self.client.observe() if refresh else self._last_observation
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
        self.task = observation.task
        self.frame_id = int(observation.frame_id)
        self.done = bool(observation.done)
        self.success = None
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
                    data=_decode_rgb_image(image.data),
                    content_type=image.content_type,
                    metadata={"width": image.width, "height": image.height},
                )
                for image in observation.images
            ],
            proprioception=[float(value) for value in observation.state],
            task=observation.task,
            metadata={
                "driver": "robocasa_remote",
                "done": observation.done,
                **dict(observation.metadata),
            },
            timestamp=time.time(),
        )

    def _envelope(self) -> Envelope:
        return Envelope(robot_id=self.robot_id)


def _decode_rgb_image(payload: bytes) -> np.ndarray:
    """Convert backend JPEG/PNG transport bytes to the media-store image contract."""
    try:
        with Image.open(io.BytesIO(payload)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise ValueError("RoboCasa runtime returned an invalid encoded image") from exc
