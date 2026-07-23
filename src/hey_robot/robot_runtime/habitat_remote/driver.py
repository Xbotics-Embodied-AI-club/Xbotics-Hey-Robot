from __future__ import annotations

import math
import time
from io import BytesIO
from typing import Any, Literal

import numpy as np
from PIL import Image

from hey_robot.protocol import Envelope, RobotAction, RobotSkillAction, RobotStatus
from hey_robot.robot_runtime.base import (
    RobotCapabilities,
    RobotDriverContext,
    RobotHealth,
)
from hey_robot.robot_runtime.habitat_remote.protocol import (
    HabitatObservation,
    HabitatRuntimeClient,
)
from hey_robot.robot_runtime.observations import DriverObservation, ObservationAsset


class HabitatRemoteDriver:
    """Adapter from Hey Robot semantic actions to one remote Habitat episode."""

    def __init__(
        self, context: RobotDriverContext, client: HabitatRuntimeClient
    ) -> None:
        self.context = context
        self.client = client
        self.robot_id = context.robot_id
        settings = context.spec.settings
        self.profile = str(settings.get("profile", "habitat3_social_spot_human_oracle"))
        self.task = str(settings.get("task", "RearrangePddlSocialNavTask-v0"))
        self.split = str(settings.get("split", "val"))
        self.seed = int(settings.get("seed", 100))
        self.controlled_agent = str(settings.get("controlled_agent", "agent_0"))
        self.requested_dataset_episode_id = _optional_string(
            settings.get("requested_dataset_episode_id")
        )
        self.default_max_steps = max(1, int(settings.get("max_skill_steps", 250)))
        self.state: Literal[
            "created", "idle", "executing", "completed", "error", "closed"
        ] = "created"
        self.episode_id: str | None = None
        self.frame_id = 0
        self.done = False
        self.success: bool | None = None
        self.last_error: str | None = None
        self.last_metrics: dict[str, Any] = {}
        self._last_observation: HabitatObservation | None = None
        self._active_operation_id: str | None = None

    async def start(self) -> None:
        health = await self.client.health()
        if not health.get("online") or not health.get("loaded"):
            self.state = "error"
            self.last_error = str(health.get("error") or "Habitat runtime unavailable")
            return
        observation = await self.client.create_episode(
            profile=self.profile,
            task=self.task,
            split=self.split,
            requested_dataset_episode_id=self.requested_dataset_episode_id,
            seed=self.seed,
            controlled_agent=self.controlled_agent,
        )
        self._accept_observation(observation)
        self.state = "completed" if observation.done else "idle"

    async def capabilities(self) -> RobotCapabilities:
        cameras = []
        if self._last_observation is not None:
            cameras = [
                asset.name or asset.role or "camera"
                for asset in self._last_observation.assets
                if asset.kind == "image"
            ]
        return RobotCapabilities(
            robot_id=self.robot_id,
            driver_type="habitat_remote",
            cameras=cameras,
            observation_modalities=["image", "depth", "state", "entities", "status"],
            supports_reset=True,
            supports_interrupt=True,
            metadata={
                "body": "habitat3_spot",
                "robot_family": self.context.spec.robot_family,
                "environment": self.context.spec.robot_environment,
                "profile": self.profile,
                "controlled_agent": self.controlled_agent,
                "runtime": "remote_simulator",
                "simulator_only": True,
                "control": "semantic_skill_or_debug_action",
            },
        )

    async def health(self) -> RobotHealth:
        online = self.state not in {"created", "closed", "error"}
        ready = online and self.episode_id is not None
        return RobotHealth(
            robot_id=self.robot_id,
            online=online,
            state=self.state,
            frame_id=self.frame_id,
            error=self.last_error,
            metrics={
                "episode_id": self.episode_id,
                "profile": self.profile,
                "task": self.task,
                "done": self.done,
                "success": self.success,
                "metrics": dict(self.last_metrics),
                "readiness": {
                    "remote_runtime": {"ok": ready},
                    "base": {"ok": ready},
                    "camera": {"ok": ready},
                    "arm": {"ok": ready},
                    "gripper": {"ok": ready},
                },
                "simulator_only": True,
            },
        )

    async def observe(self) -> DriverObservation:
        if self.episode_id is None:
            raise RuntimeError("Habitat episode has not been created")
        observation = await self.client.observe(episode_id=self.episode_id)
        self._accept_observation(observation)
        return self._driver_observation(observation)

    async def status(self) -> RobotStatus:
        state: Literal["idle", "executing", "error", "offline", "unknown"]
        if self.state == "executing":
            state = "executing"
        elif self.state == "error":
            state = "error"
        else:
            state = "idle"
        return RobotStatus(
            envelope=self._envelope(),
            frame_id=self.frame_id,
            state=state,
            task=self.task,
            success=self.success,
            error=self.last_error,
            metrics={
                "driver": "habitat_remote",
                "episode_id": self.episode_id,
                "profile": self.profile,
                "done": self.done,
                "habitat": dict(self.last_metrics),
                "simulator_only": True,
            },
        )

    async def apply_action(self, action: RobotAction) -> RobotStatus:
        try:
            self._validate_frame(action)
            if self.done:
                raise ValueError("episode is already done; reset before another action")
            if self.episode_id is None:
                raise ValueError("Habitat episode has not been created")
            self.state = "executing"
            skill = _skill_action(action)
            if skill is not None:
                if (
                    skill.name == "habitat_stop"
                    and self._active_operation_id is not None
                    and self.episode_id is not None
                ):
                    await self.client.cancel_skill(
                        episode_id=self.episode_id,
                        operation_id=self._active_operation_id,
                    )
                result = await self._execute_skill(action, skill)
                last_result = {
                    "success": result.success,
                    "summary": _summary(skill.name, result),
                    "skill": skill.to_dict(),
                    "failure_mode": result.failure_mode,
                    "steps": result.steps,
                    "trace": result.trace,
                    "privileged": bool(result.metrics.get("privileged", False)),
                }
                self.last_metrics = {
                    **dict(result.metrics),
                    "last_skill_result": last_result,
                }
                self._accept_observation(result.observation)
                self.state = "completed" if self.done else "idle"
                self.last_error = None if result.success else result.failure_mode
                status = await self.status()
                return RobotStatus(
                    envelope=status.envelope,
                    frame_id=status.frame_id,
                    state=status.state,
                    task=status.task,
                    skill_id=action.skill_id,
                    success=result.success,
                    error=None if result.success else result.failure_mode,
                    metrics=status.metrics,
                )

            raw_action = action.metadata.get("habitat_action")
            if not isinstance(raw_action, dict):
                raise ValueError(
                    "Habitat driver accepts RobotSkillAction or metadata.habitat_action"
                )
            step = await self.client.step(
                episode_id=self.episode_id,
                action=raw_action,
                expected_frame_id=self.frame_id,
            )
            self.last_metrics = dict(step.metrics)
            self._accept_observation(step.observation)
            self.state = "completed" if self.done else "idle"
            self.last_error = None
            status = await self.status()
            return RobotStatus(
                envelope=status.envelope,
                frame_id=status.frame_id,
                state=status.state,
                task=status.task,
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
                task=status.task,
                skill_id=action.skill_id,
                success=False,
                error=self.last_error,
                metrics=status.metrics,
            )

    async def reset(self) -> RobotStatus:
        if self.episode_id is None:
            self.state = "error"
            self.last_error = "Habitat episode has not been created"
            return await self.status()
        try:
            observation = await self.client.reset(
                episode_id=self.episode_id, expected_frame_id=self.frame_id
            )
            self._accept_observation(observation)
            self.done = False
            self.success = None
            self.last_error = None
            self.state = "idle"
        except Exception as exc:
            self.state = "error"
            self.last_error = f"{type(exc).__name__}: {exc}"
        return await self.status()

    async def close(self) -> None:
        try:
            if self.episode_id is not None:
                await self.client.close_episode(
                    episode_id=self.episode_id, expected_frame_id=self.frame_id
                )
                self.episode_id = None
        finally:
            await self.client.close()
            self.state = "closed"

    async def _execute_skill(self, action: RobotAction, skill: RobotSkillAction):
        assert self.episode_id is not None
        operation_id = action.action_id
        is_stop = skill.name == "habitat_stop"
        if not is_stop:
            self._active_operation_id = operation_id
        try:
            return await self.client.execute_skill(
                episode_id=self.episode_id,
                operation_id=operation_id,
                skill_name=skill.name,
                arguments=dict(skill.arguments),
                expected_frame_id=self.frame_id,
                max_steps=int(
                    skill.arguments.get("max_steps") or self.default_max_steps
                ),
            )
        finally:
            if not is_stop and self._active_operation_id == operation_id:
                self._active_operation_id = None

    def _validate_frame(self, action: RobotAction) -> None:
        expected = action.metadata.get("expected_frame_id")
        if expected is not None and int(expected) != self.frame_id:
            raise ValueError(
                f"stale action: expected_frame_id={expected}, current_frame_id={self.frame_id}"
            )
        if any(not math.isfinite(float(value)) for value in action.values):
            raise ValueError("action contains a non-finite value")

    def _accept_observation(self, observation: HabitatObservation) -> None:
        if self.episode_id is not None and observation.episode_id != self.episode_id:
            raise ValueError(
                "Habitat runtime returned an observation for another episode"
            )
        if self._last_observation is not None and observation.frame_id < self.frame_id:
            raise ValueError("Habitat runtime returned a stale observation frame")
        self.episode_id = observation.episode_id
        self.frame_id = int(observation.frame_id)
        self.done = bool(observation.done)
        self.success = bool(observation.success) if self.done else None
        self.last_metrics = {**self.last_metrics, **dict(observation.metrics)}
        self._last_observation = observation

    def _driver_observation(self, observation: HabitatObservation) -> DriverObservation:
        assets = [
            ObservationAsset(
                kind=asset.kind,
                role=asset.role,
                name=asset.name,
                data=_asset_data(asset),
                content_type=asset.content_type,
                metadata={
                    "width": asset.width,
                    "height": asset.height,
                    **dict(asset.metadata),
                },
            )
            for asset in observation.assets
        ]
        return DriverObservation(
            envelope=self._envelope(),
            frame_id=observation.frame_id,
            assets=assets,
            proprioception=[float(value) for value in observation.proprioception],
            task=observation.task or self.task,
            metadata={
                "driver": "habitat_remote",
                "episode_id": observation.episode_id,
                "done": observation.done,
                "success": observation.success,
                "entities": [dict(entity) for entity in observation.entities],
                "habitat": {
                    "metrics": dict(observation.metrics),
                    "metadata": dict(observation.metadata),
                    "profile": self.profile,
                    "controlled_agent": self.controlled_agent,
                },
            },
            timestamp=time.time(),
        )

    def _envelope(self) -> Envelope:
        return Envelope(robot_id=self.robot_id)


def _skill_action(action: RobotAction) -> RobotSkillAction | None:
    try:
        return RobotSkillAction.from_robot_action(action)
    except ValueError:
        return None


def _summary(name: str, result: Any) -> str:
    if result.cancelled:
        return f"{name} cancelled"
    if result.success:
        return f"{name} completed in {result.steps} Habitat steps"
    return f"{name} failed: {result.failure_mode or 'execution_failed'}"


def _optional_string(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _asset_data(asset: Any) -> Any:
    if asset.kind != "image":
        return asset.data
    try:
        with Image.open(BytesIO(asset.data)) as image:
            return np.asarray(image.convert("RGB"))
    except Exception as exc:
        raise ValueError(
            f"invalid Habitat image asset {asset.name or asset.role}: {exc}"
        ) from exc
