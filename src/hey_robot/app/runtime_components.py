"""Local native skill runtime composition.

This module is intentionally small: it wires stable ports together and leaves
transport, cognition, robot safety, and model inference in their own modules.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from hey_robot.bus.factory import create_bus_client
from hey_robot.bus.types import MessageBus
from hey_robot.config import DeploymentConfig
from hey_robot.foundation.clients import ModelServiceRegistry, RegistryModelRouter
from hey_robot.persistence import FileRunStore
from hey_robot.protocol import (
    SkillEvent as ProjectedSkillEvent,
    Topics,
)
from hey_robot.protocol.messages import to_payload
from hey_robot.robot_api import RobotClient
from hey_robot.robot_media import LocalMediaStore
from hey_robot.robot_runtime.clients import LocalRobotClient
from hey_robot.robot_transport import RobotService
from hey_robot.skills import SkillRegistry, registry_from_config
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import Skill, SkillCommand, SkillEvent
from hey_robot.skills.resources import ResourceManager
from hey_robot.skills.worker import SkillWorker


@dataclass(frozen=True)
class RuntimeComponents:
    registry: SkillRegistry
    agent_skills: tuple[Skill, ...]
    robot_client: RobotClient
    model_router: RegistryModelRouter
    run_store: FileRunStore
    skill_client: SkillWorker


class SkillEventProjector:
    """Read-only bus projection; RunStore remains the execution truth."""

    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus
        self._topic = Topics().skill_event

    async def start(self) -> None:
        await self._bus.connect()

    async def close(self) -> None:
        await self._bus.close()

    async def project(self, event: SkillEvent) -> None:
        result = event.result
        phase = {
            "running": "executing",
            "cancelled": "interrupted",
        }.get(event.phase, event.phase)
        await self._bus.publish(
            self._topic,
            to_payload(
                ProjectedSkillEvent(
                    envelope=event.envelope,
                    skill_id=event.run_id,
                    name=event.name,
                    phase=phase,
                    step=str(event.sequence),
                    text=event.summary,
                    frame_id=event.frame_id,
                    progress=event.progress,
                    error=result.error if result is not None else None,
                    summary=result.summary if result is not None else event.summary,
                    metadata={
                        "sequence": event.sequence,
                        "failure_mode": (
                            result.failure_mode if result is not None else None
                        ),
                        "evidence_ids": (
                            list(result.evidence_ids) if result is not None else []
                        ),
                    },
                )
            ),
        )


class ProjectionHealthStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, stats: dict[str, object]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self._path)


def build_local_runtime_components(
    config: DeploymentConfig,
    *,
    robot_service: RobotService,
) -> RuntimeComponents:
    registry = registry_from_config(config)
    agent_skills = registry.select(config.skills.tool_names)
    robot_client = LocalRobotClient(robot_service.runtimes)
    model_router = RegistryModelRouter(ModelServiceRegistry(config))
    event_projector = SkillEventProjector(
        create_bus_client(config.deployment.bus, role="skill_controller")
    )
    projection_health = ProjectionHealthStore(
        Path(config.resources.runtime_dir) / "skill_projection_health.json"
    )
    run_store = FileRunStore(
        Path(config.resources.runtime_dir) / "runs",
        artifact_store=LocalMediaStore(
            config.resources.media_root,
            max_items=config.resources.media_max_items,
        ),
    )

    def context_factory(command: SkillCommand) -> SkillContext:
        return SkillContext(
            run_id=command.run_id,
            task_id=command.task_id,
            robot_id=command.robot_id,
            robot=robot_client,
            models=model_router,
        )

    skill_client = SkillWorker(
        registry,
        resources=ResourceManager(),
        context_factory=context_factory,
        run_store=run_store,
        cancel_model=model_router.cancel,
        emergency_stop=lambda robot_id, reason: robot_client.emergency_stop(
            robot_id, reason=reason
        ),
        project_event=event_projector.project,
        start_projection=event_projector.start,
        stop_projection=event_projector.close,
        projection_health=projection_health.write,
    )
    return RuntimeComponents(
        registry=registry,
        agent_skills=agent_skills,
        robot_client=robot_client,
        model_router=model_router,
        run_store=run_store,
        skill_client=skill_client,
    )
