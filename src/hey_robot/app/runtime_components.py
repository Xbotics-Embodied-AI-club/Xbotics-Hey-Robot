"""Local native skill runtime composition.

This module is intentionally small: it wires stable ports together and leaves
transport, cognition, robot safety, and model inference in their own modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hey_robot.config import DeploymentConfig
from hey_robot.foundation.clients import ModelServiceRegistry, RegistryModelRouter
from hey_robot.persistence import FileRunStore
from hey_robot.robot_runtime.clients import LocalRobotClient, RobotClient
from hey_robot.robot_runtime.service import RobotService
from hey_robot.skills import SkillRegistry, registry_from_config
from hey_robot.skills.client import SkillClient
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import Skill, SkillCommand
from hey_robot.skills.resources import ResourceManager
from hey_robot.skills.transport import LocalSkillClient


@dataclass(frozen=True)
class SkillToolCatalog:
    """Selected native skills exposed as Agent tools."""

    skills: tuple[Skill, ...]

    def list(self) -> tuple[Skill, ...]:
        return self.skills

    def get(self, name: str) -> Skill:
        for skill in self.skills:
            if skill.name == name:
                return skill
        raise KeyError(f"unknown skill tool: {name}")


@dataclass(frozen=True)
class RuntimeComponents:
    registry: SkillRegistry
    tool_catalog: SkillToolCatalog
    robot_client: RobotClient
    model_router: RegistryModelRouter
    run_store: FileRunStore
    skill_client: SkillClient


def build_local_runtime_components(
    config: DeploymentConfig,
    *,
    robot_service: RobotService,
) -> RuntimeComponents:
    registry = registry_from_config(config)
    tool_catalog = SkillToolCatalog(registry.select(config.skills.tool_names))
    robot_client = LocalRobotClient(robot_service.runtimes)
    model_router = RegistryModelRouter(ModelServiceRegistry(config))
    run_store = FileRunStore(
        Path(config.resources.runtime_dir) / config.deployment.id / "runs"
    )

    def context_factory(command: SkillCommand) -> SkillContext:
        return SkillContext(
            run_id=command.run_id,
            task_id=command.task_id,
            robot_id=command.robot_id,
            robot=robot_client,
            models=model_router,
        )

    skill_client = LocalSkillClient(
        registry,
        resources=ResourceManager(),
        context_factory=context_factory,
        run_store=run_store,
    )
    return RuntimeComponents(
        registry=registry,
        tool_catalog=tool_catalog,
        robot_client=robot_client,
        model_router=model_router,
        run_store=run_store,
        skill_client=skill_client,
    )
