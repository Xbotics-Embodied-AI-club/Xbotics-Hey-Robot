from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from hey_robot.app.runtime_components import (
    RuntimeComponents,
    build_local_runtime_components,
)
from hey_robot.app.sidecars import managed_robocasa_backend
from hey_robot.cognition.autonomous_agent_service import AutonomousAgentService
from hey_robot.cognition.perception.scene import build_scene_captioner
from hey_robot.config import DeploymentConfig
from hey_robot.config.validation import validate_deployment
from hey_robot.gateway import GatewayService
from hey_robot.human_follow import HumanFollowService
from hey_robot.logging import HeyRobotLogger
from hey_robot.robot_media import MediaResolver
from hey_robot.robot_transport import RobotService
from hey_robot.skills import robot_action_specs_from_config

logger = logging.getLogger(__name__)


@dataclass
class ManagedService:
    name: str
    start: Callable[[], Coroutine[Any, Any, None]]
    stop: Callable[[], Coroutine[Any, Any, None]]


class ResourceInspection(TypedDict):
    runtime_dir: str
    media_root: str
    episodes_root: str
    events_max_items: int


class DeploymentInspection(TypedDict):
    deployment: str
    robots: list[str]
    agents: list[str]
    channels: list[str]
    resources: ResourceInspection
    issues: list[dict[str, object]]
    services: list[str]


class DeploymentRunner:
    """在一个 asyncio 进程中运行完整的本地部署。"""

    def __init__(
        self,
        config: DeploymentConfig,
        *,
        episode_dir: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.episode_dir = episode_dir or config.resources.episodes_root
        HeyRobotLogger.from_spec(self.config.logging)
        self.sidecar = managed_robocasa_backend(config, config_path=config_path)
        # Sidecar identity must exist before RobotService and ModelService
        # clients capture their role-specific credentials.
        self.services = self._build_services()
        self._tasks: list[asyncio.Task] = []

    def inspect(self) -> DeploymentInspection:
        return {
            "deployment": self.config.deployment.id,
            "robots": sorted(self.config.robots),
            "agents": sorted(self.config.agents),
            "channels": sorted(self.config.channels),
            "resources": {
                "runtime_dir": self.config.resources.runtime_dir,
                "media_root": self.config.resources.media_root,
                "episodes_root": self.config.resources.episodes_root,
                "events_max_items": self.config.resources.events_max_items,
            },
            "issues": [issue.__dict__ for issue in validate_deployment(self.config)],
            "services": [service.name for service in self.services],
        }

    async def run(self) -> None:
        errors = [
            issue.message
            for issue in validate_deployment(self.config)
            if issue.level == "error"
        ]
        if errors:
            raise ValueError("invalid deployment: " + "; ".join(errors))
        if self.sidecar is not None:
            await self.sidecar.start()
            self._tasks.append(
                asyncio.create_task(self.sidecar.wait(), name="robocasa-backend")
            )
        for service in self.services:
            self._tasks.append(asyncio.create_task(service.start(), name=service.name))
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self, *, timeout_s: float = 5.0) -> None:
        """取消所有服务任务并等待退出。

        Windows IOCP 事件循环在清理阶段可能阻塞，设置超时上限防止
        Ctrl+C 关闭时长时间卡住。
        """
        for task in self._tasks:
            if not task.done():
                task.cancel()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=timeout_s,
            )
        if self.sidecar is not None:
            await self.sidecar.stop()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(
                    *(service.stop() for service in reversed(self.services)),
                    return_exceptions=True,
                ),
                timeout=timeout_s,
            )

    def _build_services(self) -> list[ManagedService]:
        services: list[ManagedService] = []
        robot = None
        runtime_components: RuntimeComponents | None = None
        action_specs = robot_action_specs_from_config(self.config)
        if self.config.robots:
            robot = RobotService(
                self.config,
                action_specs=action_specs,
                scene_captioner_factory=lambda store: build_scene_captioner(
                    self.config,
                    self.config.default_agent_id(),
                    image_resolver=MediaResolver(store),
                ),
            )
            services.append(ManagedService("robot", robot.start, robot.stop))
            if bool(
                self.config.deployment.bus.options.get(
                    "human_follow_service_enabled", False
                )
            ):
                human_follow = HumanFollowService(self.config)
                services.append(
                    ManagedService(
                        "human-follow", human_follow.start, human_follow.stop
                    )
                )
        if robot is not None:
            runtime_components = build_local_runtime_components(
                self.config,
                robot_service=robot,
            )
            services.append(
                ManagedService(
                    "skills",
                    runtime_components.skill_client.start,
                    runtime_components.skill_client.close,
                )
            )
        for agent_id, spec in self.config.agents.items():
            if not spec.enabled:
                continue
            agent = AutonomousAgentService(
                self.config,
                agent_id=agent_id,
                skill_client=(
                    runtime_components.skill_client
                    if runtime_components is not None
                    else None
                ),
                agent_skills=(
                    runtime_components.agent_skills
                    if runtime_components is not None
                    else None
                ),
            )
            services.append(
                ManagedService(f"agent:{agent_id}", agent.start, agent.stop)
            )
        if self.config.channels:
            gateway = GatewayService(self.config, episode_dir=self.episode_dir)
            services.append(ManagedService("gateway", gateway.start, gateway.stop))
        return services
