from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

from hey_robot.agents import RobotAgentService, TaskSupervisorService
from hey_robot.config import DeploymentConfig
from hey_robot.config.validation import validate_deployment
from hey_robot.gateway import GatewayService
from hey_robot.human_follow import HumanFollowService
from hey_robot.logging import HeyRobotLogger
from hey_robot.robots import RobotService
from hey_robot.skills.controller import SkillControllerService


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
    """Run a complete local deployment in one asyncio process."""

    def __init__(
        self, config: DeploymentConfig, *, episode_dir: str | Path | None = None
    ) -> None:
        self.config = config
        self.episode_dir = episode_dir or config.resources.episodes_root
        HeyRobotLogger.from_spec(self.config.logging)
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
        if self.config.robots:
            robot = RobotService(self.config)
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
        if any(spec.enabled for spec in self.config.policies.values()):
            skills = SkillControllerService(self.config)
            services.append(
                ManagedService("skill-controller", skills.start, skills.stop)
            )
        if any(spec.enabled for spec in self.config.agents.values()):
            supervisor = TaskSupervisorService(
                self.config, episode_dir=self.episode_dir
            )
            services.append(
                ManagedService("task-supervisor", supervisor.start, supervisor.stop)
            )
        for agent_id, spec in self.config.agents.items():
            if not spec.enabled:
                continue
            agent = RobotAgentService(
                self.config, agent_id=agent_id, episode_dir=self.episode_dir
            )
            services.append(
                ManagedService(f"agent:{agent_id}", agent.start, agent.stop)
            )
        if self.config.channels:
            gateway = GatewayService(self.config, episode_dir=self.episode_dir)
            services.append(ManagedService("gateway", gateway.start, gateway.stop))
        return services
