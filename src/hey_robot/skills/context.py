from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from hey_robot.protocol import RobotObservation
from hey_robot.skills.apis import (
    CapabilityAPI,
    PerceptionAPI,
    RobotSkillAPI,
)


@dataclass
class SkillContext:
    skill_id: str | None = None
    robot_id: str | None = None
    robot: RobotSkillAPI | None = None
    perception: PerceptionAPI | None = None
    capabilities: CapabilityAPI | None = None
    observation: RobotObservation | None = None
    current_observation: Callable[[], RobotObservation | None] | None = None
    resolve_images: Callable[[list[Any]], list[Any]] | None = None
    logger: Any = None
    invoke: Callable[[str, dict[str, Any] | None], Any] | None = None
    progress: Callable[..., Awaitable[None]] | None = None
    human_follow: Any = None
