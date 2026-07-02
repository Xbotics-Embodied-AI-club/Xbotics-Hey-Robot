from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from hey_robot.protocol import RobotObservation
from hey_robot.skill_os.apis import (
    ModelServiceAPI,
    PerceptionAPI,
    RobotSkillAPI,
)


@dataclass
class SkillContext:
    skill_id: str | None = None
    robot_id: str | None = None
    robot: RobotSkillAPI | None = None
    perception: PerceptionAPI | None = None
    model_services: ModelServiceAPI | None = None
    observation: RobotObservation | None = None
    current_observation: Callable[[], RobotObservation | None] | None = None
    resolve_images: Callable[[list[Any]], list[Any]] | None = None
    logger: Any = None
    invoke: Callable[[str, dict[str, Any] | None], Any] | None = None
    progress: Callable[..., Awaitable[None]] | None = None
    human_follow: Any = None
    get_camera_frame: Callable[[], tuple[dict[str, Any], Any] | None] | None = None
