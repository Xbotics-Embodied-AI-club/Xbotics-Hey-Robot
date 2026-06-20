from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from hey_robot.config import RobotSpec
from hey_robot.perception import DriverObservation
from hey_robot.protocol import RobotAction, RobotStatus
from hey_robot.robots.embodiments.base import EmbodimentProfile

if TYPE_CHECKING:
    from hey_robot.skills.catalog import RobotSkillCatalog


@dataclass(frozen=True)
class RobotDriverContext:
    robot_id: str
    spec: RobotSpec
    deployment_id: str
    embodiment: EmbodimentProfile | None = None
    skill_catalog: RobotSkillCatalog | None = None


@dataclass(frozen=True)
class RobotCapabilities:
    robot_id: str
    driver_type: str
    action_dimensions: int | None = None
    control_hz: float | None = None
    cameras: list[str] = field(default_factory=list)
    observation_modalities: list[str] = field(default_factory=list)
    supports_reset: bool = True
    supports_interrupt: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotHealth:
    robot_id: str
    online: bool
    state: str
    frame_id: int | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class RobotDriver(Protocol):
    robot_id: str

    async def start(self) -> None: ...

    async def capabilities(self) -> RobotCapabilities: ...

    async def health(self) -> RobotHealth: ...

    async def observe(self) -> DriverObservation: ...

    async def status(self) -> RobotStatus: ...

    async def apply_action(self, action: RobotAction) -> RobotStatus: ...

    async def reset(self) -> RobotStatus: ...

    async def close(self) -> None: ...
