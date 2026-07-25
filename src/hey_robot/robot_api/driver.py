from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from hey_robot.protocol import RobotAction, RobotStatus
from hey_robot.robot_api.embodiment import EmbodimentProfile
from hey_robot.robot_api.observation import DriverObservation


@dataclass(frozen=True)
class RobotActionSpec:
    name: str
    parameters: dict[str, Any]
    resources: tuple[str, ...] = ()
    motion: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": dict(self.parameters),
            "resources": list(self.resources),
            "motion": self.motion,
        }


@dataclass(frozen=True)
class RobotDriverContext:
    robot_id: str
    deployment_id: str
    robot_family: str
    environment: str
    driver_kind: str
    settings: dict[str, Any] = field(default_factory=dict)
    embodiment: EmbodimentProfile | None = None
    action_specs: tuple[RobotActionSpec, ...] = ()


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


@runtime_checkable
class BaseVelocityStreamDriver(Protocol):
    async def apply_stream_velocity(
        self, *, vx: float, vy: float, wz: float, watchdog_ms: int
    ) -> Any: ...

    async def stop_base_stream(self) -> Any: ...


@runtime_checkable
class PerceptionRequestDriver(Protocol):
    def before_perception_request(
        self, skill_name: str, arguments: dict[str, Any]
    ) -> None: ...
