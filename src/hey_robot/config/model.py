from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import find_dotenv, load_dotenv


@dataclass(frozen=True)
class BusSpec:
    type: str = "nats"
    url: str = "nats://127.0.0.1:4222"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, init=False)
class DeploymentSpec:
    id: str = "local"
    bus: BusSpec = field(default_factory=BusSpec)

    def __init__(
        self,
        id: str = "local",
        bus: BusSpec | None = None,
        *,
        mode: str | None = None,
    ) -> None:
        del mode
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "bus", bus or BusSpec())


@dataclass(frozen=True)
class LoggingSpec:
    level: str = "INFO"
    file_path: str | None = None
    theme: str = "dark"
    json_format: bool = False
    json_file_path: str | None = None
    throttle_sec: float = 0.0


@dataclass(frozen=True)
class ResourceSpec:
    runtime_dir: str = "runtime"
    media_root: str = "runtime/media"
    media_max_items: int = 5000
    media_image_save_every_n: int = 1
    episodes_root: str = "runtime/episodes"
    events_max_items: int = 1000


@dataclass(frozen=True)
class ChannelSpec:
    type: str
    enabled: bool = True
    account_id: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IdentitySpec:
    enabled: bool = True
    unified_user_episodes: bool = True
    default_user_id: str | None = None
    bindings: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NotificationSpec:
    defaults: dict[str, Any] = field(default_factory=dict)
    channels: dict[str, dict[str, Any]] = field(default_factory=dict)
    kinds: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotSpec:
    type: str
    enabled: bool = True
    family: str | None = None
    environment: str | None = None
    driver: str | None = None
    embodiment_profile: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def robot_family(self) -> str:
        if self.family:
            return str(self.family)
        if self.type == "mock":
            body = (
                self.settings.get("body")
                or self.settings.get("embodiment_type")
                or "xlerobot"
            )
            return str(body)
        if self.type.endswith("_sim"):
            return self.type.removesuffix("_sim")
        return str(self.type)

    @property
    def robot_environment(self) -> str:
        if self.environment:
            return str(self.environment)
        if self.type == "mock":
            return "mock"
        if self.type.endswith("_sim"):
            return "sim"
        return "real"

    @property
    def driver_kind(self) -> str:
        if self.driver:
            return str(self.driver)
        if self.type == "mock":
            return "mock"
        if self.type.endswith("_sim"):
            return "mujoco"
        return "native"


@dataclass(frozen=True)
class PolicySpec:
    robot_id: str
    enabled: bool = True
    freq_hz: float = 20.0


@dataclass(frozen=True, init=False)
class CapabilityServiceSpec:
    type: str
    robot_id: str
    enabled: bool = True
    target: str | None = None
    skill_names: tuple[str, ...] = ()
    timeout_sec: float = 30.0
    settings: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        type: str,
        robot_id: str,
        enabled: bool = True,
        target: str | None = None,
        skill_names: tuple[str, ...] = (),
        timeout_sec: float = 30.0,
        settings: dict[str, Any] | None = None,
        *,
        resources: tuple[str, ...] | None = None,
    ) -> None:
        del resources
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "robot_id", robot_id)
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "skill_names", skill_names)
        object.__setattr__(self, "timeout_sec", timeout_sec)
        object.__setattr__(self, "settings", dict(settings or {}))


@dataclass(frozen=True, init=False)
class AgentSpec:
    enabled: bool = True
    robot_id: str | None = None
    policy_id: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        enabled: bool = True,
        robot_id: str | None = None,
        policy_id: str | None = None,
        settings: dict[str, Any] | None = None,
        *,
        type: str | None = None,
    ) -> None:
        del type
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "robot_id", robot_id)
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "settings", dict(settings or {}))


@dataclass(frozen=True)
class SkillSurfaceConfig:
    modules: tuple[str, ...] = ("hey_robot.skills.builtin",)
    enabled: tuple[str, ...] = ()
    mode: str = "production"  # "production" | "bringup"


@dataclass(frozen=True)
class DeploymentConfig:
    deployment: DeploymentSpec = field(default_factory=DeploymentSpec)
    logging: LoggingSpec = field(default_factory=LoggingSpec)
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    notifications: NotificationSpec = field(default_factory=NotificationSpec)
    identity: IdentitySpec = field(default_factory=IdentitySpec)
    channels: dict[str, ChannelSpec] = field(default_factory=dict)
    robots: dict[str, RobotSpec] = field(default_factory=dict)
    policies: dict[str, PolicySpec] = field(default_factory=dict)
    capability_services: dict[str, CapabilityServiceSpec] = field(default_factory=dict)
    agents: dict[str, AgentSpec] = field(default_factory=dict)
    skills: SkillSurfaceConfig = field(default_factory=SkillSurfaceConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DeploymentConfig:
        load_dotenv(find_dotenv(usecwd=True))
        with Path(path).open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeploymentConfig:
        deployment_data = data.get("deployment", {}) or {}
        bus_data = deployment_data.get("bus", {}) or {}
        deployment = DeploymentSpec(
            id=deployment_data.get("id", "local"),
            bus=BusSpec(
                type=bus_data.get("type", "nats"),
                url=bus_data.get("url", "nats://127.0.0.1:4222"),
                options=dict(bus_data.get("options", {}) or {}),
            ),
        )
        logging_data = data.get("logging", {}) or {}
        logging_spec = LoggingSpec(
            level=str(logging_data.get("level", "INFO")).upper(),
            file_path=logging_data.get("file_path"),
            theme=str(logging_data.get("theme", "dark")),
            json_format=bool(logging_data.get("json_format", False)),
            json_file_path=logging_data.get("json_file_path"),
            throttle_sec=float(logging_data.get("throttle_sec", 0.0)),
        )
        resources_data = data.get("resources", {}) or {}
        media_data = resources_data.get("media", {}) or {}
        episodes_data = resources_data.get("episodes", {}) or {}
        events_data = resources_data.get("events", {}) or {}
        resources = ResourceSpec(
            runtime_dir=str(resources_data.get("runtime_dir", "runtime")),
            media_root=str(media_data.get("root", "runtime/media")),
            media_max_items=int(media_data.get("max_items", 5000)),
            media_image_save_every_n=max(
                1, int(media_data.get("image_save_every_n", 1))
            ),
            episodes_root=str(episodes_data.get("root", "runtime/episodes")),
            events_max_items=int(events_data.get("retain", 1000)),
        )
        notifications_data = data.get("notifications", {}) or {}
        notifications = NotificationSpec(
            defaults=dict(notifications_data.get("defaults", {}) or {}),
            channels={
                str(name): dict(value or {})
                for name, value in dict(
                    notifications_data.get("channels", {}) or {}
                ).items()
            },
            kinds={
                str(name): dict(value or {})
                for name, value in dict(
                    notifications_data.get("kinds", {}) or {}
                ).items()
            },
        )
        identity_data = data.get("identity", {}) or {}
        identity = IdentitySpec(
            enabled=bool(identity_data.get("enabled", True)),
            unified_user_episodes=bool(
                identity_data.get("unified_user_episodes", True)
            ),
            default_user_id=(
                str(identity_data.get("default_user_id")).strip() or None
                if identity_data.get("default_user_id") is not None
                else None
            ),
            bindings={
                str(key): str(value)
                for key, value in dict(identity_data.get("bindings", {}) or {}).items()
                if str(key).strip() and str(value).strip()
            },
        )
        skills_data = data.get("skills", {}) or {}
        return cls(
            deployment=deployment,
            logging=logging_spec,
            resources=resources,
            notifications=notifications,
            identity=identity,
            channels={
                name: ChannelSpec(
                    type=(value or {}).get("type", name),
                    enabled=bool((value or {}).get("enabled", True)),
                    account_id=(value or {}).get("account_id"),
                    settings={
                        key: val
                        for key, val in (value or {}).items()
                        if key not in {"type", "enabled", "account_id"}
                    },
                )
                for name, value in (data.get("channels", {}) or {}).items()
            },
            robots={
                name: RobotSpec(
                    type=(value or {}).get("type", name),
                    enabled=bool((value or {}).get("enabled", True)),
                    family=(
                        str((value or {}).get("family")).strip() or None
                        if (value or {}).get("family") is not None
                        else None
                    ),
                    environment=(
                        str((value or {}).get("environment")).strip() or None
                        if (value or {}).get("environment") is not None
                        else None
                    ),
                    driver=(
                        str((value or {}).get("driver")).strip() or None
                        if (value or {}).get("driver") is not None
                        else None
                    ),
                    embodiment_profile=(
                        str((value or {}).get("embodiment_profile")).strip() or None
                        if (value or {}).get("embodiment_profile") is not None
                        else None
                    ),
                    settings={
                        **dict((value or {}).get("settings", {}) or {}),
                        **{
                            key: val
                            for key, val in (value or {}).items()
                            if key
                            not in {
                                "type",
                                "enabled",
                                "family",
                                "environment",
                                "driver",
                                "embodiment_profile",
                                "settings",
                            }
                        },
                    },
                )
                for name, value in (data.get("robots", {}) or {}).items()
            },
            policies={
                name: PolicySpec(
                    robot_id=(value or {}).get("robot_id", ""),
                    enabled=bool((value or {}).get("enabled", True)),
                    freq_hz=float((value or {}).get("freq_hz", 20.0)),
                )
                for name, value in (data.get("policies", {}) or {}).items()
            },
            capability_services={
                name: CapabilityServiceSpec(
                    type=(value or {}).get("type", name),
                    robot_id=(value or {}).get("robot_id", ""),
                    enabled=bool((value or {}).get("enabled", True)),
                    target=(value or {}).get("target"),
                    skill_names=tuple(
                        str(item) for item in (value or {}).get("skill_names", ()) or ()
                    ),
                    timeout_sec=float((value or {}).get("timeout_sec", 30.0)),
                    settings={
                        **dict((value or {}).get("settings", {}) or {}),
                        **{
                            key: val
                            for key, val in (value or {}).items()
                            if key
                            not in {
                                "type",
                                "robot_id",
                                "enabled",
                                "target",
                                "skill_names",
                                "resources",
                                "timeout_sec",
                                "settings",
                            }
                        },
                    },
                )
                for name, value in (data.get("capability_services", {}) or {}).items()
            },
            agents={
                name: AgentSpec(
                    enabled=bool((value or {}).get("enabled", True)),
                    robot_id=(value or {}).get("robot_id"),
                    policy_id=(value or {}).get("policy_id"),
                    settings={
                        **dict((value or {}).get("settings", {}) or {}),
                        **{
                            key: val
                            for key, val in (value or {}).items()
                            if key
                            not in {
                                "type",
                                "enabled",
                                "robot_id",
                                "policy_id",
                                "settings",
                            }
                        },
                    },
                )
                for name, value in (data.get("agents", {}) or {}).items()
            },
            skills=SkillSurfaceConfig(
                modules=tuple(
                    str(item).strip()
                    for item in skills_data.get(
                        "modules", ("hey_robot.skills.builtin",)
                    )
                    or ("hey_robot.skills.builtin",)
                    if str(item).strip()
                )
                or ("hey_robot.skills.builtin",),
                enabled=tuple(
                    str(item).strip()
                    for item in skills_data.get("enabled", ()) or ()
                    if str(item).strip()
                ),
                mode=str(skills_data.get("mode", "production")),
            ),
        )

    def default_agent_id(self) -> str:
        if "main" in self.agents:
            return "main"
        if self.agents:
            return next(iter(self.agents))
        return "main"

    def default_robot_id(self, agent_id: str | None = None) -> str | None:
        if agent_id is not None:
            agent = self.agents.get(agent_id)
            if agent is not None and agent.robot_id:
                return agent.robot_id
        default_agent = self.agents.get(self.default_agent_id())
        if default_agent is not None and default_agent.robot_id:
            return default_agent.robot_id
        if self.robots:
            return next(iter(self.robots))
        return None
