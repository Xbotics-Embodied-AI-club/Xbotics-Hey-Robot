from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

from hey_robot.protocol.messages import RobotAction, RobotStatus, SkillIntent

FeedbackMode = str


@dataclass(frozen=True)
class RobotSkillAction:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    safety_level: str = "normal"
    expected_duration_sec: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("skill action name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": dict(self.arguments),
            "safety_level": self.safety_level,
            "expected_duration_sec": self.expected_duration_sec,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RobotSkillAction:
        return cls(
            name=str(payload.get("name") or payload.get("skill") or ""),
            arguments=dict(payload.get("arguments") or payload.get("args") or {}),
            safety_level=str(payload.get("safety_level") or "normal"),
            expected_duration_sec=(
                float(payload["expected_duration_sec"])
                if payload.get("expected_duration_sec") is not None
                else None
            ),
        )

    def to_robot_action(self, intent: SkillIntent) -> RobotAction:
        return RobotAction(
            envelope=intent.envelope,
            skill_id=intent.skill_id,
            values=[],
            timestamp=time.time(),
            metadata={
                "action_type": "skill",
                "skill": self.to_dict(),
            },
        )

    @classmethod
    def from_robot_action(cls, action: RobotAction) -> RobotSkillAction:
        if action.metadata.get("action_type") != "skill":
            raise ValueError("robot action is not a skill action")
        skill = action.metadata.get("skill")
        if not isinstance(skill, dict):
            raise ValueError("skill action metadata is missing metadata.skill")
        return cls.from_dict(skill)


@dataclass(frozen=True)
class RobotSkillResult:
    success: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            **dict(self.data),
        }

    @classmethod
    def from_response(
        cls, response: dict[str, Any], *, default_message: str = ""
    ) -> RobotSkillResult:
        return cls(
            success=bool(response.get("success", False)),
            message=str(response.get("message") or default_message),
            data=dict(response),
        )


@dataclass(frozen=True)
class RobotSkillSpec:
    """Runtime contract derived from the canonical plugin SkillSpec."""

    name: str
    description: str
    level: str = "primitive"
    agent_visible: bool = True
    category: str = "general"
    input_schema: dict[str, Any] = field(default_factory=dict)
    safety_level: str = "normal"
    supported_robots: tuple[str, ...] = ()
    external_capability: str | None = None
    driver_primitives: tuple[str, ...] = ()
    required_resources: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    recovery_hints: tuple[str, ...] = ()
    timeout_sec: float = 10.0
    interruptible: bool = True
    feedback_mode: FeedbackMode = "status"
    refresh_observation: bool = True
    capability_type: str | None = None
    goal_effects: tuple[str, ...] = ()
    evidence_outputs: tuple[str, ...] = ()
    cannot_satisfy: tuple[str, ...] = ()

    def supports(self, robot_type: str | None) -> bool:
        return (
            robot_type is None
            or not self.supported_robots
            or robot_type in self.supported_robots
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "supported_robots",
            "driver_primitives",
            "required_resources",
            "preconditions",
            "success_criteria",
            "failure_modes",
            "recovery_hints",
            "goal_effects",
            "evidence_outputs",
            "cannot_satisfy",
        ):
            data[key] = list(data[key])
        return data


class RobotSkillCatalog:
    """Read-only runtime view derived from a SkillRegistry."""

    def __init__(
        self, specs: list[RobotSkillSpec] | tuple[RobotSkillSpec, ...]
    ) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def get(self, name: str) -> RobotSkillSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unknown robot skill action: {name}") from exc

    def list(self, *, robot_type: str | None = None) -> tuple[RobotSkillSpec, ...]:
        return tuple(spec for spec in self._specs.values() if spec.supports(robot_type))

    def list_agent_visible(
        self, *, robot_type: str | None = None
    ) -> tuple[RobotSkillSpec, ...]:
        return tuple(
            spec for spec in self.list(robot_type=robot_type) if spec.agent_visible
        )

    def names(self, *, robot_type: str | None = None) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.list(robot_type=robot_type))

    def agent_visible_names(self, *, robot_type: str | None = None) -> tuple[str, ...]:
        return tuple(
            spec.name for spec in self.list_agent_visible(robot_type=robot_type)
        )

    def resolve(
        self, name: str | None, *, robot_type: str | None = None
    ) -> RobotSkillSpec:
        if not name:
            raise KeyError("robot skill action name is required")
        spec = self.get(name)
        if not spec.supports(robot_type):
            raise KeyError(
                f"robot skill action {name!r} does not support robot type {robot_type!r}"
            )
        return spec


@dataclass(frozen=True)
class SkillContractDecision:
    allowed: bool
    reason: str = "accepted"
    failure_mode: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls, *, metadata: dict[str, Any] | None = None) -> SkillContractDecision:
        return cls(True, metadata=metadata or {})

    @classmethod
    def reject(
        cls,
        reason: str,
        *,
        failure_mode: str,
        metadata: dict[str, Any] | None = None,
    ) -> SkillContractDecision:
        return cls(
            False, reason=reason, failure_mode=failure_mode, metadata=metadata or {}
        )


class SkillContractRuntime:
    """Deterministic contract gate for skill scheduling and robot execution."""

    SHARED_RESOURCES: ClassVar[set[str]] = {"camera"}

    def __init__(self, catalog: RobotSkillCatalog | None = None) -> None:
        self.catalog = catalog

    def resolve(
        self, name: str | None, *, robot_type: str | None = None
    ) -> RobotSkillSpec:
        if self.catalog is None:
            if not name:
                raise KeyError("robot skill action name is required")
            return RobotSkillSpec(
                name=name,
                description="Uncataloged robot skill action.",
                required_resources=("robot",),
            )
        return self.catalog.resolve(name, robot_type=robot_type)

    def validate_action(
        self,
        action: RobotSkillAction,
        *,
        robot_type: str | None = None,
        status: RobotStatus | None = None,
        readiness: dict[str, Any] | None = None,
    ) -> tuple[RobotSkillSpec, SkillContractDecision]:
        try:
            contract = self.resolve(action.name, robot_type=robot_type)
        except KeyError as exc:
            return (
                RobotSkillSpec(
                    name=action.name or "unknown_skill",
                    description="Unknown skill action.",
                    required_resources=("robot",),
                ),
                SkillContractDecision.reject(str(exc), failure_mode="unknown_skill"),
            )
        decision = self.acceptance_decision(
            contract, status=status, readiness=readiness, arguments=action.arguments
        )
        return contract, decision

    def acceptance_decision(
        self,
        contract: RobotSkillSpec,
        *,
        status: RobotStatus | None = None,
        readiness: dict[str, Any] | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> SkillContractDecision:
        resolved_arguments = arguments or {}
        missing = self.missing_required_arguments(contract, resolved_arguments)
        if missing:
            return SkillContractDecision.reject(
                f"skill {contract.name} missing required arguments: {','.join(missing)}",
                failure_mode="invalid_arguments",
                metadata={"missing_arguments": missing, "contract": contract.to_dict()},
            )
        readiness_block = self.readiness_block(
            contract, readiness, arguments=resolved_arguments
        )
        if readiness_block is not None:
            return readiness_block
        precondition_block = self.precondition_block(contract, status)
        if precondition_block is not None:
            return precondition_block
        return SkillContractDecision.allow(metadata={"contract": contract.to_dict()})

    @staticmethod
    def missing_required_arguments(
        contract: RobotSkillSpec, arguments: dict[str, Any]
    ) -> list[str]:
        required = contract.input_schema.get("required")
        if not isinstance(required, list):
            return []
        return [
            str(key)
            for key in required
            if key not in arguments or arguments.get(key) is None
        ]

    def readiness_block(
        self,
        contract: RobotSkillSpec,
        readiness: dict[str, Any] | None,
        *,
        arguments: dict[str, Any] | None = None,
    ) -> SkillContractDecision | None:
        if not readiness:
            return None
        if self._is_exempt_from_readiness(contract):
            return None
        issues: list[str] = []
        if bool(readiness.get("emergency_stop") or readiness.get("estop")):
            issues.append("emergency stop is active")
        resources = self.normalized_resources(contract, arguments=arguments)
        for resource in sorted(resources):
            if resource in {"robot", "robot.actuation"}:
                continue
            if not self._resource_ready(resource, readiness):
                issues.append(f"{resource} is not ready")
        battery = readiness.get("battery")
        if isinstance(battery, dict):
            battery_status = str(battery.get("status") or "").lower()
            if battery_status == "critical":
                issues.append("battery critical")
            elif battery_status == "low" and contract.safety_level == "motion":
                issues.append("battery low")
        if not issues:
            return None
        return SkillContractDecision.reject(
            f"readiness gate blocked {contract.name}: {'; '.join(issues)}",
            failure_mode="readiness_failed",
            metadata={
                "issues": issues,
                "readiness": readiness,
                "contract": contract.to_dict(),
            },
        )

    @staticmethod
    def precondition_block(
        contract: RobotSkillSpec, status: RobotStatus | None
    ) -> SkillContractDecision | None:
        if status is None:
            return None
        state = str(status.state or "").lower()
        if contract.safety_level in {"observe", "stop", "emergency"}:
            return None
        if state in {"failed", "degraded", "interrupted", "emergency", "estop"}:
            return SkillContractDecision.reject(
                f"robot state {state!r} blocks {contract.safety_level} skill {contract.name}",
                failure_mode="precondition_failed",
                metadata={"state": state, "contract": contract.to_dict()},
            )
        battery = status.metrics.get("battery")
        if isinstance(battery, dict):
            battery_status = str(battery.get("status") or "").lower()
            if battery_status == "critical":
                return SkillContractDecision.reject(
                    f"battery critical blocks skill {contract.name}",
                    failure_mode="precondition_failed",
                    metadata={"battery": battery, "contract": contract.to_dict()},
                )
            if battery_status == "low" and contract.safety_level == "motion":
                return SkillContractDecision.reject(
                    f"battery low blocks motion skill {contract.name}",
                    failure_mode="precondition_failed",
                    metadata={"battery": battery, "contract": contract.to_dict()},
                )
        return None

    def resources_conflict(
        self,
        left: RobotSkillSpec,
        right: RobotSkillSpec,
        *,
        left_arguments: dict[str, Any] | None = None,
        right_arguments: dict[str, Any] | None = None,
    ) -> bool:
        return bool(
            self.shared_or_global_resources(
                left,
                right,
                left_arguments=left_arguments,
                right_arguments=right_arguments,
            )
        )

    def shared_or_global_resources(
        self,
        left: RobotSkillSpec,
        right: RobotSkillSpec,
        *,
        left_arguments: dict[str, Any] | None = None,
        right_arguments: dict[str, Any] | None = None,
    ) -> set[str]:
        left_resources = self.normalized_resources(left, arguments=left_arguments)
        right_resources = self.normalized_resources(right, arguments=right_arguments)
        if self.has_global_resource(left_resources) or self.has_global_resource(
            right_resources
        ):
            return left_resources | right_resources
        left_exclusive = self._exclusive_resources(left_resources)
        right_exclusive = self._exclusive_resources(right_resources)
        return left_exclusive & right_exclusive

    def _exclusive_resources(self, resources: set[str]) -> set[str]:
        return {
            r
            for r in resources
            if r not in self.SHARED_RESOURCES and not r.endswith("_camera")
        }

    @staticmethod
    def normalized_resources(
        contract: RobotSkillSpec, *, arguments: dict[str, Any] | None = None
    ) -> set[str]:
        resources = {
            str(resource).strip().lower()
            for resource in contract.required_resources
            if str(resource).strip()
        }
        return SkillContractRuntime._instance_resources(resources, arguments=arguments)

    @staticmethod
    def has_global_resource(resources: Iterable[str]) -> bool:
        return bool(set(resources) & {"robot", "robot.actuation"})

    @staticmethod
    def _instance_resources(
        resources: set[str], *, arguments: dict[str, Any] | None = None
    ) -> set[str]:
        if not resources:
            return {"robot"}
        resolved = set(resources)
        payload = arguments or {}
        arm = str(payload.get("arm") or "").strip().lower()
        camera = str(payload.get("camera") or "").strip().lower()
        if arm:
            if "arm" in resolved:
                resolved.remove("arm")
                resolved.add(f"{arm}_arm")
            if "gripper" in resolved:
                resolved.remove("gripper")
                resolved.add(f"{arm}_gripper")
        if camera and "camera" in resolved:
            resolved.remove("camera")
            resolved.add(f"{camera}_camera")
        return resolved or {"robot"}

    @staticmethod
    def _is_exempt_from_readiness(contract: RobotSkillSpec) -> bool:
        return (
            contract.safety_level in {"stop", "emergency"}
            or contract.name == "stop_motion"
        )

    @staticmethod
    def _resource_ready(resource: str, readiness: dict[str, Any]) -> bool:
        item = readiness.get(resource)
        if isinstance(item, dict):
            if "ok" in item:
                return bool(item["ok"])
            if "available" in item:
                return bool(item["available"])
            if "ready" in item:
                return bool(item["ready"])
            return True
        if item is not None:
            return bool(item)
        return bool(
            readiness.get(f"{resource}_available", False)
            or readiness.get(f"{resource}_ready", False)
        )
