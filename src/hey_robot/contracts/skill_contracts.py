from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

from hey_robot.protocol.messages import RobotStatus
from hey_robot.protocol.skills import RobotSkillAction

FeedbackMode = str


@dataclass(frozen=True)
class SkillContract:
    """Canonical runtime contract for a robot skill."""

    name: str
    description: str
    level: str = "primitive"
    agent_visible: bool = True
    category: str = "general"
    input_schema: dict[str, Any] = field(default_factory=dict)
    safety_level: str = "normal"
    supported_robots: tuple[str, ...] = ()
    required_model_service: str | None = None
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
            "dependencies",
        ):
            if key in data:
                data[key] = list(data[key])
        return data


class SkillContractCatalog:
    """Read-only runtime view of skill contracts."""

    def __init__(self, specs: list[SkillContract] | tuple[SkillContract, ...]) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def get(self, name: str) -> SkillContract:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unknown robot skill action: {name}") from exc

    def list(self, *, robot_type: str | None = None) -> tuple[SkillContract, ...]:
        return tuple(spec for spec in self._specs.values() if spec.supports(robot_type))

    def list_agent_visible(
        self, *, robot_type: str | None = None
    ) -> tuple[SkillContract, ...]:
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
    ) -> SkillContract:
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

    def __init__(self, catalog: SkillContractCatalog | None = None) -> None:
        self.catalog = catalog

    def resolve(
        self, name: str | None, *, robot_type: str | None = None
    ) -> SkillContract:
        if self.catalog is None:
            if not name:
                raise KeyError("robot skill action name is required")
            return SkillContract(
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
    ) -> tuple[SkillContract, SkillContractDecision]:
        try:
            contract = self.resolve(action.name, robot_type=robot_type)
        except KeyError as exc:
            return (
                SkillContract(
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
        contract: SkillContract,
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
        contract: SkillContract, arguments: dict[str, Any]
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
        contract: SkillContract,
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
        contract: SkillContract, status: RobotStatus | None
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
        left: SkillContract,
        right: SkillContract,
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
        left: SkillContract,
        right: SkillContract,
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
        contract: SkillContract, *, arguments: dict[str, Any] | None = None
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
    def _is_exempt_from_readiness(contract: SkillContract) -> bool:
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
