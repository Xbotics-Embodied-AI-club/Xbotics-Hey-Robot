from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from hey_robot.protocol import RobotStatus
from hey_robot.skills.actions import RobotSkillAction
from hey_robot.skills.catalog import RobotSkillCatalog, RobotSkillSpec


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
        if catalog is None:
            from hey_robot.skills.registry import load_skill_registry

            catalog = load_skill_registry().robot_skill_catalog()
        self.catalog = catalog

    def resolve(
        self, name: str | None, *, robot_type: str | None = None
    ) -> RobotSkillSpec:
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
            return True  # unknown keys — resource is present and reporting
        if item is not None:
            return bool(item)
        return bool(
            readiness.get(f"{resource}_available", False)
            or readiness.get(f"{resource}_ready", False)
        )
