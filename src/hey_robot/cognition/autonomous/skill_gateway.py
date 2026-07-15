"""在发布 ``SkillIntent`` 前由 Supervisor 执行的调度预检。

该预检会基于当前 ``RobotExecutionGate`` 和 ``RobotStatus`` 校验技能规格、
安全等级、类别、前置条件、目标和参数。它不发布消息，也不写入动作账本。
Skill Controller 会在实际执行前再次进行最终准入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from hey_robot.cognition.autonomous.policy import PolicyDecision
from hey_robot.protocol import RobotExecutionGate, RobotStatus


class SkillCatalogView(Protocol):
    def get(self, name: str) -> Any: ...


@dataclass(frozen=True)
class DispatchPreflight:
    catalog: SkillCatalogView

    OBSERVE_CATEGORIES = frozenset({"observe", "perception"})

    def check(
        self,
        *,
        skill_name: str,
        objective: str,
        arguments: dict[str, Any],
        intent_kind: str,
        gate: RobotExecutionGate,
        status: RobotStatus | None,
    ) -> PolicyDecision:
        if not objective or not objective.strip():
            return PolicyDecision(False, "SAFETY_REJECTED")

        try:
            spec = self.catalog.get(skill_name)
        except KeyError:
            return PolicyDecision(False, "SKILL_REJECTED")

        category = getattr(spec, "category", "general")
        if intent_kind == "skill" and (
            skill_name == "inspect_scene" or category in self.OBSERVE_CATEGORIES
        ):
            return PolicyDecision(False, "SAFETY_REJECTED")

        input_schema = getattr(spec, "input_schema", {}) or {}
        for field_name in input_schema.get("required", []):
            if field_name not in arguments:
                return PolicyDecision(False, "INVALID_TOOL_ARGUMENTS")

        if gate.state != "ready":
            return PolicyDecision(False, "ROBOT_EXECUTION_UNCERTAIN")

        if status is None:
            return PolicyDecision(False, "ROBOT_OFFLINE")

        if status.state in {"offline", "unknown", "error"}:
            return PolicyDecision(False, "ROBOT_OFFLINE")

        safety_level = getattr(spec, "safety_level", "normal")
        if (
            safety_level == "critical"
            and status.battery_percentage is not None
            and status.battery_percentage < 30.0
        ):
            return PolicyDecision(False, "SAFETY_REJECTED")

        return PolicyDecision(True)

    def validate(self, **kwargs: Any) -> PolicyDecision:
        """为迁移到 ``check`` 的调用方保留的兼容别名。"""
        return self.check(**kwargs)


SkillGateway = DispatchPreflight
