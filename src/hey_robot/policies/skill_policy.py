from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from hey_robot.config import PolicySpec
from hey_robot.policies.runtime import PolicyHealth, PolicyIOSchema
from hey_robot.protocol import RobotObservation, SkillIntent
from hey_robot.skills import (
    RobotSkillAction,
    RobotSkillCatalog,
    SkillPlanner,
)


@dataclass(frozen=True)
class ConservativeSkillPlanner:
    default_step_cm: float = 20.0
    default_turn_deg: float = 30.0
    max_step_cm: float = 80.0
    max_turn_deg: float = 120.0

    def plan(self, objective: str, policy_input: Any | None = None) -> RobotSkillAction:
        _ = policy_input
        text = _normalize(objective)
        if _contains_lateral_move(text, "left"):
            return RobotSkillAction(
                "move_base",
                {
                    "direction": "left",
                    "distance_cm": self._bounded_distance_cm(
                        objective,
                        self.default_step_cm,
                        self.max_step_cm,
                    ),
                },
                expected_duration_sec=1.0,
            )
        if _contains_lateral_move(text, "right"):
            return RobotSkillAction(
                "move_base",
                {
                    "direction": "right",
                    "distance_cm": self._bounded_distance_cm(
                        objective,
                        self.default_step_cm,
                        self.max_step_cm,
                    ),
                },
                expected_duration_sec=1.0,
            )
        semantic = SkillPlanner(
            approach_step_cm=min(self.default_step_cm, 10.0),
            max_step_cm=self.max_step_cm,
        ).plan(objective)
        if semantic is not None:
            return semantic
        if _contains(text, "backward", "back"):
            return RobotSkillAction(
                "move_base",
                {
                    "direction": "backward",
                    "distance_cm": self._bounded_distance_cm(
                        objective,
                        self.default_step_cm,
                        self.max_step_cm,
                    ),
                },
                expected_duration_sec=1.0,
            )
        if _contains(text, "forward", "ahead"):
            return RobotSkillAction(
                "move_base",
                {
                    "direction": "forward",
                    "distance_cm": self._bounded_distance_cm(
                        objective,
                        self.default_step_cm,
                        self.max_step_cm,
                    ),
                },
                expected_duration_sec=1.0,
            )
        if _contains(text, "left"):
            return RobotSkillAction(
                "turn_base",
                {
                    "direction": "left",
                    "angle_deg": self._bounded_number(
                        objective,
                        self.default_turn_deg,
                        self.max_turn_deg,
                    ),
                },
                expected_duration_sec=1.0,
            )
        if _contains(text, "right"):
            return RobotSkillAction(
                "turn_base",
                {
                    "direction": "right",
                    "angle_deg": self._bounded_number(
                        objective,
                        self.default_turn_deg,
                        self.max_turn_deg,
                    ),
                },
                expected_duration_sec=1.0,
            )
        return RobotSkillAction("inspect_scene", {"question": objective})

    @staticmethod
    def _bounded_number(text: str, default: float, maximum: float) -> float:
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        value = abs(float(match.group(0))) if match else float(default)
        return min(value, float(maximum))

    @staticmethod
    def _bounded_distance_cm(text: str, default: float, maximum: float) -> float:
        match = re.search(
            r"(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>cm|centimeter|centimeters|m|meter|meters|\u5398\u7c73|\u7c73)?",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return min(float(default), float(maximum))
        value = abs(float(match.group("value")))
        unit = (match.group("unit") or "cm").lower()
        if unit in {"m", "meter", "meters", "\u7c73"}:
            value *= 100.0
        return min(value, float(maximum))


class SkillPolicyAdapter:
    def __init__(
        self,
        policy_id: str,
        spec: PolicySpec,
        *,
        skill_catalog: RobotSkillCatalog,
    ) -> None:
        self.policy_id = policy_id
        self.spec = spec
        self.skill_catalog = skill_catalog
        self.loaded = False
        self._planner: ConservativeSkillPlanner | None = None

    async def warmup(self) -> None:
        self._planner = ConservativeSkillPlanner(
            default_step_cm=float(self.spec.settings.get("default_step_cm", 20.0)),
            default_turn_deg=float(self.spec.settings.get("default_turn_deg", 30.0)),
            max_step_cm=float(self.spec.settings.get("max_step_cm", 80.0)),
            max_turn_deg=float(self.spec.settings.get("max_turn_deg", 120.0)),
        )
        self.loaded = True

    async def predict(
        self,
        policy_input: Any,
        _observation: RobotObservation,
        intent: SkillIntent,
    ) -> RobotSkillAction:
        if self._planner is None:
            await self.warmup()
        assert self._planner is not None
        if intent.name:
            return RobotSkillAction(
                intent.name,
                dict(intent.arguments),
                expected_duration_sec=intent.timeout_sec,
            )
        return self._planner.plan(intent.objective, policy_input)

    async def close(self) -> None:
        self.loaded = False
        self._planner = None

    def health(self) -> PolicyHealth:
        return PolicyHealth(
            policy_id=self.policy_id,
            loaded=self.loaded,
            device=self.spec.device,
            metrics={"policy_type": self.spec.type, "robot_id": self.spec.robot_id},
        )

    def schema(self) -> PolicyIOSchema:
        robot_type = str(
            self.spec.settings.get("robot_skill_catalog_type")
            or self.spec.settings.get("embodiment_type")
            or self.spec.settings.get("body")
            or self.spec.robot_id
        )
        return PolicyIOSchema(
            policy_id=self.policy_id,
            policy_type=self.spec.type,
            robot_id=self.spec.robot_id,
            observation_schema={
                "modalities": ["images", "proprioception", "status", "task"]
            },
            action_schema={
                "type": "skill",
                "skills": list(self.skill_catalog.names(robot_type=robot_type)),
            },
            control_frequency_hz=float(self.spec.freq_hz),
            device=self.spec.device,
            accelerator=None,
            stateful=False,
            supports_interrupt=True,
            metadata=dict(self.spec.settings),
        )


def _normalize(text: str) -> str:
    normalized = str(text).lower().strip()
    replacements = {
        "\u5411\u524d": " forward ",
        "\u5f80\u524d": " forward ",
        "\u524d\u8fdb": " forward ",
        "\u5411\u540e": " backward ",
        "\u5f80\u540e": " backward ",
        "\u540e\u9000": " backward ",
        "\u5de6\u8f6c": " left ",
        "\u53f3\u8f6c": " right ",
        "\u5411\u5de6": " left ",
        "\u5f80\u5de6": " left ",
        "\u5de6\u8fb9": " left ",
        "\u5de6\u4fa7": " left ",
        "\u5411\u53f3": " right ",
        "\u5f80\u53f3": " right ",
        "\u53f3\u8fb9": " right ",
        "\u53f3\u4fa7": " right ",
        "\u79fb\u52a8": " move ",
        "\u5e73\u79fb": " strafe ",
        "\u4fa7\u5411": " lateral ",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return " ".join(normalized.split())


def _contains(text: str, *needles: str) -> bool:
    return any(needle.lower() in text for needle in needles)


def _contains_lateral_move(text: str, direction: str) -> bool:
    if not _contains(text, direction):
        return False
    return _contains(text, "move", "strafe", "lateral", "sideways")
