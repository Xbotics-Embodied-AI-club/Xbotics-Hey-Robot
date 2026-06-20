from __future__ import annotations

import re
from dataclasses import dataclass

from hey_robot.skills.actions import RobotSkillAction


@dataclass(frozen=True)
class SkillPlanner:
    """Small fallback mapper from plain language to the 15 classic skills."""

    approach_step_cm: float = 8.0
    max_step_cm: float = 80.0

    def plan(self, text: str) -> RobotSkillAction | None:
        normalized = _normalize(text)
        if _contains(normalized, "emergency", "estop", "urgent stop", "急停"):
            return RobotSkillAction(
                "stop_motion", {"emergency": True}, safety_level="emergency"
            )
        if _contains(normalized, "stop", "halt", "停止", "停下"):
            return RobotSkillAction("stop_motion")
        if _contains(normalized, "look around", "scan around", "环顾", "四周看看"):
            return RobotSkillAction(
                "look_around", {"question": text}, safety_level="observe"
            )
        if _contains(normalized, "marker", "aruco", "标记", "二维码"):
            if _contains(normalized, "align", "center", "face", "对齐", "居中"):
                return RobotSkillAction("detect_marker", safety_level="observe")
            return RobotSkillAction("detect_marker", safety_level="observe")
        if _contains(
            normalized, "face person", "face user", "look at person", "面对人"
        ):
            return None
        if _contains(normalized, "follow person", "follow me", "跟随", "跟着我"):
            return RobotSkillAction("human_follow")
        if _contains(normalized, "remember", "save scene", "记住", "保存场景"):
            return None
        if _contains(normalized, "recall", "memory", "remembered", "回忆", "记忆"):
            return None
        if _contains(
            normalized,
            "look",
            "see",
            "observe",
            "inspect",
            "verify",
            "what do you see",
            "看看",
            "观察",
            "看一下",
        ):
            return RobotSkillAction(
                "inspect_scene", {"question": text}, safety_level="observe"
            )
        if _contains(
            normalized,
            "reset posture",
            "safe posture",
            "arm home",
            "home arm",
            "复位",
            "回中",
        ):
            return RobotSkillAction("reset_posture")
        pose_name = _extract_named_pose(normalized)
        if pose_name:
            return RobotSkillAction("set_arm_pose", {"pose_name": pose_name})
        if _contains(normalized, "open gripper", "打开夹爪", "张开夹爪"):
            return RobotSkillAction("set_gripper", {"action": "open"})
        if _contains(normalized, "close gripper", "关闭夹爪", "夹紧"):
            return RobotSkillAction("set_gripper", {"action": "close"})
        if "gripper" in normalized and _contains(normalized, "%", "percent", "pct"):
            return RobotSkillAction(
                "set_gripper", {"opening_pct": self._bounded_number(text, 50.0, 100.0)}
            )
        if _contains(normalized, "backward", "back", "后退", "向后", "往后"):
            return RobotSkillAction(
                "move_base",
                {
                    "direction": "backward",
                    "distance_cm": self._bounded_distance_cm(
                        text, self.approach_step_cm, self.max_step_cm
                    ),
                },
                expected_duration_sec=1.0,
            )
        if _contains(normalized, "forward", "ahead", "前进", "向前", "往前"):
            return RobotSkillAction(
                "move_base",
                {
                    "direction": "forward",
                    "distance_cm": self._bounded_distance_cm(
                        text, self.approach_step_cm, self.max_step_cm
                    ),
                },
                expected_duration_sec=1.0,
            )
        if "left" in normalized or "左" in normalized:
            return RobotSkillAction(
                "turn_base",
                {
                    "direction": "left",
                    "angle_deg": self._bounded_number(text, 30.0, 120.0),
                },
                expected_duration_sec=1.0,
            )
        if "right" in normalized or "右" in normalized:
            return RobotSkillAction(
                "turn_base",
                {
                    "direction": "right",
                    "angle_deg": self._bounded_number(text, 30.0, 120.0),
                },
                expected_duration_sec=1.0,
            )
        return None

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
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return " ".join(normalized.split())


def _contains(text: str, *needles: str) -> bool:
    return any(needle.lower() in text for needle in needles)


def _extract_named_pose(text: str) -> str | None:
    patterns = (
        r"(?:pose|named pose)\s+([a-z0-9_-]+)",
        r"(?:go to|move to|arm to)\s+(?:pose\s+)?([a-z0-9_-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return str(match.group(1))
    return None
