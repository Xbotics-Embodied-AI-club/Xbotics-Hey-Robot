from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hey_robot.agents.skill_gateway import WaitPolicy


@dataclass(frozen=True)
class RoutedCommand:
    capability: str
    objective: str
    slots: dict[str, Any] = field(default_factory=dict)
    interrupt: bool = False
    wait_policy: WaitPolicy = "wait_acceptance"
    reply_text: str = "已发送指令。"


class CommandRouter:
    """Deterministic routes for short, low-ambiguity robot commands."""

    def route(self, text: str) -> RoutedCommand | None:
        normalized = _normalize(text)
        if not normalized:
            return None
        if normalized in {"stop", "halt", "停止", "停下", "别动", "不要动"} or (
            "停下" in normalized and "动作" in normalized
        ):
            return RoutedCommand(
                capability="stop_motion",
                objective="停止当前所有机器人动作",
                slots={"emergency": True},
                interrupt=True,
                wait_policy="wait_acceptance",
                reply_text="已发送停止指令，正在确认状态。",
            )
        if normalized in {"复位", "重置", "reset", "resetposture"}:
            return RoutedCommand(
                capability="reset_posture",
                objective="复位机器人姿态",
                slots={},
                interrupt=True,
                wait_policy="wait_acceptance",
                reply_text="已发送复位指令，正在确认状态。",
            )
        if _is_home_pose_command(normalized):
            return RoutedCommand(
                capability="set_arm_pose",
                objective="机械臂回到 home 位姿",
                slots={"pose_name": "home"},
                wait_policy="wait_acceptance",
                reply_text="已发送回到 home 位姿的指令。",
            )
        if _is_gripper_open_command(normalized):
            return RoutedCommand(
                capability="set_gripper",
                objective="打开夹爪",
                slots={"action": "open"},
                wait_policy="wait_acceptance",
                reply_text="已发送打开夹爪的指令。",
            )
        if _is_gripper_close_command(normalized):
            return RoutedCommand(
                capability="set_gripper",
                objective="关闭夹爪",
                slots={"action": "close"},
                wait_policy="wait_acceptance",
                reply_text="已发送关闭夹爪的指令。",
            )
        return None


def _normalize(text: str) -> str:
    return (
        str(text or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("　", "")
        .replace("。", "")
        .replace("！", "")
        .replace("!", "")
    )


def _is_home_pose_command(text: str) -> bool:
    return "home" in text and any(
        marker in text for marker in ("位姿", "位置", "姿态", "回到", "回", "到")
    )


def _is_gripper_open_command(text: str) -> bool:
    return text.startswith("夹爪") and any(
        marker in text for marker in ("打开", "张开", "完全张开", "open")
    )


def _is_gripper_close_command(text: str) -> bool:
    return text.startswith("夹爪") and any(
        marker in text for marker in ("关闭", "闭合", "夹紧", "close")
    )
