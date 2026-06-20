from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UserInteractionIntent:
    kind: str
    urgency: str = "normal"
    target: str = "task"

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "urgency": self.urgency, "target": self.target}


def classify_user_interaction(
    text: str, *, robot_busy: bool = False
) -> UserInteractionIntent:
    normalized = " ".join((text or "").lower().split())
    interrupt_markers = {
        "stop",
        "halt",
        "pause",
        "abort",
        "cancel",
        "emergency",
        "hold",
        "停",
        "停止",
        "暂停",
        "别动",
        "取消",
    }
    if any(marker in normalized for marker in interrupt_markers):
        return UserInteractionIntent(
            kind="interrupt", urgency="immediate", target="active_skill"
        )

    correction_markers = {
        "not",
        "instead",
        "actually",
        "wrong",
        "left",
        "right",
        "higher",
        "lower",
        "不是",
        "不对",
        "左",
        "右",
        "高",
        "低",
        "换",
        "改",
    }
    if any(marker in normalized for marker in correction_markers):
        return UserInteractionIntent(
            kind="correction", urgency="safe_boundary", target="active_skill"
        )

    readonly_markers = {
        "battery",
        "power",
        "status",
        "state",
        "progress",
        "what are you doing",
        "what do you see",
        "where are you",
        "电池",
        "电量",
        "状态",
        "进度",
        "你在干什么",
        "你看到",
        "看到了什么",
        "现在怎么样",
        "机械臂",
        "夹爪",
    }
    if robot_busy and any(marker in normalized for marker in readonly_markers):
        return UserInteractionIntent(
            kind="read_only", urgency="immediate", target="status"
        )

    if robot_busy:
        return UserInteractionIntent(
            kind="follow_up", urgency="safe_boundary", target="task"
        )
    return UserInteractionIntent(kind="new_task", urgency="normal", target="task")
