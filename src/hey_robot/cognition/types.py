from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hey_robot.protocol import RobotObservation, RobotStatus, SkillResult, UserTurn


@dataclass(frozen=True)
class RobotSnapshot:
    robot_id: str
    status: RobotStatus | None = None
    observation: RobotObservation | None = None
    skill_result: SkillResult | None = None

    def summary(self) -> str:
        parts = [f"robot_id={self.robot_id}"]
        if self.status is not None:
            parts.append(f"state={self.status.state}")
            if self.status.frame_id is not None:
                parts.append(f"frame_id={self.status.frame_id}")
            if self.status.task:
                parts.append(f"task={self.status.task}")
            if self.status.success is not None:
                parts.append(f"success={self.status.success}")
            if self.status.error:
                parts.append(f"error={self.status.error}")
            metric_summary = _summarize_metrics(self.status.metrics)
            if metric_summary:
                parts.append(f"metrics={metric_summary}")
        if self.observation is not None:
            parts.append(f"observation_frame={self.observation.frame_id}")
            parts.append(f"images={len(self.observation.images)}")
        if self.skill_result is not None:
            parts.append(
                f"last_skill={self.skill_result.skill_id}:{self.skill_result.status}"
            )
        return " ".join(parts)


@dataclass
class AgentCoreResult:
    reply_text: str | None = None
    skill_submitted: bool = False
    task_finished: bool = False
    tool: str = "wait"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentTurnInput:
    turn: UserTurn
    snapshot: RobotSnapshot
    memory_context: str | None = None
    recovery_context: str | None = None
    block_actuation: bool = False
    perception_context: str | None = None
    allowed_tools: set[str] | None = None


def _summarize_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return ""
    priority = ("battery", "arm", "base", "gripper")
    items = [(key, metrics[key]) for key in priority if key in metrics]
    items.extend((key, value) for key, value in metrics.items() if key not in priority)
    rendered = [f"{key}={_compact_metric_value(value)}" for key, value in items[:8]]
    return " ".join(item for item in rendered if item)


def _compact_metric_value(value: Any, *, depth: int = 0) -> str:
    if isinstance(value, dict):
        if depth >= 1:
            preview_keys = list(value)[:4]
            return "{" + ",".join(f"{key}=..." for key in preview_keys) + "}"
        preferred_keys: list[str] = [
            "status",
            "percentage",
            "voltage",
            "level",
            "state",
            "joint_count",
            "position",
            "error",
        ]
        items = [(key, value[key]) for key in preferred_keys if key in value]
        items.extend(
            (key, item) for key, item in value.items() if key not in preferred_keys
        )
        return (
            "{"
            + ",".join(
                f"{key}={_compact_metric_value(item, depth=depth + 1)}"
                for key, item in items[:6]
            )
            + "}"
        )
    if isinstance(value, list):
        if len(value) > 6:
            return f"[{len(value)} items]"
        return (
            "["
            + ",".join(_compact_metric_value(item, depth=depth + 1) for item in value)
            + "]"
        )
    text = str(value).replace("\n", " ").strip()
    return text[:80] + "..." if len(text) > 80 else text
