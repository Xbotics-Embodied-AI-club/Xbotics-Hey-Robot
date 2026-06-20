from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Literal

GoalStatus = Literal["active", "completed", "abandoned"]


@dataclass
class AutonomyGoal:
    goal_id: int
    text: str
    source: str = "system"
    status: GoalStatus = "active"
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    summary: str | None = None

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "text": self.text,
            "source": self.source,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary,
        }


@dataclass
class AutonomyEvent:
    event_type: str
    summary: str
    timestamp: float = field(default_factory=time.time)
    frame_id: int | None = None

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
        }


class AutonomyManager:
    """Small goal and event memory for always-on agent operation.

    This is intentionally policy-light. It gives the agent durable context and
    tools to manage goals, but it does not hard-code robot-specific behavior.
    """

    def __init__(
        self, *, max_events: int = 50, default_goal: str | None = None
    ) -> None:
        self.max_events = max(1, int(max_events))
        self._next_goal_id = 1
        self._goals: list[AutonomyGoal] = []
        self._events: list[AutonomyEvent] = []
        if default_goal:
            self.add_goal(default_goal, source="default", priority=-10)

    def reset(self, *, keep_goals: bool = True) -> None:
        self._events.clear()
        if not keep_goals:
            self._goals.clear()
            self._next_goal_id = 1

    def add_goal(
        self, text: str, *, source: str = "system", priority: int = 0
    ) -> AutonomyGoal:
        goal_text = (text or "").strip()
        if not goal_text:
            raise ValueError("goal text must not be empty")
        existing = self.find_active_goal(goal_text)
        if existing is not None:
            existing.updated_at = time.time()
            existing.priority = max(existing.priority, priority)
            return existing
        goal = AutonomyGoal(
            goal_id=self._next_goal_id,
            text=goal_text,
            source=source,
            priority=priority,
        )
        self._next_goal_id += 1
        self._goals.append(goal)
        return goal

    def find_active_goal(self, text: str) -> AutonomyGoal | None:
        normalized = (text or "").strip().lower()
        if not normalized:
            return None
        for goal in self._goals:
            if goal.status == "active" and goal.text.lower() == normalized:
                return goal
        return None

    def active_goal(self) -> AutonomyGoal | None:
        active = [goal for goal in self._goals if goal.status == "active"]
        if not active:
            return None
        return sorted(active, key=lambda g: (g.priority, g.created_at), reverse=True)[0]

    def complete_goal(
        self, goal_id: int | None = None, *, summary: str = ""
    ) -> AutonomyGoal:
        goal = self._resolve_goal(goal_id)
        goal.status = "completed"
        goal.updated_at = time.time()
        goal.summary = (summary or "").strip() or "completed"
        return goal

    def abandon_goal(
        self, goal_id: int | None = None, *, summary: str = ""
    ) -> AutonomyGoal:
        goal = self._resolve_goal(goal_id)
        goal.status = "abandoned"
        goal.updated_at = time.time()
        goal.summary = (summary or "").strip() or "abandoned"
        return goal

    def remember(
        self, event_type: str, summary: str, *, frame_id: int | None = None
    ) -> None:
        text = (summary or "").strip()
        if not text:
            return
        self._events.append(
            AutonomyEvent(
                event_type=event_type or "note", summary=text, frame_id=frame_id
            )
        )
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events :]

    def goals_json(self) -> str:
        return json.dumps(
            {"goals": [goal.to_dict() for goal in self._goals]}, ensure_ascii=False
        )

    def prompt_context(self) -> str:
        parts: list[str] = []
        goal = self.active_goal()
        if goal is not None:
            parts.append(
                f"Active autonomous goal: {goal.text} (id={goal.goal_id}, source={goal.source})"
            )
        active_goals = [
            g for g in self._goals if g.status == "active" and g is not goal
        ]
        if active_goals:
            parts.append(
                "Other active goals: "
                + "; ".join(f"{g.goal_id}: {g.text}" for g in active_goals[-5:])
            )
        if self._events:
            parts.append("Recent robot memory:")
            for event in self._events[-8:]:
                frame = f" frame={event.frame_id}" if event.frame_id is not None else ""
                parts.append(f"- {event.event_type}{frame}: {event.summary}")
        return "\n".join(parts)

    def _resolve_goal(self, goal_id: int | None) -> AutonomyGoal:
        if goal_id is None:
            goal = self.active_goal()
            if goal is None:
                raise ValueError("no active goal")
            return goal
        for goal in self._goals:
            if goal.goal_id == int(goal_id):
                return goal
        raise ValueError(f"unknown goal_id: {goal_id}")
