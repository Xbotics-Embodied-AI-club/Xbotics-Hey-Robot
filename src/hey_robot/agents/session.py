from __future__ import annotations

import time
from dataclasses import dataclass, field

from hey_robot.agents.types import AgentCoreResult
from hey_robot.protocol import UserTurn


@dataclass
class AgentTurnState:
    turn: UserTurn
    skill_id: str | None = None
    started_at: float = field(default_factory=time.time)
    baseline_frame_id: int | None = None
    status: str = "started"


class AgentTurnSessions:
    """In-memory session state for agent turns.

    Durable task state belongs to TaskRunManager. This class only tracks the
    live service concerns needed for turn handling: message dedupe, robot leases, and
    trace-level turn status.
    """

    def __init__(
        self, *, max_seen_messages: int = 2000, seen_trim_size: int = 1000
    ) -> None:
        self.active_turns: dict[str, AgentTurnState] = {}
        self.robot_leases: dict[str, tuple[str, float]] = {}
        self._seen_messages: list[str] = []
        self._seen_message_set: set[str] = set()
        self.max_seen_messages = max_seen_messages
        self.seen_trim_size = seen_trim_size

    def is_duplicate_or_remember(self, turn: UserTurn) -> bool:
        key = turn.envelope.message_id or turn.envelope.trace_id
        if key in self._seen_message_set:
            return True
        self._seen_message_set.add(key)
        self._seen_messages.append(key)
        if len(self._seen_messages) > self.max_seen_messages:
            keep = self._seen_messages[-self.seen_trim_size :]
            self._seen_messages = keep
            self._seen_message_set = set(keep)
        return False

    def active_robot_lease(
        self, robot_id: str | None, *, timeout_sec: float
    ) -> tuple[str, float] | None:
        if not robot_id:
            return None
        lease = self.robot_leases.get(robot_id)
        if lease is None:
            return None
        if time.time() - lease[1] > timeout_sec:
            self.robot_leases.pop(robot_id, None)
            return None
        return lease

    def lease_robot(self, robot_id: str | None, skill_id: str) -> None:
        if robot_id:
            self.robot_leases[robot_id] = (skill_id, time.time())

    def release_robot(self, robot_id: str | None) -> None:
        if robot_id:
            self.robot_leases.pop(robot_id, None)

    def record_turn_result(
        self,
        *,
        turn: UserTurn,
        result: AgentCoreResult,
        baseline_frame_id: int | None,
    ) -> AgentTurnState:
        skill_id = result.metadata.get("skill_id")
        state = AgentTurnState(
            turn=turn,
            skill_id=str(skill_id) if skill_id else None,
            baseline_frame_id=baseline_frame_id,
            status="task_finished" if result.task_finished else "responded",
        )
        self.active_turns[turn.envelope.trace_id] = state
        return state

    def update_turn_status(
        self, trace_id: str | None, status: str
    ) -> AgentTurnState | None:
        state = self.active_turns.get(trace_id or "")
        if state is None:
            return None
        state.status = status
        return state
