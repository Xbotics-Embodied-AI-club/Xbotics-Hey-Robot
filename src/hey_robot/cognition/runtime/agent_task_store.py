"""Session-scoped sustained task state for the robot agent loop."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from hey_robot.protocol import ActionProposal, Envelope, ToolOutcome
from hey_robot.protocol.messages import to_payload

TaskStatus = Literal["active", "completed", "blocked", "cancelled", "failed"]

TERMINAL_STATUSES = frozenset({"completed", "blocked", "cancelled", "failed"})


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    session_key: str
    robot_id: str
    objective: str
    ui_summary: str
    status: TaskStatus
    created_at: float
    updated_at: float
    step_count: int
    continuation_count: int
    deadline_at: float | None
    last_error: str | None
    final_recap: str | None


@dataclass(frozen=True)
class AgentTaskStep:
    step_id: str
    task_id: str
    sequence: int
    proposal: ActionProposal
    outcome: ToolOutcome
    started_at: float
    completed_at: float | None
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class CompletionCheck:
    accepted: bool
    reason: str


class AgentTaskStore:
    """SQLite-backed active task and evidence ledger.

    The store owns conversation task semantics only. Physical truth remains in
    SkillResult, RobotObservation and RobotStatus returned through ToolOutcome.
    """

    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS sustained_tasks (
                task_id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                robot_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                ui_summary TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                channel TEXT,
                chat_id TEXT,
                sender_id TEXT,
                user_id TEXT,
                agent_id TEXT,
                episode_id TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                step_count INTEGER NOT NULL DEFAULT 0,
                continuation_count INTEGER NOT NULL DEFAULT 0,
                deadline_at REAL,
                last_error TEXT,
                final_recap TEXT
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS task_steps (
                step_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                proposal_json TEXT NOT NULL,
                outcome_json TEXT NOT NULL,
                started_at REAL NOT NULL,
                completed_at REAL,
                evidence_json TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES sustained_tasks(task_id)
            )
            """
        )
        self._db.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_sustained_task
            ON sustained_tasks(session_key)
            WHERE status = 'active'
            """
        )
        self._db.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS task_steps_sequence
            ON task_steps(task_id, sequence)
            """
        )
        self._db.commit()

    def create_task(
        self,
        *,
        session_key: str,
        envelope: Envelope,
        objective: str,
        ui_summary: str = "",
        deadline_at: float | None = None,
    ) -> AgentTask:
        if self.active_task(session_key) is not None:
            raise ValueError("当前会话已有进行中的持续任务。")
        robot_id = envelope.robot_id or ""
        if not robot_id:
            raise ValueError("当前没有可用的机器人。")
        now = time.time()
        task_id = f"task_{uuid.uuid4().hex}"
        self._db.execute(
            """
            INSERT INTO sustained_tasks (
                task_id, session_key, robot_id, objective, ui_summary, status,
                channel, chat_id, sender_id, user_id, agent_id, episode_id,
                created_at, updated_at, step_count, continuation_count,
                deadline_at, last_error, final_recap
            )
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, NULL, NULL)
            """,
            (
                task_id,
                session_key,
                robot_id,
                objective,
                ui_summary,
                envelope.channel,
                envelope.chat_id,
                envelope.sender_id,
                envelope.user_id,
                envelope.agent_id,
                envelope.episode_id,
                now,
                now,
                deadline_at,
            ),
        )
        self._db.commit()
        task = self.active_task(session_key)
        if task is None:
            raise RuntimeError("failed to load newly created sustained task")
        return task

    def active_task(self, session_key: str) -> AgentTask | None:
        row = self._db.execute(
            """
            SELECT task_id, session_key, robot_id, objective, ui_summary, status,
                   created_at, updated_at, step_count, continuation_count,
                   deadline_at, last_error, final_recap
            FROM sustained_tasks
            WHERE session_key=? AND status='active'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (session_key,),
        ).fetchone()
        return _task_from_row(row) if row is not None else None

    def task(self, task_id: str) -> AgentTask | None:
        row = self._db.execute(
            """
            SELECT task_id, session_key, robot_id, objective, ui_summary, status,
                   created_at, updated_at, step_count, continuation_count,
                   deadline_at, last_error, final_recap
            FROM sustained_tasks
            WHERE task_id=?
            """,
            (task_id,),
        ).fetchone()
        return _task_from_row(row) if row is not None else None

    def recent_tasks(
        self, limit: int = 50, *, robot_id: str | None = None
    ) -> tuple[AgentTask, ...]:
        if robot_id:
            rows = self._db.execute(
                """
                SELECT task_id, session_key, robot_id, objective, ui_summary, status,
                       created_at, updated_at, step_count, continuation_count,
                       deadline_at, last_error, final_recap
                FROM sustained_tasks
                WHERE robot_id=?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (robot_id, limit),
            ).fetchall()
        else:
            rows = self._db.execute(
                """
                SELECT task_id, session_key, robot_id, objective, ui_summary, status,
                       created_at, updated_at, step_count, continuation_count,
                       deadline_at, last_error, final_recap
                FROM sustained_tasks
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(_task_from_row(row) for row in rows)

    def add_step(
        self, task_id: str, proposal: ActionProposal, outcome: ToolOutcome
    ) -> AgentTaskStep:
        task = self.task(task_id)
        if task is None or task.status != "active":
            raise ValueError("cannot append a step to a non-active task")
        sequence = int(task.step_count) + 1
        step_id = f"step_{uuid.uuid4().hex}"
        evidence_ids = _evidence_ids(step_id, proposal, outcome)
        now = time.time()
        self._db.execute(
            """
            INSERT INTO task_steps (
                step_id, task_id, sequence, proposal_json, outcome_json,
                started_at, completed_at, evidence_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step_id,
                task_id,
                sequence,
                json.dumps(to_payload(proposal), ensure_ascii=False, sort_keys=True),
                json.dumps(to_payload(outcome), ensure_ascii=False, sort_keys=True),
                now,
                now,
                json.dumps(list(evidence_ids), ensure_ascii=False),
            ),
        )
        self._db.execute(
            """
            UPDATE sustained_tasks
            SET step_count=?, updated_at=?
            WHERE task_id=?
            """,
            (sequence, now, task_id),
        )
        self._db.commit()
        return AgentTaskStep(
            step_id,
            task_id,
            sequence,
            proposal,
            outcome,
            now,
            now,
            evidence_ids,
        )

    def recent_steps(self, task_id: str, limit: int = 12) -> tuple[AgentTaskStep, ...]:
        rows = self._db.execute(
            """
            SELECT step_id, task_id, sequence, proposal_json, outcome_json,
                   started_at, completed_at, evidence_json
            FROM task_steps
            WHERE task_id=?
            ORDER BY sequence DESC
            LIMIT ?
            """,
            (task_id, limit),
        ).fetchall()
        return tuple(_step_from_row(row) for row in reversed(rows))

    def continue_task(self, task_id: str) -> int:
        task = self.task(task_id)
        if task is None or task.status != "active":
            raise ValueError("cannot continue a non-active task")
        count = task.continuation_count + 1
        self._db.execute(
            """
            UPDATE sustained_tasks
            SET continuation_count=?, updated_at=?
            WHERE task_id=?
            """,
            (count, time.time(), task_id),
        )
        self._db.commit()
        return count

    def complete_task(
        self, task_id: str, *, recap: str, evidence_ids: tuple[str, ...]
    ) -> CompletionCheck:
        check = self.check_completion(task_id, evidence_ids)
        if not check.accepted:
            return check
        self._finish(task_id, "completed", final_recap=recap)
        return check

    def control_task(self, task_id: str, status: TaskStatus, reason: str) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError("control_task status must be terminal")
        self._finish(task_id, status, last_error=reason, final_recap=reason)

    def check_completion(
        self, task_id: str, evidence_ids: tuple[str, ...]
    ) -> CompletionCheck:
        if not evidence_ids:
            return CompletionCheck(
                False, "complete_task 必须引用至少一个 evidence ID。"
            )
        steps = self.recent_steps(task_id, limit=200)
        known: dict[str, AgentTaskStep] = {}
        for step in steps:
            for evidence_id in step.evidence_ids:
                known[evidence_id] = step
        missing = [item for item in evidence_ids if item not in known]
        if missing:
            return CompletionCheck(
                False, "complete_task 引用了不存在或不属于当前任务的 evidence ID。"
            )
        referenced = [known[item] for item in evidence_ids]
        if any(step.outcome.status != "completed" for step in referenced):
            return CompletionCheck(False, "完成证据必须来自已成功完成的步骤。")
        last_world_change = max(
            (
                step.sequence
                for step in steps
                if step.outcome.status == "completed"
                and step.proposal.intent_kind == "skill"
            ),
            default=0,
        )
        if last_world_change and not any(
            step.sequence > last_world_change
            and step.proposal.intent_kind == "observation"
            and step.outcome.status == "completed"
            for step in referenced
        ):
            return CompletionCheck(
                False,
                "发生移动或转向后，完成当前场景相关任务必须引用动作后的观察证据。",
            )
        return CompletionCheck(True, "完成证据有效。")

    def projection(self, session_key: str) -> str:
        task = self.active_task(session_key)
        if task is None:
            return "当前会话没有进行中的持续任务。"
        steps = self.recent_steps(task.task_id, limit=6)
        lines = [
            (
                "当前持续任务："
                f"id={task.task_id}；objective={task.objective}；"
                f"status={task.status}；steps={task.step_count}；"
                f"continuations={task.continuation_count}。"
            )
        ]
        if steps:
            lines.append("最近证据：")
            for step in steps:
                summary = step.outcome.user_summary or step.outcome.status
                ids = ", ".join(step.evidence_ids) or "无"
                lines.append(
                    f"- #{step.sequence} {step.proposal.skill_name} "
                    f"status={step.outcome.status} evidence={ids} summary={summary}"
                )
        return "\n".join(lines)

    def close(self) -> None:
        self._db.close()

    def _finish(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        last_error: str | None = None,
        final_recap: str | None = None,
    ) -> None:
        self._db.execute(
            """
            UPDATE sustained_tasks
            SET status=?, updated_at=?, last_error=?, final_recap=?
            WHERE task_id=? AND status='active'
            """,
            (status, time.time(), last_error, final_recap, task_id),
        )
        self._db.commit()


def _task_from_row(row: tuple[Any, ...]) -> AgentTask:
    return AgentTask(
        task_id=str(row[0]),
        session_key=str(row[1]),
        robot_id=str(row[2]),
        objective=str(row[3]),
        ui_summary=str(row[4] or ""),
        status=row[5],
        created_at=float(row[6]),
        updated_at=float(row[7]),
        step_count=int(row[8]),
        continuation_count=int(row[9]),
        deadline_at=float(row[10]) if row[10] is not None else None,
        last_error=str(row[11]) if row[11] is not None else None,
        final_recap=str(row[12]) if row[12] is not None else None,
    )


def _step_from_row(row: tuple[Any, ...]) -> AgentTaskStep:
    proposal = ActionProposal(**json.loads(row[3]))
    outcome = ToolOutcome(**json.loads(row[4]))
    evidence_ids = tuple(str(item) for item in json.loads(row[7]))
    return AgentTaskStep(
        step_id=str(row[0]),
        task_id=str(row[1]),
        sequence=int(row[2]),
        proposal=proposal,
        outcome=outcome,
        started_at=float(row[5]),
        completed_at=float(row[6]) if row[6] is not None else None,
        evidence_ids=evidence_ids,
    )


def _evidence_ids(
    step_id: str, proposal: ActionProposal, outcome: ToolOutcome
) -> tuple[str, ...]:
    ids = [f"step:{step_id}"]
    if outcome.operation_id:
        prefix = "observation" if proposal.intent_kind == "observation" else "skill"
        ids.append(f"{prefix}:{outcome.operation_id}")
    ids.extend(
        item.strip()
        for item in outcome.data.get("evidence_ids", ()) or ()
        if isinstance(item, str) and item.strip()
    )
    return tuple(dict.fromkeys(ids))
