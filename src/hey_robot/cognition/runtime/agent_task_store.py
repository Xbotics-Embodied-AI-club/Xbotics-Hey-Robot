"""Session-scoped sustained task state for the robot agent loop."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from hey_robot.cognition.tools.models import PhysicalToolCall
from hey_robot.protocol import Envelope, ToolOutcome
from hey_robot.protocol.messages import from_payload, to_payload

TaskStatus = Literal["active", "paused", "completed", "blocked", "cancelled", "failed"]
StepStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

TERMINAL_STATUSES = frozenset({"completed", "blocked", "cancelled", "failed"})
TASK_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    session_key: str
    envelope: Envelope
    objective: str
    ui_summary: str
    status: TaskStatus
    created_at: float
    updated_at: float
    step_count: int
    deadline_at: float | None
    last_error: str | None
    final_recap: str | None

    @property
    def robot_id(self) -> str:
        return self.envelope.robot_id or ""


@dataclass(frozen=True)
class AgentTaskStep:
    step_id: str
    task_id: str
    sequence: int
    proposal: PhysicalToolCall
    outcome: ToolOutcome
    started_at: float
    completed_at: float | None
    evidence_ids: tuple[str, ...]
    status: StepStatus = "completed"
    run_id: str | None = None
    tool_call_id: str | None = None
    last_event_sequence: int = 0


class AgentTaskStore:
    """SQLite-backed active task and evidence ledger.

    The store owns conversation task semantics only. Physical truth remains in
    SkillResult, RobotObservation and RobotStatus returned through ToolOutcome.
    """

    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        existing = self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sustained_tasks'"
        ).fetchone()
        schema_version = int(self._db.execute("PRAGMA user_version").fetchone()[0])
        if existing is not None and schema_version != TASK_SCHEMA_VERSION:
            self._db.close()
            raise RuntimeError(
                "unsupported sustained-task runtime schema; archive or reset this "
                "deployment runtime before starting the new harness"
            )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS sustained_tasks (
                task_id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                envelope_json TEXT NOT NULL,
                interaction_id TEXT,
                objective TEXT NOT NULL,
                ui_summary TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                step_count INTEGER NOT NULL DEFAULT 0,
                deadline_at REAL,
                last_error TEXT,
                final_recap TEXT,
                resume_required INTEGER NOT NULL DEFAULT 0,
                resume_after_sequence INTEGER NOT NULL DEFAULT 0
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
                run_id TEXT,
                tool_call_id TEXT,
                status TEXT NOT NULL DEFAULT 'completed',
                last_event_sequence INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(task_id) REFERENCES sustained_tasks(task_id)
            )
            """
        )
        self._db.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS one_open_sustained_task
            ON sustained_tasks(session_key)
            WHERE status IN ('active', 'paused')
            """
        )
        self._db.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS task_steps_sequence
            ON task_steps(task_id, sequence)
            """
        )
        self._db.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS task_steps_run_id
            ON task_steps(run_id)
            WHERE run_id IS NOT NULL
            """
        )
        self._db.execute(f"PRAGMA user_version={TASK_SCHEMA_VERSION}")
        self._db.commit()

    def create_task(
        self,
        *,
        session_key: str,
        envelope: Envelope,
        interaction_id: str | None = None,
        objective: str,
        ui_summary: str = "",
        deadline_at: float | None = None,
    ) -> AgentTask:
        if self.current_task(session_key) is not None:
            raise ValueError("当前会话已有进行中的持续任务。")
        robot_id = envelope.robot_id or ""
        if not robot_id:
            raise ValueError("当前没有可用的机器人。")
        now = time.time()
        task_id = f"task_{uuid.uuid4().hex}"
        self._db.execute(
            """
            INSERT INTO sustained_tasks (
                task_id, session_key, envelope_json, interaction_id,
                objective, ui_summary, status,
                created_at, updated_at, step_count,
                deadline_at, last_error, final_recap
            )
            VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, 0, ?, NULL, NULL)
            """,
            (
                task_id,
                session_key,
                json.dumps(to_payload(envelope), ensure_ascii=False, sort_keys=True),
                interaction_id,
                objective,
                ui_summary,
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

    def update_route(
        self, task_id: str, *, envelope: Envelope, interaction_id: str
    ) -> None:
        """Associate future task continuations with the latest accepted interaction."""
        self._db.execute(
            """
            UPDATE sustained_tasks
            SET envelope_json=?, interaction_id=?, updated_at=?
            WHERE task_id=? AND status='active'
            """,
            (
                json.dumps(to_payload(envelope), ensure_ascii=False, sort_keys=True),
                interaction_id,
                time.time(),
                task_id,
            ),
        )
        self._db.commit()

    def task_interaction_id(self, task_id: str) -> str | None:
        row = self._db.execute(
            "SELECT interaction_id FROM sustained_tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return str(row[0])

    def active_task(self, session_key: str) -> AgentTask | None:
        row = self._db.execute(
            """
            SELECT task_id, session_key, envelope_json, objective, ui_summary, status,
                   created_at, updated_at, step_count,
                   deadline_at, last_error, final_recap
            FROM sustained_tasks
            WHERE session_key=? AND status='active'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (session_key,),
        ).fetchone()
        return _task_from_row(row) if row is not None else None

    def current_task(self, session_key: str) -> AgentTask | None:
        """Return the one non-terminal task, including a paused task."""
        row = self._db.execute(
            """
            SELECT task_id, session_key, envelope_json, objective, ui_summary, status,
                   created_at, updated_at, step_count,
                   deadline_at, last_error, final_recap
            FROM sustained_tasks
            WHERE session_key=? AND status IN ('active', 'paused')
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (session_key,),
        ).fetchone()
        return _task_from_row(row) if row is not None else None

    def resumable_tasks(self) -> tuple[AgentTask, ...]:
        rows = self._db.execute(
            """
            SELECT task_id, session_key, envelope_json, objective, ui_summary, status,
                   created_at, updated_at, step_count,
                   deadline_at, last_error, final_recap
            FROM sustained_tasks
            WHERE status='active' AND resume_required=1
            ORDER BY updated_at ASC
            """
        ).fetchall()
        return tuple(_task_from_row(row) for row in rows)

    def resume_after_sequence(self, task_id: str) -> int:
        row = self._db.execute(
            "SELECT resume_after_sequence FROM sustained_tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def task(self, task_id: str) -> AgentTask | None:
        row = self._db.execute(
            """
            SELECT task_id, session_key, envelope_json, objective, ui_summary, status,
                   created_at, updated_at, step_count,
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
                SELECT task_id, session_key, envelope_json, objective, ui_summary, status,
                       created_at, updated_at, step_count,
                       deadline_at, last_error, final_recap
                FROM sustained_tasks
                WHERE json_extract(envelope_json, '$.robot_id')=?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (robot_id, limit),
            ).fetchall()
        else:
            rows = self._db.execute(
                """
                SELECT task_id, session_key, envelope_json, objective, ui_summary, status,
                       created_at, updated_at, step_count,
                       deadline_at, last_error, final_recap
                FROM sustained_tasks
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(_task_from_row(row) for row in rows)

    def task_envelope(self, task_id: str) -> Envelope | None:
        task = self.task(task_id)
        return task.envelope if task is not None else None

    def has_successful_step(self, task_id: str) -> bool:
        row = self._db.execute(
            """
            SELECT 1 FROM task_steps
            WHERE task_id=? AND status='completed'
            LIMIT 1
            """,
            (task_id,),
        ).fetchone()
        return row is not None

    def add_step(
        self, task_id: str, proposal: PhysicalToolCall, outcome: ToolOutcome
    ) -> AgentTaskStep:
        task = self.task(task_id)
        if task is None or task.status != "active":
            raise ValueError("cannot append a step to a non-active task")
        sequence = int(task.step_count) + 1
        step_id = f"step_{uuid.uuid4().hex}"
        evidence_ids = _evidence_ids(step_id, outcome)
        step_status: StepStatus = (
            "failed" if outcome.status == "failed" else "completed"
        )
        now = time.time()
        self._db.execute(
            """
            INSERT INTO task_steps (
                step_id, task_id, sequence, proposal_json, outcome_json,
                started_at, completed_at, evidence_json, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step_id,
                task_id,
                sequence,
                json.dumps(
                    _physical_call_payload(proposal), ensure_ascii=False, sort_keys=True
                ),
                json.dumps(to_payload(outcome), ensure_ascii=False, sort_keys=True),
                now,
                now,
                json.dumps(list(evidence_ids), ensure_ascii=False),
                step_status,
            ),
        )
        self._db.execute(
            """
            UPDATE sustained_tasks
            SET step_count=?, updated_at=?, resume_required=1,
                resume_after_sequence=?
            WHERE task_id=?
            """,
            (sequence, now, sequence, task_id),
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
            status=step_status,
        )

    def add_pending_step(
        self,
        task_id: str,
        proposal: PhysicalToolCall,
        *,
        run_id: str,
        tool_call_id: str,
    ) -> AgentTaskStep:
        task = self.task(task_id)
        if task is None or task.status != "active":
            raise ValueError("cannot append a step to a non-active task")
        sequence = task.step_count + 1
        step_id = f"step_{uuid.uuid4().hex}"
        now = time.time()
        outcome = ToolOutcome("accepted", operation_id=run_id)
        self._db.execute(
            """
            INSERT INTO task_steps (
                step_id, task_id, sequence, proposal_json, outcome_json,
                started_at, completed_at, evidence_json, run_id, tool_call_id,
                status, last_event_sequence
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, 'pending', 0)
            """,
            (
                step_id,
                task_id,
                sequence,
                json.dumps(
                    _physical_call_payload(proposal), ensure_ascii=False, sort_keys=True
                ),
                json.dumps(to_payload(outcome), ensure_ascii=False, sort_keys=True),
                now,
                json.dumps([], ensure_ascii=False),
                run_id,
                tool_call_id,
            ),
        )
        self._db.execute(
            """
            UPDATE sustained_tasks
            SET step_count=?, updated_at=?, resume_required=0
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
            None,
            (),
            status="pending",
            run_id=run_id,
            tool_call_id=tool_call_id,
        )

    def resolve_pending_step(
        self,
        run_id: str,
        *,
        outcome: ToolOutcome,
        status: StepStatus,
        event_sequence: int,
    ) -> AgentTaskStep | None:
        row = self._db.execute(
            """
            SELECT step_id, task_id, sequence, proposal_json, outcome_json,
                   started_at, completed_at, evidence_json, status, run_id,
                   tool_call_id, last_event_sequence
            FROM task_steps WHERE run_id=?
            """,
            (run_id,),
        ).fetchone()
        if row is None or int(row[11]) >= event_sequence:
            return None
        proposal = _physical_call_from_payload(json.loads(row[3]))
        evidence_ids = _evidence_ids(str(row[0]), outcome)
        completed_at = (
            time.time() if status in {"completed", "failed", "cancelled"} else None
        )
        self._db.execute(
            """
            UPDATE task_steps
            SET outcome_json=?, evidence_json=?, status=?, completed_at=?, last_event_sequence=?
            WHERE run_id=?
            """,
            (
                json.dumps(to_payload(outcome), ensure_ascii=False, sort_keys=True),
                json.dumps(list(evidence_ids), ensure_ascii=False),
                status,
                completed_at,
                event_sequence,
                run_id,
            ),
        )
        if status in {"completed", "failed", "cancelled"}:
            self._db.execute(
                """
                UPDATE sustained_tasks
                SET resume_required=1, resume_after_sequence=?, updated_at=?
                WHERE task_id=? AND status='active'
                """,
                (int(row[2]), time.time(), str(row[1])),
            )
        self._db.commit()
        return AgentTaskStep(
            str(row[0]),
            str(row[1]),
            int(row[2]),
            proposal,
            outcome,
            float(row[5]),
            completed_at,
            evidence_ids,
            status=status,
            run_id=run_id,
            tool_call_id=str(row[10]) if row[10] is not None else None,
            last_event_sequence=event_sequence,
        )

    def apply_skill_event(
        self,
        run_id: str,
        *,
        outcome: ToolOutcome,
        status: StepStatus,
        event_sequence: int,
    ) -> AgentTaskStep | None:
        return self.resolve_pending_step(
            run_id,
            outcome=outcome,
            status=status,
            event_sequence=event_sequence,
        )

    def recent_steps(self, task_id: str, limit: int = 12) -> tuple[AgentTaskStep, ...]:
        rows = self._db.execute(
            """
            SELECT step_id, task_id, sequence, proposal_json, outcome_json,
                   started_at, completed_at, evidence_json, status, run_id,
                   tool_call_id, last_event_sequence
            FROM task_steps
            WHERE task_id=?
            ORDER BY sequence DESC
            LIMIT ?
            """,
            (task_id, limit),
        ).fetchall()
        return tuple(_step_from_row(row) for row in reversed(rows))

    def active_run_ids(self, task_id: str) -> tuple[str, ...]:
        rows = self._db.execute(
            """
            SELECT run_id FROM task_steps
            WHERE task_id=? AND run_id IS NOT NULL AND status IN ('pending', 'running')
            ORDER BY sequence ASC
            """,
            (task_id,),
        ).fetchall()
        return tuple(str(row[0]) for row in rows if row[0] is not None)

    def active_skill_steps(self) -> tuple[AgentTaskStep, ...]:
        """返回 active sustained task 的 non-terminal Skill step。

        使用 Store query 而非 in-memory service map，确保服务重启、原始 Agent turn
        已结束后仍可进行 startup reconciliation。调用方负责 transport-specific status
        lookup，并通过 ``apply_skill_event`` 写入返回的 event。
        """
        rows = self._db.execute(
            """
            SELECT s.step_id, s.task_id, s.sequence, s.proposal_json, s.outcome_json,
                   s.started_at, s.completed_at, s.evidence_json, s.status, s.run_id,
                   s.tool_call_id, s.last_event_sequence
            FROM task_steps AS s
            JOIN sustained_tasks AS t ON t.task_id = s.task_id
            WHERE t.status='active' AND s.run_id IS NOT NULL
              AND s.status IN ('pending', 'running')
            ORDER BY s.started_at ASC, s.sequence ASC
            """
        ).fetchall()
        return tuple(_step_from_row(row) for row in rows)

    def pause_task(self, task_id: str, reason: str) -> None:
        self._db.execute(
            """
            UPDATE sustained_tasks
            SET status='paused', updated_at=?, last_error=?, resume_required=0
            WHERE task_id=? AND status='active'
            """,
            (time.time(), reason, task_id),
        )
        self._db.commit()

    def resume_task(self, task_id: str) -> AgentTask:
        self._db.execute(
            """
            UPDATE sustained_tasks
            SET status='active', updated_at=?, last_error=NULL, resume_required=1
            WHERE task_id=? AND status='paused'
            """,
            (time.time(), task_id),
        )
        self._db.commit()
        task = self.task(task_id)
        if task is None or task.status != "active":
            raise ValueError("cannot resume a non-paused task")
        return task

    def close_task(self, task_id: str, *, recap: str) -> None:
        """Close an active durable turn after the model returns final text."""
        recent = self.recent_steps(task_id, limit=1)
        if recent and recent[-1].status == "failed":
            self._finish(task_id, "failed", last_error=recap, final_recap=recap)
            return
        if recent and recent[-1].status == "cancelled":
            self._finish(task_id, "cancelled", last_error=recap, final_recap=recap)
            return
        self._finish(task_id, "completed", final_recap=recap)

    def complete_from_environment(self, task_id: str, *, recap: str) -> None:
        """Accept an authoritative environment terminal without LLM re-verification."""
        task = self.task(task_id)
        if task is None or task.status != "active":
            return
        self._finish(task_id, "completed", final_recap=recap)

    def control_task(self, task_id: str, status: TaskStatus, reason: str) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError("control_task status must be terminal")
        self._finish(task_id, status, last_error=reason, final_recap=reason)

    def projection(self, session_key: str) -> str:
        task = self.active_task(session_key)
        if task is None:
            return "当前会话没有进行中的持续任务。"
        return self._project_task(task)

    def _project_task(self, task: AgentTask) -> str:
        steps = self.recent_steps(task.task_id, limit=6)
        all_steps = self.recent_steps(task.task_id, limit=max(task.step_count, 1))
        completed_counts: dict[str, int] = {}
        for step in all_steps:
            if step.status != "completed":
                continue
            completed_counts[step.proposal.name] = (
                completed_counts.get(step.proposal.name, 0) + 1
            )
        lines = [
            (
                f"当前持续任务：id={task.task_id}；status={task.status}；"
                f"objective={task.objective}；steps={task.step_count}。"
            )
        ]
        if completed_counts:
            counts = "，".join(
                f"{name}×{count}" for name, count in sorted(completed_counts.items())
            )
            lines.append(f"已确认完成动作汇总：{counts}。")
        if steps:
            lines.append("最近证据：")
            for step in steps:
                summary = step.outcome.user_summary or step.outcome.status
                ids = ", ".join(step.evidence_ids) or "无"
                lines.append(
                    f"- #{step.sequence} {step.proposal.name} "
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
            SET status=?, updated_at=?, last_error=?, final_recap=?,
                resume_required=0
            WHERE task_id=? AND status='active'
            """,
            (status, time.time(), last_error, final_recap, task_id),
        )
        self._db.commit()


def _task_from_row(row: tuple[Any, ...]) -> AgentTask:
    payload = json.loads(str(row[2]))
    if not isinstance(payload, dict):
        raise ValueError("stored task envelope must be an object")
    return AgentTask(
        task_id=str(row[0]),
        session_key=str(row[1]),
        envelope=from_payload(Envelope, payload),
        objective=str(row[3]),
        ui_summary=str(row[4] or ""),
        status=row[5],
        created_at=float(row[6]),
        updated_at=float(row[7]),
        step_count=int(row[8]),
        deadline_at=float(row[9]) if row[9] is not None else None,
        last_error=str(row[10]) if row[10] is not None else None,
        final_recap=str(row[11]) if row[11] is not None else None,
    )


def _step_from_row(row: tuple[Any, ...]) -> AgentTaskStep:
    proposal = _physical_call_from_payload(json.loads(row[3]))
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
        status=row[8] if len(row) > 8 else "completed",
        run_id=str(row[9]) if len(row) > 9 and row[9] is not None else None,
        tool_call_id=str(row[10]) if len(row) > 10 and row[10] is not None else None,
        last_event_sequence=int(row[11]) if len(row) > 11 else 0,
    )


def _evidence_ids(step_id: str, outcome: ToolOutcome) -> tuple[str, ...]:
    ids = [f"step:{step_id}"]
    if outcome.operation_id:
        ids.append(f"tool:{outcome.operation_id}")
    ids.extend(
        item.strip()
        for item in outcome.data.get("evidence_ids", ()) or ()
        if isinstance(item, str) and item.strip()
    )
    return tuple(dict.fromkeys(ids))


def _physical_call_payload(call: PhysicalToolCall) -> dict[str, Any]:
    return {"name": call.name, "arguments": dict(call.arguments)}


def _physical_call_from_payload(payload: dict[str, Any]) -> PhysicalToolCall:
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("physical tool payload must include name")
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("physical tool payload arguments must be an object")
    return PhysicalToolCall(name, dict(arguments))
