"""Single SQLite authority for goals, actions, evidence, and execution locks."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from hey_robot.protocol.messages import RobotExecutionGate


class AutonomyStore:
    SCHEMA_VERSION = 2

    def __init__(self, path: str | Path) -> None:
        location = Path(path)
        location.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(location))
        self._db.row_factory = sqlite3.Row
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS goals (goal_id TEXT PRIMARY KEY, robot_id TEXT NOT NULL,
          status TEXT NOT NULL, version INTEGER NOT NULL, snapshot TEXT NOT NULL, termination_reason TEXT,
          active_deliberation_id TEXT, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS goal_budgets (goal_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE UNIQUE INDEX IF NOT EXISTS one_active_goal ON goals(robot_id)
          WHERE status NOT IN ('completed','failed','cancelled');
        CREATE TABLE IF NOT EXISTS goal_commands (command_id TEXT PRIMARY KEY, result TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS actions (deliberation_id TEXT PRIMARY KEY, skill_id TEXT UNIQUE NOT NULL,
          goal_id TEXT NOT NULL, status TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL,
          terminal_hash TEXT, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS evidence (evidence_id TEXT PRIMARY KEY, goal_id TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS outgoing_deliberations (goal_id TEXT NOT NULL, trigger_event_id TEXT NOT NULL,
          deliberation_id TEXT UNIQUE NOT NULL, request TEXT NOT NULL, request_hash TEXT NOT NULL,
          PRIMARY KEY(goal_id, trigger_event_id));
        CREATE TABLE IF NOT EXISTS robot_execution_gate (robot_id TEXT PRIMARY KEY, version INTEGER NOT NULL,
          state TEXT NOT NULL, control_id TEXT, reason TEXT, updated_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS control_commands (control_id TEXT PRIMARY KEY, robot_id TEXT NOT NULL,
          goal_id TEXT, target_skill_id TEXT, action TEXT NOT NULL, status TEXT NOT NULL,
          payload TEXT NOT NULL, terminal_hash TEXT, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS reconcile_records (reconcile_id TEXT PRIMARY KEY,
          robot_id TEXT NOT NULL, skill_id TEXT NOT NULL, operator_id TEXT NOT NULL,
          status_payload TEXT NOT NULL, created_at REAL NOT NULL);
        """)
        row = self._db.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            self._db.execute(
                "INSERT INTO schema_version VALUES (?)", (self.SCHEMA_VERSION,)
            )
        elif row[0] == 1:
            self._db.execute(
                "UPDATE schema_version SET version=?", (self.SCHEMA_VERSION,)
            )
        elif row[0] != self.SCHEMA_VERSION:
            raise RuntimeError("autonomy sqlite schema version mismatch")
        self._db.commit()

    def command_result(self, command_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT result FROM goal_commands WHERE command_id=?", (command_id,)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def create_goal(
        self,
        *,
        command_id: str,
        goal_id: str,
        robot_id: str,
        snapshot: dict[str, Any],
        budgets: dict[str, Any],
    ) -> bool:
        with self._db:
            try:
                self._db.execute(
                    "INSERT INTO goals VALUES (?, ?, 'pending', 0, ?, NULL, NULL, ?)",
                    (goal_id, robot_id, _json(snapshot), time.time()),
                )
                self._db.execute(
                    "INSERT INTO goal_budgets VALUES (?, ?)", (goal_id, _json(budgets))
                )
                self._db.execute(
                    "INSERT INTO goal_commands VALUES (?, ?)",
                    (command_id, _json({"goal_id": goal_id, "status": "pending"})),
                )
                self._db.execute(
                    "INSERT OR IGNORE INTO robot_execution_gate VALUES (?, 0, 'ready', NULL, NULL, ?)",
                    (robot_id, time.time()),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def goal(self, goal_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT * FROM goals WHERE goal_id=?", (goal_id,)
        ).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["snapshot"] = json.loads(value["snapshot"])
        return value

    def budget_state(
        self, goal_id: str, battery_percentage: float | None
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        row = self._db.execute(
            "SELECT created_at FROM goals WHERE goal_id=?", (goal_id,)
        ).fetchone()
        budget = self._db.execute(
            "SELECT payload FROM goal_budgets WHERE goal_id=?", (goal_id,)
        ).fetchone()
        if row is None or budget is None:
            return None
        deliberations = self._db.execute(
            "SELECT COUNT(*) FROM outgoing_deliberations WHERE goal_id=?", (goal_id,)
        ).fetchone()[0]
        skills = self._db.execute(
            "SELECT COUNT(*) FROM actions WHERE goal_id=?", (goal_id,)
        ).fetchone()[0]
        return json.loads(budget[0]), {
            "elapsed_wall_time_sec": max(0.0, time.time() - row[0]),
            "deliberations_used": int(deliberations),
            "skills_used": int(skills),
            "battery_percentage": battery_percentage,
        }

    def gate(self, robot_id: str) -> RobotExecutionGate:
        row = self._db.execute(
            "SELECT * FROM robot_execution_gate WHERE robot_id=?", (robot_id,)
        ).fetchone()
        if row is None:
            with self._db:
                self._db.execute(
                    "INSERT INTO robot_execution_gate VALUES (?, 0, 'ready', NULL, NULL, ?)",
                    (robot_id, time.time()),
                )
            return self.gate(robot_id)
        return RobotExecutionGate(
            robot_id=str(row["robot_id"]),
            version=int(row["version"]),
            state=str(row["state"]),  # type: ignore[arg-type]
            control_id=str(row["control_id"]) if row["control_id"] else None,
            reason=str(row["reason"]) if row["reason"] else None,
            updated_at=float(row["updated_at"]),
        )

    def active_goal_for_robot(self, robot_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT goal_id FROM goals WHERE robot_id=? AND status NOT IN ('completed','failed','cancelled')",
            (robot_id,),
        ).fetchone()
        return None if row is None else self.goal(str(row[0]))

    def goals_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT goal_id FROM goals ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            goal_data = self.goal(str(row[0]))
            if goal_data is not None:
                result.append(goal_data)
        return result

    def actions_for_goal(self, goal_id: str) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM actions WHERE goal_id=? ORDER BY created_at", (goal_id,)
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            result.append(item)
        return result

    def schedule_deliberation(
        self,
        *,
        goal_id: str,
        trigger_event_id: str,
        deliberation_id: str,
        request: dict[str, Any],
        request_hash: str,
    ) -> bool:
        with self._db:
            goal = self._db.execute(
                "SELECT status, termination_reason FROM goals WHERE goal_id=?",
                (goal_id,),
            ).fetchone()
            if (
                goal is None
                or goal[0] not in {"pending", "waiting"}
                or goal[1] is not None
            ):
                return False
            try:
                self._db.execute(
                    "INSERT INTO outgoing_deliberations VALUES (?, ?, ?, ?, ?)",
                    (
                        goal_id,
                        trigger_event_id,
                        deliberation_id,
                        _json(request),
                        request_hash,
                    ),
                )
            except sqlite3.IntegrityError:
                return False
            self._db.execute(
                "UPDATE goals SET status='active', version=version+1, active_deliberation_id=? WHERE goal_id=?",
                (deliberation_id, goal_id),
            )
            return True

    def deliberation_request_hash(
        self, *, goal_id: str, deliberation_id: str
    ) -> str | None:
        row = self._db.execute(
            "SELECT request_hash FROM outgoing_deliberations "
            "WHERE goal_id=? AND deliberation_id=?",
            (goal_id, deliberation_id),
        ).fetchone()
        return None if row is None else str(row[0])

    def accept_action(
        self,
        *,
        goal_id: str,
        deliberation_id: str,
        skill_id: str,
        payload: dict[str, Any],
    ) -> bool:
        with self._db:
            goal = self._db.execute(
                "SELECT status, termination_reason FROM goals WHERE goal_id=?",
                (goal_id,),
            ).fetchone()
            if goal is None or goal[0] != "active" or goal[1] is not None:
                return False
            try:
                self._db.execute(
                    "INSERT INTO actions VALUES (?, ?, ?, 'persisted', 0, ?, NULL, ?)",
                    (deliberation_id, skill_id, goal_id, _json(payload), time.time()),
                )
            except sqlite3.IntegrityError:
                return False
            self._db.execute(
                "UPDATE goals SET status='waiting', version=version+1 WHERE goal_id=?",
                (goal_id,),
            )
            return True

    def fail_goal(self, goal_id: str, reason: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE goals SET status='failed', version=version+1, termination_reason=COALESCE(termination_reason, ?) WHERE goal_id=? AND status NOT IN ('completed','failed','cancelled')",
                (reason, goal_id),
            )

    def complete_goal(self, goal_id: str) -> bool:
        with self._db:
            cur = self._db.execute(
                "UPDATE goals SET status='completed', version=version+1 "
                "WHERE goal_id=? AND status='active' AND termination_reason IS NULL",
                (goal_id,),
            )
            return cur.rowcount == 1

    def cancel_goal(self, goal_id: str) -> bool:
        with self._db:
            cur = self._db.execute(
                "UPDATE goals SET status='cancelled', termination_reason='cancel', version=version+1 WHERE goal_id=? AND status IN ('pending','active')",
                (goal_id,),
            )
            return cur.rowcount == 1

    def begin_stop(
        self,
        *,
        goal_id: str,
        robot_id: str,
        control_id: str,
        target_skill_id: str | None,
        action: str,
        reason: str,
        payload: dict[str, Any],
    ) -> bool:
        """Persist stop-pending gate and command before its bus publication."""
        with self._db:
            goal = self._db.execute(
                "SELECT status FROM goals WHERE goal_id=?", (goal_id,)
            ).fetchone()
            if goal is None or goal[0] not in {"waiting", "blocked"}:
                return False
            try:
                self._db.execute(
                    "INSERT INTO control_commands VALUES (?, ?, ?, ?, ?, 'persisted', ?, NULL, ?)",
                    (
                        control_id,
                        robot_id,
                        goal_id,
                        target_skill_id,
                        action,
                        _json(payload),
                        time.time(),
                    ),
                )
            except sqlite3.IntegrityError:
                return False
            self._db.execute(
                "UPDATE robot_execution_gate SET state='stop_pending', control_id=?, reason=?, version=version+1, updated_at=? WHERE robot_id=?",
                (control_id, reason, time.time(), robot_id),
            )
            self._db.execute(
                "UPDATE goals SET termination_reason=?, version=version+1 WHERE goal_id=? AND termination_reason IS NULL",
                (reason, goal_id),
            )
            return True

    def cas_control_status(
        self,
        control_id: str,
        expected: str,
        status: str,
        terminal_hash: str | None = None,
    ) -> bool:
        with self._db:
            cur = self._db.execute(
                "UPDATE control_commands SET status=?, terminal_hash=COALESCE(?, terminal_hash) WHERE control_id=? AND status=?",
                (status, terminal_hash, control_id, expected),
            )
            return cur.rowcount == 1

    def control(self, control_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT * FROM control_commands WHERE control_id=?", (control_id,)
        ).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["payload"] = json.loads(value["payload"])
        return value

    def terminal_control(
        self, *, control_id: str, status: str, result_hash: str, idle_confirmed: bool
    ) -> str | None:
        with self._db:
            control = self._db.execute(
                "SELECT robot_id, goal_id, target_skill_id, status, terminal_hash FROM control_commands WHERE control_id=?",
                (control_id,),
            ).fetchone()
            if control is None:
                return None
            if control[4] is not None:
                return (
                    str(control[1])
                    if control[4] == result_hash and control[1] is not None
                    else None
                    if control[4] == result_hash
                    else "conflict"
                )
            self._db.execute(
                "UPDATE control_commands SET status=?, terminal_hash=? WHERE control_id=?",
                (status, result_hash, control_id),
            )
            if status == "completed" and idle_confirmed:
                if control[2]:
                    self._db.execute(
                        "UPDATE actions SET status='reconciled_idle', version=version+1 "
                        "WHERE skill_id=? AND status NOT IN ('completed','failed','interrupted','unknown','reconciled_idle')",
                        (control[2],),
                    )
                self._db.execute(
                    "UPDATE robot_execution_gate SET state='ready', control_id=NULL, reason=NULL, version=version+1, updated_at=? WHERE robot_id=?",
                    (time.time(), control[0]),
                )
                if control[1]:
                    goal_status = (
                        "cancelled"
                        if self._db.execute(
                            "SELECT termination_reason FROM goals WHERE goal_id=?",
                            (control[1],),
                        ).fetchone()[0]
                        == "cancel"
                        else "failed"
                    )
                    self._db.execute(
                        "UPDATE goals SET status=?, version=version+1 WHERE goal_id=? AND status NOT IN ('completed','failed','cancelled')",
                        (goal_status, control[1]),
                    )
            else:
                self._db.execute(
                    "UPDATE robot_execution_gate SET state='uncertain', version=version+1, updated_at=? WHERE robot_id=?",
                    (time.time(), control[0]),
                )
                if control[1]:
                    self._db.execute(
                        "UPDATE goals SET status='blocked', version=version+1 WHERE goal_id=? AND status NOT IN ('completed','failed','cancelled')",
                        (control[1],),
                    )
            return str(control[1]) if control[1] is not None else None

    def reconcile_unknown_action(
        self,
        *,
        reconcile_id: str,
        robot_id: str,
        skill_id: str,
        operator_id: str,
        status_payload: dict[str, Any],
    ) -> bool:
        """Explicitly release an UNKNOWN execution lock after authoritative idle proof."""
        with self._db:
            action = self._db.execute(
                "SELECT goal_id, status FROM actions WHERE skill_id=?", (skill_id,)
            ).fetchone()
            gate = self._db.execute(
                "SELECT state FROM robot_execution_gate WHERE robot_id=?", (robot_id,)
            ).fetchone()
            goal = (
                None
                if action is None
                else self._db.execute(
                    "SELECT robot_id FROM goals WHERE goal_id=?", (action[0],)
                ).fetchone()
            )
            if (
                action is None
                or action[1] != "unknown"
                or gate is None
                or gate[0] != "uncertain"
                or goal is None
                or goal[0] != robot_id
            ):
                return False
            try:
                self._db.execute(
                    "INSERT INTO reconcile_records VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        reconcile_id,
                        robot_id,
                        skill_id,
                        operator_id,
                        _json(status_payload),
                        time.time(),
                    ),
                )
            except sqlite3.IntegrityError:
                return False
            self._db.execute(
                "UPDATE actions SET status='reconciled_idle', version=version+1 WHERE skill_id=?",
                (skill_id,),
            )
            self._db.execute(
                "UPDATE robot_execution_gate SET state='ready', control_id=NULL, reason=NULL, version=version+1, updated_at=? WHERE robot_id=?",
                (time.time(), robot_id),
            )
            return True

    def action(self, skill_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT * FROM actions WHERE skill_id=?", (skill_id,)
        ).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["payload"] = json.loads(value["payload"])
        return value

    def active_action_for_goal(self, goal_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT skill_id FROM actions WHERE goal_id=? AND status IN "
            "('persisted','publishing','published','accepted','running') "
            "ORDER BY created_at DESC LIMIT 1",
            (goal_id,),
        ).fetchone()
        return None if row is None else self.action(str(row[0]))

    def terminal_skill_result(
        self,
        *,
        skill_id: str,
        status: str,
        result_hash: str,
        evidence: list[dict[str, Any]],
        result: dict[str, Any] | None = None,
    ) -> str | None:
        """Atomically record terminal action/evidence and reactivate its waiting goal."""
        with self._db:
            action = self._db.execute(
                "SELECT goal_id, status, terminal_hash, payload FROM actions WHERE skill_id=?",
                (skill_id,),
            ).fetchone()
            if action is None:
                return None
            if action[2] is not None:
                return action[0] if action[2] == result_hash else "conflict"
            payload = json.loads(action[3])
            if result is not None:
                payload["result"] = result
            self._db.execute(
                "UPDATE actions SET status=?, payload=?, terminal_hash=?, version=version+1 WHERE skill_id=?",
                (status, _json(payload), result_hash, skill_id),
            )
            for fact in evidence:
                self._db.execute(
                    "INSERT OR IGNORE INTO evidence VALUES (?, ?, ?)",
                    (fact["evidence_id"], action[0], _json(fact)),
                )
            if status == "completed":
                self._db.execute(
                    "UPDATE goals SET status='waiting', version=version+1 WHERE goal_id=? AND status='waiting'",
                    (action[0],),
                )
            elif status in {"failed", "interrupted"}:
                self._db.execute(
                    "UPDATE goals SET status='failed', version=version+1 WHERE goal_id=? AND status='waiting'",
                    (action[0],),
                )
            else:
                self._db.execute(
                    "UPDATE goals SET status='blocked', version=version+1 WHERE goal_id=? AND status='waiting'",
                    (action[0],),
                )
            return str(action[0])

    def mark_skill_result_conflict(self, skill_id: str) -> str | None:
        """A conflicting terminal payload makes physical execution uncertain."""
        with self._db:
            row = self._db.execute(
                "SELECT actions.goal_id, goals.robot_id FROM actions JOIN goals ON goals.goal_id=actions.goal_id WHERE actions.skill_id=?",
                (skill_id,),
            ).fetchone()
            if row is None:
                return None
            self._db.execute(
                "UPDATE actions SET status='unknown', version=version+1 WHERE skill_id=?",
                (skill_id,),
            )
            self._db.execute(
                "UPDATE robot_execution_gate SET state='uncertain', reason='IDEMPOTENCY_CONFLICT', version=version+1, updated_at=? WHERE robot_id=?",
                (time.time(), row[1]),
            )
            self._db.execute(
                "UPDATE goals SET status='blocked', version=version+1 WHERE goal_id=? AND status NOT IN ('completed','failed','cancelled')",
                (row[0],),
            )
            return str(row[0])

    def evidence_for_goal(self, goal_id: str) -> list[dict[str, Any]]:
        return [
            json.loads(row[0])
            for row in self._db.execute(
                "SELECT payload FROM evidence WHERE goal_id=? ORDER BY evidence_id",
                (goal_id,),
            ).fetchall()
        ]

    def append_evidence(self, goal_id: str, evidence: list[dict[str, Any]]) -> None:
        with self._db:
            for fact in evidence:
                self._db.execute(
                    "INSERT OR IGNORE INTO evidence VALUES (?, ?, ?)",
                    (fact["evidence_id"], goal_id, _json(fact)),
                )

    def cas_action_status(
        self,
        *,
        skill_id: str,
        expected: str,
        status: str,
        terminal_hash: str | None = None,
    ) -> bool:
        with self._db:
            cur = self._db.execute(
                "UPDATE actions SET status=?, version=version+1, terminal_hash=COALESCE(?, terminal_hash) WHERE skill_id=? AND status=?",
                (status, terminal_hash, skill_id, expected),
            )
            return cur.rowcount == 1

    def mark_action_unknown(self, *, skill_id: str, expected: str, reason: str) -> bool:
        """Atomically retain the execution lock when physical dispatch is uncertain."""
        with self._db:
            row = self._db.execute(
                "SELECT actions.goal_id, goals.robot_id FROM actions "
                "JOIN goals ON goals.goal_id=actions.goal_id "
                "WHERE actions.skill_id=? AND actions.status=?",
                (skill_id, expected),
            ).fetchone()
            if row is None:
                return False
            self._db.execute(
                "UPDATE actions SET status='unknown', version=version+1 "
                "WHERE skill_id=? AND status=?",
                (skill_id, expected),
            )
            self._db.execute(
                "UPDATE robot_execution_gate SET state='uncertain', reason=?, "
                "version=version+1, updated_at=? WHERE robot_id=?",
                (reason, time.time(), row[1]),
            )
            self._db.execute(
                "UPDATE goals SET status='blocked', version=version+1 "
                "WHERE goal_id=? AND status NOT IN ('completed','failed','cancelled')",
                (row[0],),
            )
            return True

    def recover_publishing(self) -> list[str]:
        with self._db:
            rows = self._db.execute(
                "SELECT actions.skill_id, actions.goal_id, goals.robot_id FROM actions JOIN goals ON goals.goal_id=actions.goal_id WHERE actions.status='publishing'"
            ).fetchall()
            self._db.execute(
                "UPDATE actions SET status='unknown', version=version+1 WHERE status='publishing'"
            )
            for row in rows:
                self._db.execute(
                    "UPDATE robot_execution_gate SET state='uncertain', reason='DISPATCH_INTERRUPTED', version=version+1, updated_at=? WHERE robot_id=?",
                    (time.time(), row[2]),
                )
                self._db.execute(
                    "UPDATE goals SET status='blocked', version=version+1 WHERE goal_id=? AND status NOT IN ('completed','failed','cancelled')",
                    (row[1],),
                )
            return [str(row[0]) for row in rows]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
