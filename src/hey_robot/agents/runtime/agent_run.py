from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class AgentRunRecorder:
    """Append-only transcript for one agent execution run."""

    def __init__(
        self, log_dir: str | Path = "logs", agent_run_id: str | None = None
    ) -> None:
        self.log_dir = Path(log_dir)
        self.agent_run_id = agent_run_id or str(int(time.time()))
        self.agent_run_dir = self.log_dir / "agent_runs" / self.agent_run_id

    def record_transcript(self, role: str, content: str, **metadata: Any) -> None:
        self._append_jsonl(
            "transcript.jsonl",
            {
                "timestamp": time.time(),
                "role": role,
                "content": content,
                "metadata": metadata,
            },
        )

    def record_decision(
        self,
        *,
        task: str,
        robot_state: str,
        decision: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        self._append_jsonl(
            "agent_steps.jsonl",
            {
                "timestamp": time.time(),
                "task": task,
                "robot_state": robot_state,
                "decision": decision,
                "result": result,
            },
        )

    def record_task_contract(self, contract: dict[str, Any]) -> None:
        self._append_jsonl(
            "task_contracts.jsonl",
            {
                "timestamp": time.time(),
                "contract": contract,
            },
        )

    def record_task_evidence(self, ledger: dict[str, Any]) -> None:
        self._append_jsonl(
            "task_evidence.jsonl",
            {
                "timestamp": time.time(),
                "ledger": ledger,
            },
        )

    def record_task_evaluation(
        self,
        *,
        contract: dict[str, Any],
        ledger: dict[str, Any],
        evaluation: dict[str, Any],
        final_candidate: str | None = None,
    ) -> None:
        self._append_jsonl(
            "task_evaluations.jsonl",
            {
                "timestamp": time.time(),
                "contract": contract,
                "ledger": ledger,
                "evaluation": evaluation,
                "final_candidate": final_candidate,
            },
        )

    def record_turn_trace(self, trace: dict[str, Any]) -> None:
        self._append_jsonl(
            "turn_traces.jsonl",
            {
                "timestamp": time.time(),
                "trace": trace,
            },
        )

    def _append_jsonl(self, filename: str, payload: dict[str, Any]) -> None:
        self.agent_run_dir.mkdir(parents=True, exist_ok=True)
        path = self.agent_run_dir / filename
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class AgentRunReader:
    """Read artifacts written by AgentRunRecorder and related runtime stores."""

    def __init__(
        self, log_dir: str | Path = "logs", agent_run_id: str | None = None
    ) -> None:
        self.log_dir = Path(log_dir)
        self.agent_run_id = agent_run_id

    def list_agent_runs(self) -> list[str]:
        agent_runs_dir = self.log_dir / "agent_runs"
        if not agent_runs_dir.exists():
            return []
        return sorted(item.name for item in agent_runs_dir.iterdir() if item.is_dir())

    def read_jsonl(
        self, filename: str, *, agent_run_id: str | None = None
    ) -> list[dict[str, Any]]:
        path = self._agent_run_dir(agent_run_id) / filename
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
        return records

    def latest_agent_step(
        self, *, agent_run_id: str | None = None
    ) -> dict[str, Any] | None:
        records = self.read_jsonl("agent_steps.jsonl", agent_run_id=agent_run_id)
        return records[-1] if records else None

    def recovery_summary(self, *, agent_run_id: str | None = None) -> dict[str, Any]:
        rid = agent_run_id or self.agent_run_id
        if not rid:
            agent_runs = self.list_agent_runs()
            rid = agent_runs[-1] if agent_runs else None
        if not rid:
            return {"agent_run_id": None, "has_agent_run": False}
        return {
            "agent_run_id": rid,
            "has_agent_run": True,
            "latest_agent_step": self.latest_agent_step(agent_run_id=rid),
            "tool_call_count": len(
                self.read_jsonl("tool_calls.jsonl", agent_run_id=rid)
            ),
            "agent_step_count": len(
                self.read_jsonl("agent_steps.jsonl", agent_run_id=rid)
            ),
        }

    def _agent_run_dir(self, agent_run_id: str | None = None) -> Path:
        rid = agent_run_id or self.agent_run_id
        if not rid:
            raise ValueError("agent_run_id is required")
        return self.log_dir / "agent_runs" / rid
