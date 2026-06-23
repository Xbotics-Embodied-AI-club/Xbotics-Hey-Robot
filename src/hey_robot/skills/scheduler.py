from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from hey_robot.protocol import RobotStatus, SkillIntent
from hey_robot.skills.catalog import RobotSkillSpec
from hey_robot.skills.composition import SkillExecutionPlan
from hey_robot.skills.contracts import SkillContractRuntime


@dataclass
class SkillRun:
    intent: SkillIntent
    skill_name: str
    implementation_name: str
    implementation_kind: str
    contract: RobotSkillSpec
    execution_plan: SkillExecutionPlan
    timeout_override_sec: float | None = None
    accepted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    action_published_at: float | None = None
    status_received_at: float | None = None
    steps_executed: int = 0
    terminal: bool = False
    step_summaries: list[str] = field(default_factory=list)
    task: asyncio.Task[Any] | None = None
    pending_status: asyncio.Future[RobotStatus] | None = None
    current_step: str | None = None

    @property
    def timeout_sec(self) -> float:
        return float(
            self.intent.timeout_sec
            or self.timeout_override_sec
            or self.contract.timeout_sec
        )

    @property
    def timed_out(self) -> bool:
        return time.time() - self.timeout_base_at > self.timeout_sec

    @property
    def timeout_base_at(self) -> float:
        return self.action_published_at or self.started_at or self.accepted_at


class SkillScheduler:
    """Owns active runs and deterministic resource-conflict decisions."""

    def __init__(self, contracts: SkillContractRuntime) -> None:
        self.contracts = contracts
        self.runs: dict[str, SkillRun] = {}

    def add(self, run: SkillRun) -> None:
        if run.intent.skill_id in self.runs:
            raise ValueError(f"duplicate active skill id: {run.intent.skill_id}")
        self.runs[run.intent.skill_id] = run

    def remove(self, skill_id: str) -> SkillRun | None:
        return self.runs.pop(skill_id, None)

    def conflicting_run(
        self,
        contract: RobotSkillSpec,
        arguments: dict[str, Any],
    ) -> SkillRun | None:
        for run in self.runs.values():
            if not run.terminal and self.contracts.resources_conflict(
                contract,
                run.contract,
                left_arguments=arguments,
                right_arguments=run.intent.arguments,
            ):
                return run
        return None

    def timed_out_runs(self) -> tuple[SkillRun, ...]:
        return tuple(
            run for run in self.runs.values() if not run.terminal and run.timed_out
        )
