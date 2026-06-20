from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from hey_robot.bus.client import BusClient
from hey_robot.events import RuntimeEvent
from hey_robot.events.bus import EventPublisher
from hey_robot.protocol import SkillEvent, SkillIntent, SkillResult, Topics
from hey_robot.protocol.messages import to_payload
from hey_robot.skills.catalog import RobotSkillSpec
from hey_robot.skills.composition import SkillExecutionPlan
from hey_robot.skills.contracts import SkillContractRuntime
from hey_robot.skills.scheduler import SkillRun


class SkillEventSink:
    """Publishes skill protocol events and persists scheduler diagnostics."""

    def __init__(
        self,
        *,
        bus: BusClient,
        events: EventPublisher,
        topics: Topics,
        contracts: SkillContractRuntime,
        runtime_dir: str | Path,
    ) -> None:
        self.bus = bus
        self.events = events
        self.topics = topics
        self.contracts = contracts
        self.scheduler_root = Path(runtime_dir) / "skill_scheduler"
        self.scheduler_root.mkdir(parents=True, exist_ok=True)

    async def publish_event(
        self,
        intent: SkillIntent,
        phase: str,
        *,
        run: SkillRun | None = None,
        progress: float | None = None,
        summary: str | None = None,
        error: str | None = None,
        policy_id: str | None = None,
        frame_id: int | None = None,
        steps_executed: int | None = None,
        contract: RobotSkillSpec | None = None,
        step: str | None = None,
        execution_plan: SkillExecutionPlan | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event_metadata = self._metadata(intent, contract=contract, run=run)
        event_metadata.update(dict(metadata or {}))
        if execution_plan is not None:
            event_metadata["execution_plan"] = {
                "strategy": execution_plan.strategy,
                "actions": [action.to_dict() for action in execution_plan.actions],
                "notes": list(execution_plan.notes),
            }
        await self.bus.publish(
            self.topics.skill_event,
            to_payload(
                SkillEvent(
                    envelope=intent.envelope,
                    skill_id=intent.skill_id,
                    name=intent.name or (contract.name if contract is not None else ""),
                    phase=phase,
                    step=step,
                    policy_id=policy_id,
                    frame_id=frame_id,
                    steps_executed=steps_executed,
                    progress=progress,
                    summary=summary,
                    error=error,
                    metadata=event_metadata,
                )
            ),
        )

    async def publish_result(
        self,
        intent: SkillIntent,
        status: str,
        success: bool,
        summary: str,
        *,
        run: SkillRun | None = None,
        frame_id: int | None = None,
        error: str | None = None,
        failure_mode: str | None = None,
        steps_executed: int = 0,
        contract: RobotSkillSpec | None = None,
    ) -> None:
        await self.bus.publish(
            self.topics.skill_result,
            to_payload(
                SkillResult(
                    envelope=intent.envelope,
                    skill_id=intent.skill_id,
                    name=intent.name or (contract.name if contract is not None else ""),
                    status=status,
                    success=success,
                    steps_executed=steps_executed,
                    progress=1.0 if success else 0.0,
                    summary=summary,
                    failure_mode=failure_mode,
                    frame_id=frame_id,
                    error=error,
                    metadata=self._metadata(intent, contract=contract, run=run),
                )
            ),
        )

    async def publish_scheduler_state(
        self,
        policy_id: str,
        *,
        robot_id: str,
        active_runs: dict[str, SkillRun],
        phase: str,
        intent: SkillIntent,
        contract: RobotSkillSpec | None = None,
        decision: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> dict[str, Any]:
        last_decision = {
            "phase": phase,
            "skill_id": intent.skill_id,
            "skill": intent.name or (contract.name if contract is not None else ""),
            "resources": sorted(
                self.contracts.normalized_resources(
                    contract, arguments=intent.arguments
                )
            )
            if contract is not None
            else [],
            **(decision or {}),
        }
        snapshot = self.scheduler_snapshot(
            policy_id,
            robot_id=robot_id,
            active_runs=active_runs,
            last_decision=last_decision,
        )
        self._write_scheduler_snapshot(robot_id, snapshot)
        await self.events.publish(
            RuntimeEvent.make(
                "skill_scheduler.state",
                source="skill-controller",
                severity=severity,
                trace_id=intent.envelope.trace_id,
                episode_id=intent.envelope.episode_id,
                agent_id=intent.envelope.agent_id,
                robot_id=intent.envelope.robot_id,
                payload=snapshot,
            )
        )
        return last_decision

    def scheduler_snapshot(
        self,
        policy_id: str,
        *,
        robot_id: str,
        active_runs: dict[str, SkillRun],
        last_decision: dict[str, Any] | None,
    ) -> dict[str, Any]:
        runs = [
            self._run_snapshot(run)
            for run in sorted(active_runs.values(), key=lambda item: item.accepted_at)
            if not run.terminal
        ]
        leases = {
            resource: run["skill_id"] for run in runs for resource in run["resources"]
        }
        return {
            "robot_id": robot_id,
            "policy_id": policy_id,
            "updated_at": time.time(),
            "active_runs": runs,
            "resource_leases": leases,
            "last_decision": last_decision,
        }

    def _run_snapshot(self, run: SkillRun) -> dict[str, Any]:
        return {
            "skill_id": run.intent.skill_id,
            "skill": run.skill_name,
            "objective": run.intent.objective,
            "strategy": run.execution_plan.strategy,
            "action_count": len(run.execution_plan.actions),
            "current_action": run.current_step,
            "resources": sorted(
                self.contracts.normalized_resources(
                    run.contract, arguments=run.intent.arguments
                )
            ),
            "safety_level": run.contract.safety_level,
            "implementation_name": run.implementation_name,
            "implementation_kind": run.implementation_kind,
            "accepted_at": run.accepted_at,
            "started_at": run.started_at,
            "age_sec": max(0.0, time.time() - run.accepted_at),
            "timeout_sec": run.timeout_sec,
            "steps_executed": run.steps_executed,
            "terminal": run.terminal,
        }

    @staticmethod
    def _metadata(
        intent: SkillIntent,
        *,
        contract: RobotSkillSpec | None,
        run: SkillRun | None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {"skill": intent.name or ""}
        if contract is not None:
            metadata["contract"] = contract.to_dict()
        if run is not None:
            metadata["implementation_name"] = run.implementation_name
            metadata["implementation_kind"] = run.implementation_kind
        return metadata

    def _write_scheduler_snapshot(
        self, robot_id: str, snapshot: dict[str, Any]
    ) -> None:
        path = self.scheduler_root / f"{robot_id}.json"
        tmp = self.scheduler_root / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.replace(path)
