"""The sole autonomous owner allowed to turn a proposal into SkillIntent."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from hey_robot.bus.factory import create_bus_client
from hey_robot.cognition.autonomous.policy import check_budget, dispatch_admission
from hey_robot.cognition.autonomous.store import AutonomyStore
from hey_robot.cognition.runtime.trace import RunTraceWriter
from hey_robot.cognition.task.contract import create_task_contract
from hey_robot.cognition.task.evidence import project_robot_status
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import (
    ActionProposal,
    BudgetState,
    DeliberationRequest,
    DeliberationResult,
    Envelope,
    EvidenceFact,
    GoalBudgets,
    GoalCommand,
    GoalEvent,
    GoalSnapshot,
    RobotStatus,
    SkillControl,
    SkillControlResult,
    SkillIntent,
    SkillResult,
    Topics,
)
from hey_robot.protocol.messages import from_payload, to_payload


class AutonomySupervisorService:
    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.topics = Topics()
        self.bus = create_bus_client(config.deployment.bus, role="autonomy_supervisor")
        path = (
            Path(config.resources.runtime_dir)
            / config.deployment.id
            / "autonomy.sqlite3"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.store = AutonomyStore(path)
        self.trace = RunTraceWriter(path.with_name("autonomy.trace.jsonl"))
        self._latest_status: dict[str, RobotStatus] = {}

    async def start(self) -> None:
        interrupted = self.store.recover_publishing()
        for skill_id in interrupted:
            self.trace.write(
                "action.unknown",
                details={"skill_id": skill_id, "reason": "DISPATCH_INTERRUPTED"},
            )
        await self.bus.connect()
        await self.bus.subscribe([self.topics.goal_command], self._on_goal_command)
        await self.bus.subscribe(
            [self.topics.agent_deliberation_result], self._on_deliberation_result
        )
        await self.bus.subscribe([self.topics.skill_result], self._on_skill_result)
        await self.bus.subscribe(
            [self.topics.skill_control_result], self._on_control_result
        )
        await self.bus.subscribe(
            [self.topics.robot_status, self.topics.robot_observation], self._on_snapshot
        )
        await __import__("asyncio").Event().wait()

    async def stop(self) -> None:
        await self.bus.close()

    async def _on_goal_command(self, _topic: str, payload: dict) -> None:
        command = from_payload(GoalCommand, payload)
        self.trace.write(
            "goal.command",
            details={"command_id": command.command_id, "action": command.action},
        )
        if self.store.command_result(command.command_id) is not None:
            return
        if command.action == "cancel":
            if command.goal_id:
                await self._cancel(
                    command.goal_id, command.command_id, command.envelope
                )
            return
        robot_id = command.envelope.robot_id
        if not robot_id or not command.objective or not command.success_criteria:
            return
        if not self._valid_contract(command, robot_id):
            return
        gate = self.store.gate(robot_id)
        if gate["state"] != "ready":
            return
        goal_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        contract_id = str(uuid.uuid4())
        contract = create_task_contract(
            goal_id=goal_id,
            task_id=task_id,
            contract_id=contract_id,
            objective=command.objective,
            success_criteria=command.success_criteria,
        )
        snapshot = GoalSnapshot(
            goal_id,
            0,
            task_id,
            contract_id,
            contract.contract_hash,
            command.objective,
            command.success_criteria,
            "pending",
        )
        if self.store.create_goal(
            command_id=command.command_id,
            goal_id=goal_id,
            robot_id=robot_id,
            snapshot=to_payload(snapshot),
            budgets=to_payload(command.budgets),
        ):
            self.trace.write("goal.created", goal_id=goal_id)
            await self._schedule(snapshot, command.envelope, command.command_id)
            await self._publish_goal_event(goal_id, command.envelope)

    def _valid_contract(self, command: GoalCommand, robot_id: str) -> bool:
        catalog = set(self.config.autonomy.entity_catalog)
        catalog.add(f"robot:{robot_id}")
        if not catalog:
            return False
        for criterion in command.success_criteria:
            if criterion.subject_id not in catalog:
                return False
            if criterion.object_id not in catalog and criterion.object_id != "scene":
                return False
        return True

    async def _schedule(
        self, goal: GoalSnapshot, envelope: Envelope, trigger: str
    ) -> None:
        deliberation_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{self.config.deployment.id}:{goal.goal_id}:{trigger}",
            )
        )
        evidence = tuple(
            from_payload(EvidenceFact, item)
            for item in self.store.evidence_for_goal(goal.goal_id)
        )
        robot_id = envelope.robot_id or ""
        status = self._latest_status.get(robot_id)
        state = self.store.budget_state(
            goal.goal_id, status.battery_percentage if status else None
        )
        if state is None:
            self.store.fail_goal(goal.goal_id, "PERSISTENCE_FAILED")
            return
        raw_budgets, raw_state = state
        budget_state = BudgetState(**raw_state)
        decision = check_budget(
            budgets=GoalBudgets(**raw_budgets),
            hard_wall_time_sec=self.config.autonomy.hard_max_wall_time_sec,
            hard_deliberations=self.config.autonomy.hard_max_deliberations,
            hard_skills=self.config.autonomy.hard_max_skills,
            minimum_battery=self.config.autonomy.min_battery_percentage,
            state=BudgetState(
                budget_state.elapsed_wall_time_sec,
                budget_state.deliberations_used,
                budget_state.skills_used,
                budget_state.battery_percentage
                if budget_state.battery_percentage is not None
                else 100.0,
            ),
        )
        if not decision.allowed:
            self.store.fail_goal(goal.goal_id, decision.code or "BUDGET_EXHAUSTED")
            return
        request = DeliberationRequest(
            envelope.child(robot_id=envelope.robot_id),
            deliberation_id,
            trigger,
            goal,
            status,
            None,
            (),
            evidence,
            None,
            budget_state,
        )
        raw = to_payload(request)
        request_hash = _hash(raw)
        if not self.store.schedule_deliberation(
            goal_id=goal.goal_id,
            trigger_event_id=trigger,
            deliberation_id=deliberation_id,
            request=raw,
            request_hash=request_hash,
        ):
            return
        self.trace.write(
            "deliberation.scheduled",
            goal_id=goal.goal_id,
            details={"deliberation_id": deliberation_id, "trigger": trigger},
        )
        try:
            await self.bus.publish(self.topics.agent_deliberation, raw)
        except Exception:
            self.store.fail_goal(goal.goal_id, "DELIBERATION_DISPATCH_INTERRUPTED")

    async def _cancel(self, goal_id: str, command_id: str, envelope: Envelope) -> None:
        goal = self.store.goal(goal_id)
        if goal is None:
            return
        if goal["status"] in {"pending", "active"}:
            self.store.cancel_goal(goal_id)
            self.trace.write("goal.cancelled", goal_id=goal_id)
            return
        active = self._active_action_for_goal(goal_id)
        if active is None:
            self.store.cancel_goal(goal_id)
            return
        control_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{command_id}:{active['skill_id']}")
        )
        control = SkillControl(
            envelope, control_id, "interrupt", active["skill_id"], goal_id, "cancel"
        )
        if not self.store.begin_stop(
            goal_id=goal_id,
            robot_id=goal["robot_id"],
            control_id=control_id,
            target_skill_id=active["skill_id"],
            action="interrupt",
            reason="cancel",
            payload=to_payload(control),
        ):
            return
        if not self.store.cas_control_status(control_id, "persisted", "publishing"):
            return
        try:
            await self.bus.publish(self.topics.skill_control, to_payload(control))
        except Exception:
            self.store.cas_control_status(control_id, "publishing", "unknown")
            return
        self.store.cas_control_status(control_id, "publishing", "published")

    def _active_action_for_goal(self, goal_id: str) -> dict | None:
        return self.store.active_action_for_goal(goal_id)

    async def _on_deliberation_result(self, _topic: str, payload: dict) -> None:
        result = from_payload(DeliberationResult, payload)
        self.trace.write(
            "deliberation.result",
            goal_id=result.goal_id,
            details={
                "deliberation_id": result.deliberation_id,
                "status": result.status,
            },
        )
        goal = self.store.goal(result.goal_id)
        if (
            goal is None
            or goal["status"] != "active"
            or goal["active_deliberation_id"] != result.deliberation_id
        ):
            return
        if result.status == "completed":
            self.store.complete_goal(result.goal_id)
            await self._publish_goal_event(
                result.goal_id, result.envelope, result.evaluation
            )
            return
        if result.status == "failed":
            self.store.fail_goal(
                result.goal_id,
                result.failure.code if result.failure else "MODEL_STOPPED_BEFORE_GOAL",
            )
            await self._publish_goal_event(result.goal_id, result.envelope)
            return
        assert result.proposal is not None
        await self._dispatch(
            goal, result.proposal, result.deliberation_id, result.envelope
        )

    async def _dispatch(
        self,
        goal: dict,
        proposal: ActionProposal,
        deliberation_id: str,
        envelope: Envelope,
    ) -> None:
        admission = dispatch_admission(
            gate_state=self.store.gate(goal["robot_id"])["state"],
            status=self._latest_status.get(goal["robot_id"]),
        )
        if not admission.allowed:
            self.store.fail_goal(goal["goal_id"], admission.code or "STATE_UNKNOWN")
            return
        skill_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL, f"{self.config.deployment.id}:{deliberation_id}"
            )
        )
        snapshot = goal["snapshot"]
        intent = SkillIntent(
            envelope=envelope,
            skill_id=skill_id,
            goal_id=goal["goal_id"],
            task_id=snapshot["task_id"],
            deliberation_id=deliberation_id,
            intent_kind=proposal.intent_kind,
            name=proposal.skill_name,
            objective=proposal.objective,
            arguments=proposal.arguments,
        )
        if not self.store.accept_action(
            goal_id=goal["goal_id"],
            deliberation_id=deliberation_id,
            skill_id=skill_id,
            payload=to_payload(intent),
        ):
            return
        self.trace.write(
            "intent.persisted", goal_id=goal["goal_id"], details={"skill_id": skill_id}
        )
        if not self.store.cas_action_status(
            skill_id=skill_id, expected="persisted", status="publishing"
        ):
            return
        try:
            await self.bus.publish(self.topics.skill_intent, to_payload(intent))
        except Exception:
            self.store.cas_action_status(
                skill_id=skill_id, expected="publishing", status="unknown"
            )
            return
        self.store.cas_action_status(
            skill_id=skill_id, expected="publishing", status="published"
        )
        await self._publish_goal_event(goal["goal_id"], envelope)

    async def _on_skill_result(self, _topic: str, payload: dict) -> None:
        result = from_payload(SkillResult, payload)
        self.trace.write(
            "skill.result",
            details={"skill_id": result.skill_id, "status": result.status},
        )
        action = self.store.action(result.skill_id)
        if action is None:
            return
        result_hash = _hash(
            {
                "skill_id": result.skill_id,
                "status": result.status,
                "success": result.success,
                "frame_id": result.frame_id,
                "evidence": [item.evidence_id for item in result.evidence],
            }
        )
        evidence = [
            fact
            for fact in result.evidence
            if self._valid_evidence(fact, result.envelope.robot_id)
        ]
        goal_id = self.store.terminal_skill_result(
            skill_id=result.skill_id,
            status=result.status,
            result_hash=result_hash,
            evidence=[to_payload(item) for item in evidence],
        )
        if goal_id == "conflict":
            self.store.mark_skill_result_conflict(result.skill_id)
            return
        if goal_id is None:
            return
        await self._publish_goal_event(goal_id, result.envelope)
        if result.status != "completed":
            return
        goal = self.store.goal(goal_id)
        if (
            goal is None
            or goal["status"] != "waiting"
            or goal["termination_reason"] is not None
        ):
            return
        snapshot = from_payload(GoalSnapshot, goal["snapshot"])
        await self._schedule(
            snapshot, result.envelope, f"{result.skill_id}:{result.status}"
        )

    async def _on_control_result(self, _topic: str, payload: dict) -> None:
        result = from_payload(SkillControlResult, payload)
        digest = _hash(
            {
                "control_id": result.control_id,
                "status": result.status,
                "idle": result.robot_idle_confirmed,
            }
        )
        goal_id = self.store.terminal_control(
            control_id=result.control_id,
            status=result.status,
            result_hash=digest,
            idle_confirmed=result.robot_idle_confirmed,
        )
        if goal_id and goal_id != "conflict":
            await self._publish_goal_event(goal_id, result.envelope)

    async def _on_snapshot(self, _topic: str, payload: dict) -> None:
        if _topic != self.topics.robot_status:
            return
        status = from_payload(RobotStatus, payload)
        robot_id = status.envelope.robot_id
        if not robot_id:
            return
        self._latest_status[robot_id] = status
        goal = self.store.active_goal_for_robot(robot_id)
        if goal is None:
            return
        facts = project_robot_status(goal_id=goal["goal_id"], status=status)
        self.store.append_evidence(
            goal["goal_id"],
            [
                to_payload(fact)
                for fact in facts
                if self._valid_evidence(fact, robot_id)
            ],
        )

    def _valid_evidence(self, fact: EvidenceFact, robot_id: str | None) -> bool:
        catalog = set(self.config.autonomy.entity_catalog)
        if robot_id:
            catalog.add(f"robot:{robot_id}")
        return fact.goal_id and fact.subject_id in catalog and fact.object_id in catalog

    async def _publish_goal_event(
        self,
        goal_id: str,
        envelope: Envelope,
        evaluation=None,
    ) -> None:
        goal = self.store.goal(goal_id)
        if goal is None:
            return
        snapshot = goal["snapshot"]
        active = self.store.active_action_for_goal(goal_id)
        gate = self.store.gate(goal["robot_id"])
        event = GoalEvent(
            envelope=envelope,
            event_id=str(uuid.uuid4()),
            goal_id=goal_id,
            task_id=snapshot["task_id"],
            status=goal["status"],
            active_skill_id=None if active is None else active["skill_id"],
            active_control_id=gate["control_id"],
            termination_reason=goal["termination_reason"],
            evaluation=evaluation,
        )
        await self.bus.publish(self.topics.goal_event, to_payload(event))


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
