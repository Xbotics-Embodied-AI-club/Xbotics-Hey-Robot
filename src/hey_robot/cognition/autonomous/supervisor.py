"""The sole autonomous owner allowed to turn a proposal into SkillIntent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from contextlib import suppress
from pathlib import Path

from hey_robot.bus.factory import create_bus_client
from hey_robot.cognition.autonomous.policy import check_budget, dispatch_admission
from hey_robot.cognition.autonomous.skill_gateway import SkillGateway
from hey_robot.cognition.autonomous.store import AutonomyStore
from hey_robot.cognition.runtime.trace import RunTraceWriter
from hey_robot.cognition.task.contract import create_task_contract
from hey_robot.cognition.task.evidence import project_robot_status
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import (
    ActionProposal,
    ActionSnapshot,
    BudgetState,
    DeliberationRequest,
    DeliberationResult,
    Envelope,
    EvidenceFact,
    GoalBudgets,
    GoalCommand,
    GoalEvent,
    GoalSnapshot,
    RobotObservation,
    RobotStatus,
    SkillControl,
    SkillControlResult,
    SkillIntent,
    SkillResult,
    Topics,
)
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.skill_os.registry import registry_from_config


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
        skill_catalog = registry_from_config(config).catalog(semantic_only=False)
        self.gateway = SkillGateway(skill_catalog)
        self._watchdog_interval: float = 5.0
        self._deliberation_timeout: float = 300.0
        self._heartbeat_timeout: float = 60.0
        self._latest_status: dict[str, RobotStatus] = {}
        self._latest_observation: dict[str, RobotObservation] = {}
        self._watchdog_task: asyncio.Task | None = None

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
        self._watchdog_task = asyncio.ensure_future(self._watchdog_loop())
        await asyncio.Event().wait()

    async def stop(self) -> None:
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            self._watchdog_task = None
        await self.bus.close()

    async def _on_goal_command(self, _topic: str, payload: dict) -> None:
        command = from_payload(GoalCommand, payload)
        self.trace.write(
            "goal.command",
            details={"command_id": command.command_id, "action": command.action},
        )
        if self.store.command_result(command.command_id) is not None:
            return
        if command.action == "emergency_stop":
            robot_id = command.envelope.robot_id
            if robot_id:
                await self._emergency_stop(
                    robot_id, command.command_id, command.envelope
                )
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
        if gate.state != "ready":
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
        observation = self._latest_observation.get(robot_id)
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
        actions = tuple(
            ActionSnapshot(
                skill_id=a["payload"]["skill_id"],
                deliberation_id=a["payload"]["deliberation_id"],
                intent_kind=a["payload"]["intent_kind"],
                name=a["payload"]["name"],
                objective=a["payload"]["objective"],
                arguments=a["payload"]["arguments"],
                status=a["status"],
            )
            for a in self.store.actions_for_goal(goal.goal_id)
        )
        latest_skill_result = self._latest_skill_result_for_goal(goal.goal_id)
        active_snapshot = GoalSnapshot(
            goal.goal_id,
            goal.version + 1,
            goal.task_id,
            goal.contract_id,
            goal.contract_hash,
            goal.objective,
            goal.success_criteria,
            "active",
        )
        request = DeliberationRequest(
            envelope.child(robot_id=envelope.robot_id),
            deliberation_id,
            trigger,
            active_snapshot,
            status,
            observation,
            actions,
            evidence,
            latest_skill_result,
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

    def _latest_skill_result_for_goal(self, goal_id: str) -> SkillResult | None:
        """Return the most recent terminal SkillResult for the goal, if any."""
        actions = self.store.actions_for_goal(goal_id)
        for action in reversed(actions):
            payload = action.get("payload") or {}
            result_data = payload.get("result")
            if result_data and isinstance(result_data, dict):
                return from_payload(SkillResult, result_data)
        return None

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
        await self._interrupt_action(goal, active, command_id, envelope, "cancel")

    async def _emergency_stop(
        self, robot_id: str, command_id: str, envelope: Envelope
    ) -> None:
        goal = self.store.active_goal_for_robot(robot_id)
        if goal is None:
            return
        gate = self.store.gate(robot_id)
        if gate.state == "stop_pending":
            return
        active = self._active_action_for_goal(goal["goal_id"])
        if active is None:
            self.store.fail_goal(goal["goal_id"], "EMERGENCY_STOPPED")
            return
        await self._interrupt_action(goal, active, command_id, envelope, "emergency")

    async def _interrupt_action(
        self,
        goal: dict,
        active: dict,
        command_id: str,
        envelope: Envelope,
        reason: str,
    ) -> None:
        control_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{command_id}:{active['skill_id']}")
        )
        control = SkillControl(
            envelope,
            control_id,
            "emergency_stop" if reason == "emergency" else "interrupt",
            active["skill_id"],
            goal["goal_id"],
            reason,
        )
        if not self.store.begin_stop(
            goal_id=goal["goal_id"],
            robot_id=goal["robot_id"],
            control_id=control_id,
            target_skill_id=active["skill_id"],
            action="interrupt",
            reason=reason,
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
            or result.task_id != goal["snapshot"]["task_id"]
            or result.request_hash
            != self.store.deliberation_request_hash(
                goal_id=result.goal_id, deliberation_id=result.deliberation_id
            )
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
        robot_id = goal["robot_id"]
        gate = self.store.gate(robot_id)
        status = self._latest_status.get(robot_id)
        admission = dispatch_admission(gate=gate, status=status)
        if not admission.allowed:
            self.store.fail_goal(goal["goal_id"], admission.code or "STATE_UNKNOWN")
            return
        validation = self.gateway.validate(
            skill_name=proposal.skill_name,
            objective=proposal.objective,
            arguments=proposal.arguments,
            intent_kind=proposal.intent_kind,
            gate=gate,
            status=status,
        )
        if not validation.allowed:
            self.store.fail_goal(goal["goal_id"], validation.code or "SAFETY_REJECTED")
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
            self.store.mark_action_unknown(
                skill_id=skill_id,
                expected="publishing",
                reason="DISPATCH_INTERRUPTED",
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
        intent = action["payload"]
        envelope_data = intent.get("envelope", {})
        expected_robot_id = envelope_data.get("robot_id")
        if expected_robot_id and result.envelope.robot_id != expected_robot_id:
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
            if self._valid_evidence(
                fact,
                robot_id=result.envelope.robot_id,
                goal_id=action["goal_id"],
                source_kind="skill_result",
                source_id=result.skill_id,
            )
        ]
        goal_id = self.store.terminal_skill_result(
            skill_id=result.skill_id,
            status=result.status,
            result_hash=result_hash,
            result=to_payload(result),
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
        if _topic == self.topics.robot_observation:
            observation = from_payload(RobotObservation, payload)
            robot_id = observation.envelope.robot_id
            if robot_id:
                self._latest_observation[robot_id] = observation
            return
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
                if self._valid_evidence(
                    fact,
                    robot_id=robot_id,
                    goal_id=goal["goal_id"],
                    source_kind="robot_status",
                    source_id=f"status:{robot_id}:{status.frame_id}",
                )
            ],
        )

    def _valid_evidence(
        self,
        fact: EvidenceFact,
        *,
        robot_id: str | None,
        goal_id: str,
        source_kind: str,
        source_id: str,
    ) -> bool:
        catalog = set(self.config.autonomy.entity_catalog)
        if robot_id:
            catalog.add(f"robot:{robot_id}")
        return bool(
            fact.goal_id == goal_id
            and fact.source_kind == source_kind
            and fact.source_id == source_id
            and fact.subject_id in catalog
            and fact.object_id in catalog
        )

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
            active_control_id=gate.control_id,
            termination_reason=goal["termination_reason"],
            evaluation=evaluation,
        )
        await self.bus.publish(self.topics.goal_event, to_payload(event))

    async def _watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(self._watchdog_interval)
            self._watchdog_tick()

    def _watchdog_tick(self) -> None:
        """Check deadlines, heartbeat, and budget for all non-terminal goals.

        Does NOT wake models, republish messages, or select recovery actions.
        """
        now = time.time()
        seen_robots: set[str] = set()
        for goal in self.store.goals_recent(limit=100):
            goal_id = goal["goal_id"]
            robot_id = goal["robot_id"]
            status = goal["status"]
            if status in {"completed", "failed", "cancelled"}:
                continue
            seen_robots.add(robot_id)

            # Check deliberation timeout
            active_delib = goal.get("active_deliberation_id")
            if active_delib and status == "active":
                latest = self._latest_status.get(robot_id)
                budget_state = self.store.budget_state(
                    goal_id,
                    latest.battery_percentage if latest else None,
                )
                if budget_state is not None:
                    raw_budgets, raw_state = budget_state
                    budget_state_obj = BudgetState(**raw_state)
                    hard_wall = self.config.autonomy.hard_max_wall_time_sec
                    goal_wall = GoalBudgets(**raw_budgets).max_wall_time_sec
                    if budget_state_obj.elapsed_wall_time_sec >= min(
                        hard_wall, goal_wall
                    ):
                        self.store.fail_goal(goal_id, "BUDGET_EXHAUSTED")
                        self.trace.write("watchdog.budget_exhausted", goal_id=goal_id)
                        continue
                    if budget_state_obj.deliberations_used >= min(
                        GoalBudgets(**raw_budgets).max_deliberations,
                        self.config.autonomy.hard_max_deliberations,
                    ):
                        self.store.fail_goal(goal_id, "BUDGET_EXHAUSTED")
                        self.trace.write("watchdog.budget_exhausted", goal_id=goal_id)
                        continue

            # Check skill wait timeout for WAITING goals
            if status == "waiting":
                active_action = self.store.active_action_for_goal(goal_id)
                if active_action is not None:
                    latest = self._latest_status.get(robot_id)
                    budget = self.store.budget_state(
                        goal_id, latest.battery_percentage if latest else None
                    )
                    if budget is not None:
                        raw_budgets, raw_state = budget
                        decision = check_budget(
                            budgets=GoalBudgets(**raw_budgets),
                            hard_wall_time_sec=self.config.autonomy.hard_max_wall_time_sec,
                            hard_deliberations=self.config.autonomy.hard_max_deliberations,
                            hard_skills=self.config.autonomy.hard_max_skills,
                            minimum_battery=self.config.autonomy.min_battery_percentage,
                            state=BudgetState(**raw_state),
                        )
                        if decision.code == "BUDGET_EXHAUSTED":
                            with suppress(RuntimeError):
                                asyncio.get_running_loop().create_task(
                                    self._interrupt_action(
                                        goal,
                                        active_action,
                                        f"budget:{goal_id}",
                                        Envelope(robot_id=robot_id),
                                        "budget",
                                    )
                                )
                            continue
                    created = active_action.get("created_at", 0)
                    if now - created > self._deliberation_timeout:
                        self.store.mark_action_unknown(
                            skill_id=active_action["skill_id"],
                            expected=active_action["status"],
                            reason="SKILL_TIMEOUT",
                        )
                        self.trace.write(
                            "watchdog.skill_timeout",
                            goal_id=goal_id,
                            details={"skill_id": active_action["skill_id"]},
                        )

        # Check robot heartbeat for robots with active goals
        for robot_id in seen_robots:
            status = self._latest_status.get(robot_id)
            if (
                status is not None
                and status.envelope.timestamp
                and now - status.envelope.timestamp > self._heartbeat_timeout
            ):
                self._mark_robot_offline(robot_id)

    def _mark_robot_offline(self, robot_id: str) -> None:
        goal = self.store.active_goal_for_robot(robot_id)
        if goal is None:
            return
        if goal["status"] == "blocked":
            return
        active = self.store.active_action_for_goal(goal["goal_id"])
        if active is not None:
            self.store.mark_action_unknown(
                skill_id=active["skill_id"],
                expected=active["status"],
                reason="ROBOT_OFFLINE",
            )
        else:
            self.store.fail_goal(goal["goal_id"], "ROBOT_OFFLINE")
        self.trace.write(
            "watchdog.robot_offline",
            goal_id=goal["goal_id"],
            details={"robot_id": robot_id},
        )


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
