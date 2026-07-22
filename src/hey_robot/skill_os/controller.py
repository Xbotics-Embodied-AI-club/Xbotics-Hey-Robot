from __future__ import annotations

import asyncio
import base64
import io
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from PIL import Image

from hey_robot.bus.factory import create_bus_client
from hey_robot.config import DeploymentConfig, PolicySpec
from hey_robot.contracts import SkillContract, SkillContractRuntime
from hey_robot.events import EventKind, RuntimeEvent
from hey_robot.events.bus import BusEventPublisher
from hey_robot.foundation.clients import (
    ModelServiceRegistry,
    ServiceInvocationRequest,
    ServiceInvocationResult,
)
from hey_robot.human_follow import HumanFollowServiceClient
from hey_robot.logging import HeyRobotLogger
from hey_robot.protocol import (
    RobotAction,
    RobotObservation,
    RobotStatus,
    ShortOperationCommand,
    SkillControl,
    SkillControlResult,
    SkillIntent,
    Topics,
)
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.robot_runtime.identity import resolve_robot_family
from hey_robot.robot_runtime.media import LocalMediaStore, MediaResolver
from hey_robot.robot_runtime.observations.frame_stream import decode_frame_packet
from hey_robot.skill_os.actions import RobotSkillAction
from hey_robot.skill_os.command_store import SkillCommandStore, canonical_payload_hash
from hey_robot.skill_os.composition import SkillExecutionPlan
from hey_robot.skill_os.context import SkillContext
from hey_robot.skill_os.event_sink import SkillEventSink
from hey_robot.skill_os.ports import ModelServicePort, PerceptionPort, RobotActionPort
from hey_robot.skill_os.registry import registry_from_config
from hey_robot.skill_os.runtime import SkillInvoke, SkillRuntime
from hey_robot.skill_os.scheduler import SkillRun, SkillScheduler

logger = HeyRobotLogger(name="skill")

_ORCHESTRATION_RESULT_KEYS = (
    "option_state",
    "termination_reason",
    "root_task_success",
    "episode_done",
    "requires_reobservation",
    "before_frame_id",
    "after_frame_id",
)


def _orchestration_result_metadata(data: object) -> dict[str, Any]:
    """Select small control-plane fields from a plugin result.

    Model outputs and RoboCasa traces can be large.  Only the fields needed by
    the slow Agent loop are allowed onto the protocol result metadata.
    """
    if not isinstance(data, dict):
        return {}
    metrics = data.get("metrics")
    nested = metrics if isinstance(metrics, dict) else {}
    return {
        key: data[key] if key in data else nested[key]
        for key in _ORCHESTRATION_RESULT_KEYS
        if key in data or key in nested
    }


def _model_trace_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep execution plans auditable without copying images into every event."""
    traced = {
        key: value
        for key, value in arguments.items()
        if key not in {"observation", "image_path", "images"}
    }
    observation = arguments.get("observation")
    if isinstance(observation, dict) and observation.get("frame_id") is not None:
        traced["observation_frame_id"] = observation["frame_id"]
    return traced


@dataclass
class _SkillControllerState:
    spec: PolicySpec
    scheduler: SkillScheduler
    latest_observation: RobotObservation | None = None
    latest_status: RobotStatus | None = None
    last_scheduler_decision: dict[str, Any] | None = None
    recently_finished_runs: dict[str, tuple[SkillRun, float, str | None]] | None = None
    latest_camera_frame: tuple[dict[str, Any], Any] | None = None

    @property
    def active_runs(self) -> dict[str, SkillRun]:
        return self.scheduler.runs


def _short_operation_intent(command: ShortOperationCommand) -> SkillIntent:
    proposal = command.proposal
    return SkillIntent(
        envelope=command.envelope,
        skill_id=command.operation_id,
        task_id=command.operation_id,
        intent_kind=proposal.intent_kind,
        name=proposal.skill_name,
        arguments=dict(proposal.arguments),
        objective=proposal.objective,
        timeout_sec=command.timeout_sec,
    )


class SkillControllerService:
    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.topics = Topics()
        self.bus = create_bus_client(config.deployment.bus, role="skill_controller")
        self.events = BusEventPublisher(self.bus, self.topics)
        self.media_resolver = MediaResolver(
            LocalMediaStore(
                config.resources.media_root, max_items=config.resources.media_max_items
            )
        )
        self.skill_registry = registry_from_config(config)
        self.plugin_skill_catalog = self.skill_registry.robot_skill_catalog()
        self.skill_runtime = SkillRuntime(self.skill_registry)
        self.contracts = self.skill_runtime.contracts
        self.event_sink = SkillEventSink(
            bus=self.bus,
            events=self.events,
            topics=self.topics,
            contracts=self.contracts,
            runtime_dir=config.resources.runtime_dir,
        )
        self.command_store = SkillCommandStore(
            Path(config.resources.runtime_dir) / "skill_receipts.sqlite3"
        )
        self.model_services = ModelServiceRegistry(config)
        self.states = {
            policy_id: _SkillControllerState(
                spec=spec,
                scheduler=SkillScheduler(self.contracts),
            )
            for policy_id, spec in config.policies.items()
            if spec.enabled
        }
        self._stop = asyncio.Event()
        self.human_follow = (
            HumanFollowServiceClient(self.bus, self.topics, sorted(config.robots))
            if bool(
                config.deployment.bus.options.get("human_follow_service_enabled", False)
            )
            else None
        )

    async def start(self) -> None:
        await self.bus.connect()
        if self.human_follow is not None:
            await self.human_follow.start()
        await self.bus.subscribe([self.topics.robot_observation], self._on_observation)
        await self.bus.subscribe(
            [self.topics.short_operation_command], self._on_short_operation
        )
        await self.bus.subscribe([self.topics.skill_intent], self._on_skill_intent)
        await self.bus.subscribe([self.topics.skill_control], self._on_skill_control)
        await self.bus.subscribe([self.topics.robot_status], self._on_status)
        robot_ids = list({state.spec.robot_id for state in self.states.values()})
        if robot_ids:
            await self.bus.subscribe_raw(
                [
                    self.topics.for_robot(self.topics.camera_frame, robot_id)
                    for robot_id in robot_ids
                ],
                self._on_camera_frame,
            )
        await asyncio.gather(
            *(
                self._skill_loop(policy_id, state)
                for policy_id, state in self.states.items()
            )
        )

    async def stop(self) -> None:
        self._stop.set()
        for state in self.states.values():
            for run in state.active_runs.values():
                if run.task is not None:
                    run.task.cancel()
        await self.bus.close()

    async def _on_observation(self, _topic: str, payload: dict[str, Any]) -> None:
        observation = from_payload(RobotObservation, payload)
        for state in self.states.values():
            if state.spec.robot_id == observation.envelope.robot_id:
                state.latest_observation = observation

    async def _on_camera_frame(self, _topic: str, payload: bytes) -> None:
        try:
            metadata, image = await asyncio.to_thread(decode_frame_packet, payload)
        except Exception:
            return
        robot_id = str(metadata.get("robot_id") or "")
        for state in self.states.values():
            if state.spec.robot_id == robot_id:
                state.latest_camera_frame = (metadata, image)

    async def _on_short_operation(self, _topic: str, payload: dict[str, Any]) -> None:
        command = from_payload(ShortOperationCommand, payload)
        intent = _short_operation_intent(command)
        await self._on_skill_intent(self.topics.skill_intent, to_payload(intent))

    async def _on_skill_intent(self, _topic: str, payload: dict[str, Any]) -> None:
        intent = from_payload(SkillIntent, payload)
        receipt = self.command_store.receive(
            intent.skill_id, canonical_payload_hash(payload)
        )
        if receipt == "conflict":
            await self._publish_result(
                intent,
                "unknown",
                None,
                "skill id payload conflict",
                failure_mode="IDEMPOTENCY_CONFLICT",
                error="IDEMPOTENCY_CONFLICT",
            )
            return
        if receipt == "replay":
            result = self.command_store.result(intent.skill_id)
            if result is not None:
                await self.bus.publish(self.topics.skill_result, result)
            return
        state_item = self._state_for_robot(intent.envelope.robot_id)
        if state_item is None:
            return
        policy_id, state = state_item
        try:
            await self._accept_skill(policy_id, state, intent)
        except Exception as exc:
            failure_mode = (
                "unknown_skill" if isinstance(exc, KeyError) else "internal_error"
            )
            await self._publish_event(
                intent, "failed", summary=str(exc), error=str(exc)
            )
            await self._publish_result(
                intent,
                "failed",
                False,
                str(exc),
                failure_mode=failure_mode,
                error=str(exc),
            )
            await self._publish_scheduler_state(
                policy_id,
                state,
                phase="rejected",
                intent=intent,
                decision={"reason": failure_mode, "error": str(exc)},
                severity="warn",
            )

    async def _on_skill_control(self, _topic: str, payload: dict[str, Any]) -> None:
        control = from_payload(SkillControl, payload)
        receipt = self.command_store.receive(
            control.control_id, canonical_payload_hash(payload)
        )
        if receipt == "replay":
            prior = self.command_store.result(control.control_id)
            if prior is not None:
                await self.bus.publish(self.topics.skill_control_result, prior)
            return
        if receipt == "conflict":
            result = SkillControlResult(
                control.envelope,
                control.control_id,
                control.action,
                control.target_skill_id,
                "unknown",
                False,
                "IDEMPOTENCY_CONFLICT",
            )
        else:
            stopped = False
            affected_states: list[_SkillControllerState] = []
            for state in self.states.values():
                targets = (
                    [control.target_skill_id]
                    if control.action == "interrupt" and control.target_skill_id
                    else list(state.active_runs)
                )
                for skill_id in targets:
                    run = state.active_runs.get(skill_id)
                    if run is None:
                        continue
                    if run.task is not None:
                        run.task.cancel()
                    run.terminal = True
                    state.scheduler.remove(skill_id)
                    stopped = True
                    affected_states.append(state)
            stop_dispatched = False
            for state in affected_states:
                try:
                    await self._publish_stop_motion(control, state)
                    stop_dispatched = True
                except Exception:
                    logger.exception(
                        f"failed to publish physical stop for control {control.control_id}",
                    )
            idle_confirmed = bool(affected_states) and all(
                self._idle_confirmed(state) for state in affected_states
            )
            result = SkillControlResult(
                control.envelope,
                control.control_id,
                control.action,
                control.target_skill_id,
                "completed"
                if stopped and stop_dispatched and idle_confirmed
                else "unknown",
                stopped and stop_dispatched and idle_confirmed,
                None
                if stopped and stop_dispatched and idle_confirmed
                else "physical idle state was not confirmed",
            )
        encoded = to_payload(result)
        self.command_store.terminal(control.control_id, encoded)
        await self.bus.publish(self.topics.skill_control_result, encoded)

    async def _publish_stop_motion(
        self, control: SkillControl, _state: _SkillControllerState
    ) -> None:
        """控制平面停止命令走 robot action 路径，绝不通过 SkillIntent。"""
        from hey_robot.protocol import RobotAction

        await self.bus.publish(
            self.topics.robot_action,
            to_payload(
                RobotAction(
                    envelope=control.envelope,
                    values=[],
                    skill_id=control.target_skill_id or control.control_id,
                    task_id=control.task_id or "",
                    intent_kind="skill",
                    metadata={
                        "action_type": "skill",
                        "skill": {
                            "name": "stop_motion",
                            "arguments": {
                                "emergency": control.action == "emergency_stop"
                            },
                            "safety_level": "emergency",
                            "expected_duration_sec": None,
                        },
                        "control_id": control.control_id,
                    },
                )
            ),
        )

    def _idle_confirmed(self, state: _SkillControllerState) -> bool:
        status = state.latest_status
        return bool(
            status is not None and status.state == "idle" and status.skill_id is None
        )

    async def _accept_skill(
        self, policy_id: str, state: _SkillControllerState, intent: SkillIntent
    ) -> None:
        """在机器人执行前对技能意图进行最终准入。

        Supervisor 的预检负责保护调度；此处仍必须再次校验，避免消息传输期间
        机器人状态或资源占用变化造成检查时与使用时不一致（TOCTOU）。
        """
        resolved_args = {**dict(intent.arguments), "objective": intent.objective}
        contract, decision = self.skill_runtime.validate(
            intent.name,
            resolved_args,
            enabled_only=bool(self.config.skills.enabled),
            status=state.latest_status,
            robot_type=self._robot_type(state.spec.robot_id),
        )
        if not decision.allowed:
            await self._publish_event(
                intent,
                "failed",
                summary="skill precondition failed",
                error=decision.reason,
                contract=contract,
            )
            await self._publish_result(
                intent,
                "failed",
                False,
                decision.reason,
                failure_mode=decision.failure_mode or "precondition_failed",
                error=decision.reason,
                contract=contract,
            )
            await self._publish_scheduler_state(
                policy_id,
                state,
                phase="rejected",
                intent=intent,
                contract=contract,
                decision={
                    "reason": decision.failure_mode or "precondition_failed",
                    "error": decision.reason,
                },
                severity="warn",
            )
            return
        conflict = state.scheduler.conflicting_run(contract, intent.arguments)
        if conflict is not None:
            shared = self.contracts.shared_or_global_resources(
                contract,
                conflict.contract,
                left_arguments=intent.arguments,
                right_arguments=conflict.intent.arguments,
            )
            reason = (
                f"robot {state.spec.robot_id} resource conflict with active skill "
                f"{conflict.intent.skill_id} on resources {','.join(sorted(shared))}"
            )
            await self._publish_event(
                intent,
                "failed",
                summary="skill rejected; robot resource is busy",
                error=reason,
                contract=contract,
            )
            await self._publish_result(
                intent,
                "failed",
                False,
                reason,
                failure_mode="resource_busy",
                error=reason,
                contract=contract,
            )
            await self._publish_scheduler_state(
                policy_id,
                state,
                phase="rejected",
                intent=intent,
                contract=contract,
                decision={
                    "reason": "resource_busy",
                    "error": reason,
                    "conflicting_skill_id": conflict.intent.skill_id,
                    "conflicting_skill": conflict.intent.name or conflict.contract.name,
                    "conflicting_resources": sorted(shared),
                },
                severity="warn",
            )
            return
        execution_plan = self._execution_plan(intent, contract)
        state.scheduler.add(
            SkillRun(
                intent=intent,
                skill_name=intent.name,
                implementation_name=intent.name,
                implementation_kind="plugin",
                contract=contract,
                execution_plan=execution_plan,
                timeout_override_sec=self._estimated_timeout_sec(
                    intent, contract, execution_plan
                ),
            )
        )
        await self._publish_event(
            intent,
            "accepted",
            summary="skill accepted",
            policy_id=policy_id,
            contract=contract,
            execution_plan=execution_plan,
        )
        await self._publish_scheduler_state(
            policy_id,
            state,
            phase="accepted",
            intent=intent,
            contract=contract,
            decision={"reason": "accepted"},
        )

    async def _on_status(self, _topic: str, payload: dict[str, Any]) -> None:
        status = from_payload(RobotStatus, payload)
        for state in self.states.values():
            if state.spec.robot_id != status.envelope.robot_id:
                continue
            state.latest_status = status
            if status.skill_id:
                active = sorted(state.active_runs.keys())
                run_for_log = state.active_runs.get(status.skill_id)
                logger.info(
                    "skill_status_trace received "
                    f"robot={status.envelope.robot_id} skill_id={status.skill_id} "
                    f"success={status.success} state={status.state} frame={status.frame_id} "
                    f"active_runs={active} "
                    f"run_found={run_for_log is not None} "
                    f"run_terminal={run_for_log.terminal if run_for_log is not None else None} "
                    f"pending_status={run_for_log.pending_status is not None if run_for_log is not None else None} "
                    f"pending_done={run_for_log.pending_status.done() if run_for_log is not None and run_for_log.pending_status is not None else None}"
                )
            run = state.active_runs.get(status.skill_id or "")
            if run is None or run.terminal:
                if status.skill_id and status.success is True:
                    handled = await self._handle_late_success_status(state, status)
                    if handled:
                        continue
                if status.skill_id:
                    logger.warning(
                        "skill_status_trace ignored "
                        f"robot={status.envelope.robot_id} skill_id={status.skill_id} "
                        f"reason={'missing_run' if run is None else 'terminal_run'}"
                    )
                continue
            run.status_received_at = time.time()
            future = run.pending_status
            if future is not None and not future.done():
                logger.info(
                    "skill_status_trace resolving_pending_status "
                    f"robot={status.envelope.robot_id} skill_id={status.skill_id} "
                    f"success={status.success} frame={status.frame_id}"
                )
                future.set_result(status)
                continue
            if status.skill_id and future is None:
                logger.warning(
                    "skill_status_trace no_pending_status "
                    f"robot={status.envelope.robot_id} skill_id={status.skill_id} "
                    f"task_active={run.task is not None}"
                )
            if run.task is None and status.skill_id == run.intent.skill_id:
                if status.success is True:
                    run.steps_executed += 1
                    step_summary = self._status_step_summary(status)
                    if step_summary:
                        run.step_summaries.append(step_summary)
                    await self._finish_run(
                        next(
                            policy_id
                            for policy_id, item in self.states.items()
                            if item is state
                        ),
                        state,
                        run,
                        success=True,
                        summary=step_summary or "skill completed",
                        status="completed",
                    )
                elif status.success is False:
                    await self._finish_run(
                        next(
                            policy_id
                            for policy_id, item in self.states.items()
                            if item is state
                        ),
                        state,
                        run,
                        success=False,
                        summary=status.error or "skill failed",
                        status="failed",
                        failure_mode=self._failure_mode(status),
                        error=status.error,
                    )
        await asyncio.sleep(0)

    async def _skill_loop(self, policy_id: str, state: _SkillControllerState) -> None:
        period = 1.0 / max(float(state.spec.freq_hz), 0.1)
        while not self._stop.is_set():
            await self._skill_loop_step(policy_id, state)
            await asyncio.sleep(period)

    async def _skill_loop_step_for_test(
        self, policy_id: str, state: _SkillControllerState
    ) -> None:
        await self._skill_loop_step(policy_id, state)
        await asyncio.sleep(0)

    async def _skill_loop_step(
        self, policy_id: str, state: _SkillControllerState
    ) -> None:
        if not state.active_runs:
            return
        await self._expire_timed_out_runs(policy_id, state)
        for skill_id in list(state.active_runs.keys()):
            run = state.active_runs.get(skill_id)
            if run is None or run.terminal:
                continue
            if run.task is None:
                run.started_at = time.time()
                run.task = asyncio.create_task(
                    self._execute_plugin_run(policy_id, state, run)
                )
                continue
            if run.task.done():
                exc = run.task.exception()
                if (
                    exc is not None
                    and state.active_runs.get(skill_id) is run
                    and not run.terminal
                ):
                    await self._finish_run(
                        policy_id,
                        state,
                        run,
                        success=False,
                        summary=str(exc),
                        status="failed",
                        failure_mode="internal_error",
                        error=str(exc),
                    )

    async def _execute_plugin_run(
        self, policy_id: str, state: _SkillControllerState, run: SkillRun
    ) -> None:
        intent = run.intent
        await self._publish_event(
            intent,
            "executing",
            progress=0.1,
            summary=f"executing skill {run.skill_name}",
            policy_id=policy_id,
            steps_executed=run.steps_executed,
            contract=run.contract,
            execution_plan=run.execution_plan,
        )
        result = await self.skill_runtime.execute(
            run.skill_name,
            {**dict(intent.arguments), "objective": intent.objective},
            context_factory=lambda invoke: self._plugin_context(
                policy_id, state, run, invoke
            ),
            enabled_only=False,
            status=state.latest_status,
            robot_type=self._robot_type(state.spec.robot_id),
        )
        if state.active_runs.get(intent.skill_id) is not run or run.terminal:
            return
        result_data = dict(getattr(result, "data", {}) or {})
        evidence_data = result_data.get("evidence")
        result_metadata = _orchestration_result_metadata(result_data)
        if result.success and intent.name == "inspect_scene":
            facts = list(evidence_data) if isinstance(evidence_data, list) else []
            facts.append(
                {
                    "subject_id": f"robot:{state.spec.robot_id}",
                    "predicate": "observed",
                    "object_id": "scene",
                }
            )
            evidence_data = facts
        await self._finish_run(
            policy_id,
            state,
            run,
            success=bool(result.success),
            summary=str(result.summary),
            status=cast(
                Literal["completed", "failed", "interrupted", "unknown"],
                str(result.status),
            ),
            failure_mode=getattr(result, "failure_mode", None),
            error=getattr(result, "error", None),
            evidence_data=evidence_data,
            result_metadata=result_metadata,
        )

    async def _finish_run(
        self,
        policy_id: str,
        state: _SkillControllerState,
        run: SkillRun,
        *,
        success: bool | None,
        summary: str,
        status: Literal["completed", "failed", "interrupted", "unknown"],
        failure_mode: str | None = None,
        error: str | None = None,
        evidence_data: object = None,
        result_metadata: dict[str, Any] | None = None,
    ) -> None:
        intent = run.intent
        final_summary = self._completion_summary(run, summary) if success else summary
        run.terminal = True
        state.scheduler.remove(intent.skill_id)
        self._remember_finished_run(state, run, failure_mode)
        phase = "completed" if success else "failed"
        await self._publish_event(
            intent,
            phase,
            progress=1.0 if success else 0.0,
            summary=final_summary,
            error=None if success else error,
            policy_id=policy_id,
            steps_executed=run.steps_executed,
            frame_id=state.latest_status.frame_id if state.latest_status else None,
            contract=run.contract,
            execution_plan=run.execution_plan,
        )
        await self._publish_result(
            intent,
            status,
            success,
            final_summary,
            frame_id=state.latest_status.frame_id if state.latest_status else None,
            error=None if success else error,
            failure_mode=None if success else (failure_mode or "execution_failed"),
            steps_executed=run.steps_executed,
            contract=run.contract,
            run=run,
            evidence_data=evidence_data,
            metadata=result_metadata,
        )
        await self._publish_scheduler_state(
            policy_id,
            state,
            phase=phase,
            intent=intent,
            contract=run.contract,
            decision={
                "reason": "completed"
                if success
                else (failure_mode or "execution_failed"),
                "error": None if success else error,
            },
            severity="info" if success else "warn",
        )

    def _remember_finished_run(
        self,
        state: _SkillControllerState,
        run: SkillRun,
        failure_mode: str | None,
    ) -> None:
        if state.recently_finished_runs is None:
            state.recently_finished_runs = {}
        now = time.time()
        state.recently_finished_runs[run.intent.skill_id] = (run, now, failure_mode)
        for skill_id, (_run, finished_at, _failure_mode) in list(
            state.recently_finished_runs.items()
        ):
            if now - finished_at > 5.0:
                state.recently_finished_runs.pop(skill_id, None)

    async def _handle_late_success_status(
        self, state: _SkillControllerState, status: RobotStatus
    ) -> bool:
        if state.recently_finished_runs is None or not status.skill_id:
            return False
        item = state.recently_finished_runs.get(status.skill_id)
        if item is None:
            return False
        run, finished_at, failure_mode = item
        if failure_mode != "timeout" or time.time() - finished_at > 2.0:
            return False
        run.terminal = True
        run.status_received_at = time.time()
        run.steps_executed = max(run.steps_executed, 1)
        step_summary = self._status_step_summary(status)
        if step_summary:
            run.step_summaries.append(step_summary)
        logger.warning(
            "skill_status_trace late_success_after_timeout "
            f"robot={status.envelope.robot_id} skill_id={status.skill_id} "
            f"frame={status.frame_id}"
        )
        policy_id = next(
            policy_id
            for policy_id, item_state in self.states.items()
            if item_state is state
        )
        await self._publish_event(
            run.intent,
            "completed",
            progress=1.0,
            summary=step_summary or "late success after timeout",
            policy_id=policy_id,
            frame_id=status.frame_id,
            steps_executed=run.steps_executed,
            contract=run.contract,
            execution_plan=run.execution_plan,
            metadata={"late_success_after_timeout": True},
        )
        await self._publish_result(
            run.intent,
            "completed",
            True,
            step_summary or "late success after timeout",
            frame_id=status.frame_id,
            steps_executed=run.steps_executed,
            contract=run.contract,
            run=run,
        )
        await self._publish_scheduler_state(
            policy_id,
            state,
            phase="completed",
            intent=run.intent,
            contract=run.contract,
            decision={
                "reason": "late_success_after_timeout",
                "previous_failure_mode": failure_mode,
                "error": None,
            },
            severity="info",
        )
        state.recently_finished_runs.pop(status.skill_id, None)
        return True

    async def _invoke_robot_skill(
        self,
        policy_id: str,
        state: _SkillControllerState,
        run: SkillRun,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        contract = self.plugin_skill_catalog.resolve(
            name,
            robot_type=self._robot_type(run.intent.envelope.robot_id or ""),
        )
        decision = self.contracts.acceptance_decision(
            contract,
            status=state.latest_status,
            arguments=arguments,
        )
        if not decision.allowed:
            raise RuntimeError(decision.reason)
        await self._publish_event(
            run.intent,
            "executing",
            progress=self._run_progress(run),
            summary=f"executing {name}",
            policy_id=policy_id,
            steps_executed=run.steps_executed,
            frame_id=state.latest_observation.frame_id
            if state.latest_observation
            else None,
            contract=run.contract,
            step=name,
            execution_plan=run.execution_plan,
        )
        action_intent = self._action_intent(
            run.intent, RobotSkillAction(name, arguments)
        )
        skill_action = RobotSkillAction(name, arguments)
        run.execution_plan = SkillExecutionPlan(
            actions=(*run.execution_plan.actions, skill_action),
            strategy="runtime_trace",
            notes=("Recorded from actual skill execution.",),
        )
        action = skill_action.to_robot_action(action_intent)
        future: asyncio.Future[RobotStatus] = asyncio.get_running_loop().create_future()
        run.pending_status = future
        run.current_step = name
        logger.info(
            "skill_status_trace waiting_pending_status "
            f"robot={run.intent.envelope.robot_id} skill_id={run.intent.skill_id} "
            f"action={name} timeout_sec={run.timeout_sec:g} "
            f"age_sec={max(0.0, time.time() - run.accepted_at):.3f}"
        )
        await self.bus.publish(self.topics.robot_action, to_payload(action))
        run.action_published_at = time.time()
        await self.events.publish(
            RuntimeEvent.make(
                EventKind.POLICY_ACTION,
                source="skill-controller",
                trace_id=action.envelope.trace_id,
                episode_id=action.envelope.episode_id,
                agent_id=action.envelope.agent_id,
                robot_id=action.envelope.robot_id,
                payload={
                    "policy_id": policy_id,
                    "skill_id": run.intent.skill_id,
                    "skill": run.skill_name,
                    "primitive_action": name,
                    "contract": contract.to_dict(),
                },
            )
        )
        try:
            status = await future
            logger.info(
                "skill_status_trace pending_status_received "
                f"robot={status.envelope.robot_id} skill_id={status.skill_id} "
                f"success={status.success} state={status.state} frame={status.frame_id}"
            )
        finally:
            if run.pending_status is future:
                run.pending_status = None
                run.current_step = None
                logger.info(
                    "skill_status_trace pending_status_cleared "
                    f"robot={run.intent.envelope.robot_id} skill_id={run.intent.skill_id} "
                    f"future_done={future.done()} future_cancelled={future.cancelled()}"
                )
        if status.success is False:
            raise RuntimeError(status.error or f"{name} failed")
        run.steps_executed += 1
        step_summary = self._status_step_summary(status)
        if step_summary:
            run.step_summaries.append(step_summary)
        last_result = status.metrics.get("last_skill_result")
        if isinstance(last_result, dict):
            return dict(last_result)
        return {
            "success": status.success is not False,
            "message": step_summary or f"{name} completed",
        }

    async def _invoke_native_policy_action(
        self,
        policy_id: str,
        state: _SkillControllerState,
        run: SkillRun,
        values: list[float],
        expected_frame_id: int,
        raw_values: list[float] | None,
    ) -> dict[str, Any]:
        """Route a model-native action through the canonical Robot Runtime bus."""
        del state
        if len(values) != 12:
            raise RuntimeError(
                f"native policy action must have 12 values, got {len(values)}"
            )
        action = RobotAction(
            envelope=run.intent.envelope,
            values=[float(value) for value in values],
            skill_id=run.intent.skill_id,
            task_id=run.intent.task_id,
            intent_kind="skill",
            metadata={
                "action_type": "native_policy",
                "expected_frame_id": int(expected_frame_id),
                "raw_action": list(raw_values or values),
                "action_clipped": list(raw_values or values) != list(values),
            },
        )
        future: asyncio.Future[RobotStatus] = asyncio.get_running_loop().create_future()
        run.pending_status = future
        run.current_step = "native_policy_action"
        run.execution_plan = SkillExecutionPlan(
            actions=(
                RobotSkillAction(
                    "native_policy_action",
                    {"expected_frame_id": int(expected_frame_id)},
                ),
            ),
            strategy="runtime_trace",
            notes=("Native action routed through Robot Runtime.",),
        )
        await self.bus.publish(self.topics.robot_action, to_payload(action))
        run.action_published_at = time.time()
        await self.events.publish(
            RuntimeEvent.make(
                EventKind.POLICY_ACTION,
                source="skill-controller",
                trace_id=action.envelope.trace_id,
                episode_id=action.envelope.episode_id,
                agent_id=action.envelope.agent_id,
                robot_id=action.envelope.robot_id,
                payload={
                    "policy_id": policy_id,
                    "skill_id": run.intent.skill_id,
                    "skill": run.skill_name,
                    "action_type": "native_policy",
                    "expected_frame_id": int(expected_frame_id),
                },
            )
        )
        try:
            status = await future
        finally:
            if run.pending_status is future:
                run.pending_status = None
                run.current_step = None
        if status.success is False:
            raise RuntimeError(status.error or "native policy action failed")
        run.steps_executed += 1
        return {
            "success": True,
            "frame_id": status.frame_id,
            "done": bool(status.metrics.get("done", False)),
        }

    async def _invoke_model_service(
        self,
        run: SkillRun,
        name: str,
        _arguments: dict[str, Any],
    ) -> ServiceInvocationResult:
        model_service = self.model_services.service_for(
            name, run.intent.envelope.robot_id
        )
        if model_service is None:
            raise RuntimeError(f"{name} requires a deployed model service")
        service_id, spec, client = model_service
        health = await client.health()
        if not health.online or not health.loaded or health.busy:
            reason = health.error or (
                f"model service {service_id} is busy"
                if health.busy
                else f"model service {service_id} is not deployed or model is not loaded"
            )
            raise RuntimeError(reason)
        # 将 skill 层 enriched 的参数（如 observation/images）显式传给 ModelService。
        enriched_arguments = {**run.intent.arguments, **_arguments}
        contract = self.plugin_skill_catalog.resolve(name)
        trace_arguments = _model_trace_arguments(_arguments)
        run.execution_plan = SkillExecutionPlan(
            actions=(RobotSkillAction(name, trace_arguments),),
            strategy="runtime_trace",
            notes=("Recorded from actual model service invocation.",),
        )
        run.active_model_service_id = service_id
        run.active_model_client = client
        try:
            result = await client.execute(
                ServiceInvocationRequest(
                    service_id=service_id,
                    intent=run.intent,
                    contract=contract,
                    timeout_sec=float(
                        run.intent.timeout_sec or spec.timeout_sec or run.timeout_sec
                    ),
                    arguments=enriched_arguments,
                )
            )
        finally:
            if run.active_model_service_id == service_id:
                run.active_model_service_id = None
                run.active_model_client = None
        run.steps_executed += 1
        if result.summary:
            run.step_summaries.append(result.summary)
        return result

    async def _expire_timed_out_runs(
        self, policy_id: str, state: _SkillControllerState
    ) -> None:
        for skill_id in list(state.active_runs.keys()):
            run = state.active_runs.get(skill_id)
            if run is None or run.terminal or not run.timed_out:
                continue
            cancel_metadata = await self._cancel_active_model_service(run)
            if run.task is not None:
                run.task.cancel()
            await self._finish_run(
                policy_id,
                state,
                run,
                success=False,
                summary=f"skill timed out after {run.timeout_sec:g}s",
                status="failed",
                failure_mode="timeout",
                error="skill timed out",
            )
            if cancel_metadata is not None:
                await self._publish_event(
                    run.intent,
                    "cancel_requested",
                    summary="model service cancel requested after timeout",
                    policy_id=policy_id,
                    steps_executed=run.steps_executed,
                    contract=run.contract,
                    execution_plan=run.execution_plan,
                    metadata={"model_service_cancel": cancel_metadata},
                )

    async def _publish_event(
        self,
        intent: SkillIntent,
        phase: str,
        *,
        progress: float | None = None,
        summary: str | None = None,
        error: str | None = None,
        policy_id: str | None = None,
        frame_id: int | None = None,
        steps_executed: int | None = None,
        contract: SkillContract | None = None,
        step: str | None = None,
        execution_plan: SkillExecutionPlan | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._sync_event_sink()
        await self.event_sink.publish_event(
            intent,
            phase,
            run=self._run_for_intent(intent),
            progress=progress,
            summary=summary,
            error=error,
            policy_id=policy_id,
            frame_id=frame_id,
            steps_executed=steps_executed,
            contract=contract,
            step=step,
            execution_plan=execution_plan,
            metadata=metadata,
        )

    async def _publish_result(
        self,
        intent: SkillIntent,
        status: Literal["completed", "failed", "interrupted", "unknown"],
        success: bool | None,
        summary: str,
        *,
        frame_id: int | None = None,
        error: str | None = None,
        failure_mode: str | None = None,
        steps_executed: int = 0,
        contract: SkillContract | None = None,
        run: SkillRun | None = None,
        evidence_data: object = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._sync_event_sink()
        await self.event_sink.publish_result(
            intent,
            status,
            success,
            summary,
            run=run or self._run_for_intent(intent),
            frame_id=frame_id,
            error=error,
            failure_mode=failure_mode,
            steps_executed=steps_executed,
            contract=contract,
            evidence_data=evidence_data,
            metadata=metadata,
        )

    def _state_for_robot(
        self, robot_id: str | None
    ) -> tuple[str, _SkillControllerState] | None:
        if not robot_id:
            return None
        for policy_id, state in self.states.items():
            if state.spec.robot_id == robot_id:
                return policy_id, state
        return None

    def _execution_plan(
        self,
        intent: SkillIntent,
        contract: SkillContract,
    ) -> SkillExecutionPlan:
        del intent, contract
        return SkillExecutionPlan(
            actions=(),
            strategy="runtime_trace",
            notes=("Actions are recorded from actual execution.",),
        )

    def _robot_type(self, robot_id: str) -> str | None:
        return resolve_robot_family(self.config, robot_id, fallback=robot_id)

    @staticmethod
    def _action_intent(intent: SkillIntent, action: RobotSkillAction) -> SkillIntent:
        return SkillIntent(
            envelope=intent.envelope,
            skill_id=intent.skill_id,
            task_id=intent.task_id,
            intent_kind=intent.intent_kind,
            name=action.name,
            arguments=dict(action.arguments),
            objective=intent.objective,
            priority=intent.priority,
            timeout_sec=intent.timeout_sec,
            feedback_mode=intent.feedback_mode,
        )

    @staticmethod
    def _run_progress(run: SkillRun) -> float:
        return min(0.95, 0.1 + run.steps_executed * 0.2)

    def _run_for_intent(self, intent: SkillIntent) -> SkillRun | None:
        state_item = self._state_for_robot(intent.envelope.robot_id)
        if state_item is None:
            return None
        return state_item[1].active_runs.get(intent.skill_id)

    def _plugin_context(
        self,
        policy_id: str,
        state: _SkillControllerState,
        run: SkillRun,
        invoke_skill: SkillInvoke,
    ) -> SkillContext:
        robot = RobotActionPort(
            lambda name, arguments: self._invoke_robot_skill(
                policy_id, state, run, name, arguments
            ),
            lambda values, expected_frame_id, raw_values: (
                self._invoke_native_policy_action(
                    policy_id, state, run, values, expected_frame_id, raw_values
                )
            ),
        )

        contract = run.contract
        model_settings: dict[str, Any] = {}
        if contract is not None and contract.required_model_service:
            resolved_model = self.model_services.service_for(
                contract.required_model_service, state.spec.robot_id
            )
            if resolved_model is not None:
                model_settings = dict(resolved_model[1].settings)
        requires_camera = contract is not None and "camera" in (
            contract.required_resources or ()
        )

        def base_invoke(name: str, arguments: dict[str, Any]) -> Any:
            return self._invoke_model_service(run, name, arguments)

        if requires_camera:
            _state_ref = state

            def camera_aware_invoke(name: str, arguments: dict[str, Any]):
                if "observation" not in arguments and "image_path" not in arguments:
                    frame = _state_ref.latest_camera_frame
                    if frame is not None:
                        metadata, image_array = frame
                        buf = io.BytesIO()
                        Image.fromarray(image_array).save(
                            buf, format="JPEG", quality=85
                        )
                        b64_data = base64.b64encode(buf.getvalue()).decode("ascii")
                        arguments = {
                            **arguments,
                            "observation": {
                                "frame_id": metadata.get("frame_id"),
                                "timestamp": metadata.get("timestamp")
                                or metadata.get("created_at")
                                or time.time(),
                                "images": [
                                    {
                                        "camera": metadata.get("camera", "unknown"),
                                        "format": "jpeg",
                                        "data": b64_data,
                                    }
                                ],
                                "proprioception": list(
                                    getattr(
                                        _state_ref.latest_observation,
                                        "proprioception",
                                        [],
                                    )
                                    or []
                                ),
                            },
                        }
                return self._invoke_model_service(run, name, arguments)

            model_invoke = camera_aware_invoke
        else:
            model_invoke = base_invoke

        return SkillContext(
            skill_id=run.intent.skill_id,
            robot_id=run.intent.envelope.robot_id,
            robot=robot,
            perception=PerceptionPort(robot),
            model_services=ModelServicePort(model_invoke),
            model_settings=model_settings,
            observation=state.latest_observation,
            current_observation=lambda: state.latest_observation,
            resolve_images=self.media_resolver.resolve_images,
            logger=logger,
            invoke=invoke_skill,
            progress=lambda **kwargs: self._plugin_progress(
                policy_id, state, run, **kwargs
            ),
            human_follow=self.human_follow,
            get_camera_frame=lambda: state.latest_camera_frame,
        )

    async def _plugin_progress(
        self,
        policy_id: str,
        state: _SkillControllerState,
        run: SkillRun,
        *,
        phase: str = "executing",
        summary: str | None = None,
        progress: float | None = None,
        step: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if state.active_runs.get(run.intent.skill_id) is not run or run.terminal:
            return
        run.current_step = step or phase
        await self._publish_event(
            run.intent,
            phase,
            progress=progress,
            summary=summary,
            policy_id=policy_id,
            steps_executed=run.steps_executed,
            frame_id=state.latest_observation.frame_id
            if state.latest_observation
            else None,
            contract=run.contract,
            step=step,
            execution_plan=run.execution_plan,
            metadata=metadata,
        )

    async def _publish_scheduler_state(
        self,
        policy_id: str,
        state: _SkillControllerState,
        *,
        phase: str,
        intent: SkillIntent,
        contract: SkillContract | None = None,
        decision: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> None:
        self._sync_event_sink()
        state.last_scheduler_decision = await self.event_sink.publish_scheduler_state(
            policy_id,
            robot_id=state.spec.robot_id,
            active_runs=state.active_runs,
            phase=phase,
            intent=intent,
            contract=contract,
            decision=decision,
            severity=severity,
        )

    def _sync_event_sink(self) -> None:
        self.event_sink.bus = self.bus
        self.event_sink.events = self.events

    def _estimated_timeout_sec(
        self,
        intent: SkillIntent,
        contract: SkillContract,
        execution_plan: SkillExecutionPlan,
    ) -> float | None:
        if intent.timeout_sec is not None:
            return None
        robot = self.config.robots.get(intent.envelope.robot_id or "")
        if robot is None:
            return None
        estimated = 0.0
        actions = execution_plan.actions
        if not actions and contract.level == "primitive":
            actions = (RobotSkillAction(intent.name, dict(intent.arguments)),)
        for action in actions:
            estimated += self._estimated_action_timeout_sec(action, robot.settings)
        if estimated <= 0.0:
            return None
        return max(float(contract.timeout_sec), estimated)

    @staticmethod
    def _estimated_action_timeout_sec(
        action: RobotSkillAction, robot_settings: dict[str, Any]
    ) -> float:
        motion_time_scale = float(robot_settings.get("motion_time_scale", 1.0) or 1.0)
        if action.name == "move_base":
            distance_cm = abs(float(action.arguments.get("distance_cm") or 0.0))
            speed = abs(float(robot_settings.get("default_linear_speed", 0.2) or 0.2))
            duration = distance_cm / 100.0 / max(speed, 0.01) * motion_time_scale
            return max(3.0, duration + 3.0)
        if action.name == "turn_base":
            angle_deg = abs(float(action.arguments.get("angle_deg") or 0.0))
            angular_speed = abs(
                float(robot_settings.get("default_angular_speed", 0.45) or 0.45)
            )
            duration = (
                math.radians(angle_deg) / max(angular_speed, 0.01) * motion_time_scale
            )
            return max(3.0, duration + 3.0)
        if action.name == "base_velocity_step":
            duration_ms = abs(float(action.arguments.get("duration_ms") or 250.0))
            return max(1.0, duration_ms / 1000.0 + 1.0)
        return 0.0

    async def _cancel_active_model_service(
        self, run: SkillRun
    ) -> dict[str, Any] | None:
        client = run.active_model_client
        service_id = run.active_model_service_id
        if client is None or service_id is None:
            return None
        try:
            await client.cancel(run.intent.skill_id)
        except Exception as exc:
            return {
                "service_id": service_id,
                "skill_id": run.intent.skill_id,
                "accepted": False,
                "error": str(exc),
            }
        return {
            "service_id": service_id,
            "skill_id": run.intent.skill_id,
            "accepted": True,
        }

    async def _interrupt_active(
        self, policy_id: str, state: _SkillControllerState, interrupt: SkillIntent
    ) -> None:
        del policy_id, state, interrupt  # 已删除旁路，改用 skill.control

    @staticmethod
    def _precondition_block(
        contract: SkillContract, status: RobotStatus | None
    ) -> str | None:
        decision = SkillContractRuntime.precondition_block(contract, status)
        return None if decision is None else decision.reason

    @staticmethod
    def _failure_mode(status: RobotStatus) -> str:
        if status.error:
            text = status.error.lower()
            if "safety" in text or "blocked" in text:
                return "safety_blocked"
            if "battery" in text:
                return "battery_critical"
            if "timeout" in text:
                return "timeout"
        return "execution_failed"

    @staticmethod
    def _completion_summary(run: SkillRun, fallback: str | None) -> str:
        summaries = [item.strip() for item in run.step_summaries if item.strip()]
        if len(summaries) > 1:
            return "; ".join(summaries)
        if summaries:
            return summaries[0]
        return (fallback or "skill completed").strip() or "skill completed"

    @staticmethod
    def _status_step_summary(status: RobotStatus) -> str | None:
        last_result = status.metrics.get("last_skill_result")
        if not isinstance(last_result, dict):
            return None
        summary = str(last_result.get("summary") or "").strip()
        message = str(last_result.get("message") or "").strip()
        skill = last_result.get("skill")
        skill_name = (
            str(skill.get("name") or "").strip() if isinstance(skill, dict) else ""
        )
        if summary:
            return summary
        if message:
            return message
        if skill_name:
            return f"{skill_name} completed"
        return None
