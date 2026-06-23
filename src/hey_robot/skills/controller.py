from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol

from hey_robot.bus.factory import create_bus_client
from hey_robot.capability.runtime import (
    CapabilityExecutionRequest,
    CapabilityExecutionResult,
    CapabilityRuntime,
)
from hey_robot.config import DeploymentConfig, PolicySpec
from hey_robot.events import EventKind, RuntimeEvent
from hey_robot.events.bus import BusEventPublisher
from hey_robot.human_follow import HumanFollowServiceClient
from hey_robot.logging import HeyRobotLogger
from hey_robot.media import LocalMediaStore, MediaResolver
from hey_robot.perception import CodecRegistry, ObservationActionCodec
from hey_robot.policies.runtime import PolicyRuntime, build_policy_runtime
from hey_robot.protocol import (
    RobotObservation,
    RobotStatus,
    SkillIntent,
    Topics,
)
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.robots.identity import resolve_robot_family
from hey_robot.skills.actions import RobotSkillAction
from hey_robot.skills.catalog import RobotSkillSpec
from hey_robot.skills.composition import SkillExecutionPlan
from hey_robot.skills.context import SkillContext
from hey_robot.skills.contracts import SkillContractRuntime
from hey_robot.skills.event_sink import SkillEventSink
from hey_robot.skills.ports import CapabilityPort, PerceptionPort, RobotActionPort
from hey_robot.skills.registry import registry_from_config
from hey_robot.skills.runtime import SkillInvoke, SkillRuntime
from hey_robot.skills.scheduler import SkillRun, SkillScheduler

logger = HeyRobotLogger(name="skill")


class SkillPolicyRuntime(Protocol):
    @property
    def control_period_sec(self) -> float: ...

    async def close(self) -> None: ...


@dataclass
class _SkillControllerState:
    spec: PolicySpec
    codec: ObservationActionCodec
    scheduler: SkillScheduler
    latest_observation: RobotObservation | None = None
    latest_status: RobotStatus | None = None
    runtime: SkillPolicyRuntime | None = None
    last_scheduler_decision: dict[str, Any] | None = None

    @property
    def active_runs(self) -> dict[str, SkillRun]:
        return self.scheduler.runs


class SkillControllerService:
    def __init__(self, config: DeploymentConfig, *, robot_service: Any = None) -> None:
        self.config = config
        self._robot_service = robot_service
        self.topics = Topics()
        self.bus = create_bus_client(config.deployment.bus)
        self.events = BusEventPublisher(self.bus, self.topics)
        self.codecs = CodecRegistry()
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
        self.capabilities = CapabilityRuntime(config)
        self.states = {
            policy_id: _SkillControllerState(
                spec=spec,
                codec=self.codecs.get(str(spec.settings.get("codec", spec.type))),
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
        if self._robot_service is not None:
            for service_id, svc_spec in self.config.capability_services.items():
                if svc_spec.type != "vla_service" or not svc_spec.enabled:
                    continue
                robot_id = svc_spec.robot_id
                if not robot_id:
                    continue
                driver = self._robot_service.get(robot_id)
                if driver is None:
                    continue
                io = driver.create_vla_io_adapter(**svc_spec.settings)
                if io is not None:
                    self.capabilities.set_vla_io_adapter(io)
                    logger.info(
                        f"VLA I/O adapter injected service={service_id} robot={robot_id}"
                    )
                    break
        await self.bus.connect()
        if self.human_follow is not None:
            await self.human_follow.start()
        for policy_id, state in self.states.items():
            state.runtime = self._load_policy_runtime(policy_id, state)
        await self.bus.subscribe([self.topics.robot_observation], self._on_observation)
        await self.bus.subscribe([self.topics.skill_intent], self._on_skill_intent)
        await self.bus.subscribe([self.topics.robot_status], self._on_status)
        await asyncio.gather(
            *(
                self._skill_loop(policy_id, state)
                for policy_id, state in self.states.items()
            )
        )

    async def stop(self) -> None:
        self._stop.set()
        tasks = []
        for state in self.states.values():
            if state.runtime is not None:
                tasks.append(state.runtime.close())
            for run in state.active_runs.values():
                if run.task is not None:
                    run.task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.bus.close()

    async def _on_observation(self, _topic: str, payload: dict[str, Any]) -> None:
        observation = from_payload(RobotObservation, payload)
        for state in self.states.values():
            if state.spec.robot_id == observation.envelope.robot_id:
                state.latest_observation = observation

    async def _on_skill_intent(self, _topic: str, payload: dict[str, Any]) -> None:
        intent = from_payload(SkillIntent, payload)
        state_item = self._state_for_robot(intent.envelope.robot_id)
        if state_item is None:
            return
        policy_id, state = state_item
        if intent.interrupt:
            await self._interrupt_active(policy_id, state, intent)
            return
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

    async def _accept_skill(
        self, policy_id: str, state: _SkillControllerState, intent: SkillIntent
    ) -> None:
        contract, decision = self.skill_runtime.validate(
            intent.name,
            intent.arguments,
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
                if status.skill_id:
                    logger.warning(
                        "skill_status_trace ignored "
                        f"robot={status.envelope.robot_id} skill_id={status.skill_id} "
                        f"reason={'missing_run' if run is None else 'terminal_run'}"
                    )
                continue
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
        runtime = state.runtime or self._load_policy_runtime(policy_id, state)
        state.runtime = runtime
        period = getattr(runtime, "control_period_sec", 0.05)
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
            dict(intent.arguments),
            context_factory=lambda invoke: self._plugin_context(
                policy_id, state, run, invoke
            ),
            enabled_only=False,
            status=state.latest_status,
            robot_type=self._robot_type(state.spec.robot_id),
        )
        if state.active_runs.get(intent.skill_id) is not run or run.terminal:
            return
        await self._finish_run(
            policy_id,
            state,
            run,
            success=bool(result.success),
            summary=str(result.summary),
            status=str(result.status),
            failure_mode=getattr(result, "failure_mode", None),
            error=getattr(result, "error", None),
        )

    async def _finish_run(
        self,
        policy_id: str,
        state: _SkillControllerState,
        run: SkillRun,
        *,
        success: bool,
        summary: str,
        status: str,
        failure_mode: str | None = None,
        error: str | None = None,
    ) -> None:
        intent = run.intent
        final_summary = self._completion_summary(run, summary) if success else summary
        run.terminal = True
        state.scheduler.remove(intent.skill_id)
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
        predictor = getattr(state.runtime, "predict", None)
        if callable(predictor):
            await predictor(
                SimpleNamespace(
                    observation=state.latest_observation,
                    intent=action_intent,
                )
            )
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

    async def _invoke_capability(
        self,
        run: SkillRun,
        name: str,
        _arguments: dict[str, Any],
    ) -> CapabilityExecutionResult:
        capability = self.capabilities.service_for(name, run.intent.envelope.robot_id)
        if capability is None:
            raise RuntimeError(f"{name} requires a deployed capability service")
        service_id, spec, client = capability
        health = await client.health()
        if not health.online or not health.loaded or health.busy:
            reason = health.error or (
                f"capability {service_id} is busy"
                if health.busy
                else f"capability {service_id} is not deployed or model is not loaded"
            )
            raise RuntimeError(reason)
        contract = self.plugin_skill_catalog.resolve(name)
        run.execution_plan = SkillExecutionPlan(
            actions=(
                *run.execution_plan.actions,
                RobotSkillAction(name, dict(_arguments)),
            ),
            strategy="runtime_trace",
            notes=("Recorded from actual capability execution.",),
        )
        result = await client.execute(
            CapabilityExecutionRequest(
                service_id=service_id,
                intent=run.intent,
                contract=contract,
                timeout_sec=float(
                    run.intent.timeout_sec or spec.timeout_sec or run.timeout_sec
                ),
            )
        )
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

    def _load_policy_runtime(
        self, policy_id: str, state: _SkillControllerState
    ) -> PolicyRuntime:
        return build_policy_runtime(
            policy_id,
            state.spec,
            config=self.config,
            codec=state.codec,
            media_resolver=self.media_resolver,
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
        contract: RobotSkillSpec | None = None,
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
        status: str,
        success: bool,
        summary: str,
        *,
        frame_id: int | None = None,
        error: str | None = None,
        failure_mode: str | None = None,
        steps_executed: int = 0,
        contract: RobotSkillSpec | None = None,
        run: SkillRun | None = None,
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
        contract: RobotSkillSpec,
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
            name=action.name,
            arguments=dict(action.arguments),
            objective=intent.objective,
            priority=intent.priority,
            interrupt=False,
            timeout_sec=intent.timeout_sec,
            feedback_mode=intent.feedback_mode,
            metadata={
                **dict(intent.metadata),
                "skill": intent.name or "",
                "primitive_action": action.name,
            },
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
            )
        )

        return SkillContext(
            skill_id=run.intent.skill_id,
            robot_id=run.intent.envelope.robot_id,
            robot=robot,
            perception=PerceptionPort(robot),
            capabilities=CapabilityPort(
                lambda name, arguments: self._invoke_capability(run, name, arguments)
            ),
            observation=state.latest_observation,
            current_observation=lambda: state.latest_observation,
            resolve_images=self.media_resolver.resolve_images,
            logger=logger,
            invoke=invoke_skill,
            progress=lambda **kwargs: self._plugin_progress(
                policy_id, state, run, **kwargs
            ),
            human_follow=self.human_follow,
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
        contract: RobotSkillSpec | None = None,
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
        contract: RobotSkillSpec,
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

    async def _interrupt_active(
        self, policy_id: str, state: _SkillControllerState, interrupt: SkillIntent
    ) -> None:
        self.plugin_skill_catalog.resolve(
            "stop_motion",
            robot_type=self._robot_type(state.spec.robot_id),
        )
        active_runs = [run for run in state.active_runs.values() if not run.terminal]
        for run in active_runs:
            run.terminal = True
            if run.task is not None:
                run.task.cancel()
            await self._publish_event(
                run.intent,
                "interrupted",
                summary="skill interrupted by user",
                error=interrupt.objective or "interrupt requested",
                policy_id=policy_id,
                steps_executed=run.steps_executed,
                contract=run.contract,
                execution_plan=run.execution_plan,
            )
            await self._publish_result(
                run.intent,
                "interrupted",
                False,
                "skill interrupted by user",
                error=interrupt.objective or "interrupt requested",
                failure_mode="interrupted",
                steps_executed=run.steps_executed,
                contract=run.contract,
            )
            state.scheduler.remove(run.intent.skill_id)
            await self._publish_scheduler_state(
                policy_id,
                state,
                phase="interrupted",
                intent=run.intent,
                contract=run.contract,
                decision={
                    "reason": "interrupted",
                    "error": interrupt.objective or "interrupt requested",
                },
                severity="warn",
            )
        stop_intent = SkillIntent(
            envelope=interrupt.envelope,
            skill_id=interrupt.skill_id,
            name="stop_motion",
            arguments={"emergency": True},
            objective=interrupt.objective or "interrupt active skill",
            interrupt=False,
            timeout_sec=3.0,
            feedback_mode="none",
            metadata={
                **dict(interrupt.metadata),
                "source": "skill_controller.interrupt",
            },
        )
        await self._accept_skill(policy_id, state, stop_intent)

    @staticmethod
    def _precondition_block(
        contract: RobotSkillSpec, status: RobotStatus | None
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
        message = str(last_result.get("message") or "").strip()
        skill = last_result.get("skill")
        skill_name = (
            str(skill.get("name") or "").strip() if isinstance(skill, dict) else ""
        )
        if message:
            return message
        if skill_name:
            return f"{skill_name} completed"
        return None
