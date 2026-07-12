"""Deliberation consumer with no access to the Supervisor store or Skill OS."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from hey_robot.bus.factory import create_bus_client
from hey_robot.cognition.autonomous.context_builder import (
    build_context,
)
from hey_robot.cognition.policy.task_evaluator import TaskEvaluator
from hey_robot.cognition.runtime.deliberation_store import DeliberationStore
from hey_robot.cognition.runtime.result import AgentRunRequest
from hey_robot.cognition.runtime.strict_runner import StrictAgentRunner
from hey_robot.cognition.task.contract import TaskContract
from hey_robot.cognition.tools.autonomous import (
    AgentToolDependencies,
    build_agent_tools,
)
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import (
    DeliberationRequest,
    DeliberationResult,
    FailurePayload,
    Topics,
)
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.providers import build_provider
from hey_robot.skill_os.registry import registry_from_config


class AutonomousRobotAgentService:
    def __init__(self, config: DeploymentConfig, *, agent_id: str) -> None:
        self.config = config
        self.agent_id = agent_id
        self.topics = Topics()
        self.bus = create_bus_client(config.deployment.bus, role="agent")
        root = Path(config.resources.runtime_dir) / config.deployment.id
        root.mkdir(parents=True, exist_ok=True)
        self.store = DeliberationStore(root / "agent_deliberations.sqlite3")
        catalog = registry_from_config(config).catalog(semantic_only=False)
        tools = build_agent_tools(AgentToolDependencies(catalog))
        self.runner = StrictAgentRunner(
            build_provider(config, agent_id, purpose="agent"), tools
        )
        self.evaluator = TaskEvaluator()

    async def start(self) -> None:
        await self.bus.connect()
        for request in self.store.interrupt_incomplete():
            result = self._failed(
                request,
                request_hash=_hash(to_payload(request)),
                stage="SUPERVISION",
                code="AGENT_PROCESS_INTERRUPTED",
                message="agent process restarted before deliberation became terminal",
            )
            self.store.terminal(result, _hash(to_payload(result)))
            await self.bus.publish(
                self.topics.agent_deliberation_result, to_payload(result)
            )
        await self.bus.subscribe([self.topics.agent_deliberation], self._on_request)
        await __import__("asyncio").Event().wait()

    async def stop(self) -> None:
        await self.bus.close()

    async def _on_request(self, _topic: str, payload: dict) -> None:
        request = from_payload(DeliberationRequest, payload)
        if request.envelope.agent_id and request.envelope.agent_id != self.agent_id:
            return
        request_hash = _hash(payload)
        old = self.store.result(request.deliberation_id)
        if old is not None:
            await self.bus.publish(
                self.topics.agent_deliberation_result, to_payload(old)
            )
            return
        if not self.store.schedule(
            deliberation_id=request.deliberation_id,
            request_hash=request_hash,
            goal_id=request.goal.goal_id,
            task_id=request.goal.task_id,
            request=request,
        ):
            return
        result = await self._run(request, request_hash)
        result_hash = _hash(to_payload(result))
        self.store.terminal(result, result_hash)
        await self.bus.publish(
            self.topics.agent_deliberation_result, to_payload(result)
        )

    async def _run(
        self, request: DeliberationRequest, request_hash: str
    ) -> DeliberationResult:
        self.store.transition(request.deliberation_id, "SCHEDULED", "BUILDING")
        contract = TaskContract(
            request.goal.contract_id,
            request.goal.task_id,
            request.goal.goal_id,
            request.goal.objective,
            request.goal.success_criteria,
            1,
            request.goal.contract_hash,
        )
        evaluation = self.evaluator.evaluate(contract, request.evidence)
        if evaluation.outcome == "satisfied":
            return DeliberationResult(
                request.envelope,
                request.deliberation_id,
                request_hash,
                request.goal.goal_id,
                request.goal.task_id,
                "completed",
                evaluation=evaluation,
            )
        if not self.store.transition(
            request.deliberation_id, "BUILDING", "BEFORE_MODEL_REQUEST"
        ):
            return self._failed(
                request,
                request_hash,
                "PERSISTENCE",
                "PERSISTENCE_FAILED",
                "could not record model boundary",
            )
        context = build_context(
            request,
            evaluation_text=self._evaluation_text(evaluation),
        )
        if context.failure is not None:
            return DeliberationResult(
                request.envelope,
                request.deliberation_id,
                request_hash,
                request.goal.goal_id,
                request.goal.task_id,
                "failed",
                failure=context.failure,
                evaluation=evaluation,
            )
        run = await self.runner.run(
            AgentRunRequest(
                context.messages,
                frozenset({"request_observation", "request_skill"}),
                time.monotonic() + 120.0,
                request.deliberation_id,
                request.deliberation_id,
            )
        )
        self.store.transition(
            request.deliberation_id, "BEFORE_MODEL_REQUEST", "MODEL_RESPONSE_RECEIVED"
        )
        if run.status == "action_proposed":
            return DeliberationResult(
                request.envelope,
                request.deliberation_id,
                request_hash,
                request.goal.goal_id,
                request.goal.task_id,
                "action_proposed",
                proposal=run.proposal,
                evaluation=evaluation,
            )
        if run.status == "returned":
            return self._failed(
                request,
                request_hash,
                "MODEL_PROTOCOL",
                "MODEL_STOPPED_BEFORE_GOAL",
                "model returned text before contract was satisfied",
            )
        assert run.failure is not None
        return DeliberationResult(
            request.envelope,
            request.deliberation_id,
            request_hash,
            request.goal.goal_id,
            request.goal.task_id,
            "failed",
            failure=run.failure,
            evaluation=evaluation,
        )

    @staticmethod
    def _evaluation_text(evaluation) -> str:
        if evaluation.outcome == "satisfied":
            return f"CONTRACT SATISFIED: {evaluation.reason}"
        missing = evaluation.missing_criteria_ids
        if missing:
            return f"INCONCLUSIVE — missing criteria: {', '.join(missing)}"
        return f"INCONCLUSIVE — {evaluation.reason}"

    @staticmethod
    def _failed(
        request: DeliberationRequest,
        request_hash: str,
        stage: str,
        code: str,
        message: str,
    ) -> DeliberationResult:
        return DeliberationResult(
            request.envelope,
            request.deliberation_id,
            request_hash,
            request.goal.goal_id,
            request.goal.task_id,
            "failed",
            failure=FailurePayload(stage, code, "AutonomousRobotAgentService", message),
        )


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
