from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hey_robot.capability.vla.executor import VLAExecutor
    from hey_robot.capability.vla.io_adapter import VLAIOAdapter

from hey_robot.capability.runtime.models import (
    CapabilityExecutionRequest,
    CapabilityExecutionResult,
    CapabilityHealth,
)
from hey_robot.config import CapabilityServiceSpec
from hey_robot.logging import HeyRobotLogger

logger = HeyRobotLogger(name="capability.vla.client")


class VLACapabilityClient:
    """In-process capability client that drives a VLAExecutor.

    Receives a pre-built *VLAIOAdapter* from the wiring layer so the same
    client works for simulation and real-robot without any internal branching.
    """

    def __init__(
        self,
        service_id: str,
        spec: CapabilityServiceSpec,
        *,
        io: VLAIOAdapter | None = None,
    ) -> None:
        self.service_id = service_id
        self.spec = spec
        self._io: VLAIOAdapter | None = io
        self._executor: VLAExecutor | None = None
        self._busy = False

    def set_io(self, io: VLAIOAdapter) -> None:
        self._io = io

    def _ensure_executor(self, task_prompt: str | None = None) -> VLAExecutor:
        from hey_robot.capability.vla.executor import VLAExecutor
        from hey_robot.robots.simulation.vla_adapter import (
            build_policy_client,
            build_vla_config,
        )

        if self._io is None:
            raise RuntimeError(
                "VLACapabilityClient has no I/O adapter. "
                "Call set_io() before execute()."
            )
        settings = dict(self.spec.settings)
        config = build_vla_config(settings, task_prompt=task_prompt)
        policy = build_policy_client(config)
        self._executor = VLAExecutor(self._io, policy)
        return self._executor

    async def health(self) -> CapabilityHealth:
        online = self._io is not None and self._io.ready()
        return CapabilityHealth(
            name=self.service_id,
            online=online,
            loaded=True,
            busy=self._busy,
            robot_id=self.spec.robot_id,
            metrics={
                "type": self.spec.type,
                "policy_runtime": self.spec.settings.get("policy_runtime", "fake"),
                "arm": self.spec.settings.get("arm", "right"),
            },
        )

    async def execute(
        self, request: CapabilityExecutionRequest
    ) -> CapabilityExecutionResult:
        from hey_robot.capability.vla.schemas import VLARequest
        from hey_robot.robots.simulation.vla_adapter import build_vla_config

        if self._busy:
            return CapabilityExecutionResult(
                success=False,
                status="failed",
                summary=f"capability {self.service_id} is busy",
                failure_mode="capability_busy",
            )
        self._busy = True
        try:
            task_prompt = str(
                request.intent.arguments.get("task_prompt")
                or request.intent.objective
                or ""
            )
            executor = self._ensure_executor(task_prompt=task_prompt)
            settings = dict(self.spec.settings)
            config = build_vla_config(settings, task_prompt=task_prompt)
            vla_request = VLARequest(
                config=config,
                skill_id=request.intent.skill_id,
                episode_id=request.intent.envelope.episode_id,
                arguments=dict(request.intent.arguments),
            )
            result = await asyncio.to_thread(executor.execute, vla_request)
            return CapabilityExecutionResult(
                success=result.success,
                status=result.status,
                summary=result.summary,
                failure_mode=result.failure_mode,
                error=result.error,
                metrics=result.metrics,
            )
        except Exception as exc:
            return CapabilityExecutionResult(
                success=False,
                status="failed",
                summary=f"VLA execution failed: {type(exc).__name__}: {exc}",
                failure_mode="execution_failed",
                error=str(exc),
            )
        finally:
            self._busy = False

    async def cancel(self, _skill_id: str) -> None:
        if self._executor is not None:
            self._executor.cancel()
