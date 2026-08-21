"""Local lifecycle adapter for independently hosted foundation-model workers."""

from __future__ import annotations

import asyncio
from typing import Any

from hey_robot.config import ModelServiceSpec
from hey_robot.foundation.clients.models import (
    ServiceHealth,
    ServiceInvocationRequest,
    ServiceInvocationResult,
)


class LocalFoundationClient:
    """Own a local model worker without placing model inference behind gRPC.

    Policy implementations may start their own isolated process.  This adapter
    only bridges the Harness process to that local worker and never opens a
    network model-service endpoint.
    """

    def __init__(self, service_id: str, spec: ModelServiceSpec) -> None:
        self.service_id = service_id
        self.spec = spec
        self._executor = _build_executor(service_id, spec)

    async def health(self) -> ServiceHealth:
        payload = await asyncio.to_thread(self._executor.health)
        return ServiceHealth(
            name=str(payload.get("name") or self.service_id),
            online=bool(payload.get("online")),
            loaded=bool(payload.get("loaded")),
            robot_id=str(payload.get("robot_id") or self.spec.robot_id),
            error=payload.get("error"),
            metrics=dict(payload.get("metrics") or {}),
        )

    async def execute(
        self, request: ServiceInvocationRequest
    ) -> ServiceInvocationResult:
        payload = {
            "skill_name": request.intent.name,
            "episode_id": request.intent.envelope.episode_id,
            "objective": request.intent.objective,
            "arguments": dict(request.arguments or {}),
        }
        result = await asyncio.wait_for(
            asyncio.to_thread(self._executor.execute, payload),
            timeout=request.timeout_sec,
        )
        return ServiceInvocationResult(
            success=bool(result.get("success")),
            status=str(result.get("status") or "failed"),
            summary=str(
                result.get("summary") or "foundation model returned no summary"
            ),
            failure_mode=result.get("failure_mode"),
            error=result.get("error"),
            metrics=dict(result.get("metrics") or {}),
        )

    async def cancel(self, skill_id: str) -> None:
        del skill_id
        await asyncio.to_thread(self._executor.cancel)


def _build_executor(service_id: str, spec: ModelServiceSpec) -> Any:
    if spec.type == "robot_policy":
        runtime = str(spec.settings.get("runtime") or "")
        if runtime == "lerobot":
            from hey_robot.foundation.backends.lerobot import LeRobotPolicyExecutor

            return LeRobotPolicyExecutor(service_id, spec)
        if runtime == "rldx":
            from hey_robot.foundation.backends.rldx import RLDXPolicyExecutor

            return RLDXPolicyExecutor(service_id, spec)
        if runtime == "xiaomi":
            from hey_robot.foundation.backends.xiaomi import XiaomiPolicyExecutor

            return XiaomiPolicyExecutor(service_id, spec)
        raise ValueError(f"unsupported local robot policy runtime {runtime!r}")
    if spec.type == "vln_planner":
        from hey_robot.foundation.backends.vln import VLNPlannerExecutor

        return VLNPlannerExecutor(service_id, spec)
    raise ValueError(f"unsupported local foundation model type {spec.type!r}")
