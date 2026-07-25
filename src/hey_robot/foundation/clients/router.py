"""Thin ModelRouter adapter over the current model service registry."""

from __future__ import annotations

import asyncio

from hey_robot.foundation.clients.manager import ModelServiceRegistry
from hey_robot.foundation.clients.models import (
    ModelInferenceResult,
    ServiceInvocationRequest,
)
from hey_robot.protocol import Envelope, SkillIntent


class RegistryModelRouter:
    def __init__(self, registry: ModelServiceRegistry) -> None:
        self._registry = registry

    async def infer(
        self,
        capability: str,
        request: dict,
        *,
        run_id: str,
        robot_id: str,
        timeout_sec: float | None = None,
    ) -> ModelInferenceResult:
        service = self._registry.service_for(capability, robot_id)
        if service is None:
            return ModelInferenceResult(
                False,
                f"model capability {capability!r} is unavailable",
                failure_mode="model_service_unavailable",
                error="no enabled model service provides the requested capability",
            )
        service_id, spec, client = service
        result = await client.execute(
            request=ServiceInvocationRequest(
                service_id=service_id,
                intent=SkillIntent(
                    envelope=Envelope(robot_id=robot_id),
                    skill_id=run_id,
                    task_id=run_id,
                    intent_kind="skill",
                    name=capability,
                    arguments=dict(request),
                    objective=f"infer {capability}",
                ),
                timeout_sec=timeout_sec or spec.timeout_sec,
                arguments=dict(request),
            )
        )
        return ModelInferenceResult(
            result.success,
            result.summary,
            data=dict(result.metrics),
            failure_mode=result.failure_mode,
            error=result.error,
        )

    async def cancel(self, run_id: str) -> None:
        results = await asyncio.gather(
            *(client.cancel(run_id) for client in self._registry.clients.values()),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(
                f"{len(failures)} model service cancellation request(s) failed"
            ) from failures[0]
