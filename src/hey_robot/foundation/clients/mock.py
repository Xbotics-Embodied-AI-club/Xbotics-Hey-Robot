from __future__ import annotations

from hey_robot.config import ModelServiceSpec
from hey_robot.foundation.clients.models import (
    ServiceHealth,
    ServiceInvocationRequest,
    ServiceInvocationResult,
)


class MockModelServiceClient:
    def __init__(self, service_id: str, spec: ModelServiceSpec) -> None:
        self.service_id = service_id
        self.spec = spec
        self.executed: list[ServiceInvocationRequest] = []
        self.cancelled: list[str] = []

    async def health(self) -> ServiceHealth:
        return ServiceHealth(
            name=self.service_id,
            online=bool(self.spec.settings.get("online", True)),
            loaded=bool(self.spec.settings.get("loaded", True)),
            busy=bool(self.spec.settings.get("busy", False)),
            robot_id=self.spec.robot_id,
            error=self.spec.settings.get("error"),
            metrics=dict(self.spec.settings.get("metrics", {}) or {}),
        )

    async def execute(
        self, request: ServiceInvocationRequest
    ) -> ServiceInvocationResult:
        self.executed.append(request)
        success = bool(self.spec.settings.get("success", True))
        return ServiceInvocationResult(
            success=success,
            status="completed" if success else "failed",
            summary=str(
                self.spec.settings.get("summary") or f"{request.intent.name} completed"
            ),
            failure_mode=None
            if success
            else str(self.spec.settings.get("failure_mode", "execution_failed")),
            error=None
            if success
            else str(
                self.spec.settings.get("error", "model service invocation failed")
            ),
            metrics=dict(self.spec.settings.get("result_metrics", {}) or {}),
        )

    async def cancel(self, skill_id: str) -> None:
        self.cancelled.append(skill_id)
