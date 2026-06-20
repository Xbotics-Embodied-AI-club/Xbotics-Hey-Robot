from __future__ import annotations

from hey_robot.capability.runtime.models import (
    CapabilityExecutionRequest,
    CapabilityExecutionResult,
    CapabilityHealth,
)
from hey_robot.config import CapabilityServiceSpec


class MockCapabilityClient:
    def __init__(self, service_id: str, spec: CapabilityServiceSpec) -> None:
        self.service_id = service_id
        self.spec = spec
        self.executed: list[CapabilityExecutionRequest] = []
        self.cancelled: list[str] = []

    async def health(self) -> CapabilityHealth:
        return CapabilityHealth(
            name=self.service_id,
            online=bool(self.spec.settings.get("online", True)),
            loaded=bool(self.spec.settings.get("loaded", True)),
            busy=bool(self.spec.settings.get("busy", False)),
            robot_id=self.spec.robot_id,
            error=self.spec.settings.get("error"),
            metrics=dict(self.spec.settings.get("metrics", {}) or {}),
        )

    async def execute(
        self, request: CapabilityExecutionRequest
    ) -> CapabilityExecutionResult:
        self.executed.append(request)
        success = bool(self.spec.settings.get("success", True))
        return CapabilityExecutionResult(
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
            else str(self.spec.settings.get("error", "capability execution failed")),
            metrics=dict(self.spec.settings.get("result_metrics", {}) or {}),
        )

    async def cancel(self, skill_id: str) -> None:
        self.cancelled.append(skill_id)
