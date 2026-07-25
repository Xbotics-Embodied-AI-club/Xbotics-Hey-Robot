from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from hey_robot.protocol import SkillIntent


@dataclass(frozen=True)
class ServiceHealth:
    name: str
    online: bool
    loaded: bool = True
    busy: bool = False
    robot_id: str = ""
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    current_skill_id: str | None = None
    error_code: str | None = None
    version: str | None = None


@dataclass(frozen=True)
class ServiceInvocationRequest:
    service_id: str
    intent: SkillIntent
    timeout_sec: float
    arguments: dict[str, Any] | None = None


@dataclass(frozen=True)
class ServiceInvocationResult:
    success: bool
    summary: str
    status: str = "completed"
    failure_mode: str | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None


@dataclass(frozen=True)
class PolicyStepRequest:
    policy_session_id: str | None
    skill_name: str
    atomic_command: str
    observation: dict[str, Any] = field(default_factory=dict)
    proprioception: dict[str, Any] = field(default_factory=dict)
    frame_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyStepResult:
    kind: str
    action_space: str | None = None
    embodiment: str | None = None
    horizon: int | None = None
    dt: float | None = None
    actions: list[dict[str, Any]] = field(default_factory=list)
    local_goal: dict[str, Any] = field(default_factory=dict)
    whole_body_reference: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    confidence: float | None = None
    valid: bool = True
    failure_mode: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_metrics(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "action_space": self.action_space,
            "embodiment": self.embodiment,
            "horizon": self.horizon,
            "dt": self.dt,
            "actions": self.actions,
            "local_goal": self.local_goal,
            "whole_body_reference": self.whole_body_reference,
            "done": self.done,
            "confidence": self.confidence,
            "valid": self.valid,
            "failure_mode": self.failure_mode,
            "raw": self.raw,
        }


@dataclass(frozen=True)
class ModelInferenceResult:
    success: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    failure_mode: str | None = None
    error: str | None = None


class ModelRouter(Protocol):
    async def infer(
        self,
        capability: str,
        request: dict[str, Any],
        *,
        run_id: str,
        robot_id: str,
        timeout_sec: float | None = None,
    ) -> ModelInferenceResult: ...

    async def cancel(self, run_id: str) -> None: ...


class ModelServiceClient(Protocol):
    async def health(self) -> ServiceHealth: ...

    async def execute(
        self, request: ServiceInvocationRequest
    ) -> ServiceInvocationResult: ...

    async def cancel(self, skill_id: str) -> None: ...
