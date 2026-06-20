from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from hey_robot.config import DeploymentConfig, PolicySpec
from hey_robot.media import MediaResolver
from hey_robot.perception import ObservationActionCodec
from hey_robot.protocol import RobotAction, RobotObservation, SkillIntent


@dataclass(frozen=True)
class PolicyIOSchema:
    policy_id: str
    policy_type: str
    robot_id: str
    observation_schema: dict[str, Any] = field(default_factory=dict)
    action_schema: dict[str, Any] = field(default_factory=dict)
    control_frequency_hz: float = 20.0
    device: str = "cpu"
    accelerator: str | None = None
    stateful: bool = True
    supports_interrupt: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyHealth:
    policy_id: str
    loaded: bool
    device: str
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyRuntimeInput:
    observation: RobotObservation
    intent: SkillIntent


@dataclass(frozen=True)
class PolicyRuntimeOutput:
    action: RobotAction
    metadata: dict[str, Any] = field(default_factory=dict)


class PolicyAdapter(Protocol):
    async def warmup(self) -> None: ...

    async def predict(
        self, policy_input: Any, observation: RobotObservation, intent: SkillIntent
    ) -> Any: ...

    async def close(self) -> None: ...

    def health(self) -> PolicyHealth: ...

    def schema(self) -> PolicyIOSchema: ...


class PolicyRuntime:
    """Standard policy runtime boundary for simulator and real robot policies."""

    def __init__(
        self,
        *,
        policy_id: str,
        spec: PolicySpec,
        codec: ObservationActionCodec,
        adapter: PolicyAdapter,
    ) -> None:
        self.policy_id = policy_id
        self.spec = spec
        self.codec = codec
        self.adapter = adapter

    @property
    def control_period_sec(self) -> float:
        return 1.0 / max(float(self.schema().control_frequency_hz), 0.1)

    async def warmup(self) -> None:
        await self.adapter.warmup()

    async def close(self) -> None:
        await self.adapter.close()

    def health(self) -> PolicyHealth:
        return self.adapter.health()

    def schema(self) -> PolicyIOSchema:
        return self.adapter.schema()

    async def predict(self, payload: PolicyRuntimeInput) -> PolicyRuntimeOutput:
        policy_input = self.codec.observation_to_policy_input(
            payload.observation, payload.intent
        )
        output = await self.adapter.predict(
            policy_input, payload.observation, payload.intent
        )
        base_action = self.codec.policy_output_to_action(
            output, payload.observation, payload.intent
        )
        return PolicyRuntimeOutput(
            action=RobotAction(
                envelope=base_action.envelope,
                values=base_action.values,
                action_id=base_action.action_id,
                skill_id=base_action.skill_id,
                timestamp=base_action.timestamp,
                metadata={
                    **base_action.metadata,
                    **{
                        key: value
                        for key, value in payload.intent.metadata.items()
                        if key in {"control_source", "priority", "source"}
                    },
                    "policy_id": self.policy_id,
                    "policy_type": self.spec.type,
                    "device": self.schema().device,
                },
            ),
            metadata={"policy_id": self.policy_id, "schema": self.schema().__dict__},
        )


class MockPolicyAdapter:
    def __init__(self, policy_id: str, spec: PolicySpec) -> None:
        self.policy_id = policy_id
        self.spec = spec
        self.loaded = False

    async def warmup(self) -> None:
        self.loaded = True

    async def predict(
        self, _policy_input: Any, _observation: RobotObservation, _intent: SkillIntent
    ) -> list[float]:
        return [0.0]

    async def close(self) -> None:
        self.loaded = False

    def health(self) -> PolicyHealth:
        return PolicyHealth(
            policy_id=self.policy_id, loaded=self.loaded, device=self.spec.device
        )

    def schema(self) -> PolicyIOSchema:
        return PolicyIOSchema(
            policy_id=self.policy_id,
            policy_type=self.spec.type,
            robot_id=self.spec.robot_id,
            observation_schema={"modalities": ["proprioception", "metadata"]},
            action_schema={"type": "vector", "dimensions": 1},
            control_frequency_hz=float(self.spec.freq_hz),
            device=self.spec.device,
            accelerator=_accelerator_from_device(self.spec.device),
            stateful=False,
            metadata=dict(self.spec.settings),
        )


def build_policy_runtime(
    policy_id: str,
    spec: PolicySpec,
    *,
    config: DeploymentConfig,
    codec: ObservationActionCodec,
    media_resolver: MediaResolver,
) -> PolicyRuntime:
    _ = media_resolver
    if spec.type == "mock":
        adapter: PolicyAdapter = MockPolicyAdapter(policy_id, spec)
    elif spec.type == "skill":
        from hey_robot.policies.skill_policy import SkillPolicyAdapter
        from hey_robot.skills.registry import registry_from_config

        adapter = SkillPolicyAdapter(
            policy_id,
            spec,
            skill_catalog=registry_from_config(config).robot_skill_catalog(),
        )
    else:
        raise ValueError(f"unsupported policy type: {spec.type}")
    return PolicyRuntime(policy_id=policy_id, spec=spec, codec=codec, adapter=adapter)


def _accelerator_from_device(device: str) -> str | None:
    lowered = str(device).lower()
    if lowered.startswith("cuda"):
        return "cuda"
    if lowered.startswith("mps"):
        return "mps"
    if lowered.startswith("cpu"):
        return "cpu"
    return lowered or None
