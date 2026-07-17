"""Deterministic test harness for Phase 4 simulation and fault-injection tests.

Provides:
- ScriptedReasoningProvider: returns pre-programmed model responses per turn
- FakeRobot: simulates skill execution with configurable outcomes
- build_autonomous_test_harness: assembles supervisor + agent + fakes
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hey_robot.cognition.autonomous.supervisor import AutonomySupervisorService
from hey_robot.cognition.robot_agent_service import RobotAgentService
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import (
    ActionProposal,
    DeliberationResult,
    Envelope,
    EvidenceFact,
    FailurePayload,
    GoalCommand,
    RobotStatus,
    SkillResult,
    SuccessCriterion,
    Topics,
)
from hey_robot.protocol.messages import to_payload
from hey_robot.providers import ReasoningMessage, ReasoningResponse, ReasoningToolCall

# ── Scripted Provider ──────────────────────────────────────────────────────


@dataclass
class ScriptedTurn:
    """One pre-programmed model response."""

    tool_name: str = ""
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    finish_reason: str = "stop"
    delay_sec: float = 0.0


class ScriptedReasoningProvider:
    """Returns scripted responses in order.  Raises if more turns requested than programmed."""

    def __init__(self, turns: list[ScriptedTurn]) -> None:
        self._turns = turns
        self._idx = 0
        self.calls: list[dict[str, Any]] = []

    def get_default_model(self) -> str:
        return "scripted"

    async def chat(
        self, *, messages: list[ReasoningMessage], tools: list[dict[str, Any]], **_: Any
    ) -> ReasoningResponse:
        system_text = messages[0].content if messages else ""
        self.calls.append(
            {"system": system_text, "tools": [t["function"]["name"] for t in tools]}
        )
        if self._idx >= len(self._turns):
            raise RuntimeError(f"scripted provider exhausted at turn {self._idx}")
        turn = self._turns[self._idx]
        self._idx += 1
        if turn.delay_sec:
            await asyncio.sleep(turn.delay_sec)
        if turn.tool_name:
            return ReasoningResponse(
                tool_calls=[
                    ReasoningToolCall(_new_id(), turn.tool_name, turn.tool_arguments)
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )
        return ReasoningResponse(
            content=turn.text,
            finish_reason=turn.finish_reason,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


# ── Fake Bus ────────────────────────────────────────────────────────────────


class FakeBus:
    """In-memory bus that captures published messages and optionally replays."""

    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []
        self._subscribers: dict[str, list] = {}

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))
        for handler in self._subscribers.get(topic, []):
            await handler(topic, payload)

    async def subscribe(self, topics: list[str], handler) -> None:
        for t in topics:
            self._subscribers.setdefault(t, []).append(handler)

    def publish_sync(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))


# ── Test Harness Builder ──────────────────────────────────────────────────


@dataclass
class AutonomousTestHarness:
    supervisor: AutonomySupervisorService
    agent: RobotAgentService
    bus: FakeBus
    topics: Topics
    config: DeploymentConfig
    provider: ScriptedReasoningProvider
    envelope: Envelope
    robot_id: str
    agent_id: str


def build_harness(
    tmp_path: Path,
    *,
    turns: list[ScriptedTurn],
    robot_id: str = "test_robot",
    agent_id: str = "main",
    entity_catalog: tuple[str, ...] = ("robot:test_robot", "scene"),
    enabled_skills: tuple[str, ...] = ("inspect_scene", "navigate_to", "stop_motion"),
    hard_max_wall_time_sec: float = 3600.0,
    hard_max_deliberations: int = 40,
    hard_max_skills: int = 24,
) -> AutonomousTestHarness:
    """Build supervisor + agent wired to a ScriptedProvider and FakeBus."""
    import os

    os.environ.setdefault("DEEPSEEK_MODEL", "scripted")
    os.environ.setdefault("DEEPSEEK_API_KEY", "sk-fake")
    os.environ.setdefault("DEEPSEEK_BASE_URL", "https://api.example.com")

    config = DeploymentConfig.from_dict(
        {
            "deployment": {"id": "phase4-test"},
            "resources": {"runtime_dir": str(tmp_path / "runtime")},
            "autonomy": {
                "enabled": True,
                "robot_id": robot_id,
                "entity_catalog": list(entity_catalog),
                "hard_max_wall_time_sec": hard_max_wall_time_sec,
                "hard_max_deliberations": hard_max_deliberations,
                "hard_max_skills": hard_max_skills,
            },
            "agents": {
                agent_id: {
                    "enabled": True,
                    "robot_id": robot_id,
                    "settings": {
                        "providers": {
                            "planner": {
                                "type": "deepseek",
                                "model": "scripted",
                                "api_key": "sk-fake",
                                "temperature": 0.0,
                                "max_tokens": 256,
                            }
                        }
                    },
                }
            },
            "skills": {
                "modules": ["hey_robot.skill_os.builtins"],
                "enabled": list(enabled_skills),
            },
        }
    )

    provider = ScriptedReasoningProvider(turns)
    bus = FakeBus()
    topics = Topics()

    supervisor = AutonomySupervisorService(config)
    supervisor.bus = bus  # type: ignore[assignment]

    agent = RobotAgentService(config, agent_id=agent_id)
    agent.runner._provider = provider  # type: ignore[attr-defined]
    agent.bus = bus  # type: ignore[assignment]

    return AutonomousTestHarness(
        supervisor=supervisor,
        agent=agent,
        bus=bus,
        topics=topics,
        config=config,
        provider=provider,
        envelope=Envelope(robot_id=robot_id, agent_id=agent_id),
        robot_id=robot_id,
        agent_id=agent_id,
    )


# ── Helpers ─────────────────────────────────────────────────────────────────


async def set_robot_ready(h: AutonomousTestHarness) -> None:
    """Set robot status to idle, ready for execution."""
    await h.supervisor._on_snapshot(
        h.topics.robot_status,
        to_payload(
            RobotStatus(
                h.envelope,
                state="idle",
                battery_percentage=100.0,
                location_id="room:start",
                frame_id=1,
            )
        ),
    )


async def create_goal(
    h: AutonomousTestHarness,
    *,
    objective: str = "check scene",
    criteria: tuple[SuccessCriterion, ...] = (),
) -> str:
    """Create a goal and return its goal_id."""
    if not criteria:
        criteria = (
            SuccessCriterion(
                "c1",
                "evidence_present",
                f"robot:{h.robot_id}",
                "observed",
                "scene",
                300,
            ),
        )
    command = GoalCommand(
        h.envelope,
        str(uuid.uuid4()),
        "create",
        objective=objective,
        success_criteria=criteria,
    )
    await h.supervisor._on_goal_command(h.topics.goal_command, to_payload(command))
    await asyncio.sleep(0.1)
    goals = h.supervisor.store.goals_recent(1)
    assert len(goals) == 1, f"goal was not created, got {len(goals)}"
    return goals[0]["goal_id"]


async def run_deliberation_turn(h: AutonomousTestHarness) -> bool:
    """Run one full deliberation turn: agent processes request → supervisor gets result.

    Returns True if the goal is still active (not terminal).
    """
    delibs = [p for t, p in h.bus.published if t == h.topics.agent_deliberation]
    if not delibs:
        return False
    request_payload = delibs[-1]

    # Agent processes the request
    await h.agent._on_deliberation(h.topics.agent_deliberation, request_payload)
    await asyncio.sleep(0.1)

    # Extract the deliberation result and route to supervisor
    results = [p for t, p in h.bus.published if t == h.topics.agent_deliberation_result]
    if results:
        result_payload = results[-1]
        await h.supervisor._on_deliberation_result(
            h.topics.agent_deliberation_result, result_payload
        )
        await asyncio.sleep(0.1)

    status = goal_status(h)
    return status not in ("completed", "failed", "cancelled")


async def send_deliberation_result(
    h: AutonomousTestHarness,
    *,
    proposal: ActionProposal | None = None,
    failed: bool = False,
) -> None:
    """Have the supervisor process a deliberation result with the current active goal."""
    goals = h.supervisor.store.goals_recent(1)
    assert len(goals) == 1
    g = goals[0]
    status: str = (
        "failed" if failed else ("action_proposed" if proposal else "completed")
    )
    result = DeliberationResult(
        h.envelope,
        g["active_deliberation_id"],
        h.supervisor.store.deliberation_request_hash(
            goal_id=g["goal_id"], deliberation_id=g["active_deliberation_id"]
        )
        or "",
        g["goal_id"],
        g["snapshot"]["task_id"],
        status,  # type: ignore[arg-type]
        proposal=proposal,
        failure=(
            FailurePayload("MODEL_REQUEST", "PROVIDER_TIMEOUT", "test", "timeout")
            if failed
            else None
        ),
    )
    await h.supervisor._on_deliberation_result(
        h.topics.agent_deliberation_result, to_payload(result)
    )
    await asyncio.sleep(0.1)


async def send_skill_result(
    h: AutonomousTestHarness,
    *,
    skill_id: str = "",
    success: bool = True,
    status: str = "completed",
    evidence: tuple[EvidenceFact, ...] = (),
) -> None:
    """Send a terminal SkillResult for the given skill_id (or most recent intent)."""
    if not skill_id:
        intents = [p for t, p in h.bus.published if t == h.topics.skill_intent]
        if intents:
            skill_id = intents[-1]["skill_id"]
        else:
            skill_id = "unknown_skill"
    result = SkillResult(
        h.envelope,
        skill_id,
        status=status,  # type: ignore[arg-type]
        success=success,
        evidence=evidence,
    )
    await h.supervisor._on_skill_result(h.topics.skill_result, to_payload(result))
    await asyncio.sleep(0.1)


def last_intent(h: AutonomousTestHarness) -> dict | None:
    """Return the most recently published SkillIntent payload."""
    intents = [p for t, p in h.bus.published if t == h.topics.skill_intent]
    return intents[-1] if intents else None


def last_intent_of_name(h: AutonomousTestHarness, name: str) -> dict | None:
    """Return the most recent SkillIntent with matching skill name."""
    intents = [p for t, p in h.bus.published if t == h.topics.skill_intent]
    for intent in reversed(intents):
        if intent.get("name") == name:
            return intent
    return None


def goal_status(h: AutonomousTestHarness) -> str:
    goals = h.supervisor.store.goals_recent(1)
    return goals[0]["status"] if goals else "no_goal"


def goal(h: AutonomousTestHarness) -> dict[str, Any]:
    goals = h.supervisor.store.goals_recent(1)
    assert goals, "no goal exists"
    return goals[0]


def _new_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
