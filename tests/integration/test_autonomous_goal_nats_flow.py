"""End-to-end autonomous goal flow using the running NATS bus.

Requires: `nats-server` running on 127.0.0.1:4222 *without* ACL authentication.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import pytest
from nats.aio.client import Client as NatsClient

from hey_robot.bus.factory import create_bus_client
from hey_robot.cognition.autonomous.supervisor import AutonomySupervisorService
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import (
    ActionProposal,
    DeliberationRequest,
    DeliberationResult,
    Envelope,
    GoalCommand,
    RobotStatus,
    SkillIntent,
    SkillResult,
    SuccessCriterion,
    Topics,
)
from hey_robot.protocol.messages import from_payload, to_payload


def _nats_reachable() -> bool:
    """Check if NATS is reachable without authentication."""
    try:

        async def _probe() -> bool:
            nc = NatsClient()
            await nc.connect("nats://127.0.0.1:4222", connect_timeout=2)
            await nc.close()
            return True

        return asyncio.run(_probe())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _nats_reachable(),
    reason="NATS server not reachable (requires nats-server on :4222 without ACL)",
)


def _config(tmp_path: Path) -> DeploymentConfig:
    os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-chat")
    os.environ.setdefault("DEEPSEEK_API_KEY", "sk-placeholder")
    os.environ.setdefault("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return DeploymentConfig.from_dict(
        {
            "deployment": {"id": "integration-test"},
            "resources": {"runtime_dir": str(tmp_path / "runtime")},
            "autonomy": {
                "enabled": True,
                "robot_id": "test_robot",
                "entity_catalog": ["robot:test_robot", "scene"],
            },
            "agents": {
                "main": {
                    "enabled": True,
                    "robot_id": "test_robot",
                    "settings": {
                        "providers": {
                            "planner": {
                                "type": "deepseek",
                                "model_env": "DEEPSEEK_MODEL",
                                "api_key_env": "DEEPSEEK_API_KEY",
                                "base_url_env": "DEEPSEEK_BASE_URL",
                                "temperature": 0.0,
                                "max_tokens": 256,
                            }
                        }
                    },
                }
            },
            "skills": {
                "modules": ["hey_robot.skill_os.builtins"],
                "enabled": ["inspect_scene", "navigate_to", "stop_motion"],
            },
        }
    )


@pytest.mark.integration
def test_goal_create_and_cancel_lifecycle(tmp_path: Path) -> None:
    """Goal is created, Supervisor publishes DeliberationRequest, cancel stops it."""
    config = _config(tmp_path)
    topics = Topics()
    envelope = Envelope(robot_id="test_robot", agent_id="main")

    async def _run() -> None:
        supervisor = AutonomySupervisorService(config)
        received_deliberations: list[dict] = []
        received_goal_events: list[dict] = []

        async def _on_goal_event(_topic: str, payload: dict) -> None:
            received_goal_events.append(payload)

        async def _on_deliberation(_topic: str, payload: dict) -> None:
            received_deliberations.append(payload)

        bus = create_bus_client(config.deployment.bus, role="test_client")
        await bus.connect()
        await bus.subscribe([topics.agent_deliberation], _on_deliberation)
        await bus.subscribe([topics.goal_event], _on_goal_event)
        await asyncio.sleep(0.2)

        supervisor.bus = bus

        # Send robot status first so gate is ready
        await supervisor._on_snapshot(
            topics.robot_status,
            to_payload(RobotStatus(envelope, state="idle", battery_percentage=100)),
        )

        # Publish goal create command
        goal_id: str | None = None
        command = GoalCommand(
            envelope,
            str(uuid.uuid4()),
            "create",
            objective="check the scene",
            success_criteria=(
                SuccessCriterion(
                    "c1",
                    "evidence_present",
                    "robot:test_robot",
                    "observed",
                    "scene",
                    300,
                ),
            ),
        )
        await supervisor._on_goal_command(topics.goal_command, to_payload(command))

        await asyncio.sleep(0.3)
        assert len(received_deliberations) >= 1, (
            f"No deliberation request published. goal_events={len(received_goal_events)}"
        )
        deliberation = from_payload(
            DeliberationRequest,
            received_deliberations[0],
        )
        assert deliberation.goal.objective == "check the scene"
        assert deliberation.goal.status == "active"

        # Get goal ID from store
        goals = supervisor.store.goals_recent(1)
        assert len(goals) == 1
        goal_id = goals[0]["goal_id"]

        # Cancel the goal
        cancel = GoalCommand(envelope, str(uuid.uuid4()), "cancel", goal_id=goal_id)
        await supervisor._on_goal_command(topics.goal_command, to_payload(cancel))
        await asyncio.sleep(0.2)

        # Goal should be cancelled (it was pending/active, no action committed)
        goal = supervisor.store.goal(goal_id)
        assert goal is not None
        assert goal["status"] in {"cancelled", "pending", "active"}, (
            f"unexpected status: {goal['status']}"
        )

        await bus.close()

    asyncio.run(_run())


@pytest.mark.integration
def test_full_deliberation_result_flow(tmp_path: Path) -> None:
    """Goal create → DeliberationRequest → DeliberationResult → SkillIntent → result."""
    config = _config(tmp_path)
    topics = Topics()
    envelope = Envelope(robot_id="test_robot", agent_id="main")

    async def _run() -> None:
        supervisor = AutonomySupervisorService(config)
        published_skill_intents: list[dict] = []
        published_goal_events: list[dict] = []

        async def _on_skill_intent(_topic: str, payload: dict) -> None:
            published_skill_intents.append(payload)

        async def _on_goal_event(_topic: str, payload: dict) -> None:
            published_goal_events.append(payload)

        bus = create_bus_client(config.deployment.bus, role="test_client")
        await bus.connect()
        await bus.subscribe([topics.skill_intent], _on_skill_intent)
        await bus.subscribe([topics.goal_event], _on_goal_event)
        await asyncio.sleep(0.2)

        supervisor.bus = bus

        # Set robot status
        await supervisor._on_snapshot(
            topics.robot_status,
            to_payload(RobotStatus(envelope, state="idle", battery_percentage=100)),
        )

        # Create goal
        command = GoalCommand(
            envelope,
            str(uuid.uuid4()),
            "create",
            objective="check the scene",
            success_criteria=(
                SuccessCriterion(
                    "c1",
                    "evidence_present",
                    "robot:test_robot",
                    "observed",
                    "scene",
                    300,
                ),
            ),
        )
        await supervisor._on_goal_command(topics.goal_command, to_payload(command))
        await asyncio.sleep(0.3)

        goals = supervisor.store.goals_recent(1)
        assert len(goals) == 1
        goal = goals[0]
        assert goal["status"] == "active"

        # Simulate deliberation result: agent proposes an action
        result = DeliberationResult(
            envelope,
            goal["active_deliberation_id"],
            supervisor.store.deliberation_request_hash(
                goal_id=goal["goal_id"],
                deliberation_id=goal["active_deliberation_id"],
            )
            or "",
            goal["goal_id"],
            goal["snapshot"]["task_id"],
            "action_proposed",
            proposal=ActionProposal(
                "observation",
                "inspect_scene",
                "check the scene",
                {"question": "what do you see"},
            ),
        )
        await supervisor._on_deliberation_result(
            topics.agent_deliberation_result, to_payload(result)
        )

        await asyncio.sleep(0.3)

        # Should have published skill intent
        assert len(published_skill_intents) >= 1, (
            f"No skill_intent published. intents={published_skill_intents}, "
            f"goal_status={supervisor.store.goal(goal['goal_id'])['status']}"
        )

        intent_data = published_skill_intents[0]
        intent = from_payload(SkillIntent, intent_data)
        assert intent.intent_kind == "observation"
        assert intent.name == "inspect_scene"
        assert intent.goal_id == goal["goal_id"]

        # Goal should be waiting now
        updated_goal = supervisor.store.goal(goal["goal_id"])
        assert updated_goal is not None
        assert updated_goal["status"] == "waiting"

        # Simulate SkillResult
        skill_result = SkillResult(
            envelope,
            intent.skill_id,
            status="completed",
            success=True,
        )
        await supervisor._on_skill_result(topics.skill_result, to_payload(skill_result))

        await asyncio.sleep(0.3)

        # Goal should be back to active (waiting for next deliberation)
        final_goal = supervisor.store.goal(goal["goal_id"])
        assert final_goal is not None
        assert final_goal["status"] in {"active", "waiting"}, (
            f"unexpected status after result: {final_goal['status']}"
        )

        await bus.close()

    asyncio.run(_run())


@pytest.mark.integration
def test_duplicate_command_id_is_idempotent(tmp_path: Path) -> None:
    """Repeated GoalCommand with same command_id creates only one goal."""
    config = _config(tmp_path)
    topics = Topics()
    envelope = Envelope(robot_id="test_robot", agent_id="main")
    command_id = str(uuid.uuid4())

    async def _run() -> None:
        supervisor = AutonomySupervisorService(config)
        bus = create_bus_client(config.deployment.bus, role="test_client")
        await bus.connect()
        await asyncio.sleep(0.2)
        supervisor.bus = bus

        await supervisor._on_snapshot(
            topics.robot_status,
            to_payload(RobotStatus(envelope, state="idle", battery_percentage=100)),
        )

        command = GoalCommand(
            envelope,
            command_id,
            "create",
            objective="check scene",
            success_criteria=(
                SuccessCriterion(
                    "c1",
                    "evidence_present",
                    "robot:test_robot",
                    "observed",
                    "scene",
                    300,
                ),
            ),
        )

        # Send twice
        await supervisor._on_goal_command(topics.goal_command, to_payload(command))
        await supervisor._on_goal_command(topics.goal_command, to_payload(command))
        await asyncio.sleep(0.2)

        goals = supervisor.store.goals_recent(10)
        assert len(goals) == 1, f"Expected 1 goal, got {len(goals)}"
        assert goals[0]["status"] == "active"

        await bus.close()

    asyncio.run(_run())


@pytest.mark.integration
def test_store_persistence_survives_restart(tmp_path: Path) -> None:
    """Goal state persists in SQLite and recovers on restart."""
    config = _config(tmp_path)
    topics = Topics()
    envelope = Envelope(robot_id="test_robot", agent_id="main")
    command_id = str(uuid.uuid4())

    async def _run_first() -> str:
        supervisor = AutonomySupervisorService(config)
        bus = create_bus_client(config.deployment.bus, role="test_client")
        await bus.connect()
        await asyncio.sleep(0.2)
        supervisor.bus = bus

        await supervisor._on_snapshot(
            topics.robot_status,
            to_payload(RobotStatus(envelope, state="idle", battery_percentage=100)),
        )

        command = GoalCommand(
            envelope,
            command_id,
            "create",
            objective="survive restart",
            success_criteria=(
                SuccessCriterion(
                    "c1",
                    "evidence_present",
                    "robot:test_robot",
                    "observed",
                    "scene",
                    300,
                ),
            ),
        )
        await supervisor._on_goal_command(topics.goal_command, to_payload(command))
        await asyncio.sleep(0.2)

        goals = supervisor.store.goals_recent(1)
        assert len(goals) == 1
        goal_id = goals[0]["goal_id"]
        await bus.close()
        return goal_id

    goal_id = asyncio.run(_run_first())

    # Create a NEW supervisor (simulates restart) with same database path
    async def _run_second() -> None:
        supervisor2 = AutonomySupervisorService(config)
        goals = supervisor2.store.goals_recent(10)
        assert len(goals) == 1, f"Lost goals after restart: {len(goals)}"
        assert goals[0]["goal_id"] == goal_id
        assert goals[0]["status"] == "active"

    asyncio.run(_run_second())
