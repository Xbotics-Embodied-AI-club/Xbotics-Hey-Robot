from __future__ import annotations

import asyncio
from pathlib import Path

from hey_robot.cognition.autonomous.supervisor import AutonomySupervisorService
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import (
    ActionProposal,
    DeliberationResult,
    Envelope,
    GoalCommand,
    RobotStatus,
    SkillControlResult,
    SkillResult,
    SuccessCriterion,
)
from hey_robot.protocol.messages import to_payload


class FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))


def _service(tmp_path: Path) -> AutonomySupervisorService:
    config = DeploymentConfig.from_dict(
        {
            "deployment": {"id": "test"},
            "resources": {"runtime_dir": str(tmp_path / "runtime")},
            "autonomy": {
                "enabled": True,
                "entity_catalog": ["robot:main", "room:lab", "scene"],
            },
        }
    )
    service = AutonomySupervisorService(config)
    service.bus = FakeBus()  # type: ignore[assignment]
    return service


def test_supervisor_dispatches_one_intent_for_one_deliberation(tmp_path: Path) -> None:
    service = _service(tmp_path)
    envelope = Envelope(robot_id="main", agent_id="main")
    asyncio.run(
        service._on_snapshot(
            service.topics.robot_status,
            to_payload(RobotStatus(envelope, state="idle", battery_percentage=100)),
        )
    )
    command = GoalCommand(
        envelope,
        "create",
        "create",
        objective="move",
        success_criteria=(
            SuccessCriterion(
                "at", "robot_state", "robot:main", "equals", "room:lab", 60
            ),
        ),
    )
    asyncio.run(
        service._on_goal_command(service.topics.goal_command, to_payload(command))
    )
    goal = service.store.goals_recent(1)[0]
    result = DeliberationResult(
        envelope,
        goal["active_deliberation_id"],
        "request",
        goal["goal_id"],
        goal["snapshot"]["task_id"],
        "action_proposed",
        proposal=ActionProposal("skill", "move_base", "move", {"direction": "forward"}),
    )
    asyncio.run(
        service._on_deliberation_result(
            service.topics.agent_deliberation_result, to_payload(result)
        )
    )
    asyncio.run(
        service._on_deliberation_result(
            service.topics.agent_deliberation_result, to_payload(result)
        )
    )
    intents = [
        entry
        for entry in service.bus.published
        if entry[0] == service.topics.skill_intent
    ]  # type: ignore[attr-defined]
    assert len(intents) == 1
    assert service.store.action(intents[0][1]["skill_id"])["status"] == "published"


def test_conflicting_skill_result_locks_robot_execution(tmp_path: Path) -> None:
    service = _service(tmp_path)
    envelope = Envelope(robot_id="main", agent_id="main")
    asyncio.run(
        service._on_snapshot(
            service.topics.robot_status,
            to_payload(RobotStatus(envelope, state="idle", battery_percentage=100)),
        )
    )
    command = GoalCommand(
        envelope,
        "create",
        "create",
        objective="move",
        success_criteria=(
            SuccessCriterion(
                "at", "robot_state", "robot:main", "equals", "room:lab", 60
            ),
        ),
    )
    asyncio.run(
        service._on_goal_command(service.topics.goal_command, to_payload(command))
    )
    goal = service.store.goals_recent(1)[0]
    proposal = DeliberationResult(
        envelope,
        goal["active_deliberation_id"],
        "request",
        goal["goal_id"],
        goal["snapshot"]["task_id"],
        "action_proposed",
        proposal=ActionProposal("skill", "move_base", "move", {"direction": "forward"}),
    )
    asyncio.run(
        service._on_deliberation_result(
            service.topics.agent_deliberation_result, to_payload(proposal)
        )
    )
    skill_id = next(
        payload["skill_id"]
        for topic, payload in service.bus.published
        if topic == service.topics.skill_intent
    )  # type: ignore[attr-defined]
    asyncio.run(
        service._on_skill_result(
            service.topics.skill_result,
            to_payload(
                SkillResult(envelope, skill_id, status="completed", success=True)
            ),
        )
    )
    asyncio.run(
        service._on_skill_result(
            service.topics.skill_result,
            to_payload(SkillResult(envelope, skill_id, status="failed", success=False)),
        )
    )
    assert service.store.action(skill_id)["status"] == "unknown"
    assert service.store.gate("main")["state"] == "uncertain"
    assert service.store.goal(goal["goal_id"])["status"] == "blocked"


def test_cancel_waiting_goal_persists_interrupt_and_blocks_late_dispatch(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    envelope = Envelope(robot_id="main", agent_id="main")
    asyncio.run(
        service._on_snapshot(
            service.topics.robot_status,
            to_payload(RobotStatus(envelope, state="idle", battery_percentage=100)),
        )
    )
    command = GoalCommand(
        envelope,
        "create",
        "create",
        objective="move",
        success_criteria=(
            SuccessCriterion(
                "at", "robot_state", "robot:main", "equals", "room:lab", 60
            ),
        ),
    )
    asyncio.run(
        service._on_goal_command(service.topics.goal_command, to_payload(command))
    )
    goal = service.store.goals_recent(1)[0]
    proposal = DeliberationResult(
        envelope,
        goal["active_deliberation_id"],
        "request",
        goal["goal_id"],
        goal["snapshot"]["task_id"],
        "action_proposed",
        proposal=ActionProposal("skill", "move_base", "move", {"direction": "forward"}),
    )
    asyncio.run(
        service._on_deliberation_result(
            service.topics.agent_deliberation_result, to_payload(proposal)
        )
    )
    assert service.store.goal(goal["goal_id"])["status"] == "waiting"
    cancel = GoalCommand(envelope, "cancel", "cancel", goal_id=goal["goal_id"])
    asyncio.run(
        service._on_goal_command(service.topics.goal_command, to_payload(cancel))
    )
    controls = [
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_control
    ]  # type: ignore[attr-defined]
    assert len(controls) == 1
    assert service.store.gate("main")["state"] == "stop_pending"
    asyncio.run(
        service._on_deliberation_result(
            service.topics.agent_deliberation_result, to_payload(proposal)
        )
    )
    assert (
        len(
            [
                payload
                for topic, payload in service.bus.published
                if topic == service.topics.skill_intent
            ]
        )
        == 1
    )  # type: ignore[attr-defined]


def test_unknown_control_result_keeps_execution_lock(tmp_path: Path) -> None:
    service = _service(tmp_path)
    envelope = Envelope(robot_id="main", agent_id="main")
    asyncio.run(
        service._on_snapshot(
            service.topics.robot_status,
            to_payload(RobotStatus(envelope, state="idle", battery_percentage=100)),
        )
    )
    command = GoalCommand(
        envelope,
        "create",
        "create",
        objective="move",
        success_criteria=(
            SuccessCriterion(
                "at", "robot_state", "robot:main", "equals", "room:lab", 60
            ),
        ),
    )
    asyncio.run(
        service._on_goal_command(service.topics.goal_command, to_payload(command))
    )
    goal = service.store.goals_recent(1)[0]
    proposal = DeliberationResult(
        envelope,
        goal["active_deliberation_id"],
        "request",
        goal["goal_id"],
        goal["snapshot"]["task_id"],
        "action_proposed",
        proposal=ActionProposal("skill", "move_base", "move", {"direction": "forward"}),
    )
    asyncio.run(
        service._on_deliberation_result(
            service.topics.agent_deliberation_result, to_payload(proposal)
        )
    )
    asyncio.run(
        service._on_goal_command(
            service.topics.goal_command,
            to_payload(
                GoalCommand(envelope, "cancel", "cancel", goal_id=goal["goal_id"])
            ),
        )
    )
    control = next(
        payload
        for topic, payload in service.bus.published
        if topic == service.topics.skill_control
    )  # type: ignore[attr-defined]
    result = SkillControlResult(
        envelope,
        control["control_id"],
        "interrupt",
        control["target_skill_id"],
        "unknown",
        False,
        "stop state unknown",
    )
    asyncio.run(
        service._on_control_result(
            service.topics.skill_control_result, to_payload(result)
        )
    )
    assert service.store.gate("main")["state"] == "uncertain"
    assert service.store.goal(goal["goal_id"])["status"] == "blocked"
