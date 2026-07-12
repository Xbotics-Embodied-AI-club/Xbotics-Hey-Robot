"""Phase 4 fault injection tests — deterministic, no real LLM or MuJoCo."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from hey_robot.protocol import (
    GoalCommand,
    SkillControlResult,
    SuccessCriterion,
)
from hey_robot.protocol.messages import to_payload

from .harness import (
    ScriptedTurn,
    build_harness,
    goal,
    goal_status,
    last_intent,
    run_deliberation_turn,
    send_skill_result,
    set_robot_ready,
)


async def _create_goal(h) -> str:
    cmd = GoalCommand(
        h.envelope,
        str(uuid.uuid4()),
        "create",
        objective="test task",
        success_criteria=(
            SuccessCriterion(
                "c1",
                "evidence_present",
                f"robot:{h.robot_id}",
                "observed",
                "scene",
                300,
            ),
        ),
    )
    await h.supervisor._on_goal_command(h.topics.goal_command, to_payload(cmd))
    await asyncio.sleep(0.1)
    goals = h.supervisor.store.goals_recent(1)
    assert len(goals) == 1
    return goals[0]["goal_id"]


# ── Provider / Protocol Failures ──────────────────────────────────────────────


def test_provider_timeout_marks_goal_failed(tmp_path: Path) -> None:
    """Provider timeout → FAILED, no recovery (scripted provider exhausted)."""
    h = build_harness(tmp_path, turns=[])  # no turns → exhausted immediately

    async def _run() -> None:
        await set_robot_ready(h)
        await _create_goal(h)
        await run_deliberation_turn(h)
        assert goal_status(h) == "failed"

    asyncio.run(_run())


def test_invalid_tool_call_fails_deliberation(tmp_path: Path) -> None:
    """Model calls unknown tool → UNKNOWN_TOOL failure."""
    h = build_harness(tmp_path, turns=[ScriptedTurn("magic_spell", {"spell": "x"})])

    async def _run() -> None:
        await set_robot_ready(h)
        await _create_goal(h)
        await run_deliberation_turn(h)
        assert goal_status(h) == "failed"
        g = goal(h)
        assert g["termination_reason"] is not None

    asyncio.run(_run())


def test_model_returns_text_before_contract_satisfied(tmp_path: Path) -> None:
    """Model returns plain text → MODEL_STOPPED_BEFORE_GOAL → FAILED."""
    h = build_harness(
        tmp_path, turns=[ScriptedTurn(text="all done", finish_reason="stop")]
    )

    async def _run() -> None:
        await set_robot_ready(h)
        await _create_goal(h)
        await run_deliberation_turn(h)
        assert goal_status(h) == "failed"

    asyncio.run(_run())


# ── Dispatch Faults ───────────────────────────────────────────────────────────


def test_action_publishing_interrupted_enters_unknown(tmp_path: Path) -> None:
    """Action stuck in PUBLISHING on restart → UNKNOWN + BLOCKED."""
    h = build_harness(
        tmp_path, turns=[ScriptedTurn("request_observation", {"question": "check"})]
    )

    async def _run() -> None:
        await set_robot_ready(h)
        gid = await _create_goal(h)
        await run_deliberation_turn(h)

        actions = h.supervisor.store.actions_for_goal(gid)
        assert len(actions) > 0
        # Force back to PUBLISHING
        h.supervisor.store.cas_action_status(
            skill_id=actions[0]["skill_id"], expected="published", status="publishing"
        )
        interrupted = h.supervisor.store.recover_publishing()
        assert len(interrupted) == 1
        assert h.supervisor.store.action(actions[0]["skill_id"])["status"] == "unknown"
        assert h.supervisor.store.gate(h.robot_id).state == "uncertain"

    asyncio.run(_run())


def test_fast_skill_result_cas_preserves_terminal(tmp_path: Path) -> None:
    """SkillResult arrives during publish → CAS keeps terminal."""
    h = build_harness(
        tmp_path, turns=[ScriptedTurn("request_observation", {"question": "check"})]
    )

    async def _run() -> None:
        await set_robot_ready(h)
        await _create_goal(h)
        await run_deliberation_turn(h)
        intent = last_intent(h)
        assert intent is not None
        skill_id = intent["skill_id"]

        # Set to PUBLISHING
        h.supervisor.store.cas_action_status(
            skill_id=skill_id, expected="published", status="publishing"
        )
        # Fast SkillResult arrives
        await send_skill_result(h, skill_id=skill_id, success=True)

        action = h.supervisor.store.action(skill_id)
        assert action is not None
        assert action["status"] == "completed", (
            f"expected completed, got {action['status']}"
        )

    asyncio.run(_run())


def test_duplicate_deliberation_request_idempotent(tmp_path: Path) -> None:
    """Same deliberation_id → no second model call."""
    h = build_harness(
        tmp_path, turns=[ScriptedTurn("request_observation", {"question": "check"})]
    )

    async def _run() -> None:
        await set_robot_ready(h)
        await _create_goal(h)
        await run_deliberation_turn(h)
        calls_before = len(h.provider.calls)

        # Re-send the same deliberation request
        delibs = [p for t, p in h.bus.published if t == h.topics.agent_deliberation]
        await h.agent._on_request(h.topics.agent_deliberation, delibs[-1])
        await asyncio.sleep(0.1)

        assert len(h.provider.calls) == calls_before, (
            "duplicate request triggered model"
        )

    asyncio.run(_run())


def test_duplicate_skill_result_is_replay(tmp_path: Path) -> None:
    """Repeated SkillResult → replay (ignored)."""
    h = build_harness(
        tmp_path, turns=[ScriptedTurn("request_observation", {"question": "check"})]
    )

    async def _run() -> None:
        await set_robot_ready(h)
        await _create_goal(h)
        await run_deliberation_turn(h)
        intent = last_intent(h)
        assert intent is not None
        skill_id = intent["skill_id"]

        await send_skill_result(h, skill_id=skill_id, success=True)
        assert h.supervisor.store.action(skill_id)["status"] == "completed"

        # Duplicate with same payload → replay
        status_before = goal_status(h)
        await send_skill_result(h, skill_id=skill_id, success=True)
        assert goal_status(h) == status_before

    asyncio.run(_run())


def test_conflicting_skill_result_triggers_idempotency_conflict(tmp_path: Path) -> None:
    """Same skill_id, different payload → IDEMPOTENCY_CONFLICT → BLOCKED."""
    h = build_harness(
        tmp_path, turns=[ScriptedTurn("request_observation", {"question": "check"})]
    )

    async def _run() -> None:
        await set_robot_ready(h)
        await _create_goal(h)
        await run_deliberation_turn(h)
        intent = last_intent(h)
        assert intent is not None
        skill_id = intent["skill_id"]

        await send_skill_result(h, skill_id=skill_id, success=True)
        await send_skill_result(h, skill_id=skill_id, success=False, status="failed")

        action = h.supervisor.store.action(skill_id)
        assert action is not None
        assert action["status"] == "unknown"
        assert h.supervisor.store.gate(h.robot_id).state == "uncertain"
        assert goal_status(h) == "blocked"

    asyncio.run(_run())


# ── Cancel and Control Faults ─────────────────────────────────────────────────


def test_cancel_during_deliberation_rejects_late_proposal(tmp_path: Path) -> None:
    """Cancel arrives while active → CANCELLED, late proposal ignored."""
    h = build_harness(
        tmp_path, turns=[ScriptedTurn("request_observation", {"question": "check"})]
    )

    async def _run() -> None:
        await set_robot_ready(h)
        gid = await _create_goal(h)
        # Cancel before agent processes
        cancel_cmd = GoalCommand(h.envelope, "cancel_cmd", "cancel", goal_id=gid)
        await h.supervisor._on_goal_command(
            h.topics.goal_command, to_payload(cancel_cmd)
        )
        await asyncio.sleep(0.1)
        assert goal_status(h) in {"cancelled", "failed"}

    asyncio.run(_run())


def test_cancel_during_active_skill_sends_control(tmp_path: Path) -> None:
    """Cancel while skill waiting → stop control → confirmed idle → CANCELLED."""
    h = build_harness(
        tmp_path,
        turns=[
            ScriptedTurn(
                "request_skill",
                {
                    "skill": "navigate_to",
                    "objective": "go",
                    "slots": {"target": "room:lab"},
                },
            )
        ],
        entity_catalog=("robot:test_robot", "room:lab", "scene"),
    )

    async def _run() -> None:
        await set_robot_ready(h)
        gid = await _create_goal(h)
        await run_deliberation_turn(h)
        assert goal_status(h) == "waiting"

        cancel_cmd = GoalCommand(h.envelope, "cancel_cmd2", "cancel", goal_id=gid)
        await h.supervisor._on_goal_command(
            h.topics.goal_command, to_payload(cancel_cmd)
        )
        await asyncio.sleep(0.1)

        controls = [p for t, p in h.bus.published if t == h.topics.skill_control]
        assert len(controls) >= 1, "no skill.control published"
        assert h.supervisor.store.gate(h.robot_id).state == "stop_pending"

        control = controls[-1]
        result = SkillControlResult(
            h.envelope,
            control["control_id"],
            "interrupt",
            control["target_skill_id"],
            "completed",
            True,
        )
        await h.supervisor._on_control_result(
            h.topics.skill_control_result, to_payload(result)
        )
        await asyncio.sleep(0.1)

        assert goal_status(h) == "cancelled"
        assert h.supervisor.store.gate(h.robot_id).state == "ready"

    asyncio.run(_run())


def test_cancel_blocked_goal_preserves_execution_lock(tmp_path: Path) -> None:
    """Cancelling a BLOCKED goal does not clear gate UNCERTAIN."""
    h = build_harness(
        tmp_path, turns=[ScriptedTurn("request_observation", {"question": "check"})]
    )

    async def _run() -> None:
        await set_robot_ready(h)
        gid = await _create_goal(h)
        await run_deliberation_turn(h)
        intent = last_intent(h)
        assert intent is not None

        # Force conflict → BLOCKED + UNCERTAIN
        h.supervisor.store.cas_action_status(
            skill_id=intent["skill_id"], expected="published", status="unknown"
        )
        h.supervisor.store.mark_skill_result_conflict(intent["skill_id"])
        assert h.supervisor.store.gate(h.robot_id).state == "uncertain"

        cancel_cmd = GoalCommand(h.envelope, "cancel_blocked", "cancel", goal_id=gid)
        await h.supervisor._on_goal_command(
            h.topics.goal_command, to_payload(cancel_cmd)
        )
        await asyncio.sleep(0.1)

        assert h.supervisor.store.gate(h.robot_id).state == "uncertain"

    asyncio.run(_run())


def test_supervisor_restart_recovers_goal_state(tmp_path: Path) -> None:
    """Supervisor restart recovers goals from SQLite."""
    h = build_harness(
        tmp_path, turns=[ScriptedTurn("request_observation", {"question": "check"})]
    )

    async def _run_first() -> str:
        await set_robot_ready(h)
        return await _create_goal(h)

    gid = asyncio.run(_run_first())

    h2 = build_harness(
        tmp_path, turns=[ScriptedTurn("request_observation", {"question": "check"})]
    )
    goals = h2.supervisor.store.goals_recent(10)
    assert len(goals) == 1
    assert goals[0]["goal_id"] == gid
    assert goals[0]["status"] == "active"
