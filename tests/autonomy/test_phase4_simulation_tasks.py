"""Phase 4 simulation tasks — deterministic, no real LLM or MuJoCo required."""

from __future__ import annotations

import asyncio
from pathlib import Path

from hey_robot.protocol import EvidenceFact, GoalCommand, RobotStatus, SuccessCriterion
from hey_robot.protocol.messages import to_payload

from .harness import (
    ScriptedTurn,
    build_harness,
    goal,
    goal_status,
    last_intent,
    last_intent_of_name,
    run_deliberation_turn,
    send_skill_result,
    set_robot_ready,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


async def _create_goal(h, *, objective="check scene", criteria=()):
    import uuid as _uuid

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
    cmd = GoalCommand(
        h.envelope,
        str(_uuid.uuid4()),
        "create",
        objective=objective,
        success_criteria=criteria,
    )
    await h.supervisor._on_goal_command(h.topics.goal_command, to_payload(cmd))
    await asyncio.sleep(0.1)
    goals = h.supervisor.store.goals_recent(1)
    assert len(goals) == 1
    return goals[0]["goal_id"]


# ── Basic Tasks (5) ──────────────────────────────────────────────────────────


def test_observe_room_and_report(tmp_path: Path) -> None:
    h = build_harness(
        tmp_path,
        turns=[
            ScriptedTurn("request_observation", {"question": "what is in this room"})
        ],
    )

    async def _run() -> None:
        await set_robot_ready(h)
        gid = await _create_goal(h, objective="observe room")
        assert await run_deliberation_turn(h)
        intent = last_intent(h)
        assert intent is not None
        assert intent["name"] == "inspect_scene"
        assert goal_status(h) == "waiting"
        await send_skill_result(
            h,
            skill_id=intent["skill_id"],
            success=True,
            evidence=(
                EvidenceFact(
                    "e1",
                    gid,
                    "skill_result",
                    intent["skill_id"],
                    9e18,
                    1,
                    f"robot:{h.robot_id}",
                    "observed",
                    "scene",
                ),
            ),
        )
        await asyncio.sleep(0.1)
        await run_deliberation_turn(h)
        assert goal_status(h) == "completed"

    asyncio.run(_run())


def test_navigate_to_living_room(tmp_path: Path) -> None:
    h = build_harness(
        tmp_path,
        turns=[
            ScriptedTurn("request_observation", {"question": "where am I"}),
            ScriptedTurn(
                "request_skill",
                {
                    "skill": "navigate_to",
                    "objective": "go",
                    "slots": {"target": "room:living_room"},
                },
            ),
        ],
        entity_catalog=("robot:test_robot", "room:living_room", "scene"),
    )

    async def _run() -> None:
        await set_robot_ready(h)
        gid = await _create_goal(
            h,
            objective="go to living room",
            criteria=(
                SuccessCriterion(
                    "c1",
                    "robot_state",
                    "robot:test_robot",
                    "equals",
                    "room:living_room",
                    300,
                ),
            ),
        )
        # Turn 1: observe
        assert await run_deliberation_turn(h)
        i1 = last_intent(h)
        assert i1 is not None
        assert i1["name"] == "inspect_scene"
        await send_skill_result(h, skill_id=i1["skill_id"], success=True)
        # Turn 2: navigate with location evidence
        assert await run_deliberation_turn(h)
        i2 = last_intent_of_name(h, "navigate_to")
        assert i2
        await send_skill_result(
            h,
            skill_id=i2["skill_id"],
            success=True,
            evidence=(
                EvidenceFact(
                    "e_loc",
                    gid,
                    "skill_result",
                    i2["skill_id"],
                    9e18,
                    1,
                    "robot:test_robot",
                    "equals",
                    "room:living_room",
                ),
            ),
        )
        # Turn 3: SATISFIED (zero model calls)
        await run_deliberation_turn(h)
        assert goal_status(h) == "completed"

    asyncio.run(_run())


def test_approach_table_and_pick_cup(tmp_path: Path) -> None:
    h = build_harness(
        tmp_path,
        turns=[
            ScriptedTurn("request_observation", {"question": "find cup"}),
            ScriptedTurn(
                "request_skill",
                {
                    "skill": "navigate_to",
                    "objective": "approach table",
                    "slots": {"target": "fixture:table"},
                },
            ),
            ScriptedTurn("request_observation", {"question": "is cup reachable"}),
            ScriptedTurn(
                "request_skill",
                {
                    "skill": "navigate_to",
                    "objective": "pick cup",
                    "slots": {"target": "object:cup"},
                },
            ),
        ],
        entity_catalog=("robot:test_robot", "object:cup", "fixture:table", "scene"),
    )

    async def _run() -> None:
        await set_robot_ready(h)
        gid = await _create_goal(
            h,
            objective="pick up the cup",
            criteria=(
                SuccessCriterion(
                    "c1",
                    "object_relation",
                    "object:cup",
                    "held_by",
                    "robot:test_robot",
                    300,
                ),
            ),
        )
        for _ in range(4):
            assert await run_deliberation_turn(h)
            intent = last_intent(h)
            if intent and intent["name"] == "inspect_scene":
                await send_skill_result(h, skill_id=intent["skill_id"], success=True)
            else:
                nav = last_intent_of_name(h, "navigate_to")
                if nav:
                    await send_skill_result(
                        h,
                        skill_id=nav["skill_id"],
                        success=True,
                        evidence=(
                            EvidenceFact(
                                "e_pick",
                                gid,
                                "skill_result",
                                nav["skill_id"],
                                9e18,
                                2,
                                "object:cup",
                                "held_by",
                                "robot:test_robot",
                            ),
                        )
                        if nav.get("objective") == "pick cup"
                        else (),
                    )
        await run_deliberation_turn(h)
        assert goal_status(h) == "completed"

    asyncio.run(_run())


def test_return_wand_to_dock(tmp_path: Path) -> None:
    h = build_harness(
        tmp_path,
        turns=[
            ScriptedTurn("request_observation", {"question": "where is wand"}),
            ScriptedTurn(
                "request_skill",
                {
                    "skill": "navigate_to",
                    "objective": "pick wand",
                    "slots": {"target": "object:wand"},
                },
            ),
            ScriptedTurn(
                "request_skill",
                {
                    "skill": "navigate_to",
                    "objective": "place wand",
                    "slots": {"target": "fixture:dock"},
                },
            ),
        ],
        entity_catalog=("robot:test_robot", "object:wand", "fixture:dock", "scene"),
    )

    async def _run() -> None:
        await set_robot_ready(h)
        gid = await _create_goal(
            h,
            objective="put wand back on dock",
            criteria=(
                SuccessCriterion(
                    "c1", "object_relation", "object:wand", "at", "fixture:dock", 300
                ),
            ),
        )
        for _turn_idx in range(3):
            assert await run_deliberation_turn(h)
            intent = last_intent(h)
            if intent and intent["name"] == "inspect_scene":
                await send_skill_result(h, skill_id=intent["skill_id"], success=True)
            else:
                nav = last_intent_of_name(h, "navigate_to")
                if nav and "place" in str(nav.get("objective", "")):
                    await send_skill_result(
                        h,
                        skill_id=nav["skill_id"],
                        success=True,
                        evidence=(
                            EvidenceFact(
                                "e_dock",
                                gid,
                                "skill_result",
                                nav["skill_id"],
                                9e18,
                                3,
                                "object:wand",
                                "at",
                                "fixture:dock",
                            ),
                        ),
                    )
                elif nav:
                    await send_skill_result(h, skill_id=nav["skill_id"], success=True)
        await run_deliberation_turn(h)
        assert goal_status(h) == "completed"

    asyncio.run(_run())


def test_patrol_multiple_rooms(tmp_path: Path) -> None:
    h = build_harness(
        tmp_path,
        turns=[
            ScriptedTurn("request_observation", {"question": "living room status"}),
            ScriptedTurn(
                "request_skill",
                {
                    "skill": "navigate_to",
                    "objective": "go to dining room",
                    "slots": {"target": "room:dining_room"},
                },
            ),
            ScriptedTurn("request_observation", {"question": "dining room status"}),
            ScriptedTurn(
                "request_skill",
                {
                    "skill": "navigate_to",
                    "objective": "go to kitchen",
                    "slots": {"target": "room:kitchen"},
                },
            ),
            ScriptedTurn("request_observation", {"question": "kitchen status"}),
        ],
        entity_catalog=(
            "robot:test_robot",
            "room:living_room",
            "room:dining_room",
            "room:kitchen",
            "scene",
        ),
    )

    async def _run() -> None:
        await set_robot_ready(h)
        gid = await _create_goal(
            h,
            objective="patrol all rooms",
            criteria=(
                SuccessCriterion(
                    "c1",
                    "evidence_present",
                    "room:living_room",
                    "observed",
                    "scene",
                    300,
                ),
                SuccessCriterion(
                    "c2",
                    "evidence_present",
                    "room:dining_room",
                    "observed",
                    "scene",
                    300,
                ),
                SuccessCriterion(
                    "c3", "evidence_present", "room:kitchen", "observed", "scene", 300
                ),
            ),
        )
        room_evidence = {
            0: "room:living_room",
            2: "room:dining_room",
            4: "room:kitchen",
        }
        for turn_idx in range(5):
            assert await run_deliberation_turn(h)
            intent = last_intent(h)
            if intent and intent["name"] == "inspect_scene":
                room = room_evidence.get(turn_idx, "scene")
                await send_skill_result(
                    h,
                    skill_id=intent["skill_id"],
                    success=True,
                    evidence=(
                        EvidenceFact(
                            f"e_{turn_idx}",
                            gid,
                            "skill_result",
                            intent["skill_id"],
                            9e18,
                            turn_idx,
                            room,
                            "observed",
                            "scene",
                        ),
                    ),
                )
            else:
                nav = last_intent_of_name(h, "navigate_to")
                if nav:
                    await send_skill_result(h, skill_id=nav["skill_id"], success=True)
        await run_deliberation_turn(h)
        assert goal_status(h) == "completed"

    asyncio.run(_run())


# ── Failure Tasks (6) ────────────────────────────────────────────────────────


def test_search_for_nonexistent_object_fails_by_budget(tmp_path: Path) -> None:
    h = build_harness(
        tmp_path,
        turns=[ScriptedTurn("request_observation", {"question": "find unicorn"})] * 10,
        hard_max_deliberations=3,
        entity_catalog=("robot:test_robot", "object:unicorn", "room:lab", "scene"),
    )

    async def _run() -> None:
        await set_robot_ready(h)
        await _create_goal(h, objective="find the unicorn")
        for _ in range(4):
            if not await run_deliberation_turn(h):
                break
            intent = last_intent(h)
            if intent:
                await send_skill_result(h, skill_id=intent["skill_id"], success=True)
        assert goal_status(h) == "failed"

    asyncio.run(_run())


def test_unreachable_location_fails(tmp_path: Path) -> None:
    h = build_harness(
        tmp_path,
        turns=[
            ScriptedTurn("request_observation", {"question": "where to go"}),
            ScriptedTurn(
                "request_skill",
                {
                    "skill": "navigate_to",
                    "objective": "go to mars",
                    "slots": {"target": "room:mars"},
                },
            ),
        ],
        entity_catalog=("robot:test_robot", "room:mars", "scene"),
    )

    async def _run() -> None:
        await set_robot_ready(h)
        await _create_goal(
            h,
            objective="go to mars",
            criteria=(
                SuccessCriterion(
                    "c1", "robot_state", "robot:test_robot", "equals", "room:mars", 300
                ),
            ),
        )
        assert await run_deliberation_turn(h)
        i1 = last_intent(h)
        await send_skill_result(h, skill_id=i1["skill_id"], success=True)

        assert await run_deliberation_turn(h)
        i2 = last_intent_of_name(h, "navigate_to")
        assert i2 is not None
        await send_skill_result(
            h, skill_id=i2["skill_id"], success=False, status="failed"
        )
        await asyncio.sleep(0.3)

        # Skill failure → goal should be failed
        g = goal(h)
        s = goal_status(h)
        assert s in {"failed", "blocked"}, f"expected failed/blocked, got {s}. goal={g}"

    asyncio.run(_run())


def test_observation_skill_fails_goal_fails(tmp_path: Path) -> None:
    h = build_harness(
        tmp_path, turns=[ScriptedTurn("request_observation", {"question": "check"})]
    )

    async def _run() -> None:
        await set_robot_ready(h)
        await _create_goal(h)
        await run_deliberation_turn(h)
        i1 = last_intent(h)
        assert i1 is not None
        assert i1["name"] == "inspect_scene"
        await send_skill_result(
            h, skill_id=i1["skill_id"], success=False, status="failed"
        )
        assert goal_status(h) == "failed"

    asyncio.run(_run())


def test_low_battery_blocks_physical_skill(tmp_path: Path) -> None:
    h = build_harness(
        tmp_path, turns=[ScriptedTurn("request_observation", {"question": "check"})]
    )

    async def _run() -> None:
        await set_robot_ready(h)
        await _create_goal(h)
        await run_deliberation_turn(h)
        i1 = last_intent(h)
        # Drain battery BEFORE skill result triggers reschedule
        await h.supervisor._on_snapshot(
            h.topics.robot_status,
            to_payload(
                RobotStatus(
                    h.envelope, state="idle", battery_percentage=5.0, frame_id=100
                )
            ),
        )
        await send_skill_result(h, skill_id=i1["skill_id"], success=True)
        await asyncio.sleep(0.2)
        assert goal_status(h) in {"failed", "blocked"}

    asyncio.run(_run())


def test_vln_repeating_same_turn_fails_by_budget(tmp_path: Path) -> None:
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
        ]
        * 4,
        hard_max_skills=2,
        entity_catalog=("robot:test_robot", "room:lab", "scene"),
    )

    async def _run() -> None:
        await set_robot_ready(h)
        await _create_goal(
            h,
            objective="go to lab",
            criteria=(
                SuccessCriterion(
                    "c1", "robot_state", "robot:test_robot", "equals", "room:lab", 300
                ),
            ),
        )
        for _ in range(4):
            if not await run_deliberation_turn(h):
                break
            nav = last_intent_of_name(h, "navigate_to")
            if nav:
                await send_skill_result(h, skill_id=nav["skill_id"], success=True)
        await asyncio.sleep(0.5)
        assert goal_status(h) == "failed"

    asyncio.run(_run())


def test_skill_success_but_contract_not_satisfied_continues(tmp_path: Path) -> None:
    h = build_harness(
        tmp_path,
        turns=[
            ScriptedTurn("request_observation", {"question": "check wand"}),
            ScriptedTurn(
                "request_skill",
                {
                    "skill": "navigate_to",
                    "objective": "put wand to dock",
                    "slots": {"target": "fixture:dock"},
                },
            ),
            ScriptedTurn("request_observation", {"question": "is wand at dock now"}),
        ],
        entity_catalog=("robot:test_robot", "object:wand", "fixture:dock", "scene"),
    )

    async def _run() -> None:
        await set_robot_ready(h)
        await _create_goal(
            h,
            objective="put wand to dock",
            criteria=(
                SuccessCriterion(
                    "c1", "object_relation", "object:wand", "at", "fixture:dock", 300
                ),
            ),
        )
        await run_deliberation_turn(h)
        await send_skill_result(h, skill_id=last_intent(h)["skill_id"], success=True)
        assert await run_deliberation_turn(h)  # navigate — no evidence
        nav = last_intent_of_name(h, "navigate_to")
        await send_skill_result(
            h, skill_id=nav["skill_id"], success=True
        )  # no evidence!
        assert await run_deliberation_turn(h)  # still INCONCLUSIVE, re-observes
        assert last_intent(h)["name"] == "inspect_scene"

    asyncio.run(_run())
