from __future__ import annotations

import time

import pytest

from hey_robot.protocol import (
    Envelope,
    RobotAction,
    RobotSkillAction,
    RobotStatus,
    SkillIntent,
)
from hey_robot.robot_runtime.control_plane import RobotControlPlane


def _action(**metadata) -> RobotAction:
    return RobotAction(
        envelope=Envelope(trace_id="tr1", robot_id="mock0"),
        skill_id="skill1",
        values=[],
        metadata=dict(metadata),
    )


def _intent() -> SkillIntent:
    return SkillIntent(
        envelope=Envelope(trace_id="tr1", robot_id="mock0"),
        skill_id="skill1",
        task_id="task1",
        intent_kind="skill",
        name="policy_action",
        arguments={},
        objective="test policy action",
    )


@pytest.mark.asyncio
async def test_control_plane_buffers_action_and_records_watchdog() -> None:
    plane = RobotControlPlane(max_buffer_size=4)

    async def apply(action: RobotAction) -> RobotStatus:
        return RobotStatus(
            envelope=action.envelope,
            skill_id=action.skill_id,
            success=True,
            metrics={"driver": "fake"},
        )

    status = await plane.apply_action(_action(action_type="skill"), apply_fn=apply)

    assert status.success is True
    assert status.metrics["control_plane"]["buffer_size"] == 1
    assert plane.last_watchdog is not None
    assert plane.last_watchdog["success"] is True


@pytest.mark.asyncio
async def test_control_plane_rejects_expired_deadline() -> None:
    plane = RobotControlPlane()

    async def apply(_action: RobotAction) -> RobotStatus:
        raise AssertionError("expired action should not execute")

    status = await plane.apply_action(
        _action(deadline_at=time.time() - 1.0),
        apply_fn=apply,
    )

    assert status.success is False
    assert status.metrics["last_skill_result"]["failure_mode"] == (
        "control_deadline_expired"
    )


@pytest.mark.asyncio
async def test_control_plane_preemption_runs_stop_motion_first() -> None:
    plane = RobotControlPlane()
    calls: list[str] = []

    async def apply(action: RobotAction) -> RobotStatus:
        try:
            skill = RobotSkillAction.from_robot_action(action)
            calls.append(skill.name)
        except ValueError:
            calls.append("raw")
        return RobotStatus(
            envelope=action.envelope,
            skill_id=action.skill_id,
            success=True,
        )

    status = await plane.apply_action(
        _action(preempt=True),
        apply_fn=apply,
        stop_fn=lambda action: plane.stop_motion(action, apply_fn=apply),
    )

    assert status.success is True
    assert calls == ["stop_motion", "raw"]
    assert plane.preemptions[-1]["success"] is True


def test_control_plane_maps_action_chunk_to_runtime_skill_actions() -> None:
    plane = RobotControlPlane()
    intent = _intent()

    actions = plane.map_policy_result(
        {
            "kind": "action_chunk",
            "action_space": "xlerobot_single_arm_joint",
            "embodiment": "xlerobot",
            "horizon": 2,
            "dt": 0.05,
            "policy_session_id": "skill1",
            "actions": [
                {"joints": {"shoulder_pan": 0.1}, "gripper": 1.0},
                {"joints": {"shoulder_pan": 0.2}, "gripper": 0.2},
            ],
        },
        intent=intent,
    )

    decoded = [RobotSkillAction.from_robot_action(action) for action in actions]
    assert [(item.name, item.arguments) for item in decoded] == [
        ("move_arm_joints", {"joints": {"shoulder_pan": 0.1}, "mode": "absolute"}),
        ("set_gripper", {"opening_pct": 100.0}),
        ("move_arm_joints", {"joints": {"shoulder_pan": 0.2}, "mode": "absolute"}),
        ("set_gripper", {"opening_pct": 20.0}),
    ]
    assert actions[2].metadata["action_index"] == 1
    assert actions[2].metadata["deadline_sec"] == 0.1


def test_control_plane_maps_local_goal_pixel_as_row_col() -> None:
    plane = RobotControlPlane()
    intent = _intent()

    actions = plane.map_policy_result(
        {
            "kind": "local_goal",
            "local_goal": {
                "mode": "pixel_goal",
                "pixel_goal": [240, 32],
                "image_width": 640,
            },
        },
        intent=intent,
    )

    decoded = [RobotSkillAction.from_robot_action(action) for action in actions]
    assert [(item.name, item.arguments) for item in decoded] == [
        ("turn_base", {"direction": "left", "angle_deg": 14.0})
    ]


@pytest.mark.asyncio
async def test_control_plane_applies_policy_result_sequence() -> None:
    plane = RobotControlPlane()
    intent = _intent()
    calls: list[str] = []

    async def apply(action: RobotAction) -> RobotStatus:
        calls.append(RobotSkillAction.from_robot_action(action).name)
        return RobotStatus(
            envelope=action.envelope,
            skill_id=action.skill_id,
            success=True,
        )

    status = await plane.apply_policy_result(
        {
            "kind": "action_chunk",
            "actions": [
                {"joints": {"shoulder_pan": 0.1}},
                {"gripper": 0.2},
            ],
        },
        intent=intent,
        apply_fn=apply,
    )

    assert status.success is True
    assert calls == ["move_arm_joints", "set_gripper"]
    assert plane.snapshot()["buffer_size"] == 2
