"""Behavioural tests for the analytic RoboCasa motion skills.

The fake robot deliberately applies the native action convention.  This keeps
the tests at the skill/robot boundary rather than asserting implementation
details of the servo loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from hey_robot.protocol import Envelope, RobotObservation
from hey_robot.skills import SkillContext, SkillRegistry
from hey_robot.skills.builtins import robocasa_primitives as primitives
from hey_robot.skills.models import SkillResult


@dataclass
class _NativeResult:
    success: bool
    summary: str
    observation: RobotObservation | None
    error: str | None = None


class _NativeRobot:
    def __init__(self) -> None:
        self.frame_id = 0
        self.position = np.zeros(3)
        self.base = np.zeros(3)
        self.gripper = 0.0
        self.actions: list[list[float]] = []

    def observation(self, robot_id: str = "robocasa") -> RobotObservation:
        proprioception = np.zeros(16)
        proprioception[:3] = self.position
        proprioception[7:10] = self.base
        proprioception[10:14] = [0.0, 0.0, 0.0, 1.0]
        proprioception[14:16] = self.gripper
        return RobotObservation(
            Envelope(robot_id=robot_id),
            self.frame_id,
            proprioception=proprioception.tolist(),
        )

    async def observe(self, robot_id: str, **_: Any) -> RobotObservation:
        return self.observation(robot_id)

    async def execute(
        self, robot_id: str, action: str, arguments: dict[str, Any], **_: Any
    ) -> _NativeResult:
        assert action == "embodiment_native_action"
        values = [float(value) for value in arguments["values"]]
        self.actions.append(values)
        if values[11] < 0:
            self.position += np.asarray(values[:3]) * 0.05
        else:
            self.base[0] += values[7] * 0.05
            self.base[1] += values[8] * 0.05
        self.gripper += values[6] * 0.01
        self.frame_id += 1
        return _NativeResult(True, "applied", self.observation(robot_id))


def _context(robot: _NativeRobot) -> SkillContext:
    return SkillContext("primitive-run", "task", "robocasa", robot=robot)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _clear_primitive_state() -> None:
    primitives._state_cache.clear()


async def test_arm_motion_gripper_and_delta_use_native_closed_loop_actions() -> None:
    robot = _NativeRobot()
    ctx = _context(robot)

    reached = await primitives.move_to(
        ctx, {"xyz": [0.12, 0.06, 0.06], "max_steps": 20, "tol": 0.015}
    )
    held = await primitives.set_gripper(ctx, {"gripper": 1.5, "steps": 2})
    released = await primitives.release(ctx, {"steps": 2})
    delta = await primitives.move_delta(ctx, {"dxyz": [0.02, 0.0, 0.0], "max_steps": 8})

    assert reached.success
    assert reached.data["steps"] > 0
    assert held.success
    assert released.success
    assert delta.success
    assert any(action[6] == -1.0 for action in robot.actions)
    assert all(len(action) == 12 for action in robot.actions)
    assert np.linalg.norm(robot.position - np.array([0.14, 0.06, 0.06])) < 0.03


async def test_base_motion_and_navigation_reset_arm_calibration() -> None:
    robot = _NativeRobot()
    ctx = _context(robot)
    primitives._state(ctx).pos_jac = np.eye(3)

    driven = await primitives.move_base(
        ctx, {"forward": 0.5, "lateral": -0.5, "turn": 2.0, "steps": 2}
    )
    reached = await primitives.navigate_to(
        ctx, {"xy": [0.45, 0.0], "max_steps": 12, "tol": 0.06}
    )

    assert driven.success
    assert reached.success
    assert primitives._state(ctx).pos_jac is None
    assert any(action[11] == 1.0 for action in robot.actions)
    assert reached.data["base_pos"][0] >= 0.35


async def test_localize_progress_and_registration_cover_observation_only_skills() -> (
    None
):
    robot = _NativeRobot()
    ctx = _context(robot)
    original_observation = robot.observation

    def calibrated_observation(robot_id: str = "robocasa") -> RobotObservation:
        obs = original_observation(robot_id)
        raw = {
            "task_progress": {"washed_time": 3},
            "cameras": {
                "robot0_agentview_left": {
                    "intrinsic": np.eye(3).tolist(),
                    "extrinsic_cam2world": np.eye(4).tolist(),
                    "height": 10,
                }
            },
        }
        return RobotObservation(
            obs.envelope, obs.frame_id, proprioception=obs.proprioception, raw=raw
        )

    robot.observation = calibrated_observation  # type: ignore[method-assign]
    localized = await primitives.localize(ctx, {"pixels": [[9, 0], [0, 0]], "z": 0.9})
    unavailable = await primitives.localize(
        ctx, {"pixels": [[0, 0]], "camera": "wrist"}
    )
    progress = await primitives.read_progress(ctx, {})
    registry = SkillRegistry()
    primitives.register(registry)

    assert localized.success
    assert localized.data["results"][0]["world_xyz"] == [0.0, 0.0, 0.9]
    assert not unavailable.success
    assert unavailable.failure_mode == "calibration_unavailable"
    assert progress.data["task_progress"] == {"washed_time": 3}
    assert {skill.name for skill in registry.list()} == {
        "move_to",
        "move_delta",
        "grip",
        "release",
        "scripted_grasp",
        "drive_base",
        "drive_to",
        "localize",
        "read_progress",
    }


async def test_scripted_grasp_preserves_the_failed_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = _NativeRobot()
    ctx = _context(robot)
    calls: list[dict[str, Any]] = []

    async def fake_move(_ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
        calls.append(arguments)
        return SkillResult(False, "blocked", "failed")

    monkeypatch.setattr(primitives, "move_to", fake_move)
    result = await primitives.scripted_grasp(ctx, {"xyz": [0.1, 0.2, 0.3]})

    assert not result.success
    assert result.data["stage"] == "approach"
    assert calls[0]["xyz"] == [0.1, 0.2, 0.4]


async def test_scripted_grasp_runs_all_stages_and_preserves_lift_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = _NativeRobot()
    ctx = _context(robot)
    moves: list[dict[str, Any]] = []

    async def fake_move(_ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
        moves.append(arguments)
        return SkillResult(True, "reached", "completed", data={"final": len(moves)})

    monkeypatch.setattr(primitives, "move_to", fake_move)
    result = await primitives.scripted_grasp(
        ctx,
        {"xyz": [0.1, 0.2, 0.3], "approach_z": 0.2, "grasp_z_offset": -0.02},
    )

    assert result.success
    assert result.data == {"final": 3}
    assert np.allclose(
        [move["xyz"] for move in moves],
        [[0.1, 0.2, 0.5], [0.1, 0.2, 0.28], [0.1, 0.2, 0.55]],
    )
    assert len(robot.actions) == 18


async def test_native_step_reports_unavailable_robot_and_action_failure() -> None:
    state = primitives._Obs(
        frame_id=1,
        proprioception=np.zeros(16),
        eef_world=np.zeros(3),
        base_pos=np.zeros(3),
        base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        gripper_qpos=np.zeros(2),
    )
    unavailable = SkillContext("run", "task", "robot")

    with pytest.raises(RuntimeError, match="unavailable"):
        await primitives._step(unavailable, state, [0.0] * 12)

    class FailedRobot:
        async def execute(self, *_args: Any, **_kwargs: Any) -> _NativeResult:
            return _NativeResult(False, "failed", None, error="driver error")

    with pytest.raises(RuntimeError, match="driver error"):
        await primitives._step(
            SkillContext("run", "task", "robot", robot=FailedRobot()),  # type: ignore[arg-type]
            state,
            [0.0] * 12,
        )
