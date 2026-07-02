from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from hey_robot.protocol import Envelope, ImageRef, RobotObservation
from hey_robot.skill_os.builtins.navigation import HumanFollowSkill
from hey_robot.skill_os.perception.human_follow import (
    FollowController,
    HumanFollowRunner,
    VelocityCommand,
)


def test_follow_controller_turns_before_driving_toward_off_center_target() -> None:
    controller = FollowController(
        target_distance=0.7,
        target_width_ratio=0.35,
        kp_linear=0.35,
        kp_angular=1.0,
    )
    target = SimpleNamespace(center=(90, 50), area=100)

    command = controller.compute_velocity(target, frame_width=100, frame_height=100)

    assert command is not None
    assert command.vx == 0.0
    assert command.vz > 0.0


# ── FollowController ──────────────────────────────────────────────────────


def test_follow_controller_returns_none_when_target_briefly_lost() -> None:
    controller = FollowController()
    # Simulate a few missed frames — not enough to be permanently lost
    controller.target_lost_count = 3
    controller.compute_velocity(None, frame_width=320, frame_height=320)

    command = controller.compute_velocity(None, frame_width=320, frame_height=320)

    assert command is None
    assert controller.is_searching() is True
    assert controller.is_target_lost() is False


def test_follow_controller_returns_zero_velocity_when_target_permanently_lost() -> None:
    controller = FollowController()
    # Exceed max_lost_count
    controller.target_lost_count = controller.max_lost_count + 1
    controller.compute_velocity(None, frame_width=320, frame_height=320)

    command = controller.compute_velocity(None, frame_width=320, frame_height=320)

    assert command is not None
    assert command.vx == 0.0
    assert command.vz == 0.0
    assert controller.is_target_lost() is True


def test_follow_controller_ignores_target_within_dead_zone() -> None:
    controller = FollowController(
        target_distance=0.7,
        target_width_ratio=0.35,
        kp_linear=0.35,
        kp_angular=1.0,
        dead_zone_x=0.15,
        dead_zone_area=0.95,  # very wide area dead zone so only x matters
    )
    # Target at exact center — x within dead zone
    target = SimpleNamespace(center=(160, 160), area=40000)

    command = controller.compute_velocity(target, frame_width=320, frame_height=320)

    assert command is not None
    assert abs(command.vz) < 0.02  # angular suppressed by dead zone
    assert abs(command.vx) < 0.02  # area also within wide dead zone


def test_follow_controller_produces_search_velocity() -> None:
    controller = FollowController(max_angular_speed=0.8)

    search = controller.compute_search_velocity()

    assert search.vx == 0.0
    assert search.vy == 0.0
    assert search.vz == pytest.approx(0.4)


def test_follow_controller_smooth_velocity_interpolates() -> None:
    current = VelocityCommand(0.1, 0.0, 0.3)
    target = VelocityCommand(0.5, 0.0, 0.1)

    smoothed = FollowController.smooth_velocity(current, target, alpha=0.3)

    assert smoothed.vx == pytest.approx(0.3 * 0.5 + 0.7 * 0.1)
    assert smoothed.vz == pytest.approx(0.3 * 0.1 + 0.7 * 0.3)


def test_follow_controller_constrains_forward_speed_when_off_center() -> None:
    """Do not drive toward a person until chassis is substantially aligned."""
    controller = FollowController(
        target_distance=0.7,
        target_width_ratio=0.35,
        kp_linear=0.5,
        kp_angular=1.0,
    )
    # Target far off center — vx should be scaled down by alignment
    target = SimpleNamespace(center=(280, 160), area=102400)

    command = controller.compute_velocity(target, frame_width=320, frame_height=320)

    assert command is not None
    assert command.vz > 0.2
    # vx should be reduced because chassis is not aligned
    assert command.vx < 0.05


def test_follow_controller_target_lost_count_resets_on_detection() -> None:
    controller = FollowController()
    controller.compute_velocity(None, frame_width=320, frame_height=320)
    controller.compute_velocity(None, frame_width=320, frame_height=320)
    assert controller.target_lost_count == 2

    target = SimpleNamespace(center=(160, 160), area=102400)
    controller.compute_velocity(target, frame_width=320, frame_height=320)

    assert controller.target_lost_count == 0
    assert controller.is_target_lost() is False


# ── HumanFollowRunner ─────────────────────────────────────────────────────


def test_human_follow_runner_enters_search_when_target_disappears_then_recovers(
    monkeypatch,
) -> None:
    call_count = 0

    def fake_detect(_image):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            return []
        return [
            SimpleNamespace(
                bbox=(60, 10, 70, 30),
                confidence=1.0,
                center=(65, 20),
                area=200,
            )
        ]

    monkeypatch.setattr(
        "hey_robot.skill_os.perception.human_follow.detect_people", fake_detect
    )
    phases: list[str] = []

    async def get_frame():
        return (
            {"robot_id": "r0", "frame_id": call_count + 1},
            np.zeros((100, 100, 3), dtype=np.uint8),
        )

    async def apply_velocity(_vx, _vy, _wz):
        pass

    async def emit_progress(**payload):
        phases.append(payload.get("phase", ""))

    runner = HumanFollowRunner(
        {"max_steps": 4, "target_height_ratio": 0.3},
        get_frame=get_frame,
        apply_velocity=apply_velocity,
        emit_progress=emit_progress,
        is_stopped=lambda: False,
    )

    result = asyncio.run(runner.run())

    assert result["success"] is True
    assert "searching" in phases


def test_human_follow_runner_waits_when_camera_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "hey_robot.skill_os.perception.human_follow.detect_people",
        lambda _image: [
            SimpleNamespace(
                bbox=(60, 10, 70, 30),
                confidence=1.0,
                center=(65, 20),
                area=200,
            )
        ],
    )
    call_count = 0
    progress_phases: list[str] = []

    async def get_frame():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return None
        return (
            {"robot_id": "r0", "frame_id": call_count},
            np.zeros((100, 100, 3), dtype=np.uint8),
        )

    async def apply_velocity(_vx, _vy, _wz):
        pass

    async def emit_progress(**payload):
        progress_phases.append(payload.get("phase", ""))

    runner = HumanFollowRunner(
        {"max_steps": 1, "target_height_ratio": 0.3},
        get_frame=get_frame,
        apply_velocity=apply_velocity,
        emit_progress=emit_progress,
        is_stopped=lambda: False,
    )

    asyncio.run(runner.run())

    assert "waiting_for_camera" in progress_phases


def test_human_follow_runner_stops_when_target_permanently_lost(monkeypatch) -> None:
    monkeypatch.setattr(
        "hey_robot.skill_os.perception.human_follow.detect_people",
        lambda _image: [],
    )
    frame_count = 0

    async def get_frame():
        nonlocal frame_count
        frame_count += 1
        return (
            {"robot_id": "r0", "frame_id": frame_count},
            np.zeros((100, 100, 3), dtype=np.uint8),
        )

    async def apply_velocity(_vx, _vy, _wz):
        pass

    runner = HumanFollowRunner(
        {"duration_sec": 0.3, "target_height_ratio": 0.3},
        get_frame=get_frame,
        apply_velocity=apply_velocity,
        is_stopped=lambda: False,
    )

    result = asyncio.run(runner.run())

    assert result["success"] is False
    assert result["failure_mode"] == "person_lost"


def test_human_follow_runner_completes_by_duration(monkeypatch) -> None:
    monkeypatch.setattr(
        "hey_robot.skill_os.perception.human_follow.detect_people",
        lambda _image: [
            SimpleNamespace(
                bbox=(60, 10, 70, 30),
                confidence=1.0,
                center=(65, 20),
                area=200,
            )
        ],
    )

    async def get_frame():
        return (
            {"robot_id": "r0", "frame_id": 1},
            np.zeros((100, 100, 3), dtype=np.uint8),
        )

    async def apply_velocity(_vx, _vy, _wz):
        pass

    runner = HumanFollowRunner(
        {"duration_sec": 0.05, "target_height_ratio": 0.3},
        get_frame=get_frame,
        apply_velocity=apply_velocity,
        is_stopped=lambda: False,
    )

    result = asyncio.run(runner.run())

    assert result["success"] is True
    assert "completed" in result["summary"]


def test_human_follow_runner_calls_on_start_and_on_stop(monkeypatch) -> None:
    monkeypatch.setattr(
        "hey_robot.skill_os.perception.human_follow.detect_people",
        lambda _image: [
            SimpleNamespace(
                bbox=(60, 10, 70, 30),
                confidence=1.0,
                center=(65, 20),
                area=200,
            )
        ],
    )
    events: list[str] = []

    async def get_frame():
        return (
            {"robot_id": "r0", "frame_id": 1},
            np.zeros((100, 100, 3), dtype=np.uint8),
        )

    async def apply_velocity(_vx, _vy, _wz):
        pass

    async def on_start():
        events.append("start")

    async def on_stop():
        events.append("stop")

    runner = HumanFollowRunner(
        {"max_steps": 1, "target_height_ratio": 0.3},
        get_frame=get_frame,
        apply_velocity=apply_velocity,
        is_stopped=lambda: False,
        on_start=on_start,
        on_stop=on_stop,
    )

    asyncio.run(runner.run())

    assert events == ["start", "stop"]


def test_human_follow_runner_propagates_internal_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "hey_robot.skill_os.perception.human_follow.detect_people",
        lambda _image: [
            SimpleNamespace(
                bbox=(60, 10, 70, 30),
                confidence=1.0,
                center=(65, 20),
                area=200,
            )
        ],
    )

    async def get_frame():
        return (
            {"robot_id": "r0", "frame_id": 1},
            np.zeros((100, 100, 3), dtype=np.uint8),
        )

    async def apply_velocity(_vx, _vy, _wz):
        raise RuntimeError("motor fault")

    runner = HumanFollowRunner(
        {"max_steps": 1, "target_height_ratio": 0.3},
        get_frame=get_frame,
        apply_velocity=apply_velocity,
        is_stopped=lambda: False,
    )

    result = asyncio.run(runner.run())

    assert result["success"] is False
    assert result["failure_mode"] == "internal_error"
    assert "motor fault" in result["summary"]


# ── Skill integration tests ───────────────────────────────────────────────


def test_human_follow_skill_runs_from_skill_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "hey_robot.skill_os.builtins.navigation.load_detector",
        lambda _path=None: None,
    )
    monkeypatch.setattr(
        "hey_robot.skill_os.perception.human_follow.detect_people",
        lambda _image: [
            SimpleNamespace(
                bbox=(60, 10, 70, 30),
                confidence=1.0,
                center=(65, 20),
                area=200,
            )
        ],
    )
    moves: list[tuple[str, dict]] = []
    observation = RobotObservation(
        envelope=Envelope(robot_id="mock0"),
        frame_id=1,
        images=[ImageRef(uri="media://local/images/mock0/frame.jpg", camera="front")],
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    class FakeRobot:
        async def move_base(self, **arguments):
            moves.append(("move_base", dict(arguments)))
            return {"success": True}

        async def turn_base(self, **arguments):
            moves.append(("turn_base", dict(arguments)))
            return {"success": True}

        async def base_velocity_step(self, **arguments):
            moves.append(("base_velocity_step", dict(arguments)))
            return {"success": True}

        async def stop_motion(self, **arguments):
            moves.append(("stop_motion", dict(arguments)))
            return {"success": True}

    ctx = SimpleNamespace(
        robot=FakeRobot(),
        observation=observation,
        current_observation=lambda: observation,
        resolve_images=lambda _refs: [image],
        get_camera_frame=lambda: (
            {"robot_id": "mock0", "camera": "front", "frame_id": 1},
            image,
        ),
        invoke=None,
        logger=None,
    )

    async def run():
        return await HumanFollowSkill().execute(
            ctx,
            {
                "duration_sec": 1,
                "target_height_ratio": 0.3,
            },
        )

    result = asyncio.run(run())

    assert result.success is True
    assert any(name == "base_velocity_step" for name, _ in moves)
    assert moves[-1][0] == "stop_motion"


def test_human_follow_skill_emits_user_visible_progress(monkeypatch) -> None:
    monkeypatch.setattr(
        "hey_robot.skill_os.builtins.navigation.load_detector",
        lambda _path=None: None,
    )
    monkeypatch.setattr(
        "hey_robot.skill_os.perception.human_follow.detect_people",
        lambda _image: [
            SimpleNamespace(
                id="person-1",
                bbox=(60, 10, 70, 30),
                confidence=0.91,
                center=(65, 20),
                area=200,
            )
        ],
    )
    progress_events: list[dict] = []
    observation = RobotObservation(
        envelope=Envelope(robot_id="mock0"),
        frame_id=7,
        images=[ImageRef(uri="media://local/images/mock0/frame.jpg", camera="front")],
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    class FakeRobot:
        async def move_base(self, **_arguments):
            return {"success": True}

        async def turn_base(self, **_arguments):
            return {"success": True}

        async def base_velocity_step(self, **_arguments):
            return {"success": True}

        async def stop_motion(self, **_arguments):
            return {"success": True}

    async def capture_progress(**kwargs):
        progress_events.append(kwargs)

    ctx = SimpleNamespace(
        robot=FakeRobot(),
        observation=observation,
        current_observation=lambda: observation,
        resolve_images=lambda _refs: [image],
        get_camera_frame=lambda: (
            {"robot_id": "mock0", "camera": "front", "frame_id": 1},
            image,
        ),
        invoke=None,
        logger=None,
        progress=capture_progress,
    )

    async def run():
        return await HumanFollowSkill().execute(
            ctx,
            {
                "max_steps": 1,
                "target_height_ratio": 0.3,
            },
        )

    result = asyncio.run(run())

    assert result.success is True
    steps = [event["step"] for event in progress_events]
    assert "following" in steps
    assert "completed" in steps
    following = next(event for event in progress_events if event["step"] == "following")
    ux = following["metadata"]["ux"]
    assert ux["bbox"] == [60, 10, 70, 30]
    assert ux["confidence"] == 0.91
    assert ux["frame_id"] == 7
    assert ux["camera"] == "front"
    assert ux["command"]["vx"] != 0


def test_human_follow_skill_supports_unbounded_run_with_max_steps(monkeypatch) -> None:
    monkeypatch.setattr(
        "hey_robot.skill_os.builtins.navigation.load_detector",
        lambda _path=None: None,
    )
    monkeypatch.setattr(
        "hey_robot.skill_os.perception.human_follow.detect_people",
        lambda _image: [
            SimpleNamespace(
                bbox=(60, 10, 70, 30),
                confidence=1.0,
                center=(65, 20),
                area=200,
            )
        ],
    )
    moves: list[tuple[str, dict]] = []
    observation = RobotObservation(
        envelope=Envelope(robot_id="mock0"),
        frame_id=1,
        images=[ImageRef(uri="media://local/images/mock0/frame.jpg", camera="front")],
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    class FakeRobot:
        async def move_base(self, **arguments):
            moves.append(("move_base", dict(arguments)))
            return {"success": True}

        async def turn_base(self, **arguments):
            moves.append(("turn_base", dict(arguments)))
            return {"success": True}

        async def base_velocity_step(self, **arguments):
            moves.append(("base_velocity_step", dict(arguments)))
            return {"success": True}

        async def stop_motion(self, **arguments):
            moves.append(("stop_motion", dict(arguments)))
            return {"success": True}

    next_frame_id = 0

    def current_observation():
        nonlocal next_frame_id
        next_frame_id += 1
        return RobotObservation(
            envelope=observation.envelope,
            frame_id=next_frame_id,
            images=observation.images,
        )

    def get_camera_frame():
        nonlocal next_frame_id
        next_frame_id += 1
        return (
            {"robot_id": "mock0", "camera": "front", "frame_id": next_frame_id},
            image,
        )

    ctx = SimpleNamespace(
        robot=FakeRobot(),
        observation=observation,
        current_observation=current_observation,
        resolve_images=lambda _refs: [image],
        get_camera_frame=get_camera_frame,
        invoke=None,
        logger=None,
    )

    async def run():
        return await HumanFollowSkill().execute(
            ctx,
            {
                "max_steps": 2,
                "target_height_ratio": 0.3,
            },
        )

    result = asyncio.run(run())

    assert result.success is True
    assert result.summary == "human follow completed"
    assert len(result.data["steps"]) >= 2
    assert moves[-1][0] == "stop_motion"


def test_human_follow_skill_stops_motion_on_cancel(monkeypatch) -> None:
    monkeypatch.setattr(
        "hey_robot.skill_os.builtins.navigation.load_detector",
        lambda _path=None: None,
    )
    monkeypatch.setattr(
        "hey_robot.skill_os.perception.human_follow.detect_people",
        lambda _image: [
            SimpleNamespace(
                bbox=(60, 10, 70, 30),
                confidence=1.0,
                center=(65, 20),
                area=200,
            )
        ],
    )
    moves: list[tuple[str, dict]] = []
    observation = RobotObservation(
        envelope=Envelope(robot_id="mock0"),
        frame_id=1,
        images=[ImageRef(uri="media://local/images/mock0/frame.jpg", camera="front")],
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    class FakeRobot:
        async def move_base(self, **arguments):
            moves.append(("move_base", dict(arguments)))
            await asyncio.sleep(0)
            return {"success": True}

        async def turn_base(self, **arguments):
            moves.append(("turn_base", dict(arguments)))
            await asyncio.sleep(0)
            return {"success": True}

        async def base_velocity_step(self, **arguments):
            moves.append(("base_velocity_step", dict(arguments)))
            await asyncio.sleep(0)
            return {"success": True}

        async def stop_motion(self, **arguments):
            moves.append(("stop_motion", dict(arguments)))
            return {"success": True}

    ctx = SimpleNamespace(
        robot=FakeRobot(),
        observation=observation,
        current_observation=lambda: observation,
        resolve_images=lambda _refs: [image],
        get_camera_frame=lambda: (
            {"robot_id": "mock0", "camera": "front", "frame_id": 1},
            image,
        ),
        invoke=None,
        logger=None,
    )

    async def run() -> None:
        task = asyncio.create_task(
            HumanFollowSkill().execute(
                ctx,
                {
                    "target_height_ratio": 0.3,
                },
            )
        )
        await asyncio.sleep(0.15)
        task.cancel()
        with np.testing.assert_raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

    assert moves[-1][0] == "stop_motion"
