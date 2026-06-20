from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np

from hey_robot.perception.human_follow import FollowController
from hey_robot.protocol import Envelope, ImageRef, RobotObservation
from hey_robot.skills.builtin.navigation import HumanFollowSkill


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


def test_human_follow_skill_runs_from_skill_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "hey_robot.skills.builtin.navigation.load_detector",
        lambda _path=None: None,
    )
    monkeypatch.setattr(
        "hey_robot.skills.builtin.navigation.detect_people",
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
        "hey_robot.skills.builtin.navigation.load_detector",
        lambda _path=None: None,
    )
    monkeypatch.setattr(
        "hey_robot.skills.builtin.navigation.detect_people",
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
    assert "starting" in steps
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
        "hey_robot.skills.builtin.navigation.load_detector",
        lambda _path=None: None,
    )
    monkeypatch.setattr(
        "hey_robot.skills.builtin.navigation.detect_people",
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

    ctx = SimpleNamespace(
        robot=FakeRobot(),
        observation=observation,
        current_observation=current_observation,
        resolve_images=lambda _refs: [image],
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
        "hey_robot.skills.builtin.navigation.load_detector",
        lambda _path=None: None,
    )
    monkeypatch.setattr(
        "hey_robot.skills.builtin.navigation.detect_people",
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
