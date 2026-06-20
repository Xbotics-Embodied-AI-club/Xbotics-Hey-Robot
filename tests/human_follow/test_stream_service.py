from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import numpy as np

from hey_robot.config import DeploymentConfig
from hey_robot.human_follow.service import HumanFollowService, _Session
from hey_robot.perception.frame_stream import decode_frame_packet, encode_frame_packet
from hey_robot.perception.human_follow import Detection
from hey_robot.robots.service import RobotService


class FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, dict(payload)))


def test_frame_packet_round_trip_preserves_metadata_and_rgb() -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :, 0] = 200

    metadata, decoded = decode_frame_packet(
        encode_frame_packet(image, {"robot_id": "xlerobot", "frame_id": 7})
    )

    assert metadata == {"robot_id": "xlerobot", "frame_id": 7}
    assert decoded.shape == image.shape
    assert float(decoded[:, :, 0].mean()) > 180


def test_human_follow_service_streams_velocity_inside_single_session(
    monkeypatch,
) -> None:
    config = DeploymentConfig.from_dict({"robots": {"xlerobot": {"type": "xlerobot"}}})
    service = HumanFollowService(config)
    service.bus = FakeBus()  # type: ignore[assignment]
    service._frames["xlerobot"] = (
        {"robot_id": "xlerobot", "camera": "front", "frame_id": 1},
        np.zeros((100, 100, 3), dtype=np.uint8),
    )
    service._frame_events["xlerobot"].set()
    monkeypatch.setattr(
        "hey_robot.human_follow.service.detect_people",
        lambda _image: [Detection((40, 10, 60, 70), 0.9)],
    )
    session = _Session(
        robot_id="xlerobot",
        skill_id="skill-1",
        session_id="session-1",
        arguments={"max_steps": 1},
    )
    service._sessions[session.session_id] = session

    asyncio.run(service._run_session(session))

    payloads = [payload for _topic, payload in service.bus.published]
    actions = [payload.get("action") for payload in payloads if "action" in payload]
    assert actions == ["open", "velocity", "close"]
    velocity = next(
        payload for payload in payloads if payload.get("action") == "velocity"
    )
    assert velocity["skill_id"] == "skill-1"
    assert velocity["sequence"] == 1
    assert any(payload.get("kind") == "result" for payload in payloads)


def test_robot_service_velocity_stream_rejects_stale_and_out_of_order() -> None:
    class FakeDriver:
        def __init__(self) -> None:
            self.velocities: list[dict] = []
            self.stops = 0

        async def apply_stream_velocity(self, **velocity):
            self.velocities.append(velocity)

        async def stop_base_stream(self):
            self.stops += 1

    driver = FakeDriver()
    service = object.__new__(RobotService)
    service.runtimes = {"xlerobot": SimpleNamespace(driver=driver)}
    service._base_streams = {}

    async def run() -> None:
        base = {"robot_id": "xlerobot", "session_id": "session-1"}
        await service._on_base_velocity_stream("", {**base, "action": "open"})
        await service._on_base_velocity_stream(
            "",
            {
                **base,
                "action": "velocity",
                "sequence": 1,
                "vx": 0.1,
                "expires_at": time.time() + 1,
            },
        )
        await service._on_base_velocity_stream(
            "",
            {
                **base,
                "action": "velocity",
                "sequence": 1,
                "vx": 0.2,
                "expires_at": time.time() + 1,
            },
        )
        await service._on_base_velocity_stream(
            "",
            {
                **base,
                "action": "velocity",
                "sequence": 2,
                "vx": 0.3,
                "expires_at": time.time() - 1,
            },
        )
        await service._on_base_velocity_stream("", {**base, "action": "close"})

    asyncio.run(run())

    assert len(driver.velocities) == 1
    assert driver.velocities[0]["vx"] == 0.1
    assert driver.stops == 1
