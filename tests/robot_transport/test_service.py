from __future__ import annotations

from hey_robot.protocol import (
    ArtifactRef,
    Envelope,
    ImageRef,
    RobotObservation,
    RobotStatus,
)
from hey_robot.robot_transport import RobotService


def test_robot_service_preserves_motion_status_fields() -> None:
    service = object.__new__(RobotService)
    status = RobotStatus(
        Envelope(robot_id="mock0"),
        state="executing",
        location_id="room:living_room",
        motion_state="moving",
        battery_percentage=72.0,
    )

    projected = service._status_for_publish(status)

    assert projected.location_id == "room:living_room"
    assert projected.motion_state == "moving"
    assert projected.battery_percentage == 72.0


def test_robot_service_does_not_publish_invalid_camera_images_to_scene_memory() -> None:
    black_frame = RobotObservation(
        envelope=Envelope(robot_id="mock0"),
        frame_id=1,
        images=[ImageRef(uri="media://local/images/mock0/black.jpg", camera="front")],
        raw={
            "perception": {
                "image_count": 1,
                "valid_image_count": 0,
                "image_quality_issues": ["black_frame"],
            }
        },
    )
    valid_frame = RobotObservation(
        envelope=Envelope(robot_id="mock0"),
        frame_id=2,
        images=[ImageRef(uri="media://local/images/mock0/frame.jpg", camera="front")],
        raw={"perception": {"image_count": 1, "valid_image_count": 1}},
    )
    artifact_only = RobotObservation(
        envelope=Envelope(robot_id="mock0"),
        frame_id=3,
        artifacts=[
            ArtifactRef(
                uri="media://local/artifacts/mock0/state.json",
                artifact_type="policy_observation",
            )
        ],
        raw={"perception": {"image_count": 0, "valid_image_count": 0}},
    )

    assert RobotService._should_publish_observation(black_frame) is False
    assert RobotService._should_publish_observation(valid_frame) is True
    assert RobotService._should_publish_observation(artifact_only) is True
