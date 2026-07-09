from __future__ import annotations

from hey_robot.protocol import Envelope, ImageRef, RobotObservation
from hey_robot.skill_os.builtins.manipulation import _observation_payload
from hey_robot.vla.so101_schema import SO101_STATE_SCHEMA


def test_observation_payload_includes_explicit_single_arm_vla_state() -> None:
    observation = RobotObservation(
        envelope=Envelope(trace_id="tr1", robot_id="sim_robot", timestamp=123.0),
        frame_id=42,
        images=[ImageRef(uri="media://local/front.jpg", camera="front")],
        proprioception=[99.0, 98.0, 97.0],
        raw={
            "arm_status": {
                "joint_states": {
                    "shoulder_pan": 0.1,
                    "shoulder_lift": 0.2,
                    "elbow_flex": 0.3,
                    "wrist_flex": 0.4,
                    "wrist_roll": 0.5,
                },
                "gripper_opening_pct": 25.0,
            },
            "active_arm": "right",
        },
    )

    payload = _observation_payload(observation)

    assert payload is not None
    assert payload["frame_id"] == 42
    assert payload["state"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.25]
    assert payload["state_schema"] == SO101_STATE_SCHEMA
    assert payload["active_arm"] == "right"
    assert payload["proprioception"] == [99.0, 98.0, 97.0]
    assert payload["images"][0]["camera"] == "front"


def test_observation_payload_preserves_custom_vla_state_schema() -> None:
    observation = RobotObservation(
        envelope=Envelope(trace_id="tr1", robot_id="sim_robot"),
        frame_id=7,
        raw={
            "vla_state": [1, 2, 3, 4, 5, 0.6],
            "vla_state_schema": "custom_schema",
            "active_arm": "left",
        },
    )

    payload = _observation_payload(observation)

    assert payload is not None
    assert payload["state"] == [1.0, 2.0, 3.0, 4.0, 5.0, 0.6]
    assert payload["state_schema"] == "custom_schema"
    assert payload["active_arm"] == "left"


def test_observation_payload_does_not_invent_vla_state_without_arm_status() -> None:
    observation = RobotObservation(
        envelope=Envelope(trace_id="tr1", robot_id="sim_robot"),
        frame_id=8,
        proprioception=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        raw={"base_pose": [0.0, 0.0, 0.0]},
    )

    payload = _observation_payload(observation)

    assert payload is not None
    assert "state" not in payload
    assert payload["proprioception"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
