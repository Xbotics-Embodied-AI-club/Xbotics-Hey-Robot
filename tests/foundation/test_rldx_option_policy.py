from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from hey_robot.foundation.backends.rldx.option_policy import RLDXChunkPolicy


class _Client:
    def __init__(self) -> None:
        self.options = []

    def ping(self):
        return True

    def close(self):
        pass

    def get_action(self, _observation, options):
        self.options.append(options)
        return {
            "action.end_effector_position": np.ones((1, 8, 3), np.float32),
            "action.end_effector_rotation": np.ones((1, 8, 3), np.float32),
            "action.gripper_close": np.ones((1, 8, 1), np.float32),
            "action.base_motion": np.ones((1, 8, 4), np.float32),
            "action.control_mode": np.ones((1, 8, 1), np.float32),
        }


def _image() -> str:
    output = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _payload(frame_id: int):
    return {
        "frame_id": frame_id,
        "proprioception": [0.0] * 16,
        "images": [
            {"camera": camera, "data": _image()}
            for camera in ("camera1", "camera2", "camera3")
        ],
    }


def test_rldx_chunk_policy_preserves_memory_within_a_session() -> None:
    client = _Client()
    policy = RLDXChunkPolicy(
        client,
        settings={
            "action_dimensions": 12,
            "execution_horizon": 8,
            "base_clip": 0.1,
            "camera_names": ["camera1", "camera2", "camera3"],
            "video_delta_indices": [-6, -4, -2, 0],
        },
        observation_encoder=lambda value: value,
    )

    policy.reset("episode-1")
    assert len(policy.predict(_payload(0), "rinse sink")) == 8
    assert len(policy.predict(_payload(8), "rinse sink")) == 8

    assert client.options == [
        {"session_ids": ["episode-1"], "reset_memory": [True]},
        {"session_ids": ["episode-1"], "reset_memory": [False]},
    ]
