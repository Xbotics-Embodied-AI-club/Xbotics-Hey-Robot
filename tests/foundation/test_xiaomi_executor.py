from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import pytest
from PIL import Image

from hey_robot.config import ModelServiceSpec
from hey_robot.foundation.backends.xiaomi.executor import (
    XiaomiPolicyExecutor,
    _action_chunk,
    _quat_xyzw_to_axis_angle,
)


def _spec(**settings: Any) -> ModelServiceSpec:
    return ModelServiceSpec(
        type="robot_policy",
        robot_id="robocasa365",
        provides=("manipulate",),
        settings={
            "runtime": "xiaomi",
            "policy_path": "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365",
            "policy_device": "cuda:1",
            "embodiment": "robocasa",
            "robot_type": "robocasa365",
            "action_space": "robocasa_12d",
            "action_dimensions": 12,
            "state_dimensions": 16,
            "camera_names": ["camera1", "camera2", "camera3"],
            "prompt_mode": "environment_root",
            "execution_horizon": 16,
            **settings,
        },
    )


def _image(camera: str) -> dict[str, Any]:
    output = io.BytesIO()
    Image.new("RGB", (4, 3), color=(10, 20, 30)).save(output, format="PNG")
    return {
        "camera": camera,
        "data": base64.b64encode(output.getvalue()).decode(),
    }


def _payload(session_id: str = "episode-1", frame_id: int = 3) -> dict[str, Any]:
    state = np.zeros(16, dtype=np.float32)
    state[0:3] = [1, 2, 3]
    state[3:7] = [0, 0, 0, 1]
    state[7:10] = [4, 5, 6]
    state[10:14] = [0, 0, 0, 1]
    state[14:16] = [7, 8]
    return {
        "skill_name": "manipulate",
        "episode_id": session_id,
        "objective": "agent wording",
        "arguments": {
            "task_prompt": "agent subgoal",
            "observation": {
                "frame_id": frame_id,
                "proprioception": state.tolist(),
                "images": [_image("camera1"), _image("camera2"), _image("camera3")],
                "raw": {"policy_task": "close the fridge"},
            },
        },
    }


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, dict[str, np.ndarray], str]] = []
        self.closed = False

    def ping(self) -> bool:
        return True

    def infer(self, states, images, instruction):
        self.calls.append((states, images, instruction))
        return np.arange(16 * 12, dtype=np.float32).reshape(16, 12) / 100

    def close(self) -> None:
        self.closed = True


class _Process:
    def __init__(self) -> None:
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout: float | None = None):
        del timeout
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def _loaded_executor(client: _Client, **settings: Any) -> XiaomiPolicyExecutor:
    executor = XiaomiPolicyExecutor(
        "policy", _spec(**settings), client_factory=lambda *_: client
    )
    executor._client = client
    executor._loaded = True
    return executor


def test_xiaomi_executor_maps_history_state_and_caches_sixteen_actions() -> None:
    client = _Client()
    executor = _loaded_executor(client, action_low=-10, action_high=10)

    first = executor.execute(_payload())
    second = executor.execute(_payload())

    assert first["success"] is True
    assert second["success"] is True
    assert len(client.calls) == 1
    states, images, instruction = client.calls[0]
    assert states.shape == (4, 14)
    assert states[0].tolist() == [
        1.0,
        2.0,
        3.0,
        0.0,
        0.0,
        0.0,
        7.0,
        8.0,
        4.0,
        5.0,
        6.0,
        0.0,
        0.0,
        0.0,
    ]
    assert set(images) == {"camera1", "camera2", "camera3"}
    assert images["camera1"].shape == (4, 3, 4, 3)
    assert instruction == "close the fridge"
    assert first["metrics"]["inference_performed"] is True
    assert second["metrics"]["inference_performed"] is False
    assert executor.health()["metrics"]["queued_actions"] == 14


def test_xiaomi_executor_uses_agent_subgoal_when_configured() -> None:
    client = _Client()
    executor = _loaded_executor(client, prompt_mode="agent_subgoal")

    result = executor.execute(_payload())

    assert result["success"] is True
    assert client.calls[0][2] == "agent subgoal"


def test_xiaomi_executor_preserves_official_quantized_action_range() -> None:
    class OfficialRangeClient(_Client):
        def infer(self, states, images, instruction):
            self.calls.append((states, images, instruction))
            actions = np.zeros((16, 12), dtype=np.float32)
            actions[0, 6] = -1.0078125
            actions[0, 11] = 1.015625
            return actions

    client = OfficialRangeClient()
    executor = _loaded_executor(client)

    result = executor.execute(_payload())
    arguments = result["metrics"]["policy_result"]["actions"][0]["arguments"]

    assert arguments["values"][6] == pytest.approx(-1.0078125)
    assert arguments["values"][11] == pytest.approx(1.015625)
    assert result["metrics"]["action_clipped"] is False


def test_xiaomi_executor_preserves_history_when_agent_subgoal_changes() -> None:
    client = _Client()
    executor = _loaded_executor(client, prompt_mode="agent_subgoal")

    first = _payload(frame_id=3)
    executor.execute(first)
    second = _payload(frame_id=4)
    second["arguments"]["task_prompt"] = "next semantic stage"
    result = executor.execute(second)

    assert result["success"] is True
    assert len(client.calls) == 2
    assert client.calls[1][2] == "next semantic stage"
    assert len(executor._state_history) == 2
    assert len(executor._image_history) == 2
    assert executor.health()["metrics"]["queued_actions"] == 15


def test_xiaomi_executor_resets_history_for_a_new_episode() -> None:
    client = _Client()
    executor = _loaded_executor(client, prompt_mode="agent_subgoal")

    executor.execute(_payload(session_id="episode-1", frame_id=3))
    result = executor.execute(_payload(session_id="episode-2", frame_id=1))

    assert result["success"] is True
    assert len(client.calls) == 2
    assert len(executor._state_history) == 1
    assert len(executor._image_history) == 1


def test_xiaomi_executor_rejects_incomplete_observation() -> None:
    client = _Client()
    executor = _loaded_executor(client)
    payload = _payload()
    payload["arguments"]["observation"]["images"].pop()

    result = executor.execute(payload)

    assert result["success"] is False
    assert result["failure_mode"] == "observation_schema_mismatch"
    assert client.calls == []


def test_xiaomi_executor_accepts_co_located_raw_rgb_observations() -> None:
    client = _Client()
    executor = _loaded_executor(client)
    payload = _payload()
    observation = payload["arguments"]["observation"]
    observation["raw_pixels"] = {
        name: np.full((3, 4, 3), index, dtype=np.uint8)
        for index, name in enumerate(("camera1", "camera2", "camera3"))
    }
    observation.pop("images")

    result = executor.execute(payload)

    assert result["success"] is True
    assert client.calls[0][1]["camera2"].shape == (4, 3, 4, 3)
    assert np.all(client.calls[0][1]["camera3"] == 2)


@pytest.mark.parametrize(
    "raw_pixels",
    [
        {"camera1": np.zeros((3, 4, 3), dtype=np.uint8)},
        {
            "camera1": np.zeros((3, 4, 3), dtype=np.uint8),
            "camera2": np.zeros((3, 4, 3), dtype=np.uint8),
            "camera3": np.zeros((3, 4), dtype=np.uint8),
        },
    ],
)
def test_xiaomi_executor_rejects_invalid_co_located_raw_rgb_observations(
    raw_pixels: dict[str, np.ndarray],
) -> None:
    client = _Client()
    executor = _loaded_executor(client)
    payload = _payload()
    observation = payload["arguments"]["observation"]
    observation["raw_pixels"] = raw_pixels
    observation.pop("images")

    result = executor.execute(payload)

    assert result["success"] is False
    assert result["failure_mode"] == "observation_schema_mismatch"
    assert client.calls == []


def test_xiaomi_executor_manages_official_server_process(tmp_path) -> None:
    repo = tmp_path / "Xiaomi-Robotics-1"
    (repo / "deploy").mkdir(parents=True)
    (repo / "deploy/server.py").touch()
    python = tmp_path / "python"
    python.touch()
    client = _Client()
    process = _Process()
    spawns: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def spawn(*args: Any, **kwargs: Any) -> _Process:
        spawns.append((args, kwargs))
        return process

    executor = XiaomiPolicyExecutor(
        "policy",
        _spec(
            xiaomi_repo=str(repo),
            xiaomi_python=str(python),
            server_port=10086,
        ),
        client_factory=lambda *_: client,
        process_factory=spawn,
    )

    executor.load()

    command = spawns[0][0][0]
    assert command[:3] == [
        str(python.absolute()),
        "-u",
        str((repo / "deploy/server.py").resolve()),
    ]
    assert command[command.index("--port") : command.index("--port") + 2] == [
        "--port",
        "10086",
    ]
    assert spawns[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    executor.close()
    assert client.closed is True
    assert process.terminated is True


def test_xiaomi_executor_validates_static_contract() -> None:
    for settings, message in (
        ({"action_dimensions": 11}, "action_dimensions"),
        ({"state_dimensions": 14}, "state_dimensions"),
        ({"execution_horizon": 17}, "execution_horizon"),
        ({"prompt_mode": "invalid"}, "prompt_mode"),
        ({"crop_ratio": 2}, "crop_ratio"),
        ({"observation_delta_indices": [-6, 0, -2]}, "observation_delta_indices"),
    ):
        with pytest.raises(ValueError, match=message):
            XiaomiPolicyExecutor("policy", _spec(**settings))


def test_xiaomi_action_and_quaternion_helpers() -> None:
    actions = np.zeros((1, 16, 14), dtype=np.float32)
    assert len(_action_chunk(actions, dimensions=12, execution_horizon=16)) == 16
    rotation = _quat_xyzw_to_axis_angle(
        np.asarray([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    )
    assert rotation == pytest.approx([0.0, 0.0, np.pi / 2])
