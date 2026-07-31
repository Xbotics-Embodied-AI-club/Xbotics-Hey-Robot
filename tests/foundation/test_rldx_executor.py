from __future__ import annotations

import base64
import io
import sys
import types
from typing import Any

import numpy as np
import pytest
from PIL import Image

from hey_robot.config import ModelServiceSpec
from hey_robot.foundation.backends.rldx.executor import (
    RLDXPolicyExecutor,
    _action_chunk,
    _decode_wire_value,
    _encode_wire_value,
)
from hey_robot.foundation.backends.rldx.server import _parser, main


def _spec(**settings: Any) -> ModelServiceSpec:
    return ModelServiceSpec(
        type="robot_policy",
        robot_id="robocasa365",
        provides=("manipulate",),
        settings={
            "runtime": "rldx",
            "policy_path": "RLWRLD/RLDX-1-FT-RC365",
            "policy_device": "cpu",
            "embodiment": "robocasa",
            "action_space": "robocasa_12d",
            "action_dimensions": 12,
            "state_dimensions": 16,
            "camera_names": ["camera1", "camera2", "camera3"],
            "prompt_mode": "agent_subgoal",
            "execution_horizon": 8,
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


def _payload(session_id: str = "episode-1") -> dict[str, Any]:
    return {
        "skill_name": "manipulate",
        "episode_id": session_id,
        "objective": "agent wording",
        "arguments": {
            "task_prompt": "agent subgoal",
            "observation": {
                "frame_id": 3,
                "proprioception": [float(value) for value in range(16)],
                "images": [_image("camera1"), _image("camera2"), _image("camera3")],
                "raw": {"policy_task": "boil the kettle"},
            },
        },
    }


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.closed = False

    def ping(self) -> bool:
        return True

    def get_action(self, observation, options):
        self.calls.append((observation, options))
        action = {
            "action.end_effector_position": np.ones((1, 16, 3), np.float32),
            "action.end_effector_rotation": np.full((1, 16, 3), 2, np.float32),
            "action.gripper_close": np.full((1, 16, 1), 3, np.float32),
            "action.base_motion": np.full((1, 16, 4), 4, np.float32),
            "action.control_mode": np.full((1, 16, 1), 5, np.float32),
        }
        return [action, {"ok": True}]

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


def _loaded_executor(client: _Client, **settings: Any) -> RLDXPolicyExecutor:
    executor = RLDXPolicyExecutor(
        "policy", _spec(**settings), client_factory=lambda *_: client
    )
    executor._client = client
    executor._loaded = True
    return executor


def test_rldx_executor_maps_observation_and_caches_official_eight_step_chunk() -> None:
    client = _Client()
    executor = _loaded_executor(client, action_low=-10, action_high=10)

    first = executor.execute(_payload())
    second = executor.execute(_payload())

    assert first["success"] is True
    assert second["success"] is True
    assert len(client.calls) == 1
    observation, options = client.calls[0]
    assert observation["video.robot0_agentview_left"].shape == (1, 4, 3, 4, 3)
    assert observation["video.robot0_agentview_left"].dtype == np.uint8
    assert observation["state.end_effector_position_relative"].tolist() == [
        [[0.0, 1.0, 2.0]]
    ]
    assert observation["state.end_effector_rotation_relative"].shape == (1, 1, 4)
    assert observation["state.base_position"].tolist() == [[[7.0, 8.0, 9.0]]]
    assert observation["state.base_rotation"].shape == (1, 1, 4)
    assert observation["state.gripper_qpos"].tolist() == [[[14.0, 15.0]]]
    assert observation["annotation.human.task_description"] == ["agent subgoal"]
    assert options == {"session_ids": ["episode-1"], "reset_memory": [True]}
    values = first["metrics"]["policy_result"]["actions"][0]["arguments"]["values"]
    assert values == [1.0] * 3 + [2.0] * 3 + [3.0] + [4.0] * 4 + [5.0]
    assert first["metrics"]["inference_performed"] is True
    assert second["metrics"]["inference_performed"] is False
    assert executor.health()["metrics"]["queued_actions"] == 6


def test_rldx_executor_resets_memory_for_a_new_episode() -> None:
    client = _Client()
    executor = _loaded_executor(client)

    executor.execute(_payload("episode-1"))
    executor.execute(_payload("episode-2"))

    assert len(client.calls) == 2
    assert client.calls[1][1] == {
        "session_ids": ["episode-2"],
        "reset_memory": [True],
    }


def test_rldx_executor_rejects_incomplete_observation() -> None:
    client = _Client()
    executor = _loaded_executor(client)
    payload = _payload()
    payload["arguments"]["observation"]["images"].pop()

    result = executor.execute(payload)

    assert result["success"] is False
    assert result["failure_mode"] == "observation_schema_mismatch"
    assert client.calls == []


def test_rldx_executor_manages_official_server_process(tmp_path) -> None:
    repo = tmp_path / "RLDX-1"
    (repo / "rldx/eval").mkdir(parents=True)
    python = tmp_path / "python"
    python.touch()
    client = _Client()
    process = _Process()
    spawns: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def spawn(*args: Any, **kwargs: Any) -> _Process:
        spawns.append((args, kwargs))
        return process

    executor = RLDXPolicyExecutor(
        "policy",
        _spec(
            rldx_repo=str(repo),
            rldx_python=str(python),
            server_host="127.0.0.1",
            server_port=15555,
        ),
        client_factory=lambda *_: client,
        process_factory=spawn,
    )

    assert executor.health()["loaded"] is False
    executor.load()
    assert executor.health()["loaded"] is True
    command = spawns[0][0][0]
    assert command[:3] == [
        str(python),
        "-m",
        "hey_robot.foundation.backends.rldx.server",
    ]
    assert command[command.index("--port") : command.index("--port") + 2] == [
        "--port",
        "15555",
    ]
    assert command[command.index("--device") : command.index("--device") + 2] == [
        "--device",
        "cpu",
    ]
    assert command[-2:] == ["--image-max-area", "65536"]
    executor.close()
    assert client.closed is True
    assert process.terminated is True


def test_rldx_server_parser_and_main_configure_official_server(monkeypatch) -> None:
    parser = _parser()
    args = parser.parse_args(["--model-path", "checkpoint", "--port", "15555"])
    assert args.model_path == "checkpoint"
    assert args.port == 15555

    calls: list[tuple[str, object]] = []

    class _Processor:
        @staticmethod
        def from_pretrained(path: Any, **kwargs: Any) -> None:
            calls.append((path, kwargs))

    class _Policy:
        def __init__(self, **kwargs):
            calls.append(("policy", kwargs))
            _Processor.from_pretrained("checkpoint")

    class _Wrapper:
        def __init__(self, policy, **kwargs):
            calls.append(("wrapper", (policy, kwargs)))

    class _Server:
        def __init__(self, **kwargs):
            calls.append(("server", kwargs))

        def run(self):
            calls.append(("run", True))

    modules = {
        "rldx.data.embodiment_tags": types.SimpleNamespace(
            EmbodimentTag={"GENERAL_EMBODIMENT": "tag"}
        ),
        "rldx.policy.rldx_policy": types.SimpleNamespace(
            RLDXPolicy=_Policy, RLDXSimPolicyWrapper=_Wrapper
        ),
        "rldx.policy.server_client": types.SimpleNamespace(PolicyServer=_Server),
        "transformers": types.SimpleNamespace(AutoProcessor=_Processor),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "argv", ["rldx-server", "--model-path", "checkpoint"])

    main()

    assert calls[0] == (
        "policy",
        {
            "embodiment_tag": "tag",
            "model_path": "checkpoint",
            "device": "cuda",
            "strict": True,
        },
    )
    assert calls[1] == ("checkpoint", {"image_max_area": 65536})
    assert calls[-1] == ("run", True)


def test_rldx_executor_cancel_and_validation_failures() -> None:
    client = _Client()
    executor = _loaded_executor(client)
    invalid_skill = _payload()
    invalid_skill["skill_name"] = "navigate_to"
    assert executor.execute(invalid_skill)["failure_mode"] == "invalid_task"

    executor.cancel()
    cancelled = executor.execute(_payload())
    assert cancelled["status"] == "cancelled"
    assert executor.health()["metrics"]["active_session"] == "episode-1"


def test_rldx_executor_rejects_invalid_static_contract() -> None:
    for settings, message in (
        ({"action_dimensions": 0}, "action_dimensions"),
        ({"state_dimensions": 15}, "state_dimensions"),
        ({"execution_horizon": 17}, "execution_horizon"),
        ({"prompt_mode": "invalid"}, "prompt_mode"),
        ({"video_delta_indices": [-6, 0, -2]}, "video_delta_indices"),
    ):
        with pytest.raises(ValueError, match=message):
            RLDXPolicyExecutor("policy", _spec(**settings))


def test_rldx_action_chunk_rejects_schema_mismatch() -> None:
    valid = {
        "action.end_effector_position": np.zeros((1, 16, 3), np.float32),
        "action.end_effector_rotation": np.zeros((1, 16, 3), np.float32),
        "action.gripper_close": np.zeros((1, 16, 1), np.float32),
        "action.base_motion": np.zeros((1, 16, 4), np.float32),
        "action.control_mode": np.zeros((1, 16, 1), np.float32),
    }
    keys = (
        "end_effector_position",
        "end_effector_rotation",
        "gripper_close",
        "base_motion",
        "control_mode",
    )
    for response in (
        "invalid",
        {key: value for key, value in valid.items() if key != "action.control_mode"},
        {**valid, "action.control_mode": np.zeros((16, 1), np.float32)},
        {**valid, "action.control_mode": np.zeros((1, 7, 1), np.float32)},
    ):
        try:
            _action_chunk(
                response,
                action_keys=keys,
                dimensions=12,
                execution_horizon=8,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected action schema mismatch")


def test_rldx_wire_helpers_leave_plain_values_unchanged() -> None:
    value = {"plain": True}

    assert _encode_wire_value(value) is value
    assert _decode_wire_value(value) is value
