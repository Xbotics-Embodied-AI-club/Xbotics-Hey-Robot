from __future__ import annotations

import base64
import io
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from hey_robot.foundation.backends.lerobot import executor


def _encoded_image() -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_policy_observation_uses_configured_feature_mapping(monkeypatch) -> None:
    envs = ModuleType("lerobot.envs")

    def preprocess(value):
        assert set(value["pixels"]) == {"front"}
        return {
            "observation.images.front": "image-tensor",
            "observation.state": "state-tensor",
        }

    envs.preprocess_observation = preprocess  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot.envs", envs)
    sample = executor._policy_observation(
        {
            "proprioception": [0.0] * 6,
            "images": [
                {"camera": "ignored", "data": _encoded_image()},
                {"camera": "front", "data": _encoded_image()},
            ],
        },
        "pick up the cup",
        settings={
            "state_dimensions": 6,
            "camera_names": ["front"],
            "robot_type": "so101",
            "observation_features": {
                "observation.images.training_front": "observation.images.front"
            },
        },
        input_features={
            "observation.images.training_front": (3, 4, 4),
            "observation.state": (6,),
        },
    )

    assert sample["observation.images.training_front"] == "image-tensor"
    assert sample["task"] == ["pick up the cup"]
    assert sample["robot_type"] == "so101"


@pytest.mark.parametrize(
    ("payload", "settings", "message"),
    [
        (
            {"proprioception": [0.0, 1.0], "images": []},
            {"state_dimensions": 3},
            "policy state must have shape",
        ),
        (
            {"proprioception": [0.0], "images": []},
            {"state_dimensions": 1, "camera_names": ["front"]},
            "policy requires cameras",
        ),
        (
            {
                "proprioception": [0.0],
                "images": [{"camera": "front", "data": "not-base64"}],
            },
            {"state_dimensions": 1, "camera_names": ["front"]},
            "invalid front image",
        ),
    ],
)
def test_policy_observation_rejects_invalid_runtime_input(
    payload, settings, message
) -> None:
    with pytest.raises(executor.PolicyExecutionError, match=message):
        executor._policy_observation(
            payload,
            "task",
            settings=settings,
            input_features={},
        )


def test_policy_observation_rejects_invalid_feature_contract(monkeypatch) -> None:
    envs = ModuleType("lerobot.envs")
    envs.preprocess_observation = lambda _value: {  # type: ignore[attr-defined]
        "observation.state": "state"
    }
    monkeypatch.setitem(sys.modules, "lerobot.envs", envs)

    with pytest.raises(executor.PolicyExecutionError, match=r"source .* unavailable"):
        executor._policy_observation(
            {"proprioception": [0.0], "images": []},
            "task",
            settings={
                "observation_features": {
                    "observation.images.policy": "observation.images.runtime"
                }
            },
            input_features={},
        )

    with pytest.raises(executor.PolicyExecutionError, match="features are missing"):
        executor._policy_observation(
            {"proprioception": [0.0], "images": []},
            "task",
            settings={},
            input_features={"observation.images.required": (3, 4, 4)},
        )


def test_fallback_observation_and_action_helpers() -> None:
    pytest.importorskip("torch", reason="LeRobot policy runtime is optional")
    sample = executor._fallback_preprocess_observation(
        {
            "agent_pos": np.zeros(6, dtype=np.float32),
            "pixels": {"front": np.zeros((4, 4, 3), dtype=np.uint8)},
        }
    )
    assert tuple(sample["observation.images.front"].shape) == (1, 3, 4, 4)
    assert tuple(sample["observation.state"].shape) == (1, 6)
    assert executor._action_to_numpy({"action": [[1.0, 2.0]]}).tolist() == [
        1.0,
        2.0,
    ]
    assert executor._feature_shapes(
        {"action": {"shape": [6]}, "ignored": {"shape": "6"}}
    ) == {"action": (6,)}

    with pytest.raises(executor.PolicyExecutionError, match="rank 1"):
        executor._action_to_numpy(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(executor.PolicyExecutionError, match="non-finite"):
        executor._validate_action(np.asarray([np.nan]), 1)


def test_direct_runtime_resets_and_processes_action(monkeypatch) -> None:
    seeds: list[int | None] = []

    class Policy:
        def __init__(self) -> None:
            self.reset_count = 0

        def reset(self) -> None:
            self.reset_count += 1

        def select_action(self, sample):
            assert sample == {"processed": True}
            return {"action": [[0.25, -0.5]]}

    policy = Policy()
    runtime = executor._DirectPolicyRuntime(
        policy_path="policy",
        policy_type="fastwam",
        device="cpu",
        settings={},
        input_features={},
        policy=policy,
        preprocessor=lambda _sample: {"processed": True},
        postprocessor=lambda action: action,
    )
    monkeypatch.setattr(executor, "_seed_policy", seeds.append)
    monkeypatch.setattr(executor, "_policy_observation", lambda *_args, **_kwargs: {})

    runtime.reset(9)
    action = runtime.select_action({}, "task")

    assert seeds == [9]
    assert policy.reset_count == 1
    assert action.tolist() == [0.25, -0.5]
    assert runtime.cancel() is None
    assert runtime.close() is None


def test_policy_seed_covers_cpu_and_cuda_rngs(monkeypatch) -> None:
    seeds: list[tuple[str, int]] = []
    torch = ModuleType("torch")
    torch.manual_seed = lambda seed: seeds.append(("cpu", seed))  # type: ignore[attr-defined]
    torch.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        manual_seed_all=lambda seed: seeds.append(("cuda", seed)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    executor._seed_policy(None)
    executor._seed_policy(11)

    assert seeds == [("cpu", 11), ("cuda", 11)]


class _Connection:
    def __init__(self, commands):
        self.commands = iter(commands)
        self.sent = []
        self.closed = False

    def recv(self):
        return next(self.commands)

    def send(self, value):
        self.sent.append(value)

    def close(self):
        self.closed = True


class _Runtime:
    def __init__(self) -> None:
        self.resets = []

    def reset(self, seed=None):
        self.resets.append(seed)

    def select_action(self, _observation, _task):
        return np.asarray([0.5], dtype=np.float32)


def test_isolated_policy_worker_handles_standard_commands(monkeypatch) -> None:
    runtime = _Runtime()
    connection = _Connection([("reset", 7), ("select_action", {}, "task"), ("close",)])
    monkeypatch.setattr(executor, "_load_direct_policy_runtime", lambda *_: runtime)

    executor._policy_process_main(connection, "policy", "cpu", {})

    assert runtime.resets == [7]
    assert connection.sent[0] == {"ok": True}
    assert connection.sent[2]["result"].tolist() == [0.5]
    assert connection.closed is True


def test_isolated_policy_worker_reports_invalid_command(monkeypatch) -> None:
    connection = _Connection([("unknown",)])
    monkeypatch.setattr(executor, "_load_direct_policy_runtime", lambda *_: _Runtime())

    executor._policy_process_main(connection, "policy", "cpu", {})

    assert connection.sent[-1]["ok"] is False
    assert "unknown policy process command" in connection.sent[-1]["error"]
