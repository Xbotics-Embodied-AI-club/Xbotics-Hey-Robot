from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from hey_robot.config import DeploymentConfig, ModelServiceSpec
from hey_robot.foundation.backends.lerobot import LeRobotPolicyExecutor
from hey_robot.foundation.backends.lerobot.executor import _image_bytes
from hey_robot.foundation.transport.grpc.server import (
    RobotPolicyService,
    VLNPlannerService,
    build_model_service,
)


def _spec(**settings: Any) -> ModelServiceSpec:
    return ModelServiceSpec(
        type="robot_policy",
        robot_id="robot",
        provides=("manipulate",),
        settings={
            "runtime": "lerobot",
            "policy_path": "fake-policy",
            "policy_device": "cpu",
            "embodiment": "robocasa",
            "action_space": "robocasa_12d",
            "action_dimensions": 12,
            "prompt_mode": "agent_subgoal",
            **settings,
        },
    )


def _payload(*, session_id: str = "episode-1", task: str = "close fridge"):
    return {
        "skill_name": "manipulate",
        "episode_id": session_id,
        "arguments": {
            "task_prompt": task,
            "seed": 7,
            "observation": {
                "frame_id": 3,
                "proprioception": [0.0] * 16,
                "images": [],
            },
        },
    }


class _Runtime:
    policy_type = "fake"

    def __init__(self, action: np.ndarray | None = None) -> None:
        self.action = action if action is not None else np.zeros(12, dtype=np.float32)
        self.resets: list[int | None] = []
        self.tasks: list[str] = []
        self.cancelled = False
        self.closed = False

    def reset(self, seed: int | None = None) -> None:
        self.resets.append(seed)

    def select_action(self, _observation, task: str) -> np.ndarray:
        self.tasks.append(task)
        return self.action

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        self.closed = True


def test_policy_executor_returns_standard_native_action_chunk() -> None:
    runtime = _Runtime(np.array([1.5, *([0.0] * 11)], dtype=np.float32))
    executor = LeRobotPolicyExecutor(
        "policy",
        _spec(),
        runtime_loader=lambda *_args: runtime,
    )

    result = executor.execute(_payload())

    assert result["success"] is True
    policy_result = result["metrics"]["policy_result"]
    assert policy_result["kind"] == "action_chunk"
    assert policy_result["action_space"] == "robocasa_12d"
    assert policy_result["embodiment"] == "robocasa"
    assert policy_result["horizon"] == 1
    action = policy_result["actions"][0]
    assert action["name"] == "embodiment_native_action"
    assert action["arguments"]["values"][0] == 1.0
    assert action["arguments"]["raw_values"][0] == 1.5
    assert result["metrics"]["action_clipped"] is True
    assert runtime.resets == [7]


def test_policy_executor_health_is_ready_only_after_policy_load() -> None:
    runtime = _Runtime()
    executor = LeRobotPolicyExecutor(
        "policy", _spec(), runtime_loader=lambda *_args: runtime
    )

    assert executor.health()["loaded"] is False
    executor.load()
    assert executor.health()["loaded"] is True


def test_policy_executor_resolves_configured_local_media_uri(tmp_path) -> None:
    image = tmp_path / "images" / "camera1.png"
    image.parent.mkdir()
    image.write_bytes(b"image-bytes")

    assert (
        _image_bytes(
            {"uri": "media://local/images/camera1.png"},
            {"media_root": str(tmp_path)},
        )
        == b"image-bytes"
    )

    with pytest.raises(ValueError, match="unsafe local media URI"):
        _image_bytes(
            {"uri": "media://local/../secret"},
            {"media_root": str(tmp_path)},
        )


def test_policy_executor_resets_only_when_session_or_task_changes() -> None:
    runtime = _Runtime()
    executor = LeRobotPolicyExecutor(
        "policy", _spec(), runtime_loader=lambda *_args: runtime
    )

    assert executor.execute(_payload())["success"] is True
    assert executor.execute(_payload())["success"] is True
    assert executor.execute(_payload(task="open drawer"))["success"] is True
    assert executor.execute(_payload(session_id="episode-2"))["success"] is True

    assert runtime.resets == [7, 7, 7]


def test_policy_executor_rejects_action_dimension_mismatch() -> None:
    runtime = _Runtime(np.zeros(6, dtype=np.float32))
    executor = LeRobotPolicyExecutor(
        "policy", _spec(), runtime_loader=lambda *_args: runtime
    )

    result = executor.execute(_payload())

    assert result["success"] is False
    assert result["failure_mode"] == "action_schema_mismatch"


def test_policy_executor_cancel_and_close_reach_runtime() -> None:
    first_runtime = _Runtime()
    second_runtime = _Runtime()
    runtimes = iter((first_runtime, second_runtime))
    executor = LeRobotPolicyExecutor(
        "policy", _spec(), runtime_loader=lambda *_args: next(runtimes)
    )
    assert executor.execute(_payload())["success"] is True

    executor.cancel()
    cancelled = executor.execute(_payload())
    resumed = executor.execute(_payload())
    executor.close()

    assert cancelled["status"] == "cancelled"
    assert resumed["success"] is True
    assert first_runtime.cancelled is True
    assert first_runtime.closed is True
    assert second_runtime.closed is True


def test_policy_executor_rejects_unprovided_skill() -> None:
    executor = LeRobotPolicyExecutor(
        "policy", _spec(), runtime_loader=lambda *_args: _Runtime()
    )
    payload = _payload()
    payload["skill_name"] = "navigate_to"

    result = executor.execute(payload)

    assert result["success"] is False
    assert result["failure_mode"] == "invalid_task"


def test_robot_policy_service_uses_the_configured_lerobot_runtime() -> None:
    config = DeploymentConfig.from_dict(
        {
            "model_services": {
                "policy": {
                    "type": "robot_policy",
                    "robot_id": "robot",
                    "provides": ["manipulate"],
                    "settings": dict(_spec().settings),
                }
            }
        }
    )

    service = RobotPolicyService(config, service_id="policy")

    assert isinstance(service.executor, LeRobotPolicyExecutor)


def test_build_model_service_supports_robot_policy_and_vln() -> None:
    config = DeploymentConfig.from_dict(
        {
            "model_services": {
                "policy": {
                    "type": "robot_policy",
                    "robot_id": "robot",
                    "provides": ["manipulate"],
                    "settings": dict(_spec().settings),
                },
                "planner": {
                    "type": "vln_planner",
                    "robot_id": "robot",
                    "provides": ["navigate_to"],
                    "settings": {"mock_mode": True},
                },
            }
        }
    )

    assert isinstance(
        build_model_service(config, service_id="policy"), RobotPolicyService
    )
    assert isinstance(
        build_model_service(config, service_id="planner"), VLNPlannerService
    )


def test_robot_policy_service_rejects_unknown_runtime() -> None:
    config = DeploymentConfig.from_dict(
        {
            "model_services": {
                "policy": {
                    "type": "robot_policy",
                    "robot_id": "robot",
                    "provides": ["manipulate"],
                    "settings": {**dict(_spec().settings), "runtime": "other"},
                }
            }
        }
    )

    with pytest.raises(ValueError, match="unsupported runtime 'other'"):
        RobotPolicyService(config, service_id="policy")


def test_build_model_service_rejects_unknown_type() -> None:
    config = DeploymentConfig.from_dict(
        {"model_services": {"unknown": {"type": "other", "robot_id": "robot"}}}
    )

    with pytest.raises(ValueError, match="unsupported model service type"):
        build_model_service(config, service_id="unknown")
