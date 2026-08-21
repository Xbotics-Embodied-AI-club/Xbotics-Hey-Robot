from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from hey_robot.robocasa_backend.contract import CAMERA_RENAME_MAP
from hey_robot.robocasa_backend.option_runtime import (
    ExecutorChunkPolicy,
    RoboCasaOptionRuntime,
    encode_rldx_observation,
)


@dataclass
class _Trial:
    frame_id: int
    observation: dict[str, Any]


@dataclass
class _Outcome:
    observation: dict[str, Any]
    done: bool


class _Manager:
    def __init__(self) -> None:
        self.trial = _Trial(7, {"pixels": {"camera1": "frame"}})
        self.calls: list[tuple[list[Any], int]] = []

    def current_trial(self) -> _Trial:
        return self.trial

    def step_chunk(
        self, actions: list[Any], *, expected_frame_id: int
    ) -> list[_Outcome]:
        self.calls.append((actions, expected_frame_id))
        return [_Outcome({"after": 1}, False), _Outcome({"after": 2}, True)]

    def get_task_progress(self) -> dict[str, Any]:
        return {"washed": True}

    def get_execution_diagnostics(self) -> dict[str, Any]:
        return {"steps": 2}


class _Executor:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.payload: dict[str, Any] | None = None

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payload = payload
        return self.result


def _observation() -> dict[str, Any]:
    pixels = {}
    for source in CAMERA_RENAME_MAP:
        pixels[source.removeprefix("observation.images.")] = np.zeros(
            (3, 4, 3), dtype=np.uint8
        )
    return {"frame_id": 11, "agent_pos": [1.0] * 16, "pixels": pixels}


def test_runtime_adapts_trial_observation_actions_and_diagnostics() -> None:
    manager = _Manager()
    runtime = RoboCasaOptionRuntime(manager)  # type: ignore[arg-type]

    assert runtime.observe() == {"pixels": {"camera1": "frame"}, "frame_id": 7}
    assert runtime.step_block([[1], [2]]) == ({"after": 2}, True)
    assert manager.calls == [([[1], [2]], 7)]
    assert runtime.progress() == {"washed": True}
    assert runtime.diagnostics() == {"steps": 2}


def test_runtime_rejects_an_empty_policy_block() -> None:
    manager = _Manager()
    manager.step_chunk = lambda *_args, **_kwargs: []  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="empty action block"):
        RoboCasaOptionRuntime(manager).step_block([])  # type: ignore[arg-type]


def test_chunk_policy_encodes_inputs_and_extracts_native_actions() -> None:
    executor = _Executor(
        {
            "success": True,
            "metrics": {
                "policy_result": {
                    "actions": [
                        {"arguments": {"values": [0.1] * 12}},
                        {"arguments": {"values": [0.2] * 12}},
                    ]
                }
            },
        }
    )
    policy = ExecutorChunkPolicy(executor)

    with pytest.raises(RuntimeError, match="not initialized"):
        policy.predict(_observation(), "put cup away")
    policy.reset("episode-8")
    assert policy.predict(_observation(), "put cup away") == [[0.1] * 12, [0.2] * 12]
    assert executor.payload is not None
    assert executor.payload["arguments"]["observation"]["raw"] == {
        "policy_task": "put cup away"
    }
    assert executor.payload["episode_id"] == "episode-8"


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ({"success": False, "error": "worker failed"}, "worker failed"),
        ({"success": True, "metrics": {"policy_result": {"actions": []}}}, "no native"),
    ],
)
def test_chunk_policy_rejects_failed_or_empty_worker_responses(
    result: dict[str, Any], message: str
) -> None:
    policy = ExecutorChunkPolicy(_Executor(result))
    policy.reset("episode")

    with pytest.raises(RuntimeError, match=message):
        policy.predict(_observation(), "do it")


def test_encoder_requires_all_cameras_and_produces_png_payloads() -> None:
    encoded = encode_rldx_observation(_observation())

    assert encoded["frame_id"] == 11
    assert encoded["proprioception"] == [1.0] * 16
    assert {image["camera"] for image in encoded["images"]} == {
        "camera1",
        "camera2",
        "camera3",
    }
    with pytest.raises(ValueError, match="missing robot0_agentview_left"):
        encode_rldx_observation({"pixels": {}})
