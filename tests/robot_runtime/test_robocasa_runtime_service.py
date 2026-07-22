from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from hey_robot.robocasa_runtime.v1 import robocasa_runtime_pb2 as pb
from hey_robot.robot_runtime.robocasa_remote.episode_manager import EpisodeManager
from hey_robot.robot_runtime.robocasa_remote.runtime_server import (
    RoboCasaRuntimeService,
    _assets_available,
    _json_safe,
    _validate_observation,
)


def _observation():
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    return {
        "agent_pos": np.zeros(16, dtype=np.float32),
        "pixels": {
            "robot0_agentview_left": frame,
            "robot0_agentview_right": frame,
            "robot0_eye_in_hand": frame,
        },
    }


class _Space:
    def contains(self, action) -> bool:
        return np.asarray(action).shape == (12,)


class _Env:
    action_space = _Space()
    task_description = "Close the refrigerator door."

    def __init__(self) -> None:
        self.closed = False

    def step(self, _action):
        return _observation(), 1.0, True, False, {"is_success": True}

    def close(self) -> None:
        self.closed = True


class _Context:
    def __init__(self, token: str) -> None:
        self.token = token

    def invocation_metadata(self):
        return (("authorization", f"Bearer {self.token}"),)

    async def abort(self, code, detail):
        raise RuntimeError(f"{code.name}: {detail}")


@pytest.mark.asyncio
async def test_runtime_service_covers_authenticated_trial_lifecycle() -> None:
    env = _Env()
    manager = EpisodeManager(
        allowed_tasks=frozenset({"CloseFridge"}),
        env_factory=lambda _spec: (env, _observation()),
    )
    prepared = []
    service = RoboCasaRuntimeService(
        manager=manager,
        evaluator_token="eval",  # noqa: S106
        data_token="data",  # noqa: S106
        prepare_trial=lambda: prepared.append(True),
    )
    evaluator = _Context("eval")
    data = _Context("data")

    initial = await service.BeginTrial(
        pb.BeginTrialRequest(
            trial_id="trial-1",
            task="CloseFridge",
            seed=1000,
            split="target",
            registries=["lightwheel"],
        ),
        evaluator,
    )
    assert initial.frame_id == 0
    assert [image.camera for image in initial.images] == [
        "camera1",
        "camera2",
        "camera3",
    ]
    assert prepared == [True]
    assert service.busy is True

    observed = await service.Observe(pb.EmptyRequest(), data)
    assert observed.task == "CloseFridge"
    step = await service.Step(
        pb.StepRequest(
            action=[0.0] * 12,
            raw_action=[0.0] * 12,
            expected_frame_id=0,
        ),
        data,
    )
    assert step.done is True
    assert step.observation.frame_id == 1

    truth = await service.ReadTruth(pb.EmptyRequest(), evaluator)
    assert truth.official_success is True
    assert truth.frame_id == 1
    ended = await service.EndTrial(pb.EndTrialRequest(reason="done"), evaluator)
    assert ended == pb.EndTrialResponse(ended=True)
    assert env.closed is True
    assert service.busy is False
    second_end = await service.EndTrial(pb.EndTrialRequest(), evaluator)
    assert second_end.ended is False


@pytest.mark.asyncio
async def test_runtime_service_rejects_wrong_role_and_action_schema() -> None:
    service = RoboCasaRuntimeService(
        evaluator_token="eval",  # noqa: S106
        data_token="data",  # noqa: S106
    )
    with pytest.raises(RuntimeError, match="credential is required"):
        await service._authorize(_Context("data"), role="evaluator")
    with pytest.raises(ValueError, match="exactly 12 finite"):
        await service.Step(
            pb.StepRequest(action=[0.0], raw_action=[0.0]), _Context("data")
        )


def test_runtime_helpers_validate_assets_observations_and_numpy(
    tmp_path, monkeypatch
) -> None:
    for relative in (
        "textures",
        "generative_textures",
        "fixtures",
        "objects/lightwheel",
    ):
        (tmp_path / relative).mkdir(parents=True)
    marker = tmp_path / ".robocasa-assets-ready"
    marker.touch()
    monkeypatch.setenv("ROBOCASA_MODEL_ASSET_ROOT", str(tmp_path))
    monkeypatch.setenv("ROBOCASA_ASSET_READY_FILE", str(marker))
    assert _assets_available() is True
    _validate_observation(_observation())
    with pytest.raises(RuntimeError, match="16 finite"):
        _validate_observation({"agent_pos": [0.0], "pixels": {}})
    assert _json_safe({"x": np.asarray([1]), "y": (np.float32(2),)}) == {
        "x": [1],
        "y": [2.0],
    }

    fake = types.ModuleType("lerobot.envs.robocasa")
    fake.ACTION_DIM = 12
    fake.OBS_STATE_DIM = 16
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.envs", types.ModuleType("lerobot.envs"))
    monkeypatch.setitem(sys.modules, "lerobot.envs.robocasa", fake)
