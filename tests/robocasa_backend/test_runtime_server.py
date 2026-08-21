from __future__ import annotations

import asyncio
import sys
import time
import types

import numpy as np
import pytest

from hey_robot.foundation.options import OptionResult, OptionStatus
from hey_robot.robocasa_backend.episode_manager import EpisodeManager
from hey_robot.robocasa_backend.rpc.v1 import (
    robocasa_runtime_pb2 as pb,
)
from hey_robot.robocasa_backend.runtime_server import (
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


class _SlowEnv(_Env):
    def step(self, _action):
        time.sleep(0.05)
        return _observation(), 0.0, False, False, {}


class _Runner:
    def __init__(self, manager, *, sleep: float = 0.0) -> None:
        self.manager = manager
        self.sleep = sleep
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        if self.sleep:
            time.sleep(self.sleep)
        outcome = self.manager.step_chunk([[0.0] * 12], expected_frame_id=0)[-1]
        return OptionResult(
            status=OptionStatus.SUCCESS if outcome.done else OptionStatus.BUDGET,
            actions_executed=1,
            chunks_executed=1,
            environment_done=outcome.done,
            progress=self.manager.get_task_progress(),
            diagnostics=self.manager.get_execution_diagnostics(),
        )


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
    runners = []
    service = RoboCasaRuntimeService(
        manager=manager,
        evaluator_token="eval",  # noqa: S106
        data_token="data",  # noqa: S106
        create_option_runner=lambda active: (
            runners.append(_Runner(active)) or runners[-1]
        ),
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
    assert len(runners) == 1
    assert service.busy is True

    observed = await service.Observe(pb.EmptyRequest(), data)
    assert observed.task == "CloseFridge"
    step = await service.Step(
        pb.StepRequest(
            session_id="trial-1",
            instruction="Close the refrigerator door.",
            max_actions=8,
        ),
        data,
    )
    assert step.done is True
    assert step.observation.frame_id == 1
    assert step.actions_executed == 1
    assert runners[0].requests[0].instruction == "Close the refrigerator door."

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
    with pytest.raises(ValueError, match="session_id, instruction, and max_actions"):
        await service.Step(pb.StepRequest(), _Context("data"))


@pytest.mark.asyncio
async def test_runtime_service_times_out_a_stuck_environment_step() -> None:
    manager = EpisodeManager(
        allowed_tasks=frozenset({"CloseFridge"}),
        env_factory=lambda _spec: (_SlowEnv(), _observation()),
    )
    service = RoboCasaRuntimeService(
        manager=manager,
        create_option_runner=lambda active: _Runner(active, sleep=0.05),
        option_timeout_sec=0.01,
    )
    context = _Context("")
    await service.BeginTrial(
        pb.BeginTrialRequest(trial_id="slow", task="CloseFridge", seed=1000),
        context,
    )

    with pytest.raises(RuntimeError, match=r"exceeded 0\.0s"):
        await service.Step(
            pb.StepRequest(
                session_id="slow", instruction="Close the door", max_actions=8
            ),
            context,
        )
    assert "policy option exceeded" in str(service._last_error)
    await asyncio.sleep(0.06)


def test_runtime_helpers_validate_assets_observations_and_numpy(
    tmp_path, monkeypatch
) -> None:
    for relative in (
        "textures",
        "generative_textures",
        "fixtures",
        "objects/objaverse",
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


def test_episode_manager_records_execution_owned_video_and_trajectory(tmp_path) -> None:
    manager = EpisodeManager(
        allowed_tasks=frozenset({"CloseFridge"}),
        env_factory=lambda _spec: (_Env(), _observation()),
    )
    trial = manager.begin_trial(
        manager.new_spec(
            task="CloseFridge",
            seed=0,
            execution_artifact_dir=str(tmp_path),
        )
    )
    manager.step([0.0] * 12, expected_frame_id=trial.frame_id)
    truth = manager.read_truth()
    manager.end_trial()

    artifacts = truth["execution_artifacts"]
    assert artifacts["recorded_frames"] == 2
    assert (tmp_path / "video.mp4").stat().st_size > 0
    assert len((tmp_path / "trajectory.jsonl").read_text().splitlines()) == 2
