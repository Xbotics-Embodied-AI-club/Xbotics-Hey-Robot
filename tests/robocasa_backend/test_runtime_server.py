from __future__ import annotations

import asyncio
import sys
import time
import types

import numpy as np
import pytest

from hey_robot.foundation.options import OptionResult, OptionStatus
from hey_robot.robocasa_backend.episode_manager import (
    ActiveTrial,
    EpisodeManager,
    TrialSpec,
    _render_world_map,
    _task_grasp_contact,
    _validate_observation as validate_episode_observation,
)
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


def test_task_grasp_contact_matches_rpent_robocasa_predicate() -> None:
    class TaskEnv:
        def __init__(self) -> None:
            self.robots = [types.SimpleNamespace(gripper="panda-gripper")]
            self.objects = {"spatula": object(), "onion": object()}

        def _check_grasp(self, gripper, obj):
            assert gripper == "panda-gripper"
            return obj is self.objects["spatula"]

    assert _task_grasp_contact(TaskEnv()) == (True, "spatula")


def test_task_grasp_contact_tolerates_unsupported_or_malformed_tasks() -> None:
    class UnsupportedObjectTask:
        def __init__(self) -> None:
            self.robots = [types.SimpleNamespace(gripper="panda-gripper")]
            self.objects = {"unsupported": object(), "miss": object()}

        def _check_grasp(self, _gripper, obj):
            if obj is self.objects["unsupported"]:
                raise TypeError("unsupported object")
            return False

    assert _task_grasp_contact(UnsupportedObjectTask()) == (False, None)
    assert _task_grasp_contact(types.SimpleNamespace()) == (False, None)


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        ({"agent_pos": [0.0], "pixels": {}}, "state must be 16 finite values"),
        (
            {"agent_pos": np.zeros(16, dtype=np.float32), "pixels": {}},
            "observation must contain three cameras",
        ),
    ],
)
def test_episode_manager_rejects_malformed_environment_observations(
    observation, message
) -> None:
    with pytest.raises(Exception, match=message):
        validate_episode_observation(observation)


def test_episode_manager_localizes_pixels_with_simulator_metric_depth(
    monkeypatch,
) -> None:
    manager = EpisodeManager(allowed_tasks=frozenset({"CloseFridge"}))
    task_env = types.SimpleNamespace(sim=object())
    manager._active = ActiveTrial(
        spec=TrialSpec(trial_id="trial", task="CloseFridge", seed=0),
        env=types.SimpleNamespace(_env=types.SimpleNamespace(env=task_env)),
        observation={"pixels": {"camera": np.zeros((2, 3, 3), dtype=np.uint8)}},
        frame_id=4,
    )
    world = np.asarray(
        [
            [[1.11111, 2.22222, 3.33333], [np.nan, 0.0, 0.0], [7.0, 8.0, 9.0]],
            [[4.44444, 5.55555, 6.66666], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
        ]
    )
    depth = np.asarray([[0.2, 0.3, 0.0], [0.4, np.nan, 0.6]])
    monkeypatch.setattr(
        "hey_robot.robocasa_backend.episode_manager._render_world_map",
        lambda *_args, **_kwargs: (world, depth),
    )

    result = manager.localize_pixels(
        camera="camera", pixels=[[0, 0], [0, 1], [0, 3], [1]], expected_frame_id=4
    )

    assert result["method"] == "simulator_metric_depth"
    assert result["results"] == [
        {
            "pixel": [0, 0],
            "world_xyz": [1.1111, 2.2222, 3.3333],
            "depth_m": 0.2,
            "valid": True,
            "error": None,
        },
        {
            "pixel": [0, 1],
            "world_xyz": None,
            "depth_m": None,
            "valid": False,
            "error": "invalid metric depth",
        },
        {
            "pixel": [0, 3],
            "world_xyz": None,
            "valid": False,
            "error": "pixel (0,3) out of bounds (2x3)",
        },
        {
            "pixel": [1],
            "world_xyz": None,
            "valid": False,
            "error": "pixel must be [row, col]",
        },
    ]
    assert result["summary"] == {
        "valid_count": 1,
        "total_count": 4,
        "median_xyz": [1.1111, 2.2222, 3.3333],
    }


def test_episode_manager_localization_rejects_stale_missing_and_unavailable_camera() -> (
    None
):
    manager = EpisodeManager(allowed_tasks=frozenset({"CloseFridge"}))
    manager._active = ActiveTrial(
        spec=TrialSpec(trial_id="trial", task="CloseFridge", seed=0),
        env=types.SimpleNamespace(
            _env=types.SimpleNamespace(env=types.SimpleNamespace())
        ),
        observation={"pixels": {}},
        frame_id=4,
    )

    with pytest.raises(Exception, match="expected frame 3, current frame 4"):
        manager.localize_pixels(camera="camera", pixels=[], expected_frame_id=3)
    with pytest.raises(Exception, match="camera 'camera' is unavailable"):
        manager.localize_pixels(camera="camera", pixels=[], expected_frame_id=4)

    manager._active.observation["pixels"]["camera"] = np.zeros(
        (2, 3, 3), dtype=np.uint8
    )
    with pytest.raises(Exception, match="simulator camera is unavailable"):
        manager.localize_pixels(camera="camera", pixels=[], expected_frame_id=4)


def test_episode_manager_creates_the_registered_robocasa_environment(
    monkeypatch,
) -> None:
    configured = []
    monkeypatch.setattr(
        "hey_robot.robocasa_backend.egl_config.configure_headless_egl",
        lambda: configured.append(True),
    )
    instances = []

    class RoboCasaEnv:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            instances.append(self)

        def reset(self, *, seed: int):
            return _observation(), {"seed": seed}

    lerobot = types.ModuleType("lerobot")
    envs = types.ModuleType("lerobot.envs")
    robocasa_env = types.ModuleType("lerobot.envs.robocasa")
    robocasa_env.DEFAULT_CAMERAS = ("left", "right", "wrist")
    robocasa_env.RoboCasaEnv = RoboCasaEnv
    robocasa = types.ModuleType("robocasa")
    utils = types.ModuleType("robocasa.utils")
    registry = types.ModuleType("robocasa.utils.dataset_registry")
    registry.ATOMIC_TASK_DATASETS = {"CloseFridge": {"horizon": 77}}
    registry.COMPOSITE_TASK_DATASETS = {}
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.envs", envs)
    monkeypatch.setitem(sys.modules, "lerobot.envs.robocasa", robocasa_env)
    monkeypatch.setitem(sys.modules, "robocasa", robocasa)
    monkeypatch.setitem(sys.modules, "robocasa.utils", utils)
    monkeypatch.setitem(sys.modules, "robocasa.utils.dataset_registry", registry)

    spec = TrialSpec(
        trial_id="trial",
        task="CloseFridge",
        seed=9,
        split="pretrain",
        registries=("lightwheel",),
    )
    env, observation = EpisodeManager._create_environment(spec)

    assert configured == [True]
    assert env is instances[0]
    assert np.array_equal(observation["agent_pos"], np.zeros(16, dtype=np.float32))
    assert set(observation["pixels"]) == {
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    }
    assert instances[0].kwargs == {
        "task": "CloseFridge",
        "camera_name": ("left", "right", "wrist"),
        "obs_type": "pixels_agent_pos",
        "obj_registries": ("lightwheel",),
        "split": "pretrain",
        "episode_length": 77,
        "horizon": 77,
    }


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
    localization_calls = []

    def localize_pixels(**kwargs):
        localization_calls.append(kwargs)
        return {
            "frame_id": 0,
            "camera": kwargs["camera"],
            "method": "simulator_metric_depth",
            "results": [
                {
                    "pixel": [1, 2],
                    "world_xyz": [0.1, 0.2, 0.3],
                    "valid": True,
                }
            ],
            "summary": {"valid_count": 1, "total_count": 1},
        }

    manager.localize_pixels = localize_pixels  # type: ignore[method-assign]
    localized = await service.LocalizePixels(
        pb.LocalizePixelsRequest(
            camera="camera1",
            pixels=[pb.ImagePixel(row=1, col=2)],
            expected_frame_id=0,
        ),
        data,
    )
    assert localized.frame_id == 0
    assert localized.camera == "camera1"
    assert localized.localization["method"] == "simulator_metric_depth"
    assert localization_calls == [
        {
            "camera": "robot0_agentview_left",
            "pixels": [[1, 2]],
            "expected_frame_id": 0,
        }
    ]
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


def test_depth_world_map_matches_rpent_top_down_back_projection(
    monkeypatch,
) -> None:
    camera_utils = types.ModuleType("robosuite.utils.camera_utils")
    camera_utils.get_real_depth_map = lambda _sim, depth: depth
    camera_utils.get_camera_transform_matrix = lambda _sim, _camera, _height, _width: (
        np.eye(4)
    )
    utils = types.ModuleType("robosuite.utils")
    utils.camera_utils = camera_utils
    robosuite = types.ModuleType("robosuite")
    robosuite.utils = utils
    monkeypatch.setitem(sys.modules, "robosuite", robosuite)
    monkeypatch.setitem(sys.modules, "robosuite.utils", utils)
    monkeypatch.setitem(sys.modules, "robosuite.utils.camera_utils", camera_utils)

    class Sim:
        def render(self, **_kwargs):
            return np.zeros((2, 2, 3)), np.asarray([[0.1, 0.2], [0.3, 0.4]])

    world, depth = _render_world_map(
        Sim(), camera="robot0_agentview_left", height=2, width=2
    )

    assert np.allclose(depth, [[0.3, 0.4], [0.1, 0.2]])
    assert np.allclose(world[0, 1], [0.4, 0.0, 0.4])


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
