from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace

import grpc
import numpy as np
import pytest

from evaluation.robocasa365.worker.benchmark import _command as benchmark_command
from evaluation.robocasa365.worker.lerobot_eval_wrapper import (
    _checkpoint_rename_map,
    _has_rename_map,
    _policy_path,
)
from evaluation.robocasa365.worker.option_runner import (
    OptionRequest,
    RoboCasaOptionRunner,
    _PolicyBundle,
)
from evaluation.robocasa365.worker.policy_probe import validate_feature_contract
from evaluation.robocasa365.worker.rollout import (
    RoboCasaRolloutRunner,
    RolloutError,
    RolloutRequest,
    RolloutResult,
    parse_eval_info,
    request_from_payload,
)
from evaluation.robocasa365.worker.runtime_server import RoboCasaRuntimeService
from evaluation.robocasa365.worker.server import RoboCasaModelService
from hey_robot.config import DeploymentConfig
from hey_robot.foundation.clients import ServiceInvocationRequest
from hey_robot.foundation.clients.models import ServiceInvocationResult
from hey_robot.foundation.contract.v1 import model_service_pb2, model_service_pb2_grpc
from hey_robot.foundation.transport.grpc.client import GrpcModelServiceClient
from hey_robot.protocol import Envelope, SkillIntent
from hey_robot.robocasa_runtime.v1 import (
    robocasa_runtime_pb2,
    robocasa_runtime_pb2_grpc,
)
from hey_robot.skill_os import SkillRuntime, load_skill_registry
from hey_robot.skill_os.context import SkillContext


def _fake_robocasa_observation() -> dict[str, object]:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    return {
        "agent_pos": np.zeros((16,), dtype=np.float32),
        "pixels": {
            "robot0_agentview_left": frame,
            "robot0_agentview_right": frame,
            "robot0_eye_in_hand": frame,
        },
    }


def test_robocasa_rollout_skill_forwards_task_level_arguments_only() -> None:
    class FakeServices:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        async def call(
            self, name: str, arguments: dict[str, object]
        ) -> ServiceInvocationResult:
            self.calls.append((name, arguments))
            return ServiceInvocationResult(
                success=True,
                status="completed",
                summary="RoboCasa CloseFridge: 1/1 successful",
                metrics={"success_count": 1, "output_dir": "skill-1"},
            )

    async def run_once() -> None:
        services = FakeServices()
        runtime = SkillRuntime(load_skill_registry(enabled=("robocasa_rollout",)))
        result = await runtime.execute(
            "robocasa_rollout",
            {"task": "CloseFridge", "seed": 1000},
            context_factory=lambda invoke: SkillContext(
                model_services=services, invoke=invoke
            ),
        )

        assert result.success is True
        assert result.data["metrics"]["success_count"] == 1
        assert services.calls == [
            (
                "robocasa_rollout",
                {
                    "task": "CloseFridge",
                    "seed": 1000,
                    "n_episodes": 1,
                    "obj_registries": ["lightwheel"],
                    "record_video": True,
                },
            )
        ]

    asyncio.run(run_once())


def test_robocasa_option_skill_forwards_agent_option_arguments() -> None:
    class FakeServices:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        async def call(
            self, name: str, arguments: dict[str, object]
        ) -> ServiceInvocationResult:
            self.calls.append((name, arguments))
            return ServiceInvocationResult(
                success=True,
                status="completed",
                summary="RoboCasa option CloseFridge: success after 2 steps",
                metrics={
                    "session_id": "session-1",
                    "steps": 2,
                    "episode_success": True,
                },
            )

    async def run_once() -> None:
        services = FakeServices()
        runtime = SkillRuntime(load_skill_registry(enabled=("robocasa_option",)))
        result = await runtime.execute(
            "robocasa_option",
            {
                "task": "CloseFridge",
                "option_command": "Close the fridge door.",
                "session_id": "session-1",
                "seed": 1000,
                "max_steps": 4,
            },
            context_factory=lambda invoke: SkillContext(
                model_services=services, invoke=invoke
            ),
        )

        assert result.success is True
        assert result.data["metrics"]["steps"] == 2
        assert services.calls == [
            (
                "robocasa_option",
                {
                    "task": "CloseFridge",
                    "option_command": "Close the fridge door.",
                    "session_id": "session-1",
                    "seed": 1000,
                    "max_steps": 4,
                    "reset_episode": False,
                    "close_episode": False,
                    "device": "cuda",
                    "objective": "Close the fridge door.",
                },
            )
        ]

    asyncio.run(run_once())


def test_rollout_request_and_command_are_allowlisted(tmp_path) -> None:
    runner = RoboCasaRolloutRunner(output_root=tmp_path)
    request = request_from_payload(
        {
            "skill_id": "skill:close-fridge",
            "objective": "close the fridge",
            "timeout_sec": 120.0,
            "arguments": {
                "task": "CloseFridge",
                "seed": 1001,
                "n_episodes": 1,
                "obj_registries": ["lightwheel"],
            },
        }
    )

    assert request.skill_id == "skill:close-fridge"
    command = runner._command(request, tmp_path / "skill_close-fridge")
    assert command[0] == "lerobot-eval"
    assert "--env.task=CloseFridge" in command
    assert "--env.split=target" in command
    assert "--eval.n_episodes=1" in command
    assert all("close the fridge" not in item for item in command)

    invalid = RolloutRequest(skill_id="bad", objective="", task="DeleteKitchen")
    with pytest.raises(RolloutError, match="not allowed") as exc_info:
        runner._validate_request(invalid)
    assert exc_info.value.failure_mode == "invalid_task"

    seed_zero = request_from_payload(
        {
            "skill_id": "seed-zero",
            "arguments": {"task": "CloseFridge", "seed": 0},
        }
    )
    assert seed_zero.seed == 0


def test_pi052_policy_contract_requires_three_cameras_state_and_action() -> None:
    @dataclass
    class Feature:
        shape: tuple[int, ...]

    config = SimpleNamespace(
        type="pi05",
        input_features={
            "observation.images.robot0_agentview_left": Feature((3, 256, 256)),
            "observation.images.robot0_agentview_right": Feature((3, 256, 256)),
            "observation.images.robot0_eye_in_hand": Feature((3, 256, 256)),
            "observation.state": Feature((16,)),
        },
        output_features={"action": Feature((12,))},
    )

    result = validate_feature_contract(config)

    assert result["valid"] is True
    config.output_features = {"action": Feature((7,))}
    result = validate_feature_contract(config)
    assert result["valid"] is False
    assert "expected (12,)" in result["errors"][0]


def test_policy_contract_accepts_raw_checkpoint_config() -> None:
    result = validate_feature_contract(
        {
            "type": "pi052",
            "input_features": {
                "observation.images.robot0_agentview_left": {"shape": [3, 256, 256]},
                "observation.images.robot0_agentview_right": {"shape": [3, 256, 256]},
                "observation.images.robot0_eye_in_hand": {"shape": [3, 256, 256]},
                "observation.state": {"shape": [16]},
            },
            "output_features": {"action": {"shape": [12]}},
        }
    )

    assert result["valid"] is True
    assert result["policy_type"] == "pi052"


def test_smolvla_policy_contract_accepts_camera_aliases_and_6d_state() -> None:
    result = validate_feature_contract(
        {
            "type": "smolvla",
            "input_features": {
                "observation.images.camera1": {"shape": [3, 256, 256]},
                "observation.images.camera2": {"shape": [3, 256, 256]},
                "observation.images.camera3": {"shape": [3, 256, 256]},
                "observation.state": {"shape": [6]},
            },
            "output_features": {"action": {"shape": [12]}},
        }
    )

    assert result["valid"] is True
    assert result["policy_type"] == "smolvla"


def test_robocasa_option_runner_keeps_session_across_bounded_options() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.closed = False
            self.steps = 0

        def step(self, action):
            assert len(action) == 12
            self.steps += 1
            success = self.steps >= 2
            return (
                _fake_robocasa_observation(),
                1.0,
                success,
                False,
                {"is_success": success},
            )

        def close(self) -> None:
            self.closed = True

    class FakePreprocessor:
        def __init__(self) -> None:
            self.samples: list[dict[str, object]] = []

        def __call__(self, sample):
            self.samples.append(sample)
            return sample

    class FakePolicy:
        def select_action(self, processed):
            assert processed["task"] == ["Close the fridge door."]
            return np.zeros((1, 12), dtype=np.float32)

    class FakePostprocessor:
        def __call__(self, action):
            return action

    fake_env = FakeEnv()
    fake_preprocessor = FakePreprocessor()

    def env_factory(task: str, seed: int):
        assert task == "CloseFridge"
        assert seed == 1000
        return fake_env, _fake_robocasa_observation()

    def policy_loader(policy_path: str, device: str):
        assert policy_path == "fake-policy"
        assert device == "cpu"
        return _PolicyBundle(
            policy_path=policy_path,
            policy_type="fake",
            device=device,
            input_features={
                "observation.images.camera1": (3, 256, 256),
                "observation.images.camera2": (3, 256, 256),
                "observation.images.camera3": (3, 256, 256),
                "observation.state": (6,),
            },
            policy=FakePolicy(),
            preprocessor=fake_preprocessor,
            postprocessor=FakePostprocessor(),
        )

    runner = RoboCasaOptionRunner(env_factory=env_factory, policy_loader=policy_loader)
    first = runner.run(
        OptionRequest(
            skill_id="skill-1",
            task="CloseFridge",
            objective="Close the fridge door.",
            option_command="Close the fridge door.",
            policy_path="fake-policy",
            device="cpu",
            max_steps=1,
        )
    )

    assert first.success is False
    assert first.failure_mode == "option_timeout"
    assert runner.active_sessions == 1
    assert fake_env.closed is False

    second = runner.run(
        OptionRequest(
            skill_id="skill-1",
            task="CloseFridge",
            objective="Close the fridge door.",
            option_command="Close the fridge door.",
            policy_path="fake-policy",
            device="cpu",
            max_steps=2,
            close_episode=True,
        )
    )

    assert second.success is True
    assert second.metrics["frame_id"] == 2
    assert runner.active_sessions == 0
    assert fake_env.closed is True
    assert tuple(fake_preprocessor.samples[0]["observation.state"].shape) == (1, 16)
    assert "observation.images.robot0_agentview_left" in fake_preprocessor.samples[0]


def test_rollout_eval_binary_can_come_from_environment(tmp_path) -> None:
    runner = RoboCasaRolloutRunner(
        output_root=tmp_path,
        environ={"ROBOCASA_EVAL_BINARY": "/venv/bin/lerobot-eval"},
    )

    assert runner.eval_binary == "/venv/bin/lerobot-eval"


def test_policy_cache_requires_weights_not_only_config(tmp_path) -> None:
    snapshot = (
        tmp_path / "hub" / "models--lerobot--pi052_robocasa" / "snapshots" / "revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    runner = RoboCasaRolloutRunner(environ={"HF_HOME": str(tmp_path)})

    assert runner._policy_cached("lerobot/pi052_robocasa") is False
    (snapshot / "model.safetensors").touch()
    assert runner._policy_cached("lerobot/pi052_robocasa") is True


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--policy.path=lerobot/pi052_robocasa"], "lerobot/pi052_robocasa"),
        (["--policy.path", "local/policy"], "local/policy"),
        (["--env.type=robocasa"], None),
    ],
)
def test_eval_wrapper_extracts_policy_path(arguments, expected) -> None:
    assert _policy_path(arguments) == expected


def test_eval_wrapper_reads_checkpoint_camera_map(tmp_path) -> None:
    checkpoint = tmp_path / "policy"
    checkpoint.mkdir()
    (checkpoint / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "rename_observations_processor",
                        "config": {
                            "rename_map": {
                                "observation.images.left": "observation.images.camera1"
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _checkpoint_rename_map(str(checkpoint)) == {
        "observation.images.left": "observation.images.camera1"
    }
    assert _has_rename_map(["--rename_map={}"]) is True
    assert _has_rename_map(["--policy.path=policy"]) is False


def test_administrative_benchmark_uses_raw_robocasa_camera_names(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("ROBOCASA_RENAME_MAP", raising=False)
    monkeypatch.setenv("ROBOCASA_EVAL_BINARY", "/venv/bin/robocasa-eval")
    command = benchmark_command(
        argparse.Namespace(
            task="CloseFridge",
            episodes=20,
            seed=1000,
            policy_path="lerobot/smolvla_robocasa",
            output_dir=tmp_path / "benchmark",
        )
    )

    assert command[0] == "/venv/bin/robocasa-eval"
    assert "--env.split=target" in command
    assert not any(item.startswith("--rename_map=") for item in command)


def test_rollout_accepts_explicit_camera_rename_override(tmp_path) -> None:
    runner = RoboCasaRolloutRunner(
        output_root=tmp_path,
        environ={
            "ROBOCASA_RENAME_MAP": '{"observation.images.old":"observation.images.new"}'
        },
    )
    request = RolloutRequest(skill_id="rename", objective="", task="CloseFridge")

    command = runner._command(request, tmp_path / "rename")

    assert '--rename_map={"observation.images.old":"observation.images.new"}' in command


def test_eval_info_parsing_distinguishes_process_completion_from_task_success(
    tmp_path,
) -> None:
    path = tmp_path / "eval_info.json"
    path.write_text(
        json.dumps({"per_episode": [{"success": False}], "pc_success": 0.0}),
        encoding="utf-8",
    )

    metrics = parse_eval_info(path, n_episodes=1)

    assert metrics == {
        "success_count": 0,
        "success_rate": 0.0,
        "episode_successes": [0],
    }


def test_eval_info_parsing_accepts_percent_success_fallback(tmp_path) -> None:
    path = tmp_path / "eval_info.json"
    path.write_text(json.dumps({"aggregated": {"pc_success": 100.0}}), encoding="utf-8")

    metrics = parse_eval_info(path, n_episodes=1)

    assert metrics["success_count"] == 1
    assert metrics["success_rate"] == 1.0


def test_eval_info_parsing_accepts_current_lerobot_per_task_metrics(tmp_path) -> None:
    path = tmp_path / "eval_info.json"
    path.write_text(
        json.dumps(
            {
                "per_task": [
                    {
                        "task_group": "CloseFridge",
                        "task_id": 0,
                        "metrics": {"successes": [True]},
                    }
                ],
                "overall": {"pc_success": 100.0},
            }
        ),
        encoding="utf-8",
    )

    metrics = parse_eval_info(path, n_episodes=1)

    assert metrics == {
        "success_count": 1,
        "success_rate": 1.0,
        "episode_successes": [1],
    }


def test_completed_evaluator_with_failed_episode_returns_task_unsuccessful(
    tmp_path,
) -> None:
    class CompletedProcess:
        def poll(self) -> int:
            return 0

    def fake_popen(command, **kwargs):
        del kwargs
        output_dir = next(
            item.removeprefix("--output_dir=")
            for item in command
            if item.startswith("--output_dir=")
        )
        (tmp_path / "skill-failure" / "eval_info.json").write_text(
            json.dumps({"per_episode": [{"success": False}]}), encoding="utf-8"
        )
        assert output_dir == str(tmp_path / "skill-failure")
        return CompletedProcess()

    runner = RoboCasaRolloutRunner(output_root=tmp_path, popen_factory=fake_popen)
    result = runner.run(
        RolloutRequest(skill_id="skill-failure", objective="", task="CloseFridge")
    )

    assert result.success is False
    assert result.failure_mode == "task_unsuccessful"
    assert result.summary == "RoboCasa CloseFridge: 0/1 successful"


def test_cancelled_evaluator_returns_cancelled_status(tmp_path) -> None:
    class RunningProcess:
        def poll(self):
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout=None):
            del timeout
            return 0

    runner = RoboCasaRolloutRunner(output_root=tmp_path)

    def fake_popen(command, **kwargs):
        del command, kwargs
        runner._cancelled.set()
        return RunningProcess()

    runner._popen_factory = fake_popen
    result = runner.run(
        RolloutRequest(skill_id="skill-cancelled", objective="", task="CloseFridge")
    )

    assert result.success is False
    assert result.status == "cancelled"
    assert result.failure_mode == "rollout_cancelled"


def test_worker_rejects_unknown_skill_without_running_a_rollout() -> None:
    async def run_once() -> None:
        service = RoboCasaModelService(RoboCasaRolloutRunner())
        response = await service.ExecuteSkill(
            model_service_pb2.ExecuteSkillRequest(
                skill_id="skill-1", skill_name="manipulate"
            ),
            None,
        )

        assert response.success is False
        assert response.failure_mode == "invalid_task"
        assert response.error_code == "UNSUPPORTED_SKILL"

    asyncio.run(run_once())


def test_robocasa_worker_round_trips_through_model_service_v1() -> None:
    class FakeRunner:
        busy = False
        current_skill_id = None

        def __init__(self) -> None:
            self.requests: list[RolloutRequest] = []

        def health(self) -> dict[str, object]:
            return {
                "online": True,
                "loaded": True,
                "metrics": {"asset_profile": "lightwheel"},
            }

        def run(self, request: RolloutRequest) -> RolloutResult:
            self.requests.append(request)
            return RolloutResult(
                success=True,
                status="completed",
                summary="RoboCasa CloseFridge: 1/1 successful",
                metrics={"success_count": 1, "success_rate": 1.0},
            )

        def cancel(self, skill_id: str | None = None) -> bool:
            del skill_id
            return False

    async def run_once() -> None:
        config = DeploymentConfig.from_dict(
            {
                "model_services": {
                    "robocasa365": {
                        "type": "vla_policy",
                        "robot_id": "",
                        "target": "127.0.0.1:0",
                        "provides": ["robocasa_rollout"],
                        "timeout_sec": 20,
                    }
                }
            }
        )
        spec = config.model_services["robocasa365"]
        runner = FakeRunner()
        server = grpc.aio.server()
        model_service_pb2_grpc.add_ModelServiceServicer_to_server(
            RoboCasaModelService(runner),
            server,  # type: ignore[arg-type]
        )
        port = server.add_insecure_port("127.0.0.1:0")
        object.__setattr__(spec, "target", f"127.0.0.1:{port}")
        await server.start()
        try:
            intent = SkillIntent(
                envelope=Envelope(trace_id="tr-robocasa", episode_id="ep-robocasa"),
                skill_id="skill-robocasa",
                task_id="task-robocasa",
                intent_kind="skill",
                name="robocasa_rollout",
                objective="close the refrigerator",
                arguments={"task": "CloseFridge", "seed": 1000},
            )
            contract = (
                load_skill_registry().robot_skill_catalog().get("robocasa_rollout")
            )
            result = await GrpcModelServiceClient("robocasa365", spec).execute(
                ServiceInvocationRequest(
                    service_id="robocasa365",
                    intent=intent,
                    contract=contract,
                    timeout_sec=20.0,
                )
            )
        finally:
            await server.stop(grace=0.1)

        assert result.success is True
        assert result.metrics["success_count"] == 1.0
        assert runner.requests[0].task == "CloseFridge"
        assert runner.requests[0].seed == 1000

    asyncio.run(run_once())


def test_remote_runtime_health_round_trips_without_creating_a_gpu_episode() -> None:
    async def run_once() -> None:
        server = grpc.aio.server()
        robocasa_runtime_pb2_grpc.add_RoboCasaRuntimeServicer_to_server(
            RoboCasaRuntimeService(), server
        )
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        try:
            response = await robocasa_runtime_pb2_grpc.RoboCasaRuntimeStub(
                channel
            ).GetHealth(robocasa_runtime_pb2.HealthRequest(), timeout=5)
            assert response.online is True
            # The host test environment intentionally lacks the container's
            # LeRobot/RoboCasa packages; health remains inspectable either way.
            assert isinstance(response.loaded, bool)
        finally:
            await channel.close()
            await server.stop(0)

    asyncio.run(run_once())


def test_task_and_frame_services_share_an_exclusive_resource_lock() -> None:
    async def run_once() -> None:
        resource_lock = asyncio.Lock()
        runtime_service = RoboCasaRuntimeService(resource_lock=resource_lock)
        model_service = RoboCasaModelService(resource_lock=resource_lock)

        await resource_lock.acquire()
        try:
            runtime_health = await runtime_service.GetHealth(
                robocasa_runtime_pb2.HealthRequest(), None
            )
            model_health = await model_service.GetHealth(
                model_service_pb2.GetHealthRequest(), None
            )
            assert runtime_health.busy is True
            assert model_health.busy is True

            with pytest.raises(RuntimeError, match="task-level rollout"):
                await runtime_service.CreateEpisode(
                    robocasa_runtime_pb2.CreateEpisodeRequest(
                        task="CloseFridge", seed=1000
                    ),
                    None,
                )
            response = await model_service.ExecuteSkill(
                model_service_pb2.ExecuteSkillRequest(
                    skill_id="blocked", skill_name="robocasa_rollout"
                ),
                None,
            )
            assert response.success is False
            assert response.failure_mode == "model_service_busy"
        finally:
            resource_lock.release()

    asyncio.run(run_once())
