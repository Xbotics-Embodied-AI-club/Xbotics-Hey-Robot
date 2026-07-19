from __future__ import annotations

import argparse
import asyncio
import json

import grpc
import pytest

from deploy.robocasa365.benchmark import _command as benchmark_command
from deploy.robocasa365.rollout import (
    RoboCasaRolloutRunner,
    RolloutError,
    RolloutRequest,
    RolloutResult,
    parse_eval_info,
    request_from_payload,
)
from deploy.robocasa365.runtime_server import RoboCasaRuntimeService
from deploy.robocasa365.server import RoboCasaModelService
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


def test_administrative_benchmark_uses_target_training_schema(tmp_path) -> None:
    command = benchmark_command(
        argparse.Namespace(
            task="CloseFridge",
            episodes=20,
            seed=1000,
            policy_path="lerobot/smolvla_robocasa",
            output_dir=tmp_path / "benchmark",
        )
    )

    assert "--env.split=target" in command
    rename = next(item for item in command if item.startswith("--rename_map="))
    assert (
        '"observation.images.robot0_agentview_right":"observation.images.camera2"'
        in rename
    )
    assert (
        '"observation.images.robot0_eye_in_hand":"observation.images.camera3"' in rename
    )


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
