from __future__ import annotations

import os
import subprocess

import pytest

from hey_robot.app.robocasa_backend import (
    _executor_environment,
    _load_backend_specs,
    _runtime_environment,
    _separate_egl_device,
    serve,
)
from hey_robot.config import ModelServiceSpec, RobotSpec


def test_canonical_config_builds_complete_backend_environment(monkeypatch) -> None:
    model, robot = _load_backend_specs("configs/evaluation/robocasa365.agent.yaml")
    monkeypatch.setattr(
        "hey_robot.app.robocasa_backend._separate_egl_device", lambda: "7"
    )

    environment = _executor_environment(model, robot)

    assert environment["ROBOCASA_POLICY"] == "lerobot/pi052_robocasa"
    assert environment["ROBOCASA_POLICY_DEVICE"] == "cuda"
    assert environment["ROBOCASA_PROMPT_MODE"] == "environment_root"
    assert environment["ROBOCASA_OPTION_HORIZON"] == "50"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["MUJOCO_GL"] == "egl"
    assert environment["MUJOCO_EGL_DEVICE_ID"] == "7"


def test_runtime_environment_ignores_empty_optional_values() -> None:
    spec = RobotSpec(
        type="robocasa",
        settings={
            "mujoco_gl": "egl",
            "model_asset_root": "",
            "asset_ready_file": "/ready",
        },
    )

    assert _runtime_environment(spec) == {
        "MUJOCO_GL": "egl",
        "ROBOCASA_ASSET_READY_FILE": "/ready",
    }


def test_separate_egl_device_uses_second_gpu_when_available(monkeypatch) -> None:
    monkeypatch.setattr("hey_robot.app.robocasa_backend.shutil.which", lambda _: "gpu")

    def two_gpus(*args: object, **_kwargs: object):
        return subprocess.CompletedProcess(args[0], 0, "0\n1\n", "")

    monkeypatch.setattr("hey_robot.app.robocasa_backend.subprocess.run", two_gpus)
    assert _separate_egl_device() == "1"

    def unavailable(*_args: object, **_kwargs: object):
        raise OSError("missing")

    monkeypatch.setattr("hey_robot.app.robocasa_backend.subprocess.run", unavailable)
    assert _separate_egl_device() == "0"


def test_backend_spec_requires_one_generic_manipulate_provider(monkeypatch) -> None:
    from hey_robot.config import DeploymentConfig

    invalid = DeploymentConfig.from_dict(
        {
            "robots": {"r": {"type": "robocasa"}},
            "model_services": {
                "m": {
                    "type": "robocasa_lerobot_policy",
                    "robot_id": "r",
                    "provides": ["legacy_rollout"],
                    "settings": {
                        "policy_path": "p",
                        "policy_device": "cpu",
                        "prompt_mode": "environment_root",
                        "option_horizon": 50,
                    },
                }
            },
        }
    )
    monkeypatch.setattr(DeploymentConfig, "from_yaml", lambda _: invalid)

    with pytest.raises(ValueError, match="provide only manipulate"):
        _load_backend_specs("unused.yaml")


def test_executor_environment_has_no_implicit_policy_default(monkeypatch) -> None:
    monkeypatch.delenv("ROBOCASA_POLICY", raising=False)
    model = ModelServiceSpec(
        type="robocasa_lerobot_policy",
        robot_id="r",
        provides=("manipulate",),
        settings={},
    )
    environment = _executor_environment(model, RobotSpec(type="robocasa"))
    assert "ROBOCASA_POLICY" not in environment


@pytest.mark.asyncio
async def test_serve_composes_standard_model_and_runtime_services(monkeypatch) -> None:
    # serve() runs in a dedicated production process. Isolate its process-level
    # CUDA/EGL environment when invoking it inside the shared pytest process.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    model, robot = _load_backend_specs("configs/evaluation/robocasa365.agent.yaml")
    calls: list[object] = []

    class Server:
        def add_insecure_port(self, address: str) -> None:
            calls.append(("port", address))

        async def start(self) -> None:
            calls.append("start")

        async def wait_for_termination(self) -> None:
            calls.append("wait")

        async def stop(self, grace: float) -> None:
            calls.append(("stop", grace))

    class Executor:
        def __init__(self, *, environ) -> None:
            calls.append(("executor", environ["ROBOCASA_POLICY"]))

        def close(self) -> None:
            calls.append("executor_close")

    monkeypatch.setattr(
        "hey_robot.app.robocasa_backend._load_backend_specs",
        lambda _path: (model, robot),
    )
    monkeypatch.setattr("hey_robot.app.robocasa_backend.grpc.aio.server", Server)
    monkeypatch.setattr(
        "hey_robot.app.robocasa_backend.RoboCasaLeRobotPolicyExecutor", Executor
    )
    monkeypatch.setattr(
        "hey_robot.app.robocasa_backend.model_service_pb2_grpc.add_ModelServiceServicer_to_server",
        lambda servicer, server: calls.append(("model", servicer, server)),
    )
    monkeypatch.setattr(
        "hey_robot.app.robocasa_backend.runtime_pb2_grpc.add_RoboCasaRuntimeServicer_to_server",
        lambda servicer, server: calls.append(("runtime", servicer, server)),
    )

    await serve(
        "127.0.0.1",
        9092,
        evaluator_token="evaluator",  # noqa: S106
        data_token="data",  # noqa: S106
        config_path="deployment.yaml",
    )

    assert ("port", "127.0.0.1:9092") in calls
    assert ("executor", "lerobot/pi052_robocasa") in calls
    assert "executor_close" in calls
    assert ("stop", 1.0) in calls
