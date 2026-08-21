from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import pytest

from hey_robot.app.robocasa_backend import (
    _build_policy_backend,
    _chunk_policy,
    _load_backend_specs,
    _runtime_environment,
    _separate_egl_device,
    serve,
)
from hey_robot.config import RobotSpec


def test_canonical_config_builds_runtime_environment(monkeypatch) -> None:
    robot, _, _ = _load_backend_specs("configs/evaluation/robocasa365.rldx.yaml")
    monkeypatch.setattr(
        "hey_robot.app.robocasa_backend._separate_egl_device", lambda: "7"
    )

    environment = _runtime_environment(robot)

    assert environment["MUJOCO_GL"] == "egl"
    assert environment["MUJOCO_EGL_DEVICE_ID"] == "0"


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


def test_backend_spec_requires_one_managed_robocasa_robot(monkeypatch) -> None:
    from hey_robot.config import DeploymentConfig

    invalid = DeploymentConfig.from_dict(
        {
            "robots": {"r": {"type": "robocasa"}},
        }
    )
    monkeypatch.setattr(DeploymentConfig, "from_yaml", lambda _: invalid)

    with pytest.raises(ValueError, match="exactly one managed RoboCasa robot"):
        _load_backend_specs("unused.yaml")


def test_backend_rejects_unknown_foundation_runtime() -> None:
    from hey_robot.config import ModelServiceSpec

    with pytest.raises(ValueError, match="unsupported RoboCasa foundation runtime"):
        _build_policy_backend("policy", ModelServiceSpec("robot_policy", "robot"))


def test_chunk_policy_uses_rldx_native_chunk_adapter() -> None:
    from hey_robot.config import ModelServiceSpec

    class Backend:
        def create_chunk_policy(self, encoder):
            return encoder

    rldx = ModelServiceSpec("robot_policy", "robot", settings={"runtime": "rldx"})
    generic = ModelServiceSpec("robot_policy", "robot", settings={"runtime": "lerobot"})

    assert _chunk_policy(Backend(), rldx).__name__ == "encode_rldx_observation"
    assert _chunk_policy(object(), generic).__class__.__name__ == "ExecutorChunkPolicy"


def test_main_enforces_token_pairing_and_runs_the_backend(monkeypatch) -> None:
    from hey_robot.app import robocasa_backend

    monkeypatch.setattr(sys, "argv", ["robocasa-backend", "--config", "config.yaml"])
    with pytest.raises(SystemExit, match="2"):
        robocasa_backend.main()

    called: dict[str, object] = {}

    async def fake_serve(*args: Any, **kwargs: Any) -> None:
        called["args"] = args
        called["kwargs"] = kwargs

    monkeypatch.setattr(robocasa_backend, "serve", fake_serve)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robocasa-backend",
            "--config",
            "config.yaml",
            "--insecure-local",
            "--port",
            "9000",
        ],
    )
    robocasa_backend.main()

    assert called["args"] == ("0.0.0.0", 9000)  # noqa: S104
    assert called["kwargs"] == {
        "evaluator_token": None,
        "data_token": None,
        "config_path": "config.yaml",
    }


@pytest.mark.asyncio
async def test_serve_composes_only_runtime_service(monkeypatch) -> None:
    # serve() runs in a dedicated production process. Isolate its process-level
    # CUDA/EGL environment when invoking it inside the shared pytest process.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    backend_specs = _load_backend_specs("configs/evaluation/robocasa365.rldx.yaml")
    calls: list[object] = []

    class Server:
        def __init__(self, **_kwargs) -> None:
            pass

        def add_insecure_port(self, address: str) -> None:
            calls.append(("port", address))

        async def start(self) -> None:
            calls.append("start")

        async def wait_for_termination(self) -> None:
            calls.append("wait")

        async def stop(self, grace: float) -> None:
            calls.append(("stop", grace))

    monkeypatch.setattr(
        "hey_robot.app.robocasa_backend._load_backend_specs",
        lambda _path: backend_specs,
    )
    monkeypatch.setattr("hey_robot.app.robocasa_backend.grpc.aio.server", Server)
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
    assert sum(isinstance(call, tuple) and call[0] == "runtime" for call in calls) == 1
    assert ("stop", 1.0) in calls
