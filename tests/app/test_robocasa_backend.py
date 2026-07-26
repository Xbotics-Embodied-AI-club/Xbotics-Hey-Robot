from __future__ import annotations

import os
import subprocess

import pytest

from hey_robot.app.robocasa_backend import (
    _load_backend_spec,
    _runtime_environment,
    _separate_egl_device,
    serve,
)
from hey_robot.config import RobotSpec


def test_canonical_config_builds_runtime_environment(monkeypatch) -> None:
    robot = _load_backend_spec("configs/evaluation/robocasa365.yaml")
    monkeypatch.setattr(
        "hey_robot.app.robocasa_backend._separate_egl_device", lambda: "7"
    )

    environment = _runtime_environment(robot)

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


def test_backend_spec_requires_one_managed_robocasa_robot(monkeypatch) -> None:
    from hey_robot.config import DeploymentConfig

    invalid = DeploymentConfig.from_dict(
        {
            "robots": {"r": {"type": "robocasa"}},
        }
    )
    monkeypatch.setattr(DeploymentConfig, "from_yaml", lambda _: invalid)

    with pytest.raises(ValueError, match="exactly one managed RoboCasa robot"):
        _load_backend_spec("unused.yaml")


@pytest.mark.asyncio
async def test_serve_composes_only_runtime_service(monkeypatch) -> None:
    # serve() runs in a dedicated production process. Isolate its process-level
    # CUDA/EGL environment when invoking it inside the shared pytest process.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    robot = _load_backend_spec("configs/evaluation/robocasa365.yaml")
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

    monkeypatch.setattr(
        "hey_robot.app.robocasa_backend._load_backend_spec",
        lambda _path: robot,
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
