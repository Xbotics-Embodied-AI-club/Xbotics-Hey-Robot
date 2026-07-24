from __future__ import annotations

import asyncio
import json

import pytest

from hey_robot.app.sidecars import ManagedRoboCasaBackend, managed_robocasa_backend
from hey_robot.config import DeploymentConfig
from hey_robot.foundation.clients.models import ServiceHealth


def _config(tmp_path) -> DeploymentConfig:
    return DeploymentConfig.from_dict(
        {
            "resources": {"runtime_dir": str(tmp_path)},
            "robots": {
                "r": {
                    "type": "robocasa",
                    "settings": {
                        "managed_backend": True,
                        "target": "grpc://127.0.0.1:9092",
                        "backend_python": "backend-python",
                    },
                }
            },
            "model_services": {
                "m": {
                    "type": "robot_policy",
                    "robot_id": "r",
                    "target": "grpc://127.0.0.1:9091",
                    "provides": ["manipulate"],
                    "settings": {
                        "policy_path": "p",
                        "policy_device": "cpu",
                        "runtime": "lerobot",
                        "embodiment": "robocasa",
                        "action_space": "robocasa_12d",
                        "action_dimensions": 12,
                        "prompt_mode": "environment_root",
                        "option_horizon": 50,
                    },
                }
            },
        }
    )


class _Process:
    def __init__(self) -> None:
        self.returncode = None
        self.terminated = False

    async def wait(self):
        if self.terminated:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True


@pytest.mark.asyncio
async def test_managed_backend_owns_credentials_process_and_cleanup(
    tmp_path, monkeypatch
) -> None:
    sidecar = ManagedRoboCasaBackend(_config(tmp_path), config_path="deployment.yaml")
    processes = [_Process(), _Process()]
    spawns: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def create(*args: object, **kwargs: object):
        spawns.append((args, kwargs))
        return processes[len(spawns) - 1]

    async def ready(target):
        assert target == "grpc://127.0.0.1:9092"

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(sidecar, "_wait_ready", ready)

    await sidecar.start()
    credentials = json.loads(sidecar.credentials_path.read_text())
    assert credentials["evaluator_token"] != credentials["data_token"]
    assert sidecar.credentials_path.stat().st_mode & 0o777 == 0o600
    assert spawns[0][0][:3] == (
        "backend-python",
        "-m",
        "hey_robot.app.robocasa_backend",
    )
    assert spawns[1][0][:4] == (
        "backend-python",
        "-m",
        "hey_robot.cli.main",
        "model-service",
    )

    await sidecar.stop()
    assert all(process.terminated for process in processes)
    assert not sidecar.credentials_path.exists()


@pytest.mark.asyncio
async def test_unexpected_backend_exit_is_propagated(tmp_path) -> None:
    sidecar = ManagedRoboCasaBackend(_config(tmp_path), config_path="deployment.yaml")
    runtime_process = _Process()
    model_process = _Process()
    runtime_process.returncode = 17
    model_process.returncode = 17
    sidecar.runtime_process = runtime_process
    sidecar.model_process = model_process
    with pytest.raises(RuntimeError, match="unexpectedly with 17"):
        await sidecar.wait()


@pytest.mark.asyncio
async def test_model_spawn_failure_cleans_up_runtime_process(
    tmp_path, monkeypatch
) -> None:
    sidecar = ManagedRoboCasaBackend(_config(tmp_path), config_path="deployment.yaml")
    runtime_process = _Process()
    calls = 0

    async def create(*_args: object, **_kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 1:
            return runtime_process
        raise OSError("model spawn failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    with pytest.raises(OSError, match="model spawn failed"):
        await sidecar.start()

    assert runtime_process.terminated is True
    assert not sidecar.credentials_path.exists()


@pytest.mark.asyncio
async def test_backend_health_gate_checks_both_standard_planes(
    tmp_path, monkeypatch
) -> None:
    sidecar = ManagedRoboCasaBackend(_config(tmp_path), config_path="deployment.yaml")
    runtime_process = _Process()
    model_process = _Process()
    sidecar.runtime_process = runtime_process
    sidecar.model_process = model_process

    class RuntimeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def health(self):
            return {"online": True, "loaded": True, "error": None}

        async def close(self):
            pass

    class ModelClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def health(self):
            return ServiceHealth(name="m", online=True, loaded=True)

        async def close(self):
            pass

    monkeypatch.setattr(
        "hey_robot.app.sidecars.GrpcRoboCasaRuntimeClient", RuntimeClient
    )
    monkeypatch.setattr("hey_robot.app.sidecars.GrpcModelServiceClient", ModelClient)

    await sidecar._wait_ready("grpc://127.0.0.1:9092")
    assert runtime_process.returncode is None
    assert model_process.returncode is None


def test_managed_backend_factory_has_one_deployment_entry(tmp_path) -> None:
    config = _config(tmp_path)
    assert managed_robocasa_backend(config, config_path="deployment.yaml") is not None
    assert managed_robocasa_backend(DeploymentConfig(), config_path=None) is None
    with pytest.raises(ValueError, match="requires its config path"):
        managed_robocasa_backend(config, config_path=None)
