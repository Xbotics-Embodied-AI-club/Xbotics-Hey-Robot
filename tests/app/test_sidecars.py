from __future__ import annotations

import asyncio
import json

import pytest

from hey_robot.app.sidecars import ManagedRoboCasaBackend, managed_robocasa_backend
from hey_robot.config import DeploymentConfig


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
                        "service_python": "model-python",
                        "embodiment": "robocasa",
                        "action_space": "robocasa_12d",
                        "action_dimensions": 12,
                        "prompt_mode": "environment_root",
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
    processes = [_Process()]
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
    assert len(spawns) == 1

    await sidecar.stop()
    assert all(process.terminated for process in processes)
    assert not sidecar.credentials_path.exists()


def test_managed_backend_accepts_rldx_policy(tmp_path) -> None:
    config = _config(tmp_path)
    config.model_services["m"].settings["runtime"] = "rldx"

    sidecar = ManagedRoboCasaBackend(config, config_path="deployment.yaml")

    assert sidecar.service_id == "m"


def test_managed_backend_accepts_xiaomi_policy(tmp_path) -> None:
    config = _config(tmp_path)
    config.model_services["m"].settings["runtime"] = "xiaomi"

    sidecar = ManagedRoboCasaBackend(config, config_path="deployment.yaml")

    assert sidecar.service_id == "m"


@pytest.mark.asyncio
async def test_unexpected_backend_exit_is_propagated(tmp_path) -> None:
    sidecar = ManagedRoboCasaBackend(_config(tmp_path), config_path="deployment.yaml")
    runtime_process = _Process()
    runtime_process.returncode = 17
    sidecar.runtime_process = runtime_process
    with pytest.raises(RuntimeError, match="unexpectedly with 17"):
        await sidecar.wait()


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_backend_health_gate_checks_runtime_plane(tmp_path, monkeypatch) -> None:
    sidecar = ManagedRoboCasaBackend(_config(tmp_path), config_path="deployment.yaml")
    runtime_process = _Process()
    sidecar.runtime_process = runtime_process

    class RuntimeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def health(self):
            return {"online": True, "loaded": True, "error": None}

        async def close(self):
            pass

    monkeypatch.setattr(
        "hey_robot.app.sidecars.GrpcRoboCasaRuntimeClient", RuntimeClient
    )
    await sidecar._wait_ready("grpc://127.0.0.1:9092")
    assert runtime_process.returncode is None


def test_managed_backend_factory_has_one_deployment_entry(tmp_path) -> None:
    config = _config(tmp_path)
    assert managed_robocasa_backend(config, config_path="deployment.yaml") is not None
    assert managed_robocasa_backend(DeploymentConfig(), config_path=None) is None
    with pytest.raises(ValueError, match="requires its config path"):
        managed_robocasa_backend(config, config_path=None)
