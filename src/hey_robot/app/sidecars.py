from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
from pathlib import Path
from urllib.parse import urlparse

from hey_robot.config import DeploymentConfig
from hey_robot.foundation.transport.grpc.client import GrpcModelServiceClient
from hey_robot.robot_backends.robocasa_remote.client import (
    GrpcRoboCasaRuntimeClient,
)

_SOURCE_ROOT = Path(__file__).resolve().parents[2]


class ManagedRoboCasaBackend:
    """Own one configured RoboCasa sidecar for a DeploymentRunner."""

    def __init__(self, config: DeploymentConfig, *, config_path: str | Path) -> None:
        self.config = config
        self.config_path = str(config_path)
        managed = [
            (robot_id, spec)
            for robot_id, spec in config.robots.items()
            if spec.type == "robocasa"
            and bool(spec.settings.get("managed_backend", False))
        ]
        if len(managed) != 1:
            raise ValueError("exactly one managed RoboCasa backend is required")
        self.robot_id, self.robot_spec = managed[0]
        models = [
            (service_id, spec)
            for service_id, spec in config.model_services.items()
            if spec.enabled
            and spec.robot_id == self.robot_id
            and spec.type == "robot_policy"
            and str(spec.settings.get("runtime") or "") == "lerobot"
            and str(spec.settings.get("embodiment") or "") == "robocasa"
            and tuple(spec.provides) == ("manipulate",)
        ]
        if len(models) != 1:
            raise ValueError(
                "managed RoboCasa requires exactly one manipulate policy provider"
            )
        self.service_id, self.model_spec = models[0]
        self.runtime_process: asyncio.subprocess.Process | None = None
        self.model_process: asyncio.subprocess.Process | None = None
        self._stopping = False
        self.evaluator_token = os.environ.setdefault(
            "ROBOCASA_EVALUATOR_TOKEN", secrets.token_hex(32)
        )
        self.data_token = os.environ.setdefault(
            "ROBOCASA_DATA_TOKEN", secrets.token_hex(32)
        )
        self.credentials_path = (
            Path(config.resources.runtime_dir) / "robocasa.credentials.json"
        )

    async def start(self) -> None:
        runtime_target = str(
            self.robot_spec.settings.get("target") or "grpc://127.0.0.1:9092"
        )
        model_target = str(self.model_spec.target or "grpc://127.0.0.1:9091")
        runtime_host, runtime_port = _loopback_endpoint(runtime_target, 9092)
        model_host, model_port = _loopback_endpoint(model_target, 9091)
        if (runtime_host, runtime_port) == (model_host, model_port):
            raise RuntimeError(
                "managed RoboCasa runtime and model service must use distinct targets"
            )
        self.credentials_path.parent.mkdir(parents=True, exist_ok=True)
        self.credentials_path.write_text(
            json.dumps(
                {
                    "evaluator_token": self.evaluator_token,
                    "data_token": self.data_token,
                }
            ),
            encoding="utf-8",
        )
        self.credentials_path.chmod(0o600)
        python = str(
            os.environ.get("HEY_ROBOT_ROBOCASA_BACKEND_PYTHON")
            or self.robot_spec.settings.get("backend_python")
            or sys.executable
        )
        runtime_environment = _service_environment()
        model_environment = _service_environment()
        model_environment["ROBOCASA_DATA_TOKEN"] = self.data_token
        if bool(self.model_spec.settings.get("offline", False)):
            model_environment.update(
                {
                    "ROBOCASA_OFFLINE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
            )
        self.runtime_process = await asyncio.create_subprocess_exec(
            python,
            "-m",
            "hey_robot.app.robocasa_backend",
            "--host",
            runtime_host,
            "--port",
            str(runtime_port),
            "--config",
            self.config_path,
            "--evaluator-token",
            self.evaluator_token,
            "--data-token",
            self.data_token,
            env=runtime_environment,
        )
        try:
            self.model_process = await asyncio.create_subprocess_exec(
                python,
                "-m",
                "hey_robot.cli.main",
                "model-service",
                "--config",
                self.config_path,
                "--service-id",
                self.service_id,
                "--host",
                model_host,
                "--port",
                str(model_port),
                env=model_environment,
            )
            await self._wait_ready(runtime_target)
        except BaseException:
            await self.stop()
            raise

    async def wait(self) -> None:
        processes = [self.runtime_process, self.model_process]
        if any(process is None for process in processes):
            raise RuntimeError("RoboCasa backend was not started")
        tasks = [
            asyncio.create_task(process.wait())
            for process in processes
            if process is not None
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        returncode = next(iter(done)).result()
        if not self._stopping:
            raise RuntimeError(
                f"managed RoboCasa service exited unexpectedly with {returncode}"
            )

    async def stop(self) -> None:
        self._stopping = True
        processes = [self.model_process, self.runtime_process]
        for process in processes:
            if process is not None and process.returncode is None:
                process.terminate()
        for process in processes:
            if process is None or process.returncode is not None:
                continue
            try:
                await asyncio.wait_for(process.wait(), timeout=10.0)
            except TimeoutError:
                process.kill()
                await process.wait()
        self.credentials_path.unlink(missing_ok=True)

    async def _wait_ready(self, runtime_target: str) -> None:
        runtime = GrpcRoboCasaRuntimeClient(
            runtime_target, timeout_sec=5.0, role="data"
        )
        model = GrpcModelServiceClient(self.service_id, self.model_spec)
        timeout = float(
            self.robot_spec.settings.get("backend_startup_timeout_sec", 60.0)
        )
        last_error = "backend did not answer"
        try:
            async with asyncio.timeout(timeout):
                while True:
                    for name, process in (
                        ("runtime", self.runtime_process),
                        ("model", self.model_process),
                    ):
                        if process is not None and process.returncode is not None:
                            raise RuntimeError(
                                f"managed RoboCasa {name} service exited during "
                                f"health gate with {process.returncode}"
                            )
                    try:
                        runtime_health, model_health = await asyncio.gather(
                            runtime.health(), model.health()
                        )
                        if (
                            runtime_health.get("online")
                            and runtime_health.get("loaded")
                            and model_health.online
                            and model_health.loaded
                        ):
                            return
                        last_error = str(
                            runtime_health.get("error")
                            or model_health.error
                            or "backend dependencies are not loaded"
                        )
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                    await asyncio.sleep(0.2)
        except TimeoutError as exc:
            raise RuntimeError(
                f"managed RoboCasa backend health gate timed out: {last_error}"
            ) from exc
        finally:
            await runtime.close()
            await model.close()


def _loopback_endpoint(target: str, default_port: int) -> tuple[str, int]:
    parsed = urlparse(target)
    host = parsed.hostname or "127.0.0.1"
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("a managed service must use a loopback target")
    return host, parsed.port or default_port


def _service_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(_SOURCE_ROOT), environment.get("PYTHONPATH", "")))
    )
    return environment


def managed_robocasa_backend(
    config: DeploymentConfig, *, config_path: str | Path | None
) -> ManagedRoboCasaBackend | None:
    enabled = any(
        spec.type == "robocasa" and bool(spec.settings.get("managed_backend", False))
        for spec in config.robots.values()
    )
    if not enabled:
        return None
    if config_path is None:
        raise ValueError("managed RoboCasa deployment requires its config path")
    return ManagedRoboCasaBackend(config, config_path=config_path)
