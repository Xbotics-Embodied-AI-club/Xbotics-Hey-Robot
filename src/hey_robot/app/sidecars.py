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
from hey_robot.robot_runtime.robocasa_remote.client import (
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
            and spec.type == "robocasa_lerobot_policy"
            and tuple(spec.provides) == ("manipulate",)
        ]
        if len(models) != 1:
            raise ValueError(
                "managed RoboCasa requires exactly one manipulate policy provider"
            )
        self.service_id, self.model_spec = models[0]
        self.process: asyncio.subprocess.Process | None = None
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
        target = str(self.robot_spec.settings.get("target") or "grpc://127.0.0.1:9092")
        parsed = urlparse(target)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 9092
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError("a managed RoboCasa backend must use a loopback target")
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
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(_SOURCE_ROOT), environment.get("PYTHONPATH", "")))
        )
        self.process = await asyncio.create_subprocess_exec(
            python,
            "-m",
            "hey_robot.app.robocasa_backend",
            "--host",
            host,
            "--port",
            str(port),
            "--config",
            self.config_path,
            "--evaluator-token",
            self.evaluator_token,
            "--data-token",
            self.data_token,
            env=environment,
        )
        try:
            await self._wait_ready(target)
        except BaseException:
            await self.stop()
            raise

    async def wait(self) -> None:
        if self.process is None:
            raise RuntimeError("RoboCasa backend was not started")
        returncode = await self.process.wait()
        if not self._stopping:
            raise RuntimeError(
                f"managed RoboCasa backend exited unexpectedly with {returncode}"
            )

    async def stop(self) -> None:
        self._stopping = True
        process = self.process
        if process is None or process.returncode is not None:
            self.credentials_path.unlink(missing_ok=True)
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10.0)
        except TimeoutError:
            process.kill()
            await process.wait()
        self.credentials_path.unlink(missing_ok=True)

    async def _wait_ready(self, target: str) -> None:
        runtime = GrpcRoboCasaRuntimeClient(target, timeout_sec=5.0, role="data")
        model = GrpcModelServiceClient(self.service_id, self.model_spec)
        timeout = float(
            self.robot_spec.settings.get("backend_startup_timeout_sec", 60.0)
        )
        last_error = "backend did not answer"
        try:
            async with asyncio.timeout(timeout):
                while True:
                    if self.process is not None and self.process.returncode is not None:
                        raise RuntimeError(
                            "managed RoboCasa backend exited during health gate "
                            f"with {self.process.returncode}"
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
