from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import signal
import subprocess
from contextlib import suppress
from pathlib import Path

import grpc

from hey_robot.config import DeploymentConfig, RobotSpec
from hey_robot.robocasa_backend.contract import ALLOWED_TASKS
from hey_robot.robocasa_backend.episode_manager import EpisodeManager
from hey_robot.robocasa_backend.rpc.v1 import (
    robocasa_runtime_pb2_grpc as runtime_pb2_grpc,
)
from hey_robot.robocasa_backend.runtime_server import (
    RoboCasaRuntimeService,
)


async def serve(
    host: str,
    port: int,
    *,
    evaluator_token: str | None = None,
    data_token: str | None = None,
    config_path: str | Path,
) -> None:
    """Serve only the RoboCasa environment and evaluator control plane."""
    robot_spec = _load_backend_spec(config_path)
    os.environ.update(_runtime_environment(robot_spec))
    server = grpc.aio.server()
    manager = EpisodeManager(allowed_tasks=ALLOWED_TASKS)
    runtime_pb2_grpc.add_RoboCasaRuntimeServicer_to_server(
        RoboCasaRuntimeService(
            resource_lock=asyncio.Lock(),
            manager=manager,
            evaluator_token=evaluator_token,
            data_token=data_token,
        ),
        server,
    )
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(handled_signal, shutdown.set)
    server_task = asyncio.create_task(server.wait_for_termination())
    shutdown_task = asyncio.create_task(shutdown.wait())
    try:
        await asyncio.wait(
            {server_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        server_task.cancel()
        shutdown_task.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.shield(server.stop(grace=1.0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hey Robot RoboCasa365 environment backend"
    )
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int, default=9092)
    parser.add_argument(
        "--evaluator-token", default=os.environ.get("ROBOCASA_EVALUATOR_TOKEN")
    )
    parser.add_argument("--data-token", default=os.environ.get("ROBOCASA_DATA_TOKEN"))
    parser.add_argument("--insecure-local", action="store_true")
    parser.add_argument(
        "--config",
        required=True,
        help="Canonical Hey Robot deployment YAML used by this backend",
    )
    args = parser.parse_args()
    if bool(args.evaluator_token) != bool(args.data_token):
        parser.error("set both evaluator and data tokens")
    if not args.insecure_local and not args.evaluator_token:
        parser.error("set evaluator/data tokens or explicitly pass --insecure-local")
    with suppress(KeyboardInterrupt):
        asyncio.run(
            serve(
                args.host,
                args.port,
                evaluator_token=args.evaluator_token or None,
                data_token=args.data_token or None,
                config_path=args.config,
            )
        )


def _load_backend_spec(config_path: str | Path) -> RobotSpec:
    config = DeploymentConfig.from_yaml(config_path)
    candidates = [
        spec
        for spec in config.robots.values()
        if spec.type == "robocasa" and bool(spec.settings.get("managed_backend", False))
    ]
    if len(candidates) != 1:
        raise ValueError(
            "backend config must contain exactly one managed RoboCasa robot"
        )
    return candidates[0]


def _runtime_environment(robot_spec: RobotSpec) -> dict[str, str]:
    settings = robot_spec.settings
    mapping = {
        "mujoco_gl": "MUJOCO_GL",
        "mujoco_device": "MUJOCO_EGL_DEVICE_ID",
        "model_asset_root": "ROBOCASA_MODEL_ASSET_ROOT",
        "asset_ready_file": "ROBOCASA_ASSET_READY_FILE",
    }
    environment = {
        variable: str(settings[setting])
        for setting, variable in mapping.items()
        if setting in settings and str(settings[setting]).strip()
    }
    if environment.get("MUJOCO_EGL_DEVICE_ID") == "auto_separate":
        environment["MUJOCO_EGL_DEVICE_ID"] = _separate_egl_device()
    return environment


def _separate_egl_device() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return "0"
    try:
        completed = subprocess.run(  # noqa: S603
            [executable, "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "0"
    gpu_count = len([line for line in completed.stdout.splitlines() if line.strip()])
    return "1" if gpu_count >= 2 else "0"


if __name__ == "__main__":
    main()
