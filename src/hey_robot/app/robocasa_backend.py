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

from hey_robot.config import DeploymentConfig, ModelServiceSpec, RobotSpec
from hey_robot.foundation.backends.vla.lerobot.robocasa_executor import (
    RoboCasaLeRobotPolicyExecutor,
)
from hey_robot.foundation.contract.v1 import model_service_pb2_grpc
from hey_robot.foundation.transport.grpc.server import (
    ModelServiceServicer,
    ModelServiceState,
)
from hey_robot.robocasa_runtime.v1 import (
    robocasa_runtime_pb2_grpc as runtime_pb2_grpc,
)
from hey_robot.robot_runtime.robocasa_remote.contract import ALLOWED_TASKS
from hey_robot.robot_runtime.robocasa_remote.episode_manager import EpisodeManager
from hey_robot.robot_runtime.robocasa_remote.runtime_server import (
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
    """Serve the standard model plane and RoboCasa runtime on one endpoint."""
    model_spec, robot_spec = _load_backend_specs(config_path)
    executor_environment = _executor_environment(model_spec, robot_spec)
    # PI0.5 runs in a spawned process and therefore inherits the process
    # environment, not only the executor's configuration dictionary.
    os.environ.update(executor_environment)
    server = grpc.aio.server()
    manager = EpisodeManager(allowed_tasks=ALLOWED_TASKS)
    executor = RoboCasaLeRobotPolicyExecutor(environ=executor_environment)
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(
        ModelServiceServicer(
            ModelServiceState("robocasa365", model_spec),
            executor,
            bearer_token=data_token,
        ),
        server,
    )
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
        executor.close()
        with suppress(asyncio.CancelledError):
            await asyncio.shield(server.stop(grace=1.0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hey Robot RoboCasa365 model and simulator backend"
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


def _load_backend_specs(
    config_path: str | Path,
) -> tuple[ModelServiceSpec, RobotSpec]:
    config = DeploymentConfig.from_yaml(config_path)
    candidates = [
        spec
        for spec in config.model_services.values()
        if spec.enabled and spec.type == "robocasa_lerobot_policy"
    ]
    if len(candidates) != 1:
        raise ValueError(
            "backend config must contain exactly one robocasa_lerobot_policy"
        )
    model_spec = candidates[0]
    if tuple(model_spec.provides) != ("manipulate",):
        raise ValueError("RoboCasa policy must provide only manipulate")
    try:
        robot_spec = config.robots[model_spec.robot_id]
    except KeyError as exc:
        raise ValueError(
            f"model robot {model_spec.robot_id!r} is missing from config"
        ) from exc
    return model_spec, robot_spec


def _executor_environment(
    model_spec: ModelServiceSpec, robot_spec: RobotSpec
) -> dict[str, str]:
    environment = dict(os.environ)
    settings = model_spec.settings
    mapping = {
        "policy_path": "ROBOCASA_POLICY",
        "policy_device": "ROBOCASA_POLICY_DEVICE",
        "prompt_mode": "ROBOCASA_PROMPT_MODE",
        "option_horizon": "ROBOCASA_OPTION_HORIZON",
        "load_timeout_sec": "ROBOCASA_POLICY_LOAD_TIMEOUT",
        "request_timeout_sec": "ROBOCASA_POLICY_REQUEST_TIMEOUT",
    }
    for setting, variable in mapping.items():
        if setting in settings:
            environment[variable] = str(settings[setting])
    if bool(settings.get("offline", False)):
        environment.update(
            {
                "ROBOCASA_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    environment.update(_runtime_environment(robot_spec))
    return environment


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
