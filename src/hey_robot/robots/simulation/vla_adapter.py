from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hey_robot.capability.vla.executor import VLAExecutor
    from hey_robot.capability.vla.policy_client import VLAPolicyClient
    from hey_robot.capability.vla.schemas import VLAConfig
    from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

from hey_robot.logging import HeyRobotLogger

logger = HeyRobotLogger(name="simulation.vla_adapter")


def build_vla_config(
    settings: dict[str, Any],
    *,
    task_prompt: str | None = None,
) -> VLAConfig:
    """Build a VLAConfig from a settings dict (e.g. from deployment config)."""
    from hey_robot.capability.vla.schemas import VLAConfig

    return VLAConfig(
        policy_runtime=str(settings.get("policy_runtime", "fake")),
        policy_endpoint=str(
            settings.get("policy_endpoint")
            or f"{settings.get('policy_host', '127.0.0.1')}:{settings.get('policy_port', 5555)}"
        ),
        policy_type=str(settings.get("policy_type", "act")),
        policy_model_path=str(settings.get("policy_model_path", "")),
        task_prompt=str(
            task_prompt
            or settings.get("task_prompt")
            or settings.get("task")
            or "Pick up the object."
        ),
        camera_names=tuple(
            str(c) for c in settings.get("cameras", ("front", "right_wrist"))
        ),
        arm=str(settings.get("arm", "right")),
        fps=int(settings.get("fps", 25)),
        action_horizon=int(settings.get("action_horizon", 16)),
        execution_time_sec=float(settings.get("execution_time", 10.0)),
        action_mode=str(settings.get("action_mode", "absolute_joint_position_rad")),
        camera_key_map=dict(settings.get("camera_key_map", {}) or {}),
        state_key_map=dict(settings.get("state_key_map", {}) or {}),
        language_key=str(settings.get("language_key", "language")),
        device=str(settings.get("device", "cpu")),
    )


def build_policy_client(config: VLAConfig) -> VLAPolicyClient:
    """Build the appropriate policy client for a given config."""
    from hey_robot.capability.vla.policy_client import (
        FakePolicyClient,
        GrootZmqPolicyClient,
        LerobotPolicyClient,
    )

    runtime = config.policy_runtime.lower()
    if runtime == "fake":
        return FakePolicyClient(action_horizon=config.action_horizon)
    if runtime == "groot_zmq":
        host, _, port_str = config.policy_endpoint.partition(":")
        port = int(port_str) if port_str else 5555
        return GrootZmqPolicyClient(host=host or "127.0.0.1", port=port)
    if runtime == "lerobot":
        if not config.policy_model_path:
            raise ValueError(
                "policy_model_path is required for lerobot runtime. "
                "Set it to a HuggingFace repo id or local path."
            )
        return LerobotPolicyClient(
            policy_type=config.policy_type,
            model_path=config.policy_model_path,
            device=config.device,
            action_horizon=config.action_horizon,
            camera_key_map=dict(config.camera_key_map),
        )
    raise ValueError(f"unsupported policy_runtime: {config.policy_runtime!r}")


def create_executor(
    sim: XLeRobotSimDriver,
    *,
    settings: dict[str, Any],
    task_prompt: str | None = None,
) -> VLAExecutor:
    """Convenience factory: build a fully wired VLAExecutor from settings."""
    from hey_robot.capability.vla.executor import VLAExecutor
    from hey_robot.robots.simulation.sim_vla_io_adapter import SimVLAIOAdapter

    config = build_vla_config(settings, task_prompt=task_prompt)
    policy = build_policy_client(config)
    io = SimVLAIOAdapter(sim, arm=config.arm)
    logger.info(
        f"Created VLAExecutor arm={config.arm} runtime={config.policy_runtime} "
        f"cameras={list(config.camera_names)}"
    )
    return VLAExecutor(io, policy)
