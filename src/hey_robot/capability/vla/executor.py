from __future__ import annotations

import time
from typing import Any

from hey_robot.capability.vla.action import decode_action_chunk, get_action_horizon
from hey_robot.capability.vla.io_adapter import VLAIOAdapter
from hey_robot.capability.vla.observation import build_groot_observation
from hey_robot.capability.vla.policy_client import VLAPolicyClient
from hey_robot.capability.vla.schemas import VLAConfig, VLARequest, VLAResult
from hey_robot.logging import HeyRobotLogger

logger = HeyRobotLogger(name="capability.vla.executor")


class VLAExecutor:
    """Generic VLA control loop — render → observe → infer → decode → act.

    Depends on a *VLAIOAdapter* for I/O and a *VLAPolicyClient* for inference.
    The same executor works for simulation and real-robot by swapping the adapter.
    """

    def __init__(
        self,
        io: VLAIOAdapter,
        policy: VLAPolicyClient,
    ) -> None:
        self._io = io
        self._policy = policy
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def execute(self, request: VLARequest) -> VLAResult:
        config = request.config
        self._cancelled = False
        started_at = time.monotonic()
        step_count = 0
        last_error: str | None = None

        try:
            self._policy.reset()

            if not self._policy.ping():
                return VLAResult(
                    success=False,
                    summary="VLA policy server unavailable",
                    status="failed",
                    failure_mode="policy_server_unavailable",
                    metrics={"duration_sec": round(time.monotonic() - started_at, 3)},
                )

            deadline = started_at + config.execution_time_sec

            while time.monotonic() < deadline and not self._cancelled:
                loop_start = time.monotonic()
                frames = self._io.capture_frames(list(config.camera_names))
                if not frames:
                    last_error = "camera render produced no frames"
                    break

                joint_state = self._io.read_joint_state(config.arm)

                obs = build_groot_observation(
                    frames=frames,
                    joint_state_rad=joint_state,
                    task_prompt=config.task_prompt,
                    camera_key_map=dict(config.camera_key_map),
                    state_key_map=dict(config.state_key_map),
                    language_key=config.language_key,
                )

                chunk = self._policy.get_action(obs)
                horizon = get_action_horizon(chunk) or config.action_horizon

                for t in range(horizon):
                    if self._cancelled:
                        break
                    if time.monotonic() >= deadline:
                        break
                    current = self._io.read_joint_state(config.arm)
                    targets = decode_action_chunk(
                        chunk,
                        t=t,
                        action_mode=config.action_mode,
                        current_joint_state_rad=current,
                    )
                    self._io.apply_action(config.arm, targets)
                    self._io.advance(1.0 / config.fps)
                    step_count += 1

                loop_elapsed = time.monotonic() - loop_start
                logger.debug(f"vla loop step={step_count} elapsed={loop_elapsed:.3f}s")

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.error(f"VLAExecutor failed: {last_error}")
            return VLAResult(
                success=False,
                summary=f"VLA execution failed: {last_error}",
                status="failed",
                failure_mode="execution_failed",
                error=last_error,
                metrics=_build_metrics(started_at, step_count, config),
            )

        cancelled = self._cancelled

        if cancelled:
            return VLAResult(
                success=False,
                summary="VLA execution cancelled",
                status="cancelled",
                failure_mode="cancelled",
                metrics=_build_metrics(started_at, step_count, config),
            )

        if last_error:
            return VLAResult(
                success=False,
                summary=f"VLA execution error: {last_error}",
                status="failed",
                failure_mode="camera_render_failed",
                error=last_error,
                metrics=_build_metrics(started_at, step_count, config),
            )

        return VLAResult(
            success=True,
            summary=f"VLA execution completed ({step_count} steps)",
            status="completed",
            metrics=_build_metrics(started_at, step_count, config),
        )


def _build_metrics(
    started_at: float,
    step_count: int,
    config: VLAConfig,
) -> dict[str, Any]:
    return {
        "duration_sec": round(time.monotonic() - started_at, 3),
        "steps": step_count,
        "fps": config.fps,
        "arm": config.arm,
        "action_mode": config.action_mode,
    }
