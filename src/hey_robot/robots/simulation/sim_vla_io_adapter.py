from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hey_robot.capability.vla.io_adapter import VLAIOAdapter

if TYPE_CHECKING:
    from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver


class SimVLAIOAdapter(VLAIOAdapter):
    """Wrap *XLeRobotSimDriver* as a VLA I/O adapter for the generic control loop."""

    def __init__(self, sim: XLeRobotSimDriver, *, arm: str = "right") -> None:
        self._sim = sim
        self._arm = arm

    # ------------------------------------------------------------------
    # VLAIOAdapter interface
    # ------------------------------------------------------------------

    def capture_frames(self, camera_names: list[str]) -> dict[str, np.ndarray]:
        return self._sim.render_camera_frames(camera_names)

    def read_joint_state(self, arm: str) -> np.ndarray:
        return self._sim.read_arm_state_vector(arm)

    def apply_action(self, arm: str, targets_rad: np.ndarray) -> None:
        self._sim.write_arm_targets(arm, targets_rad)

    def advance(self, dt: float) -> None:
        self._sim.step_control(dt)

    def reset(self) -> None:
        pass  # simulation state is reset upstream by the skill layer

    def ready(self) -> bool:
        readiness = self._sim.vla_readiness()
        return bool(readiness.get("vla", {}).get("ok", False))
