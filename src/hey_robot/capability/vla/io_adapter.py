from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VLAIOAdapter(ABC):
    """I/O contract for the VLA control loop.

    Abstracts the three operations that differ between simulation and
    real-robot: image capture, joint-state read, and action application.
    """

    @abstractmethod
    def capture_frames(self, camera_names: list[str]) -> dict[str, np.ndarray]:
        """Return {camera_name: (H, W, 3) uint8 ndarray} for each camera."""

    @abstractmethod
    def read_joint_state(self, arm: str) -> np.ndarray:
        """Return 6-D canonical joint vector in radians for *arm*."""

    @abstractmethod
    def apply_action(self, arm: str, targets_rad: np.ndarray) -> None:
        """Write 6-D canonical joint targets (radians) to the arm."""

    @abstractmethod
    def advance(self, dt: float) -> None:
        """Advance the environment by *dt* seconds."""

    @abstractmethod
    def reset(self) -> None:
        """Reset per-episode state (e.g. policy internal counters)."""

    @abstractmethod
    def ready(self) -> bool:
        """Return True when the underlying system is ready for inference."""
