from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np

from hey_robot.capability.vla.io_adapter import VLAIOAdapter
from hey_robot.logging import HeyRobotLogger

if TYPE_CHECKING:
    from hey_robot.robots.xlerobot.hardware.native import NativeXLeRobotClient

logger = HeyRobotLogger(name="xlerobot.vla_adapter")

# SO101 joint names in VLA canonical order:
#   shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
_SO101_JOINT_ORDER = (
    "base",
    "shoulder",
    "elbow",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class RealRobotVLAIOAdapter(VLAIOAdapter):
    """Wrap *NativeXLeRobotClient* as a VLA I/O adapter for the real XLeRobot.

    Handles SO101 joint-name ↔ canonical-order mapping and degree ↔ radian
    conversion so the generic :class:`VLAExecutor` can drive real hardware.
    """

    def __init__(
        self,
        client: NativeXLeRobotClient,
        *,
        arm_name: str = "arm",
        camera_devices: dict[str, int] | None = None,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
    ) -> None:
        self._client = client
        self._arm_name = arm_name
        self._arm = client.arms[arm_name]
        self._camera_devices = dict(camera_devices or {})
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._camera_fps = camera_fps
        self._captures: dict[str, object] = {}  # {name: cv2.VideoCapture}

    # ------------------------------------------------------------------
    # VLAIOAdapter interface
    # ------------------------------------------------------------------

    def capture_frames(self, camera_names: list[str]) -> dict[str, np.ndarray]:
        import cv2

        frames: dict[str, np.ndarray] = {}
        for name in camera_names:
            cap = self._captures.get(name)
            if cap is None:
                device_id = self._camera_devices.get(name)
                if device_id is None:
                    logger.warning(
                        f"no camera device configured for {name!r}, skipping"
                    )
                    continue
                cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_height)
                cap.set(cv2.CAP_PROP_FPS, self._camera_fps)
                if not cap.isOpened():
                    logger.warning(f"failed to open camera {name} (device {device_id})")
                    cap.release()
                    continue
                self._captures[name] = cap

            ok, frame_bgr = cap.read()  # type: ignore[attr-defined]
            if ok and frame_bgr is not None:
                frames[name] = frame_bgr[:, :, :3][:, :, ::-1].copy()  # BGR → RGB
            else:
                logger.warning(f"camera {name!r} returned no frame")

        return frames

    def read_joint_state(self, arm: str) -> np.ndarray:
        _ = arm
        self._arm._read_positions()
        angles_deg = self._arm._current_angles
        vec = np.zeros(6, dtype=np.float64)
        for i, joint_name in enumerate(_SO101_JOINT_ORDER):
            deg = angles_deg.get(joint_name, 0.0)
            vec[i] = math.radians(float(deg))
        return vec

    def apply_action(self, arm: str, targets_rad: np.ndarray) -> None:
        _ = arm
        joints: dict[str, float] = {}
        for i, joint_name in enumerate(_SO101_JOINT_ORDER):
            if i < len(targets_rad):
                joints[joint_name] = math.degrees(float(targets_rad[i]))
        self._arm.set_joints(joints)

    def advance(self, dt: float) -> None:
        time.sleep(dt)

    def reset(self) -> None:
        pass

    def ready(self) -> bool:
        return self._arm.initialized

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        for name, cap in list(self._captures.items()):
            cap.release()  # type: ignore[attr-defined]
            del self._captures[name]
