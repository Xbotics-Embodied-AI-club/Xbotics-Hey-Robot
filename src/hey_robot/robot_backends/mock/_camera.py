from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from hey_robot.robot_api import RobotDriverContext


class _CameraMixin:
    # ── attributes from _MockRobotDriverBase.__init__ ──────────────────────
    world: dict[str, Any]
    frame_id: int
    arm_joints: dict[str, float]
    gripper_opening_pct: float
    context: RobotDriverContext
    settings: dict[str, Any]
    _DEFAULT_CAMERA_NAMES: ClassVar[tuple[str, ...]]

    if TYPE_CHECKING:

        def _object_visible(self, name: str) -> bool: ...

        @staticmethod
        def _clamp(value: float, low: float, high: float) -> float: ...

        def _battery_status(self) -> dict[str, Any]: ...

    def _front_view(self) -> np.ndarray:
        image = np.full((160, 240, 3), 235, dtype=np.uint8)
        image[95:135, 25:215] = np.array([170, 190, 205], dtype=np.uint8)
        slots = {
            "front_workspace": (120, 78),
            "table": (62, 58),
            "bin": (178, 104),
            "shelf": (178, 50),
        }
        for name, obj in self.world.get("objects", {}).items():
            if not self._object_visible(str(name)):
                continue
            x, y = slots.get(str(obj.get("location") or "front_workspace"), (120, 78))
            jitter = (sum(ord(ch) for ch in str(name)) % 13) - 6
            x = int(self._clamp(x + jitter, 16, 224))
            object_color = np.array(obj.get("color") or [210, 80, 70], dtype=np.uint8)
            image[y - 10 : y + 10, x - 10 : x + 10] = object_color
            image[y - 14 : y - 10, x - 8 : x + 8] = np.array(
                [35, 35, 35], dtype=np.uint8
            )
        obstacle_x = int(118 + 30 * math.sin(self.frame_id / 3.0))
        image[86:120, obstacle_x : obstacle_x + 14] = np.array(
            [60, 65, 70], dtype=np.uint8
        )
        arm_x = int(self._clamp(120 + self.arm_joints["shoulder_pan"] / 3.0, 20, 220))
        arm_y = int(self._clamp(82 - self.arm_joints["shoulder_lift"] / 2.0, 20, 130))
        image[arm_y : arm_y + 8, 110:arm_x] = np.array([40, 150, 95], dtype=np.uint8)
        grip = int(self.gripper_opening_pct / 100.0 * 20)
        image[arm_y - 8 : arm_y + 16, arm_x : arm_x + 4] = np.array(
            [35, 95, 80], dtype=np.uint8
        )
        image[arm_y - grip // 2 : arm_y - grip // 2 + 4, arm_x - 8 : arm_x + 10] = (
            np.array([35, 95, 80], dtype=np.uint8)
        )
        image[arm_y + grip // 2 : arm_y + grip // 2 + 4, arm_x - 8 : arm_x + 10] = (
            np.array([35, 95, 80], dtype=np.uint8)
        )
        bar = self.frame_id % image.shape[1]
        image[:4, :bar] = np.array([30, 130, 220], dtype=np.uint8)
        battery = self._battery_status()["status"]
        battery_color = {
            "normal": [45, 170, 80],
            "low": [230, 165, 35],
            "critical": [220, 55, 55],
        }.get(str(battery), [120, 120, 120])
        image[8:18, 204:232] = np.array(battery_color, dtype=np.uint8)
        return image

    def _left_wrist_view(self) -> np.ndarray:
        image = np.full((160, 240, 3), 225, dtype=np.uint8)
        image[28:132, 42:198] = np.array([235, 239, 242], dtype=np.uint8)
        image[54:118, 64:176] = np.array([190, 214, 228], dtype=np.uint8)
        image[70:108, 88:152] = np.array([40, 150, 95], dtype=np.uint8)
        image[78:100, 150:192] = np.array([35, 95, 80], dtype=np.uint8)
        return image

    def _right_wrist_view(self) -> np.ndarray:
        image = np.full((160, 240, 3), 228, dtype=np.uint8)
        image[24:128, 34:190] = np.array([238, 240, 236], dtype=np.uint8)
        image[58:122, 58:170] = np.array([210, 196, 182], dtype=np.uint8)
        image[72:108, 84:146] = np.array([35, 95, 80], dtype=np.uint8)
        image[68:112, 144:188] = np.array([230, 175, 45], dtype=np.uint8)
        return image

    def _default_camera(self) -> str:
        if self.context.embodiment and self.context.embodiment.default_camera:
            return self.context.embodiment.default_camera
        return "front"

    def _build_camera_status_map(
        self,
        *,
        frame_id: int | None,
        camera_ok: bool,
        image_shape: list[int] | None,
        drop_reason: str | None,
    ) -> dict[str, Any]:
        return {
            name: {
                "ok": camera_ok,
                "frame_available": camera_ok,
                "frame_id": frame_id,
                "image_shape": image_shape,
                "owner": "mock",
                "drop_reason": drop_reason,
            }
            for name in self._camera_names()
        }

    def _camera_names(self) -> tuple[str, ...]:
        if self.context.embodiment is not None:
            raw = self.context.embodiment.camera_layout.get("cameras")
            if isinstance(raw, (list, tuple)):
                names = tuple(str(item) for item in raw if str(item).strip())
                if names:
                    return names
        return self._DEFAULT_CAMERA_NAMES

    def _camera_available_for_observe(self, observe_count: int) -> bool:
        if not bool(self.settings.get("camera_available", True)):
            return False
        warmup_missing = int(self.settings.get("camera_warmup_missing_frames", 0) or 0)
        if observe_count <= warmup_missing:
            return False
        drop_every = int(self.settings.get("camera_drop_every_n_observe", 0) or 0)
        return not (drop_every > 0 and observe_count % drop_every == 0)

    def _camera_drop_reason(self, observe_count: int) -> str | None:
        if not bool(self.settings.get("camera_available", True)):
            return "camera_unavailable"
        warmup_missing = int(self.settings.get("camera_warmup_missing_frames", 0) or 0)
        if observe_count <= warmup_missing:
            return "camera_warmup"
        drop_every = int(self.settings.get("camera_drop_every_n_observe", 0) or 0)
        if drop_every > 0 and observe_count % drop_every == 0:
            return "intermittent_drop"
        return None
