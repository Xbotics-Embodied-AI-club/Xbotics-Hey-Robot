# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
# Modified for Xbotics Hey Robot: dock oracle perception.
from __future__ import annotations

import re
import time

import numpy as np

from hey_robot.robot_runtime.simulation.so101_mobile.arm import So101MobileArmKernel
from hey_robot.robot_runtime.simulation.so101_mobile.scenario import (
    DOCK_ALIASES,
    WAND_ALIASES,
)
from hey_robot.robot_runtime.simulation.so101_mobile.session import (
    So101MobileSession,
)
from hey_robot.robot_runtime.simulation.so101_mobile.types import (
    Detection,
    Pose3D,
    TrackedObject,
)

WIDTH = 640
HEIGHT = 480

ALL_ALIASES = {**WAND_ALIASES, **DOCK_ALIASES}


class DockOracle:
    """Simulation-only ground-truth perception for dock and wand."""

    source = "mujoco_ground_truth"
    simulation_only = True

    def __init__(self, session: So101MobileSession, arm: So101MobileArmKernel) -> None:
        self.session = session
        self.arm = arm

    def color_frame(self, camera: str = "front") -> np.ndarray:
        return self.session.render(camera, bgr=True)

    def depth_frame(self) -> np.ndarray:
        return np.zeros((HEIGHT, WIDTH), dtype=np.uint16)

    def detect(self, query: str) -> list[Detection]:
        normalized = str(query or "").lower().strip()
        generic = (
            normalized in {"all", "objects", "everything", "all objects"}
            or any(token in normalized for token in ("所有", "物体"))
            or any(
                re.search(rf"\b{re.escape(token)}\b", normalized)
                for token in ("all", "objects", "everything")
            )
        )
        results: list[Detection] = []
        for name in self.arm.get_object_positions():
            aliases = ALL_ALIASES.get(name, (name,))
            if not generic and not any(alias in normalized for alias in aliases):
                continue
            center_x, center_y, half = WIDTH // 2, HEIGHT // 2, 40
            results.append(
                Detection(
                    label=name.replace("_", " "),
                    bbox=(
                        center_x - half,
                        center_y - half,
                        center_x + half,
                        center_y + half,
                    ),
                )
            )
        return results

    def track(self, detections: list[Detection]) -> list[TrackedObject]:
        positions = self.arm.get_object_positions()
        tracked: list[TrackedObject] = []
        for index, detection in enumerate(detections):
            name = detection.label.replace(" ", "_")
            position = positions.get(name)
            pose = (
                Pose3D(*[float(value) for value in position])
                if position is not None
                else None
            )
            tracked.append(
                TrackedObject(
                    track_id=index,
                    label=detection.label,
                    bbox_2d=detection.bbox,
                    pose=pose,
                    confidence=detection.confidence,
                )
            )
        return tracked

    def locate(
        self,
        query: str,
        *,
        sample_count: int = 1,
        sample_interval: float = 0.0,
    ) -> tuple[str, list[list[float]]] | None:
        detections = self.detect(query)
        if not detections and not str(query or "").strip():
            positions = self.arm.get_object_positions()
            if not positions:
                return None
            nearest = min(
                positions.items(),
                key=lambda item: item[1][0] ** 2 + item[1][1] ** 2,
            )[0]
            detections = self.detect(nearest)
        if not detections:
            return None
        samples: list[list[float]] = []
        object_name = detections[0].label.replace(" ", "_")
        for index in range(max(1, int(sample_count))):
            tracked = self.track(detections)
            if tracked and tracked[0].pose is not None:
                samples.append(tracked[0].pose.as_list())
            if index + 1 < sample_count and sample_interval > 0:
                time.sleep(float(sample_interval))
        if not samples:
            return None
        return object_name, samples

    def caption(self) -> str:
        names = list(self.arm.get_object_positions())
        if not names:
            return "I don't see the cat wand or dock."
        parts = []
        for name in names:
            label = name.replace("_", " ")
            if name == "cat_wand":
                parts.append(f"the {label} (in dock)")
            else:
                parts.append(f"the {label}")
        return "I can see: " + ", ".join(parts) + "."
