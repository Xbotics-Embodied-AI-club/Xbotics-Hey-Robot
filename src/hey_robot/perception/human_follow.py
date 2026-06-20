from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DEFAULT_DETECTOR_MODEL = Path("models/yolo26n.pt")
_DETECTOR_MODEL: Any | None = None


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int = 0
    class_name: str = "person"

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class Target:
    id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    age: int = 0
    time_since_update: int = 0
    history: list[tuple[int, int, int, int]] = field(default_factory=list)

    def update(self, detection: Detection) -> None:
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.age += 1
        self.time_since_update = 0
        self.history.append(self.bbox)
        if len(self.history) > 10:
            self.history.pop(0)

    def mark_missed(self) -> None:
        self.time_since_update += 1

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def predict(self) -> tuple[int, int, int, int]:
        if len(self.history) < 2:
            return self.bbox
        vx = 0.0
        vy = 0.0
        for prev, curr in zip(self.history[:-1], self.history[1:], strict=False):
            vx += (curr[0] - prev[0]) + (curr[2] - prev[2])
            vy += (curr[1] - prev[1]) + (curr[3] - prev[3])
        vx /= max(1, (len(self.history) - 1) * 2)
        vy /= max(1, (len(self.history) - 1) * 2)
        x1, y1, x2, y2 = self.bbox
        return (int(x1 + vx), int(y1 + vy), int(x2 + vx), int(y2 + vy))


def detect_people(image: Any) -> list[Detection]:
    if image is None:
        return []
    try:
        import numpy as np
    except Exception:
        return []
    model = _DETECTOR_MODEL
    if model is None:
        raise RuntimeError(
            "human follow detector is not loaded; call load_detector() during startup"
        )
    frame = np.asarray(image)
    if frame.size == 0:
        return []
    try:
        results = model(frame, conf=0.5, classes=[0], imgsz=320, verbose=False)
    except Exception:
        return []
    detections: list[Detection] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            detections.append(Detection((x1, y1, x2, y2), conf))
    return detections


def load_detector(model_path: str | None = None) -> None:
    global _DETECTOR_MODEL
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "human follow requires `ultralytics` for YOLO detection"
        ) from exc
    resolved = (
        Path(model_path).resolve() if model_path else _DEFAULT_DETECTOR_MODEL.resolve()
    )
    if not resolved.exists():
        raise FileNotFoundError(f"human follow detector model is missing: {resolved}")
    _DETECTOR_MODEL = YOLO(str(resolved))


class TargetTracker:
    _id_counter = 0

    def __init__(self, *, max_age: int = 30, min_iou: float = 0.3) -> None:
        self.max_age = max_age
        self.min_iou = min_iou
        self.targets: list[Target] = []
        self.primary_target: Target | None = None

    def update(self, detections: list[Detection]) -> Target | None:
        matched, unmatched_targets, unmatched_detections = self._match(detections)
        for target, detection in matched:
            target.update(detection)
        for target in unmatched_targets:
            target.mark_missed()
        for detection in unmatched_detections:
            self.targets.append(self._create_target(detection))
        self.targets = [
            target for target in self.targets if target.time_since_update < self.max_age
        ]
        self.primary_target = self._select_primary()
        return self.primary_target

    def _create_target(self, detection: Detection) -> Target:
        type(self)._id_counter += 1
        target = Target(type(self)._id_counter, detection.bbox, detection.confidence)
        target.history.append(detection.bbox)
        return target

    def _match(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[Target, Detection]], list[Target], list[Detection]]:
        if not self.targets or not detections:
            return [], list(self.targets), list(detections)
        matched: list[tuple[Target, Detection]] = []
        used: set[int] = set()
        for target in self.targets:
            best_idx = -1
            best_iou = self.min_iou
            predicted = (
                target.predict() if target.time_since_update > 0 else target.bbox
            )
            for index, detection in enumerate(detections):
                if index in used:
                    continue
                overlap = compute_iou(predicted, detection.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_idx = index
            if best_idx >= 0:
                used.add(best_idx)
                matched.append((target, detections[best_idx]))
        unmatched_targets = [
            target
            for target in self.targets
            if not any(item[0] is target for item in matched)
        ]
        unmatched_detections = [
            detection for index, detection in enumerate(detections) if index not in used
        ]
        return matched, unmatched_targets, unmatched_detections

    def _select_primary(self) -> Target | None:
        valid = [target for target in self.targets if target.time_since_update == 0]
        if not valid:
            return None
        return max(valid, key=lambda target: target.area * max(target.confidence, 0.1))


def compute_iou(
    box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
    area2 = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)
    union = area1 + area2 - intersection
    return 0.0 if union <= 0 else intersection / union


@dataclass(frozen=True)
class VelocityCommand:
    vx: float
    vy: float
    vz: float


class FollowController:
    REFERENCE_WIDTH = 320
    REFERENCE_HEIGHT = 320
    REFERENCE_AREA = REFERENCE_WIDTH * REFERENCE_HEIGHT

    def __init__(
        self,
        *,
        target_distance: float = 1.0,
        target_width_ratio: float = 0.25,
        target_height_ratio: float = 1.0,
        kp_linear: float = 0.001,
        kp_angular: float = 0.003,
        max_linear_speed: float = 0.3,
        max_backward_speed: float = 0.2,
        allow_backward: bool = True,
        max_angular_speed: float = 0.8,
        dead_zone_x: float = 0.15,
        dead_zone_area: float = 0.1,
    ) -> None:
        self.kp_linear = kp_linear
        self.kp_angular = kp_angular
        self.max_linear_speed = max_linear_speed
        self.max_backward_speed = max_backward_speed
        self.allow_backward = allow_backward
        self.max_angular_speed = max_angular_speed
        self.dead_zone_x = dead_zone_x
        self.dead_zone_area = dead_zone_area
        self.target_area = min(
            self.REFERENCE_AREA * 0.85,
            (
                self.REFERENCE_WIDTH
                * target_width_ratio
                * self.REFERENCE_HEIGHT
                * target_height_ratio
                * (1.0 / max(target_distance, 0.2)) ** 2
            ),
        )
        self.target_lost_count = 0
        self.max_lost_count = 60
        self.last_time = time.time()

    def compute_velocity(
        self, target: Target | None, *, frame_width: int, frame_height: int
    ) -> VelocityCommand | None:
        if target is None:
            self.target_lost_count += 1
            if self.target_lost_count > self.max_lost_count:
                return VelocityCommand(0.0, 0.0, 0.0)
            return None
        self.target_lost_count = 0
        cx, _cy = target.center
        area = target.area
        norm_cx = (cx / max(1.0, float(frame_width)) - 0.5) * 2.0
        norm_area = (
            area / max(1.0, float(frame_width * frame_height)) * self.REFERENCE_AREA
        )
        error_x = 0.0 if abs(norm_cx) < self.dead_zone_x else norm_cx
        error_area = (self.target_area - norm_area) / max(self.target_area, 1.0)
        error_area = 0.0 if abs(error_area) < self.dead_zone_area else error_area
        vz = self._clamp(
            error_x * self.kp_angular, -self.max_angular_speed, self.max_angular_speed
        )
        vx = error_area * self.kp_linear
        if vx > 0.0:
            # Do not drive toward a person until the chassis is substantially aligned.
            alignment_scale = self._clamp(1.0 - abs(error_x) / 0.5, 0.0, 1.0)
            vx *= alignment_scale
        if self.allow_backward:
            vx = self._clamp(vx, -self.max_backward_speed, self.max_linear_speed)
        else:
            vx = self._clamp(vx, 0.0, self.max_linear_speed)
        return VelocityCommand(vx, 0.0, vz)

    def compute_search_velocity(self) -> VelocityCommand:
        return VelocityCommand(0.0, 0.0, self.max_angular_speed * 0.5)

    def is_target_lost(self) -> bool:
        return self.target_lost_count > self.max_lost_count

    def is_searching(self) -> bool:
        return 0 < self.target_lost_count <= self.max_lost_count

    @staticmethod
    def smooth_velocity(
        current: VelocityCommand, target: VelocityCommand, *, alpha: float = 0.3
    ) -> VelocityCommand:
        return VelocityCommand(
            alpha * target.vx + (1 - alpha) * current.vx,
            alpha * target.vy + (1 - alpha) * current.vy,
            alpha * target.vz + (1 - alpha) * current.vz,
        )

    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(value, max_val))
