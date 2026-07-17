from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DEFAULT_DETECTOR_MODEL = Path("models/yolo26n.pt")
_DETECTOR_MODEL: Any | None = None
# opt-in：设了 S600_DETECT_URL 就把人体检测交给 S600 BPU（HeyRobotModelApis /v1/detect），
# 否则维持本地 ultralytics(CPU)。
_REMOTE_DETECT_URL: str | None = None


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


def _detect_remote(image: Any) -> list[Detection]:
    """把帧发给 S600 BPU 检测端点（/v1/detect），转成 Detection 列表。"""
    detect_url = _REMOTE_DETECT_URL
    if detect_url is None:
        return []
    try:
        import cv2
        import httpx
        import numpy as np
    except Exception:
        return []
    frame = np.asarray(image)
    if frame.size == 0:
        return []
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return []
    try:
        resp = httpx.post(
            detect_url,
            files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
            data={"score_threshold": "0.5", "person_only": "true"},
            timeout=10.0,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return []
    detections: list[Detection] = []
    for item in payload.get("detections", []):
        x1, y1, x2, y2 = item["box"]
        detections.append(
            Detection(
                (int(x1), int(y1), int(x2), int(y2)), float(item.get("score", 0.0))
            )
        )
    return detections


def detect_people(image: Any) -> list[Detection]:
    if image is None:
        return []
    if _REMOTE_DETECT_URL:
        return _detect_remote(image)
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
    global _DETECTOR_MODEL, _REMOTE_DETECT_URL
    remote = os.environ.get("S600_DETECT_URL", "").strip()
    if remote:
        # opt-in：人体检测走 S600 BPU，不加载本地 ultralytics
        _REMOTE_DETECT_URL = (
            remote
            if remote.rstrip("/").endswith("/detect")
            else remote.rstrip("/") + "/v1/detect"
        )
        _DETECTOR_MODEL = "remote-s600-bpu"
        return
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
            # 底盘基本对齐前，不要向人体目标前进。
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


class HumanFollowRunner:
    """共享的人体跟随控制循环，与帧来源和速度输出方式无关。

    NATS 服务和本地技能都使用这一套实现。调用方注入 get_frame、apply_velocity、
    emit_progress 和 is_stopped 等回调，把 runner 适配到各自的传输方式。
    """

    def __init__(
        self,
        arguments: dict[str, Any],
        *,
        get_frame: Any = None,
        apply_velocity: Any = None,
        emit_progress: Any = None,
        is_stopped: Any = None,
        on_start: Any = None,
        on_stop: Any = None,
    ) -> None:
        self._args = arguments
        self._get_frame = get_frame
        self._apply_velocity = apply_velocity
        self._emit_progress = emit_progress
        self._is_stopped = is_stopped
        self._on_start = on_start
        self._on_stop = on_stop

        self.tracker = TargetTracker(
            max_age=int(arguments.get("max_tracking_age") or 30),
            min_iou=float(arguments.get("min_iou_threshold") or 0.3),
        )
        self.controller = FollowController(
            target_distance=float(arguments.get("target_distance_m") or 0.7),
            target_width_ratio=float(arguments.get("target_width_ratio") or 0.35),
            target_height_ratio=float(arguments.get("target_height_ratio") or 1.0),
            kp_linear=float(arguments.get("kp_linear") or 0.35),
            kp_angular=float(arguments.get("kp_angular") or 1.0),
            max_linear_speed=float(arguments.get("max_linear_speed") or 0.3),
            max_backward_speed=float(arguments.get("max_backward_speed") or 0.2),
            allow_backward=bool(arguments.get("allow_backward", True)),
            max_angular_speed=float(arguments.get("max_angular_speed") or 1.0),
            dead_zone_x=float(arguments.get("dead_zone_x") or 0.15),
            dead_zone_area=float(arguments.get("dead_zone_area") or 0.1),
        )

    async def run(self) -> dict[str, Any]:
        """执行人体跟随控制循环，并返回结果字典。"""
        import asyncio

        duration_raw = self._args.get("duration_sec")
        max_steps = int(self._args.get("max_steps") or 0)
        if duration_raw is None and max_steps <= 0:
            duration_raw = 120.0
        duration = float(duration_raw) if duration_raw is not None else None
        deadline = time.monotonic() + duration if duration else None

        current = VelocityCommand(0.0, 0.0, 0.0)
        last_frame_id: int | None = None
        steps = 0

        if self._on_start is not None:
            await self._on_start()

        result: dict[str, Any] = {"success": True, "summary": "human follow stopped"}
        try:
            while not self._is_stopped():
                if deadline is not None and time.monotonic() >= deadline:
                    result["summary"] = "human follow completed"
                    break
                if max_steps > 0 and steps >= max_steps:
                    result["summary"] = "human follow completed"
                    break

                frame = await self._get_frame()
                if frame is None:
                    if self._emit_progress is not None:
                        await self._emit_progress(
                            phase="waiting_for_camera",
                            summary="waiting for camera frame",
                        )
                    continue

                metadata, image = frame
                frame_id = int(metadata.get("frame_id") or 0)
                if frame_id == last_frame_id:
                    await asyncio.sleep(0.01)
                    continue
                last_frame_id = frame_id

                detections = await asyncio.to_thread(detect_people, image)
                target = self.tracker.update(detections)
                height, width = image.shape[:2]
                command = self.controller.compute_velocity(
                    target, frame_width=width, frame_height=height
                )
                phase = "following"
                if command is None:
                    if not self.controller.is_searching():
                        continue
                    command = self.controller.compute_search_velocity()
                    phase = "searching"
                if self.controller.is_target_lost():
                    result = {
                        "success": False,
                        "summary": "person lost during human follow",
                        "failure_mode": "person_lost",
                        "error": "person lost during human follow",
                    }
                    break

                current = self.controller.smooth_velocity(current, command, alpha=0.3)
                steps += 1
                await self._apply_velocity(current.vx, current.vy, current.vz)

                if self._emit_progress is not None:
                    await self._emit_progress(
                        phase=phase,
                        summary="following" if phase == "following" else "searching",
                        frame_id=frame_id,
                        detections=detections,
                        target=target,
                        command={
                            "vx": current.vx,
                            "vy": current.vy,
                            "wz": current.vz,
                        },
                    )

                # 已经处于跟随窗口内时提前退出
                if (
                    target is not None
                    and abs(current.vx) < 0.02
                    and abs(current.vz) < 0.02
                    and duration is not None
                ):
                    break

        except asyncio.CancelledError:
            result = {"success": False, "summary": "human follow interrupted"}
            raise
        except Exception as exc:
            result = {
                "success": False,
                "summary": str(exc),
                "failure_mode": "internal_error",
                "error": str(exc),
            }
        finally:
            if self._on_stop is not None:
                await self._on_stop()

        return result
