from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hey_robot.media import LocalMediaStore
from hey_robot.perception import (
    DriverObservation,
    PerceptionService,
    PerceptionSnapshot,
)
from hey_robot.protocol import RobotAction, RobotObservation, RobotStatus, SkillIntent
from hey_robot.robots.base import RobotCapabilities, RobotDriver, RobotHealth
from hey_robot.robots.safety import RobotSafetyError, RobotSafetySupervisor
from hey_robot.skills import RobotSkillAction


@dataclass
class RobotRuntimeSnapshot:
    robot_id: str
    capabilities: RobotCapabilities
    health: RobotHealth
    status: RobotStatus


class RobotRuntime:
    """Runtime boundary around a concrete robot driver.

    Drivers only talk to hardware or simulation. The runtime owns deployable
    semantics that must be consistent across supported embodiments: lifecycle,
    observation materialization, skill acceptance, action application, and
    health/capability inspection.
    """

    def __init__(
        self,
        driver: RobotDriver,
        media_store: LocalMediaStore,
        *,
        safety: RobotSafetySupervisor | None = None,
        image_save_every_n: int = 1,
    ) -> None:
        self.driver = driver
        self.media_store = media_store
        self.perception = PerceptionService(
            driver, media_store, image_save_every_n=image_save_every_n
        )
        self.robot_id = driver.robot_id
        self.safety = safety or RobotSafetySupervisor()
        self._capabilities: RobotCapabilities | None = None

    async def start(self) -> RobotRuntimeSnapshot:
        await self.driver.start()
        self._capabilities = await self.driver.capabilities()
        return await self.snapshot()

    async def close(self) -> None:
        await self.driver.close()

    async def snapshot(self) -> RobotRuntimeSnapshot:
        return RobotRuntimeSnapshot(
            robot_id=self.robot_id,
            capabilities=await self.capabilities(),
            health=await self.health(),
            status=await self.status(),
        )

    async def capabilities(self) -> RobotCapabilities:
        if self._capabilities is None:
            self._capabilities = await self.driver.capabilities()
        return self._capabilities

    async def health(self) -> RobotHealth:
        return await self.driver.health()

    async def observe(self) -> RobotObservation:
        return (await self.perception.refresh(reason="runtime.observe")).observation

    async def latest_observation(
        self, *, max_age_ms: int | None = None
    ) -> RobotObservation | None:
        snapshot = self.perception.latest(max_age_ms=max_age_ms)
        return snapshot.observation if snapshot is not None else None

    async def refresh_observation(
        self, *, reason: str | None = None
    ) -> PerceptionSnapshot:
        return await self.perception.refresh(reason=reason)

    async def status(self) -> RobotStatus:
        return await self.driver.status()

    async def apply_action(self, action: RobotAction) -> RobotStatus:
        perception_skill = _perception_skill_name(action)
        if perception_skill is not None:
            return await self._apply_perception_skill(action, perception_skill)
        decision = self.safety.evaluate_action(
            action,
            capabilities=await self.capabilities(),
            health=await self.health(),
        )
        if not decision.allowed:
            raise RobotSafetyError(
                decision.reason or "robot action blocked by safety supervisor"
            )
        return await self.driver.apply_action(action)

    async def reset(self) -> RobotStatus:
        return await self.driver.reset()

    def build_observation(self, observation: DriverObservation) -> RobotObservation:
        return self.perception.build_observation(observation)

    async def _apply_perception_skill(
        self, action: RobotAction, skill_name: str
    ) -> RobotStatus:
        skill_action = RobotSkillAction.from_robot_action(action)
        before_request = getattr(self.driver, "before_perception_request", None)
        if callable(before_request):
            before_request(skill_name, dict(skill_action.arguments))
        if skill_name == "look_around":
            result = await self._look_around(action, dict(skill_action.arguments))
            return await self._perception_status(action, result=result)
        if skill_name == "detect_marker":
            snapshot = await self._current_perception_snapshot(reason=skill_name)
            result = self._detect_marker(snapshot, dict(skill_action.arguments))
            return await self._perception_status(
                action, snapshot=snapshot, result=result
            )
        snapshot = await self._current_perception_snapshot(reason=skill_name)
        result = self._inspect_scene(snapshot, dict(skill_action.arguments))
        return await self._perception_status(action, snapshot=snapshot, result=result)

    async def _current_perception_snapshot(self, *, reason: str) -> PerceptionSnapshot:
        return await self.perception.refresh(reason=reason)

    async def _perception_status(
        self,
        action: RobotAction,
        *,
        result: dict[str, Any],
        snapshot: PerceptionSnapshot | None = None,
    ) -> RobotStatus:
        status = await self.status()
        frame_id = (
            snapshot.observation.frame_id if snapshot is not None else status.frame_id
        )
        success = bool(result.get("success", False))
        return RobotStatus(
            envelope=status.envelope,
            frame_id=frame_id,
            state=status.state,
            task=status.task,
            skill_id=action.skill_id,
            success=success,
            error=None if success else str(result.get("message") or "skill failed"),
            metrics={**status.metrics, "last_skill_result": result},
        )

    def _inspect_scene(
        self, snapshot: PerceptionSnapshot, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        summary = _observation_summary(
            snapshot.observation, question=arguments.get("question")
        )
        return {
            "success": snapshot.has_images,
            "skill": "inspect_scene",
            "message": "scene inspected"
            if snapshot.has_images
            else "camera image unavailable",
            "summary": summary,
            "failure_mode": None if snapshot.has_images else "camera_unavailable",
            **snapshot.summary(),
        }

    async def _look_around(
        self, action: RobotAction, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        observations: list[dict[str, Any]] = []
        first = await self._current_perception_snapshot(reason="look_around:start")
        observations.append(self._inspect_scene(first, arguments))
        for direction, angle in (("left", 25.0), ("right", 50.0), ("left", 25.0)):
            motion = await self._apply_internal_skill(
                action,
                "turn_base",
                {"direction": direction, "angle_deg": angle},
            )
            if motion.success is False:
                return {
                    "success": False,
                    "skill": "look_around",
                    "message": f"look_around motion failed: {motion.error}",
                    "failure_mode": "base_motion_failed",
                    "observations": observations,
                }
            snapshot = await self._current_perception_snapshot(reason="look_around")
            observations.append(self._inspect_scene(snapshot, arguments))
        ok = any(item.get("success") for item in observations)
        return {
            "success": ok,
            "skill": "look_around",
            "message": "look_around completed" if ok else "no usable camera image",
            "failure_mode": None if ok else "camera_unavailable",
            "observations": observations,
            "summary": _join_summaries(observations),
        }

    def _detect_marker(
        self, snapshot: PerceptionSnapshot, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        image = self._resolve_image(
            snapshot.observation, camera=arguments.get("camera")
        )
        detections = _detect_markers(image) if image is not None else []
        marker_id = arguments.get("marker_id")
        if marker_id is not None:
            detections = [
                item for item in detections if item.get("id") == int(marker_id)
            ]
        found = bool(detections)
        return {
            "success": found,
            "skill": "detect_marker",
            "message": "marker detected" if found else "marker not found",
            "failure_mode": None if found else "marker_not_found",
            "markers": detections,
            **snapshot.summary(),
        }

    async def _apply_internal_skill(
        self, parent: RobotAction, name: str, arguments: dict[str, Any]
    ) -> RobotStatus:
        internal = RobotSkillAction(name, arguments).to_robot_action(
            SkillIntent(envelope=parent.envelope, skill_id=parent.skill_id, name=name)
        )
        decision = self.safety.evaluate_action(
            internal,
            capabilities=await self.capabilities(),
            health=await self.health(),
        )
        if not decision.allowed:
            return RobotStatus(
                envelope=parent.envelope,
                skill_id=parent.skill_id,
                success=False,
                error=decision.reason,
                metrics={
                    "last_skill_result": {
                        "success": False,
                        "skill": name,
                        "message": decision.reason,
                        "failure_mode": "safety_blocked",
                    }
                },
            )
        return await self.driver.apply_action(internal)

    def _resolve_image(
        self, observation: RobotObservation, *, camera: object | None = None
    ):
        refs = observation.images
        if camera:
            preferred = [ref for ref in refs if ref.camera == str(camera)]
            if preferred:
                refs = preferred
        if not refs:
            return None
        try:
            return self.media_store.resolve_image(refs[0])
        except Exception:
            return None


def _perception_skill_name(action: RobotAction) -> str | None:
    try:
        skill = RobotSkillAction.from_robot_action(action)
    except ValueError:
        return None
    if skill.name in {
        "inspect_scene",
        "look_around",
        "detect_marker",
        "human_follow",
    }:
        return skill.name
    return None


def _observation_summary(
    observation: RobotObservation, *, question: object | None = None
) -> str:
    parts = [
        f"frame={observation.frame_id}",
        f"images={len(observation.images)}",
        f"artifacts={len(observation.artifacts)}",
    ]
    if question:
        parts.append(f"question={str(question).strip()}")
    scene = observation.raw.get("scene")
    if scene:
        parts.append(f"scene={scene}")
    camera = observation.raw.get("camera")
    if isinstance(camera, dict):
        parts.append(
            "camera="
            + (
                "available"
                if camera.get("frame_available") or camera.get("ok")
                else "unavailable"
            )
        )
    return "; ".join(parts)


def _join_summaries(items: list[dict[str, Any]]) -> str:
    summaries = [
        str(item.get("summary") or item.get("message") or "").strip()
        for item in items
        if item.get("summary") or item.get("message")
    ]
    return " | ".join(summaries[:5])


def _detect_markers(image: Any) -> list[dict[str, Any]]:
    if image is None:
        return []
    try:
        import cv2
        import numpy as np
    except Exception:
        return []
    arr = np.asarray(image)
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    detections: list[dict[str, Any]] = []
    aruco = getattr(cv2, "aruco", None)
    if aruco is not None:
        try:
            dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            params = aruco.DetectorParameters()
            if hasattr(aruco, "ArucoDetector"):
                corners, ids, _ = aruco.ArucoDetector(dictionary, params).detectMarkers(
                    gray
                )
            else:
                corners, ids, _ = aruco.detectMarkers(
                    gray, dictionary, parameters=params
                )
            if ids is not None:
                for marker_corners, marker_id in zip(
                    corners, ids.flatten(), strict=False
                ):
                    pts = marker_corners.reshape(-1, 2)
                    detections.append(
                        _marker_detection(int(marker_id), pts, gray.shape)
                    )
        except Exception:
            detections = []
    if detections:
        return detections
    return _detect_square_markers(gray)


def _detect_square_markers(gray: Any) -> list[dict[str, Any]]:
    try:
        import cv2
        import numpy as np
    except Exception:
        return []
    arr = np.asarray(gray)
    if arr.size == 0:
        return []
    blur = cv2.GaussianBlur(arr, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[dict[str, Any]] = []
    image_area = float(arr.shape[0] * arr.shape[1])
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(80.0, image_area * 0.001):
            continue
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        detections.append(_marker_detection(None, approx.reshape(-1, 2), arr.shape))
    ranked = sorted(
        (
            (float(item.get("area", 0.0)), -index, item)
            for index, item in enumerate(detections)
        ),
        reverse=True,
    )
    return [item for _, _, item in ranked[:5]]


def _marker_detection(marker_id: int | None, pts: Any, shape: Any) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(pts, dtype=float)
    x_min = float(arr[:, 0].min())
    y_min = float(arr[:, 1].min())
    x_max = float(arr[:, 0].max())
    y_max = float(arr[:, 1].max())
    width = int(shape[1])
    height = int(shape[0])
    return {
        "id": marker_id,
        "center": [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0],
        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
        "area": max(0.0, x_max - x_min) * max(0.0, y_max - y_min),
        "image_size": [width, height],
        "confidence": 0.9 if marker_id is not None else 0.45,
    }


def _marker_area_key(item: dict[str, Any]) -> float:
    return float(item.get("area", 0.0))
