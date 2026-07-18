from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from hey_robot.logging import HeyRobotLogger
from hey_robot.protocol import (
    RobotAction,
    RobotObservation,
    RobotSkillAction,
    RobotStatus,
    SceneEntity,
    SkillIntent,
)
from hey_robot.robot_runtime.base import RobotCapabilities, RobotDriver, RobotHealth
from hey_robot.robot_runtime.control_plane import RobotControlPlane
from hey_robot.robot_runtime.media import LocalMediaStore
from hey_robot.robot_runtime.observations import (
    DriverObservation,
    PerceptionService,
    PerceptionSnapshot,
)
from hey_robot.robot_runtime.safety import RobotSafetyError, RobotSafetySupervisor

logger = HeyRobotLogger(name="robot_runtime")


class SceneCaptioner(Protocol):
    """可选语义图像描述能力的运行时端口。"""

    async def caption(
        self, observation: RobotObservation, status: RobotStatus | None = None
    ) -> Any: ...


@dataclass
class RobotRuntimeSnapshot:
    robot_id: str
    capabilities: RobotCapabilities
    health: RobotHealth
    status: RobotStatus


class RobotRuntime:
    """围绕具体机器人驱动的运行时边界。

    驱动只与硬件或仿真交互。运行时拥有所有支持的机器人本体都必须保持一致的
    可部署语义：生命周期、观测实体化、Skill 准入、动作应用，以及健康状态和
    能力检查。
    """

    def __init__(
        self,
        driver: RobotDriver,
        media_store: LocalMediaStore,
        *,
        safety: RobotSafetySupervisor | None = None,
        scene_captioner: SceneCaptioner | None = None,
        image_save_every_n: int = 1,
    ) -> None:
        self.driver = driver
        self.media_store = media_store
        self.perception = PerceptionService(
            driver, media_store, image_save_every_n=image_save_every_n
        )
        self.robot_id = driver.robot_id
        self.safety = safety or RobotSafetySupervisor()
        self.scene_captioner = scene_captioner
        self.control_plane = RobotControlPlane()
        self._capabilities: RobotCapabilities | None = None
        self._scene_entities: tuple[SceneEntity, ...] = ()
        self._scene_entities_frame_id: int | None = None

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
        return self._with_scene_entities(
            (await self.perception.refresh(reason="runtime.observe")).observation
        )

    async def latest_observation(
        self, *, max_age_ms: int | None = None
    ) -> RobotObservation | None:
        snapshot = self.perception.latest(max_age_ms=max_age_ms)
        return snapshot.observation if snapshot is not None else None

    async def refresh_observation(
        self, *, reason: str | None = None
    ) -> PerceptionSnapshot:
        snapshot = await self.perception.refresh(reason=reason)
        return replace(
            snapshot, observation=self._with_scene_entities(snapshot.observation)
        )

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
        return await self.control_plane.apply_action(
            action,
            apply_fn=self.driver.apply_action,
            stop_fn=lambda current: self.control_plane.stop_motion(
                current, apply_fn=self.driver.apply_action
            ),
        )

    async def reset(self) -> RobotStatus:
        return await self.driver.reset()

    def build_observation(self, observation: DriverObservation) -> RobotObservation:
        return self._with_scene_entities(self.perception.build_observation(observation))

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
        result = await self._inspect_scene(snapshot, dict(skill_action.arguments))
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

    async def _inspect_scene(
        self, snapshot: PerceptionSnapshot, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        caption, entities = await self._caption_scene(snapshot.observation)
        summary = (
            f"scene={caption}"
            if caption
            else _observation_summary(
                snapshot.observation, question=arguments.get("question")
            )
        )
        return {
            "success": snapshot.has_images,
            "skill": "inspect_scene",
            "message": (
                "scene recognized"
                if caption
                else "camera image captured but scene recognition unavailable"
                if snapshot.has_images
                else "camera image unavailable"
            ),
            "summary": summary,
            "failure_mode": None if snapshot.has_images else "camera_unavailable",
            "semantic_available": bool(caption),
            "entities": [_entity_payload(item) for item in entities],
            **snapshot.summary(),
        }

    async def _caption_scene(
        self, observation: RobotObservation
    ) -> tuple[str | None, tuple[SceneEntity, ...]]:
        """在启用视觉描述器时，返回模型生成的场景摘要。

        原始相机元数据不会被当作场景描述：成功采集到一帧图像，并不能证明图像中
        可见什么内容。
        """
        if self.scene_captioner is None or not observation.images:
            return None, ()
        try:
            understanding = await self.scene_captioner.caption(
                observation, await self.status()
            )
        except Exception:
            logger.exception(
                f"场景理解调用异常: robot={self.robot_id} frame={observation.frame_id}"
            )
            return None, ()
        metadata = getattr(understanding, "metadata", None)
        confidence = getattr(understanding, "confidence", 0.0)
        if not isinstance(metadata, dict):
            logger.warning(
                f"场景理解结果无元数据，已丢弃: robot={self.robot_id} "
                f"frame={observation.frame_id}"
            )
            return None, ()
        if metadata.get("error") or not confidence > 0.0:
            logger.warning(
                f"场景理解结果不可用，已丢弃: robot={self.robot_id} "
                f"frame={observation.frame_id} confidence={confidence} "
                f"reason={metadata.get('error') or metadata.get('raw') or 'unknown'}"
            )
            return None, ()
        summary = understanding.summary.strip()
        entities = tuple(
            entity
            for entity in getattr(understanding, "entities", ())
            if isinstance(entity, SceneEntity)
            and entity.frame_id == observation.frame_id
        )
        if entities:
            self._scene_entities = entities
            self._scene_entities_frame_id = observation.frame_id
        return summary or None, entities

    def _with_scene_entities(self, observation: RobotObservation) -> RobotObservation:
        cached_frame_id = self._scene_entities_frame_id
        if cached_frame_id is not None and observation.frame_id > cached_frame_id + 6:
            self._scene_entities = ()
            self._scene_entities_frame_id = None
        entities = {item.entity_id: item for item in observation.entities}
        for item in self._scene_entities:
            entities.setdefault(item.entity_id, item)
        return replace(observation, entities=list(entities.values()))

    async def _look_around(
        self, action: RobotAction, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        observations: list[dict[str, Any]] = []
        first = await self._current_perception_snapshot(reason="look_around:start")
        observations.append(await self._inspect_scene(first, arguments))
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
            observations.append(await self._inspect_scene(snapshot, arguments))
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
            SkillIntent(
                envelope=parent.envelope,
                skill_id=parent.skill_id,
                task_id=parent.task_id,
                intent_kind=parent.intent_kind,
                name=name,
                arguments=dict(arguments),
                objective=f"internal {name} for {parent.skill_id}",
            )
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


def _entity_payload(entity: SceneEntity) -> dict[str, object]:
    return {
        "entity_id": entity.entity_id,
        "type": entity.entity_type,
        "attributes": entity.attributes,
        "relations": [
            {"predicate": relation.predicate, "object_id": relation.object_id}
            for relation in entity.relations
        ],
        "frame_id": entity.frame_id,
    }
