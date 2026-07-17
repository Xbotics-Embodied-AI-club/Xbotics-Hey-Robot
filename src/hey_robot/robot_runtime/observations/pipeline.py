from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hey_robot.protocol import (
    ArtifactRef,
    ImageRef,
    RobotObservation,
    SceneEntity,
    SceneRelation,
)
from hey_robot.robot_runtime.media import LocalMediaStore
from hey_robot.robot_runtime.observations.observation import (
    DriverObservation,
    ObservationAsset,
)


@dataclass(frozen=True)
class ObservationSchema:
    robot_id: str
    driver_type: str
    cameras: list[str]
    modalities: list[str]
    proprioception_dim: int | None = None
    metadata: dict[str, Any] | None = None


class ObservationPipeline:
    """将驱动本地观测转换为协议观测。

    大型数组和二进制类资源会实体化到媒体存储。面向总线的 `RobotObservation`
    只携带少量元数据以及媒体和产物引用。
    """

    def __init__(
        self, media_store: LocalMediaStore, *, image_save_every_n: int = 1
    ) -> None:
        self.media_store = media_store
        self.image_save_every_n = max(1, int(image_save_every_n))
        self._latest_image_refs: dict[tuple[str, str], ImageRef] = {}

    def build(self, observation: DriverObservation) -> RobotObservation:
        robot_id = observation.envelope.robot_id or "robot"
        images = list(observation.images)
        artifacts: list[ArtifactRef] = []
        raw = dict(observation.metadata)
        entities = _scene_entities(raw.get("entities"), observation.frame_id)
        image_quality: list[dict[str, Any]] = []

        for index, asset in enumerate(observation.assets):
            if asset.kind == "image":
                image_quality.append(
                    _image_quality(asset.data, asset=asset, index=index)
                )
                images.append(
                    self._put_image(
                        asset,
                        robot_id=robot_id,
                        frame_id=observation.frame_id,
                        index=index,
                    )
                )
                continue
            artifacts.append(
                self._put_artifact(
                    asset, robot_id=robot_id, frame_id=observation.frame_id
                )
            )
        if image_quality:
            raw["image_quality"] = image_quality

        return RobotObservation(
            envelope=observation.envelope,
            frame_id=observation.frame_id,
            images=images,
            artifacts=artifacts,
            proprioception=observation.proprioception,
            task=observation.task,
            entities=entities,
            raw=_json_safe(raw),
        )

    def _put_image(
        self, asset: ObservationAsset, *, robot_id: str, frame_id: int, index: int
    ) -> ImageRef:
        camera = asset.name or asset.role or f"cam{index}"
        key = (robot_id, camera)
        latest = self._latest_image_refs.get(key)
        if latest is not None and frame_id % self.image_save_every_n != 0:
            return latest
        ref = self.media_store.put_image(
            asset.data,
            robot_id=robot_id,
            frame_id=frame_id,
            camera=camera,
            metadata={"role": asset.role, **asset.metadata},
        )
        self._latest_image_refs[key] = ref
        return ref

    def _put_artifact(
        self, asset: ObservationAsset, *, robot_id: str, frame_id: int
    ) -> ArtifactRef:
        artifact_type = asset.metadata.get("artifact_type") or asset.kind
        if artifact_type == "policy_observation":
            return self.media_store.put_npz_artifact(
                asset.data,
                artifact_type=str(artifact_type),
                role=asset.role,
                name=asset.name,
                robot_id=robot_id,
                frame_id=frame_id,
                metadata={
                    key: value
                    for key, value in asset.metadata.items()
                    if key != "artifact_type"
                },
            )
        return self.media_store.put_json_artifact(
            _json_safe(asset.data),
            artifact_type=str(artifact_type),
            role=asset.role,
            name=asset.name,
            robot_id=robot_id,
            frame_id=frame_id,
            metadata={
                key: value
                for key, value in asset.metadata.items()
                if key != "artifact_type"
            },
        )


def _image_quality(
    image: Any, *, asset: ObservationAsset, index: int
) -> dict[str, Any]:
    arr = np.asarray(image)
    quality: dict[str, Any] = {
        "index": index,
        "role": asset.role,
        "name": asset.name,
        "valid": False,
    }
    if arr.size == 0:
        return {**quality, "issue": "empty_image"}
    if arr.ndim not in {2, 3}:
        return {**quality, "issue": "invalid_shape", "shape": list(arr.shape)}
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 3:
        arr = arr[:, :, :3].mean(axis=2)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    dark_ratio = float(np.mean(arr <= 5.0))
    issue = None
    if mean < 8.0 or dark_ratio > 0.98:
        issue = "black_frame"
    return {
        **quality,
        "valid": issue is None,
        "issue": issue,
        "shape": list(np.asarray(image).shape),
        "mean_luma": round(mean, 3),
        "std_luma": round(std, 3),
        "dark_pixel_ratio": round(dark_ratio, 4),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scene_entities(value: Any, frame_id: int) -> list[SceneEntity]:
    if not isinstance(value, list):
        return []
    entities: list[SceneEntity] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get("entity_id") or "").strip()
        entity_type = str(item.get("type") or item.get("entity_type") or "").strip()
        if not entity_id or not entity_type:
            continue
        item_frame_id = item.get("frame_id", frame_id)
        if not isinstance(item_frame_id, int):
            continue
        entities.append(
            SceneEntity(
                entity_id,
                entity_type,
                item_frame_id,
                dict(item.get("attributes") or {}),
                _scene_relations(item.get("relations")),
            )
        )
    return entities


def _scene_relations(value: Any) -> list[SceneRelation]:
    if not isinstance(value, list):
        return []
    relations: list[SceneRelation] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        predicate = str(item.get("predicate") or "").strip()
        object_id = str(item.get("object_id") or "").strip()
        if predicate and object_id:
            relations.append(SceneRelation(predicate, object_id))
    return relations
