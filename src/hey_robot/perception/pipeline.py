from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hey_robot.media import LocalMediaStore
from hey_robot.perception.observation import DriverObservation, ObservationAsset
from hey_robot.protocol import ArtifactRef, ImageRef, RobotObservation


@dataclass(frozen=True)
class ObservationSchema:
    robot_id: str
    driver_type: str
    cameras: list[str]
    modalities: list[str]
    proprioception_dim: int | None = None
    metadata: dict[str, Any] | None = None


class ObservationPipeline:
    """Converts driver-local observations into protocol observations.

    Large arrays and binary-like assets are materialized into the media store.
    The bus-facing RobotObservation carries only small metadata plus media and
    artifact references.
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
