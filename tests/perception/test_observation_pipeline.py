from __future__ import annotations

import asyncio

import numpy as np

from hey_robot.media import LocalMediaStore
from hey_robot.perception import (
    DriverObservation,
    ObservationAsset,
    ObservationPipeline,
    PerceptionService,
)
from hey_robot.protocol import Envelope


def test_observation_pipeline_materializes_driver_assets(tmp_path) -> None:
    media_store = LocalMediaStore(tmp_path)
    pipeline = ObservationPipeline(media_store)
    observation = DriverObservation(
        envelope=Envelope(robot_id="mock0"),
        frame_id=1,
        assets=[
            ObservationAsset(
                kind="image",
                role="camera",
                name="front",
                data=np.zeros((8, 8, 3), dtype=np.uint8),
            ),
            ObservationAsset(
                kind="json",
                role="policy_observation",
                name="policy_obs",
                data={"pixels": {"image": np.zeros((2, 2, 3), dtype=np.uint8)}},
                metadata={"artifact_type": "policy_observation"},
            ),
        ],
        metadata={"keep": "yes"},
    )

    materialized = pipeline.build(observation)

    assert len(materialized.images) == 1
    assert len(materialized.artifacts) == 1
    assert materialized.raw["keep"] == "yes"
    assert materialized.raw["image_quality"][0]["issue"] == "black_frame"
    assert media_store.resolve_image(materialized.images[0]).shape == (8, 8, 3)
    loaded = media_store.load_npz_artifact(materialized.artifacts[0])
    assert loaded["pixels"]["image"].dtype == np.uint8
    assert loaded["pixels"]["image"].shape == (2, 2, 3)
    assert materialized.artifacts[0].content_type == "application/x.numpy-npz"


def test_observation_pipeline_marks_valid_image_quality(tmp_path) -> None:
    media_store = LocalMediaStore(tmp_path)
    pipeline = ObservationPipeline(media_store)
    image = np.full((8, 8, 3), 80, dtype=np.uint8)

    materialized = pipeline.build(
        DriverObservation(
            envelope=Envelope(robot_id="mock0"),
            frame_id=1,
            assets=[
                ObservationAsset(kind="image", role="camera", name="front", data=image)
            ],
        )
    )

    assert materialized.raw["image_quality"][0]["valid"] is True
    assert materialized.raw["image_quality"][0]["issue"] is None


def test_perception_snapshot_treats_black_frame_as_no_valid_image(tmp_path) -> None:
    class Driver:
        robot_id = "mock0"

        async def observe(self) -> DriverObservation:
            return DriverObservation(
                envelope=Envelope(robot_id="mock0"),
                frame_id=1,
                assets=[
                    ObservationAsset(
                        kind="image",
                        role="camera",
                        name="front",
                        data=np.zeros((8, 8, 3), dtype=np.uint8),
                    )
                ],
            )

    snapshot = asyncio.run(
        PerceptionService(Driver(), LocalMediaStore(tmp_path)).refresh()
    )

    assert snapshot.has_images is False
    assert snapshot.summary()["valid_image_count"] == 0
    assert snapshot.summary()["image_count"] == 1


def test_perception_build_observation_preserves_valid_image_count(tmp_path) -> None:
    import numpy as np

    class Driver:
        robot_id = "mock0"

        async def observe(self) -> DriverObservation:
            raise AssertionError("build_observation must not call the driver")

    service = PerceptionService(Driver(), LocalMediaStore(tmp_path))
    driver_obs = DriverObservation(
        envelope=Envelope(robot_id="mock0"),
        frame_id=4,
        assets=[
            ObservationAsset(
                kind="image",
                role="camera",
                name="front",
                data=np.full((8, 8, 3), 128, dtype=np.uint8),
            )
        ],
    )

    observation = service.build_observation(driver_obs)

    assert observation.frame_id == 4
    assert len(observation.images) == 1
    snapshot = service.latest()
    assert snapshot is not None
    assert snapshot.has_images is True
    assert snapshot.summary()["image_count"] == 1
    assert snapshot.summary()["valid_image_count"] == 1


def test_observation_pipeline_reuses_last_image_ref_when_sampling(tmp_path) -> None:
    media_store = LocalMediaStore(tmp_path)
    pipeline = ObservationPipeline(media_store, image_save_every_n=2)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    pipeline.build(
        DriverObservation(
            envelope=Envelope(robot_id="mock0"),
            frame_id=1,
            assets=[
                ObservationAsset(kind="image", role="camera", name="front", data=image)
            ],
        )
    )
    second = pipeline.build(
        DriverObservation(
            envelope=Envelope(robot_id="mock0"),
            frame_id=2,
            assets=[
                ObservationAsset(kind="image", role="camera", name="front", data=image)
            ],
        )
    )
    third = pipeline.build(
        DriverObservation(
            envelope=Envelope(robot_id="mock0"),
            frame_id=3,
            assets=[
                ObservationAsset(kind="image", role="camera", name="front", data=image)
            ],
        )
    )

    assert len(list((tmp_path / "images" / "mock0" / "front").glob("*.jpg"))) == 2
    assert second.images[0].uri == third.images[0].uri
