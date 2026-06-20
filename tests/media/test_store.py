from __future__ import annotations

import numpy as np
import pytest

from hey_robot.media import LocalMediaStore, MediaResolver, MediaSigner, MediaStoreError
from hey_robot.protocol import ArtifactRef, ImageRef


def test_local_media_store_image_roundtrip(tmp_path) -> None:
    store = LocalMediaStore(tmp_path)
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 200

    ref = store.put_image(image, robot_id="r1", frame_id=3, camera="front")
    restored = store.resolve_image(ref)

    assert ref.uri.startswith("media://local/images/")
    assert restored.shape == image.shape


def test_local_media_store_npz_artifact_preserves_numpy_types(tmp_path) -> None:
    store = LocalMediaStore(tmp_path)
    payload = {
        "pixels": {"image": np.zeros((2, 3, 3), dtype=np.uint8)},
        "robot_state": {"eef": {"pos": np.array([1.0, 2.0, 3.0], dtype=np.float32)}},
    }

    ref = store.put_npz_artifact(
        payload,
        artifact_type="policy_observation",
        role="policy_observation",
        robot_id="mock0",
        frame_id=1,
    )
    restored = store.load_npz_artifact(ref)

    assert ref.content_type == "application/x.numpy-npz"
    assert restored["pixels"]["image"].dtype == np.uint8
    assert restored["pixels"]["image"].shape == (2, 3, 3)
    assert restored["robot_state"]["eef"]["pos"].dtype == np.float32


def test_local_media_store_recreates_missing_runtime_dirs(tmp_path) -> None:
    store = LocalMediaStore(tmp_path / "media")
    payload = {"pixels": {"image": np.zeros((2, 2, 3), dtype=np.uint8)}}
    artifact_dir = tmp_path / "media" / "artifacts" / "mock0" / "policy_observation"
    artifact_dir.mkdir(parents=True)
    artifact_dir.rmdir()

    ref = store.put_npz_artifact(
        payload,
        artifact_type="policy_observation",
        role="policy_observation",
        robot_id="mock0",
        frame_id=0,
        name="policy_raw_obs",
    )

    assert store.path_for_uri(ref.uri).is_file()
    restored = store.load_npz_artifact(ref)
    assert restored["pixels"]["image"].shape == (2, 2, 3)


def test_local_media_store_retains_only_recent_images(tmp_path) -> None:
    store = LocalMediaStore(tmp_path, max_items=2)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    for frame_id in range(4):
        store.put_image(image, robot_id="xlerobot", frame_id=frame_id, camera="front")

    image_files = sorted((tmp_path / "images" / "xlerobot" / "front").glob("*.jpg"))
    assert len(image_files) == 2
    assert all(
        ("frame_00000002" in path.name or "frame_00000003" in path.name)
        for path in image_files
    )


def test_local_media_store_json_bytes_recent_and_resolver_roundtrip(tmp_path) -> None:
    store = LocalMediaStore(tmp_path, max_items=10)
    json_ref = store.put_json_artifact(
        {"objects": ["bottle", "trash"]},
        artifact_type="scene",
        role="observation",
        robot_id="robot/1",
        frame_id=2,
        name="front scene",
    )
    bytes_ref = store.put_bytes(
        b"hello",
        media_type="operator_upload",
        content_type="application/json",
        name="note.json",
        metadata={"source": "test"},
    )

    assert store.load_artifact(json_ref) == {"objects": ["bottle", "trash"]}
    assert bytes_ref.uri.startswith("media://local/uploads/operator_upload/")
    assert bytes_ref.metadata == {"source": "test"}

    recent_all = store.recent(limit=10)
    assert {item["uri"] for item in recent_all} >= {json_ref.uri, bytes_ref.uri}
    assert store.recent(kind="missing", limit=1) == []

    resolver = MediaResolver(store)
    assert resolver.load_artifact(
        [json_ref], role="observation", artifact_type="scene"
    ) == {"objects": ["bottle", "trash"]}
    assert resolver.load_artifact([json_ref], role="other") is None


def test_local_media_store_rejects_unsafe_uris_and_unsupported_content(
    tmp_path,
) -> None:
    store = LocalMediaStore(tmp_path)

    with pytest.raises(MediaStoreError, match="unsupported image content type"):
        store.put_image(
            np.zeros((2, 2, 3), dtype=np.uint8),
            robot_id="r1",
            frame_id=1,
            content_type="image/webp",
        )
    with pytest.raises(MediaStoreError, match="unsupported media content type"):
        store.put_bytes(b"x", media_type="upload", content_type="text/plain")
    with pytest.raises(MediaStoreError, match="unsupported media uri"):
        store.path_for_uri("file:///tmp/image.jpg")
    with pytest.raises(MediaStoreError, match="unsafe media uri"):
        store.path_for_uri("media://local/images/../secret.jpg")
    with pytest.raises(MediaStoreError, match="path escapes media root"):
        store.uri_for_path(tmp_path.parent / "outside.jpg")

    bad_json = ArtifactRef(
        uri="media://local/missing.json",
        artifact_type="bad",
        content_type="application/json",
    )
    with pytest.raises(MediaStoreError, match="media object not found"):
        store.load_json_artifact(bad_json)

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image_ref = store.put_image(image, robot_id="r1", frame_id=2)
    unsupported_artifact = ArtifactRef(
        uri=image_ref.uri,
        artifact_type="image",
        content_type="image/jpeg",
    )
    with pytest.raises(MediaStoreError, match="unsupported artifact content type"):
        store.load_artifact(unsupported_artifact)


def test_local_media_store_image_normalization_and_resolver_skip_failures(
    tmp_path,
) -> None:
    store = LocalMediaStore(tmp_path)
    grayscale = np.ones((3, 4), dtype=np.float32) * 0.5
    rgba = np.zeros((1, 3, 4, 4), dtype=np.uint8)
    rgba[..., 0] = 255

    gray_ref = store.put_image(
        grayscale, robot_id="r1", frame_id=1, content_type="image/png"
    )
    rgba_ref = store.put_image(
        rgba, robot_id="r1", frame_id=2, content_type="image/png"
    )

    assert store.resolve_image(gray_ref).shape == (3, 4, 3)
    assert store.resolve_image(rgba_ref).shape == (3, 4, 3)

    resolver = MediaResolver(store)
    missing = ImageRef(uri="media://local/images/missing.png")
    assert len(resolver.resolve_images([missing, gray_ref])) == 1

    with pytest.raises(MediaStoreError, match="expected image"):
        store.put_image(
            np.zeros((1, 2, 3, 4, 5), dtype=np.uint8), robot_id="r1", frame_id=3
        )


def test_local_media_store_typed_npz_preserves_containers_scalars_and_rejects_unknown_types(
    tmp_path,
) -> None:
    store = LocalMediaStore(tmp_path)
    payload = {
        "tuple": (np.array([1, 2], dtype=np.int16), "done"),
        "list": [np.float32(1.5), None, True],
    }

    ref = store.put_npz_artifact(payload, artifact_type="policy", role="observation")
    restored = store.load_artifact(ref)

    assert isinstance(restored["tuple"], tuple)
    assert restored["tuple"][0].dtype == np.int16
    assert restored["list"] == [pytest.approx(1.5), None, True]

    with pytest.raises(MediaStoreError, match="cannot encode set"):
        store.put_npz_artifact({"bad": {1, 2}}, artifact_type="bad")

    not_npz = ArtifactRef(
        uri=ref.uri, artifact_type="policy", content_type="application/json"
    )
    with pytest.raises(MediaStoreError, match="artifact is not typed NPZ"):
        store.load_npz_artifact(not_npz)


def test_media_signer_signs_verifies_and_rejects_tampering(tmp_path) -> None:
    store = LocalMediaStore(tmp_path)
    ref = store.put_bytes(
        b"hello", media_type="upload", content_type="application/octet-stream"
    )
    signer = MediaSigner(store, secret=b"unit-test-key", route_prefix="/media/")

    signed = signer.sign(ref.uri)

    assert signed is not None
    _, signature, payload = [part for part in signed.split("/") if part]
    assert signer.verify(signature, payload) == ref.uri
    assert signer.sign("media://local/missing.bin") is None

    with pytest.raises(MediaStoreError, match="invalid media signature"):
        signer.verify("bad", payload)
