from __future__ import annotations

import numpy as np

from hey_robot.media import LocalMediaStore, MediaResolver


def test_image_resolver_resolves_many(tmp_path) -> None:
    store = LocalMediaStore(tmp_path)
    ref = store.put_image(np.ones((4, 5, 3), dtype=np.uint8), robot_id="r", frame_id=1)

    images = MediaResolver(store).resolve_images([ref])

    assert len(images) == 1
    assert images[0].shape == (4, 5, 3)
