from __future__ import annotations

import numpy as np

from hey_robot.capability.vla.observation import (
    _recursive_add_extra_dim,
    build_groot_observation,
)


def _frame(h: int = 240, w: int = 320) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _joint_state() -> np.ndarray:
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)


class TestBuildGrootObservation:
    def test_produces_expected_top_level_keys(self) -> None:
        obs = build_groot_observation(
            frames={"front": _frame()},
            joint_state_rad=_joint_state(),
            task_prompt="pick up the cube",
        )
        assert set(obs.keys()) == {"video", "state", "language"}

    def test_video_uses_camera_key_map_and_stores_uint8(self) -> None:
        obs = build_groot_observation(
            frames={"front": _frame()},
            joint_state_rad=_joint_state(),
            task_prompt="task",
            camera_key_map={"front": "camera1"},
        )
        assert "front" not in obs["video"]
        assert "camera1" in obs["video"]
        assert obs["video"]["camera1"].dtype == np.uint8

    def test_state_splits_arm_and_gripper(self) -> None:
        obs = build_groot_observation(
            frames={"front": _frame()},
            joint_state_rad=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.8], dtype=np.float64),
            task_prompt="task",
        )
        assert "single_arm" in obs["state"]
        assert "gripper" in obs["state"]
        np.testing.assert_array_almost_equal(
            obs["state"]["single_arm"].ravel(), [1, 2, 3, 4, 5]
        )
        np.testing.assert_array_almost_equal(obs["state"]["gripper"].ravel(), [0.8])

    def test_state_key_map_renames_sub_keys(self) -> None:
        obs = build_groot_observation(
            frames={"front": _frame()},
            joint_state_rad=_joint_state(),
            task_prompt="task",
            state_key_map={"single_arm": "arm_state", "gripper": "grip"},
        )
        assert "arm_state" in obs["state"]
        assert "grip" in obs["state"]

    def test_language_block_contains_task_description(self) -> None:
        obs = build_groot_observation(
            frames={"front": _frame()},
            joint_state_rad=_joint_state(),
            task_prompt="open the drawer",
        )
        lang = obs["language"]
        assert lang["annotation.human.task_description"] == [["open the drawer"]]

    def test_custom_language_key(self) -> None:
        obs = build_groot_observation(
            frames={"front": _frame()},
            joint_state_rad=_joint_state(),
            task_prompt="task",
            language_key="instruction",
        )
        assert "language" not in obs
        assert "instruction" in obs
        assert obs["instruction"]["annotation.human.task_description"] == [["task"]]

    def test_batch_time_dims_prepend_two_leading_axes(self) -> None:
        obs = build_groot_observation(
            frames={"front": _frame(240, 320)},
            joint_state_rad=_joint_state(),
            task_prompt="task",
            add_batch_time_dims=True,
        )
        assert obs["video"]["front"].shape == (1, 1, 240, 320, 3)
        assert obs["state"]["single_arm"].shape == (1, 1, 5)
        assert obs["state"]["gripper"].shape == (1, 1, 1)

    def test_no_batch_time_dims_preserves_original_shapes(self) -> None:
        obs = build_groot_observation(
            frames={"front": _frame(240, 320)},
            joint_state_rad=_joint_state(),
            task_prompt="task",
            add_batch_time_dims=False,
        )
        assert obs["video"]["front"].shape == (240, 320, 3)
        assert obs["state"]["single_arm"].shape == (5,)
        assert obs["state"]["gripper"].shape == (1,)

    def test_multiple_cameras(self) -> None:
        obs = build_groot_observation(
            frames={"front": _frame(100, 200), "wrist": _frame(100, 200)},
            joint_state_rad=_joint_state(),
            task_prompt="task",
        )
        assert set(obs["video"].keys()) == {"front", "wrist"}


class TestRecursiveAddExtraDim:
    def test_adds_newaxis_to_ndarray_leaves(self) -> None:
        d: dict = {"a": np.array([1, 2, 3], dtype=np.float32)}
        result = _recursive_add_extra_dim(d)
        assert result["a"].shape == (1, 3)

    def test_recurses_into_nested_dicts(self) -> None:
        d: dict = {"outer": {"inner": np.array([1.0], dtype=np.float32)}}
        result = _recursive_add_extra_dim(d)
        assert result["outer"]["inner"].shape == (1, 1)

    def test_wraps_scalar_leaves_in_lists(self) -> None:
        d: dict = {"key": "value"}
        result = _recursive_add_extra_dim(d)
        assert result["key"] == ["value"]

    def test_returns_same_dict_type(self) -> None:
        d: dict = {"x": np.array([1])}
        result = _recursive_add_extra_dim(d)
        assert isinstance(result, dict)
