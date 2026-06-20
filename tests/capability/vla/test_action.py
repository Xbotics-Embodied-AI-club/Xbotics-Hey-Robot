from __future__ import annotations

import numpy as np
import pytest

from hey_robot.capability.vla.action import (
    _extract_arm_action_vector,
    decode_action_chunk,
    get_action_horizon,
)


def _joint_state() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _groot_chunk(b: int = 1, t: int = 16, arm_dim: int = 5, grip_dim: int = 1) -> dict:
    return {
        "single_arm": np.zeros((b, t, arm_dim), dtype=np.float32),
        "gripper": np.full((b, t, grip_dim), 0.5, dtype=np.float32),
    }


def _legacy_chunk(t: int = 16, dim: int = 6) -> dict:
    return {"action": np.ones((t, dim), dtype=np.float64)}


class TestDecodeActionChunk:
    # ── GR00T format ──────────────────────────────────────────────────

    def test_groot_absolute_mode_returns_raw_vector(self) -> None:
        chunk = _groot_chunk()
        targets = decode_action_chunk(
            chunk,
            t=0,
            action_mode="absolute_joint_position_rad",
            current_joint_state_rad=_joint_state(),
        )
        assert targets.shape == (6,)
        assert targets.dtype == np.float64
        np.testing.assert_array_equal(targets[:5], np.zeros(5))
        np.testing.assert_array_equal(targets[5:], [0.5])

    def test_groot_delta_mode_adds_to_current_state(self) -> None:
        chunk = _groot_chunk()
        current = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.0], dtype=np.float64)
        targets = decode_action_chunk(
            chunk,
            t=0,
            action_mode="delta_joint_position_rad",
            current_joint_state_rad=current,
        )
        # arm dims (0) + current, gripper (0.5) + current
        np.testing.assert_array_equal(targets[:5], current[:5])
        assert targets[5] == 0.5

    def test_groot_normalized_mode_maps_to_joint_range(self) -> None:
        chunk = {
            "single_arm": np.zeros((1, 1, 5), dtype=np.float32),
            "gripper": np.zeros((1, 1, 1), dtype=np.float32),
        }
        targets = decode_action_chunk(
            chunk,
            t=0,
            action_mode="normalized_joint_position",
            current_joint_state_rad=_joint_state(),
        )
        # zero maps to midway in [-1,+1] → (lo + hi) / 2
        lo = np.array([-3.14, -0.1, -0.2, -1.8, -3.14, 0.0])
        hi = np.array([3.14, 3.45, 3.14, 1.8, 3.14, 0.08])
        expected = (lo + hi) / 2.0
        np.testing.assert_array_almost_equal(targets, expected, decimal=5)

    def test_groot_without_batch_dim(self) -> None:
        chunk = {
            "single_arm": np.zeros((16, 5), dtype=np.float32),
            "gripper": np.full((16, 1), 0.7, dtype=np.float32),
        }
        targets = decode_action_chunk(
            chunk,
            t=3,
            action_mode="absolute_joint_position_rad",
            current_joint_state_rad=_joint_state(),
        )
        assert targets[5] == pytest.approx(0.7)

    # ── Legacy format ─────────────────────────────────────────────────

    def test_legacy_action_key_flat(self) -> None:
        chunk = {"action": np.tile(np.arange(6, dtype=np.float64), (10, 1))}
        targets = decode_action_chunk(
            chunk,
            t=2,
            action_mode="absolute_joint_position_rad",
            current_joint_state_rad=_joint_state(),
        )
        np.testing.assert_array_equal(targets, np.arange(6, dtype=np.float64))

    def test_legacy_actions_key(self) -> None:
        chunk = {"actions": np.tile(np.arange(6, dtype=np.float64), (10, 1))}
        targets = decode_action_chunk(
            chunk,
            t=5,
            action_mode="absolute_joint_position_rad",
            current_joint_state_rad=_joint_state(),
        )
        np.testing.assert_array_equal(targets, np.arange(6, dtype=np.float64))

    def test_legacy_1d_vector_at_t0(self) -> None:
        chunk = {"action": np.arange(6, dtype=np.float64)}
        targets = decode_action_chunk(
            chunk,
            t=0,
            action_mode="absolute_joint_position_rad",
            current_joint_state_rad=_joint_state(),
        )
        assert targets.shape == (6,)

    # ── Errors ────────────────────────────────────────────────────────

    def test_raises_on_missing_action_key(self) -> None:
        with pytest.raises(ValueError, match="missing recognised action key"):
            decode_action_chunk(
                {"unknown": np.zeros((16, 6))},
                t=0,
                action_mode="absolute_joint_position_rad",
                current_joint_state_rad=_joint_state(),
            )


class TestGetActionHorizon:
    def test_groot_format_3d(self) -> None:
        assert get_action_horizon(_groot_chunk(t=16)) == 16

    def test_groot_format_2d(self) -> None:
        chunk = {
            "single_arm": np.zeros((16, 5), dtype=np.float32),
            "gripper": np.zeros((16, 1), dtype=np.float32),
        }
        assert get_action_horizon(chunk) == 16

    def test_legacy_action(self) -> None:
        assert get_action_horizon({"action": np.zeros((10, 6))}) == 10

    def test_legacy_action_chunk(self) -> None:
        assert get_action_horizon({"action_chunk": np.zeros((8, 6))}) == 8

    def test_priority_groot_over_legacy(self) -> None:
        chunk = {
            "single_arm": np.zeros((1, 16, 5), dtype=np.float32),
            "gripper": np.zeros((1, 16, 1), dtype=np.float32),
            "action": np.zeros((100, 6)),
        }
        assert get_action_horizon(chunk) == 16

    def test_empty_dict_returns_zero(self) -> None:
        assert get_action_horizon({}) == 0

    def test_1d_vector_returns_zero(self) -> None:
        assert get_action_horizon({"action": np.zeros(6)}) == 0


class TestExtractArmActionVector:
    def test_groot_batch_time_dim(self) -> None:
        chunk = {
            "single_arm": np.arange(80, dtype=np.float32).reshape(1, 16, 5),
            "gripper": np.full((1, 16, 1), 0.9, dtype=np.float32),
        }
        vec = _extract_arm_action_vector(chunk, 0)
        assert vec is not None
        assert vec.shape == (6,)
        np.testing.assert_array_almost_equal(vec[:5], np.arange(5, dtype=np.float64))
        assert vec[5] == pytest.approx(0.9)

    def test_groot_t_out_of_range_returns_none(self) -> None:
        chunk = _groot_chunk(t=4)
        assert _extract_arm_action_vector(chunk, 10) is None

    def test_legacy_action_2d(self) -> None:
        chunk = {"action": np.tile(np.arange(6, dtype=np.float64), (10, 1))}
        vec = _extract_arm_action_vector(chunk, 3)
        assert vec is not None
        np.testing.assert_array_equal(vec, np.arange(6, dtype=np.float64))

    def test_legacy_action_chunk_key(self) -> None:
        chunk = {"action_chunk": np.array([[1.0] * 6, [2.0] * 6], dtype=np.float64)}
        vec = _extract_arm_action_vector(chunk, 1)
        assert vec is not None
        np.testing.assert_array_equal(vec, [2.0] * 6)

    def test_legacy_output_key(self) -> None:
        chunk = {"output": np.array([[1.0] * 6], dtype=np.float64)}
        vec = _extract_arm_action_vector(chunk, 0)
        assert vec is not None

    def test_legacy_t_out_of_range_returns_none(self) -> None:
        assert _extract_arm_action_vector({"action": np.zeros((4, 6))}, 5) is None
