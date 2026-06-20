from __future__ import annotations

from typing import Any

import numpy as np

from hey_robot.capability.vla.executor import VLAExecutor, _build_metrics
from hey_robot.capability.vla.io_adapter import VLAIOAdapter
from hey_robot.capability.vla.policy_client import FakePolicyClient
from hey_robot.capability.vla.schemas import VLAConfig, VLARequest


class _MockIO(VLAIOAdapter):
    """Configurable in-memory I/O adapter for executor tests."""

    def __init__(
        self,
        frames: dict[str, np.ndarray] | None = None,
        joint_state: np.ndarray | None = None,
    ) -> None:
        self.frames = frames or {"front": np.zeros((240, 320, 3), dtype=np.uint8)}
        self.joint_state = joint_state or np.zeros(6, dtype=np.float64)
        self.applied_actions: list[np.ndarray] = []
        self.advance_calls: list[float] = []
        self.reset_calls = 0
        self._ready = True

    def capture_frames(self, camera_names: list[str]) -> dict[str, np.ndarray]:
        return {
            n: self.frames.get(n, np.zeros((240, 320, 3), dtype=np.uint8))
            for n in camera_names
        }

    def read_joint_state(self, _arm: str) -> np.ndarray:
        return self.joint_state.copy()

    def apply_action(self, _arm: str, targets_rad: np.ndarray) -> None:
        self.applied_actions.append(targets_rad.copy())

    def advance(self, dt: float) -> None:
        self.advance_calls.append(dt)

    def reset(self) -> None:
        self.reset_calls += 1

    def ready(self) -> bool:
        return self._ready


def _config(**overrides: Any) -> VLAConfig:
    defaults: dict[str, Any] = {
        "fps": 50,
        "action_horizon": 2,
        "execution_time_sec": 0.2,
        "camera_names": ("front",),
        "arm": "right",
        "action_mode": "absolute_joint_position_rad",
        "task_prompt": "test",
    }
    defaults.update(overrides)
    return VLAConfig(**defaults)


class TestVLAExecutor:
    def test_successful_run_applies_actions_and_advances(self) -> None:
        io = _MockIO()
        policy = FakePolicyClient(action_horizon=2)
        executor = VLAExecutor(io, policy)

        result = executor.execute(VLARequest(config=_config()))

        assert result.success is True
        assert result.status == "completed"
        assert "steps" in result.metrics
        assert result.metrics["steps"] > 0
        assert len(io.applied_actions) == result.metrics["steps"]
        assert len(io.advance_calls) == result.metrics["steps"]

    def test_policy_server_unavailable(self) -> None:
        io = _MockIO()

        class UnpingablePolicy(FakePolicyClient):
            def ping(self) -> bool:
                return False

        executor = VLAExecutor(io, UnpingablePolicy(action_horizon=2))
        result = executor.execute(VLARequest(config=_config()))

        assert result.success is False
        assert result.failure_mode == "policy_server_unavailable"

    def test_cancel_before_execute_is_reset_by_execute(self) -> None:
        io = _MockIO()
        policy = FakePolicyClient(action_horizon=2)
        executor = VLAExecutor(io, policy)
        executor.cancel()
        # execute() resets _cancelled — a stale cancel should not poison execution
        result = executor.execute(VLARequest(config=_config()))

        assert result.success is True
        assert result.status == "completed"

    def test_cancel_during_execution_via_io_hook(self) -> None:
        io = _MockIO()
        policy = FakePolicyClient(action_horizon=16)
        executor = VLAExecutor(io, policy)

        original_read = io.read_joint_state
        call_count = [0]

        def _read_then_cancel(arm: str) -> np.ndarray:
            call_count[0] += 1
            if call_count[0] >= 3:
                executor.cancel()
            return original_read(arm)

        io.read_joint_state = _read_then_cancel  # type: ignore[method-assign]

        result = executor.execute(VLARequest(config=_config(execution_time_sec=5.0)))

        assert result.status == "cancelled"
        assert result.failure_mode == "cancelled"

    def test_no_frames_from_camera_fails_with_camera_render_failed(self) -> None:
        io = _MockIO()

        def empty_frames(_cameras: list[str]) -> dict[str, np.ndarray]:
            return {}

        io.capture_frames = empty_frames  # type: ignore[method-assign]

        policy = FakePolicyClient(action_horizon=2)
        executor = VLAExecutor(io, policy)

        result = executor.execute(VLARequest(config=_config()))

        assert result.success is False
        assert result.failure_mode == "camera_render_failed"

    def test_policy_exception_captured_as_execution_failed(self) -> None:
        io = _MockIO()

        class ExplodingPolicy(FakePolicyClient):
            def get_action(self, _observation: dict[str, Any]) -> dict[str, Any]:
                raise RuntimeError("inference crashed")

        executor = VLAExecutor(io, ExplodingPolicy(action_horizon=2))
        result = executor.execute(VLARequest(config=_config()))

        assert result.success is False
        assert result.failure_mode == "execution_failed"
        assert "inference crashed" in result.error

    def test_applied_actions_are_6d_radians(self) -> None:
        io = _MockIO()
        policy = FakePolicyClient(action_horizon=2)
        executor = VLAExecutor(io, policy)

        executor.execute(VLARequest(config=_config()))

        for action in io.applied_actions:
            assert action.shape == (6,)
            assert action.dtype == np.float64

    def test_respects_execution_time_deadline(self) -> None:
        io = _MockIO()
        policy = FakePolicyClient(action_horizon=100)
        executor = VLAExecutor(io, policy)

        result = executor.execute(
            VLARequest(config=_config(execution_time_sec=0.05, fps=100))
        )

        assert result.metrics["duration_sec"] < 1.0


class TestBuildMetrics:
    def test_includes_all_keys(self) -> None:
        cfg = VLAConfig(fps=30, arm="left", action_mode="delta_joint_position_rad")
        m = _build_metrics(100.0, 42, cfg)
        assert set(m.keys()) == {"duration_sec", "steps", "fps", "arm", "action_mode"}
        assert m["steps"] == 42
        assert m["fps"] == 30
        assert m["arm"] == "left"
        assert m["action_mode"] == "delta_joint_position_rad"
