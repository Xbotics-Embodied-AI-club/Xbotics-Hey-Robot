from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from hey_robot.capability.runtime.models import (
    CapabilityExecutionRequest,
)
from hey_robot.capability.vla.capability_client import VLACapabilityClient
from hey_robot.capability.vla.io_adapter import VLAIOAdapter
from hey_robot.config import CapabilityServiceSpec
from hey_robot.protocol import Envelope, SkillIntent


class _MockIO(VLAIOAdapter):
    def __init__(self, *, ready: bool = True) -> None:
        self._ready = ready

    def capture_frames(self, camera_names: list[str]) -> dict[str, np.ndarray]:
        return {n: np.zeros((240, 320, 3), dtype=np.uint8) for n in camera_names}

    def read_joint_state(self, _arm: str) -> np.ndarray:
        return np.zeros(6, dtype=np.float64)

    def apply_action(self, arm: str, targets_rad: np.ndarray) -> None:
        pass

    def advance(self, dt: float) -> None:
        pass

    def reset(self) -> None:
        pass

    def ready(self) -> bool:
        return self._ready


def _spec(**overrides: Any) -> CapabilityServiceSpec:
    defaults: dict[str, Any] = {
        "type": "vla_service",
        "robot_id": "xlerobot",
        "enabled": True,
        "skill_names": ["vla_manipulation"],
        "settings": {
            "policy_runtime": "fake",
            "arm": "right",
            "fps": 100,
            "action_horizon": 2,
            "execution_time": 0.1,
            "action_mode": "absolute_joint_position_rad",
            "task_prompt": "test",
            "cameras": ["front"],
            "camera_devices": {"front": 0},
            "camera_width": 320,
            "camera_height": 240,
            "camera_fps": 30,
            "camera_key_map": {"front": "camera1"},
        },
    }
    defaults.update(overrides)
    return CapabilityServiceSpec(**defaults)


def test_health_reports_online_when_io_ready() -> None:
    client = VLACapabilityClient("arm_vla", _spec(), io=_MockIO())
    health = asyncio.run(client.health())
    assert health.online is True
    assert health.loaded is True
    assert health.busy is False
    assert health.metrics["type"] == "vla_service"


def test_health_reports_offline_when_io_not_ready() -> None:
    client = VLACapabilityClient("arm_vla", _spec(), io=_MockIO(ready=False))
    health = asyncio.run(client.health())
    assert health.online is False


def test_health_reports_offline_when_io_is_none() -> None:
    client = VLACapabilityClient("arm_vla", _spec())
    health = asyncio.run(client.health())
    assert health.online is False


def test_execute_rejects_when_already_busy() -> None:
    client = VLACapabilityClient("arm_vla", _spec(), io=_MockIO())

    intent = SkillIntent(
        envelope=Envelope(robot_id="xlerobot"),
        skill_id="s1",
        name="vla_manipulation",
        objective="grasp",
    )
    request = CapabilityExecutionRequest(
        service_id="arm_vla",
        intent=intent,
        contract=None,
        timeout_sec=5.0,
    )

    client._busy = True
    result = asyncio.run(client.execute(request))

    assert result.success is False
    assert result.failure_mode == "capability_busy"


def test_cancel_before_executor_created_does_not_raise() -> None:
    client = VLACapabilityClient("arm_vla", _spec(), io=_MockIO())
    asyncio.run(client.cancel("any-skill"))


def test_set_io_makes_health_online() -> None:
    client = VLACapabilityClient("arm_vla", _spec())
    assert asyncio.run(client.health()).online is False
    client.set_io(_MockIO())
    assert asyncio.run(client.health()).online is True


def test_execute_runs_to_completion_with_fake_policy() -> None:
    client = VLACapabilityClient("arm_vla", _spec(), io=_MockIO())

    intent = SkillIntent(
        envelope=Envelope(robot_id="xlerobot", episode_id="ep1"),
        skill_id="s1",
        name="vla_manipulation",
        objective="grasp the cube",
        arguments={"task_prompt": "pick the red cube"},
    )
    request = CapabilityExecutionRequest(
        service_id="arm_vla",
        intent=intent,
        contract=None,
        timeout_sec=5.0,
    )

    result = asyncio.run(client.execute(request))
    assert result.success is True
    assert result.status == "completed"
    assert result.metrics is not None
    assert "steps" in result.metrics
    assert result.metrics["steps"] > 0


def test_execute_fails_when_no_io_set() -> None:
    client = VLACapabilityClient("arm_vla", _spec())

    intent = SkillIntent(
        envelope=Envelope(robot_id="xlerobot"),
        skill_id="s1",
        name="vla_manipulation",
        objective="grasp",
    )
    request = CapabilityExecutionRequest(
        service_id="arm_vla",
        intent=intent,
        contract=None,
        timeout_sec=5.0,
    )

    result = asyncio.run(client.execute(request))
    assert result.success is False
    assert result.failure_mode == "execution_failed"
    assert "no i/o" in result.error.lower()


def test_execute_falls_back_to_objective_when_no_task_prompt_arg() -> None:
    client = VLACapabilityClient("arm_vla", _spec(), io=_MockIO())

    intent = SkillIntent(
        envelope=Envelope(robot_id="xlerobot", episode_id="ep1"),
        skill_id="s1",
        name="vla_manipulation",
        objective="pick something up",
    )
    request = CapabilityExecutionRequest(
        service_id="arm_vla",
        intent=intent,
        contract=None,
        timeout_sec=5.0,
    )

    result = asyncio.run(client.execute(request))
    assert result.success is True
