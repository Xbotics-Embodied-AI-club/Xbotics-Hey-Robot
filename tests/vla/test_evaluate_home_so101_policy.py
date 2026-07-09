from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module() -> Any:
    path = Path("scripts/vla/evaluate_home_so101_policy.py")
    spec = importlib.util.spec_from_file_location("evaluate_home_so101_policy", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVAL_MODULE = _load_module()


class FakeDriver:
    def __init__(self) -> None:
        self.last_arm_status = {"gripper_opening_pct": 10.0}
        self.positions = {
            "cube": (0.1, 0.2, 0.05),
            "target": (0.12, 0.22, 0.05),
            "far_target": (1.0, 1.0, 0.05),
        }

    def _body_position_base(self, name: str) -> tuple[float, float, float]:
        return self.positions[name]


def test_success_gripper_closed() -> None:
    assert EVAL_MODULE._success(FakeDriver(), mode="gripper_closed")


def test_success_object_lifted_uses_explicit_object_body() -> None:
    assert EVAL_MODULE._success(
        FakeDriver(),
        mode="object_lifted",
        object_body="cube",
        min_lift_m=0.03,
    )


def test_success_object_near_target_uses_distance_threshold() -> None:
    assert EVAL_MODULE._success(
        FakeDriver(),
        mode="object_near_target",
        object_body="cube",
        target_body="target",
        max_distance_m=0.05,
    )
    assert not EVAL_MODULE._success(
        FakeDriver(),
        mode="object_near_target",
        object_body="cube",
        target_body="far_target",
        max_distance_m=0.05,
    )


def test_success_requires_object_body_for_object_modes() -> None:
    with pytest.raises(ValueError, match="requires"):
        EVAL_MODULE._success(FakeDriver(), mode="object_lifted")
