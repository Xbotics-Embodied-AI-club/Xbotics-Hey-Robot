from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module() -> Any:
    path = Path("scripts/vla/smoke_lerobot_policy_endpoint.py")
    spec = importlib.util.spec_from_file_location("smoke_lerobot_policy_endpoint", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SMOKE_MODULE = _load_module()


def _valid_response() -> dict[str, Any]:
    return {
        "action_chunk": {
            "kind": "action_chunk",
            "action_space": "xlerobot_single_arm_joint",
            "actions": [
                {
                    "joints": {
                        "shoulder_pan": 0.0,
                        "shoulder_lift": 0.1,
                        "elbow_flex": 0.2,
                        "wrist_flex": 0.3,
                        "wrist_roll": 0.4,
                    },
                    "gripper": 0.5,
                }
            ],
            "done": False,
        }
    }


def test_validate_action_chunk_accepts_hey_robot_schema() -> None:
    chunk = SMOKE_MODULE._validate_action_chunk(_valid_response())

    assert chunk["action_space"] == "xlerobot_single_arm_joint"


def test_validate_action_chunk_rejects_missing_joint() -> None:
    response = _valid_response()
    del response["action_chunk"]["actions"][0]["joints"]["wrist_roll"]

    with pytest.raises(ValueError, match="missing joints"):
        SMOKE_MODULE._validate_action_chunk(response)


def test_validate_action_chunk_rejects_gripper_out_of_range() -> None:
    response = _valid_response()
    response["action_chunk"]["actions"][0]["gripper"] = 1.5

    with pytest.raises(ValueError, match="gripper"):
        SMOKE_MODULE._validate_action_chunk(response)


def test_health_url_matches_predict_endpoint() -> None:
    assert (
        SMOKE_MODULE._health_url("http://127.0.0.1:18080/predict")
        == "http://127.0.0.1:18080/health"
    )
