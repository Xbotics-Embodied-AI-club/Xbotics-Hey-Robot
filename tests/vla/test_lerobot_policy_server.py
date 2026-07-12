from __future__ import annotations

import base64
import importlib.util
import io
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

pytest.importorskip("starlette", reason="VLA HTTP server dependency is not installed")


def _load_server_module() -> Any:
    path = Path("scripts/vla/serve_lerobot_policy.py")
    spec = importlib.util.spec_from_file_location("serve_lerobot_policy", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SERVER_MODULE = _load_server_module()


def _image_data() -> str:
    image = Image.new("RGB", (8, 8), color=(10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _server() -> Any:
    return SERVER_MODULE.LeRobotPolicyServer(
        policy_type="custom_policy",
        checkpoint="checkpoint",
        device="cpu",
        image_size=(8, 8),
        camera_features={
            "front": "observation.images.front",
            "right_wrist": "observation.images.handeye",
        },
        state_key="observation.state",
        action_units="rad",
        action_scale=1.0,
        gripper_index=5,
    )


def test_load_policy_class_accepts_any_factory_registered_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = object()
    policies_module = types.ModuleType("lerobot.policies")
    factory_module = types.ModuleType("lerobot.policies.factory")
    factory_module.get_policy_class = lambda policy_type: (  # type: ignore[attr-defined]
        expected if policy_type == "custom_policy" else None
    )
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.policies", policies_module)
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory_module)

    assert SERVER_MODULE._load_policy_class("custom_policy") is expected


def test_load_policy_class_does_not_fallback_to_policy_specific_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_module = types.ModuleType("lerobot.policies.factory")

    def raise_unknown(policy_type: str) -> None:
        raise KeyError(policy_type)

    factory_module.get_policy_class = raise_unknown  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(
        sys.modules, "lerobot.policies", types.ModuleType("lerobot.policies")
    )
    monkeypatch.setitem(
        sys.modules,
        "lerobot.policies.smolvla.modeling_smolvla",
        types.ModuleType("lerobot.policies.smolvla.modeling_smolvla"),
    )
    monkeypatch.setitem(
        sys.modules,
        "lerobot.policies.factory",
        factory_module,
    )

    with pytest.raises(KeyError):
        SERVER_MODULE._load_policy_class("smolvla")


def test_build_batch_uses_explicit_camera_and_state_schema() -> None:
    request = SERVER_MODULE.PredictRequest(
        task="pick up the object",
        observation={
            "images": [
                {"camera": "front", "data": _image_data()},
                {"camera": "right_wrist", "data": _image_data()},
            ],
            "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.75],
        },
    )

    batch = _server()._build_batch(request)

    assert batch["observation.images.front"].shape == (1, 3, 8, 8)
    assert batch["observation.images.handeye"].shape == (1, 3, 8, 8)
    assert torch.allclose(
        batch["observation.state"],
        torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.75]]),
    )
    assert batch["task"] == ["pick up the object"]


def test_build_batch_rejects_missing_explicit_state() -> None:
    request = SERVER_MODULE.PredictRequest(
        observation={
            "images": [{"camera": "front", "data": _image_data()}],
        }
    )

    with pytest.raises(ValueError, match=r"observation\.state"):
        _server()._build_batch(request)


def test_action_conversion_produces_hey_robot_action_chunk() -> None:
    chunk = _server()._to_action_chunk([0.1, 0.2, 0.3, 0.4, 0.5, 1.2])

    assert chunk["kind"] == "action_chunk"
    assert chunk["action_space"] == "xlerobot_single_arm_joint"
    assert chunk["actions"][0]["joints"]["shoulder_pan"] == 0.1
    assert chunk["actions"][0]["gripper"] == 1.0
