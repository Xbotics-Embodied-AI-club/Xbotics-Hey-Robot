from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from hey_robot.config import DeploymentConfig
from hey_robot.foundation.backends.vln.internvla_n1_system2 import (
    InternVLAN1System2Executor,
)
from hey_robot.skill_os.builtins.navigation_adapter import planner_output_to_primitive


def _spec(settings: dict | None = None):
    config = DeploymentConfig.from_dict(
        {
            "capability_services": {
                "vln_nav": {
                    "type": "vln_service",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "target": "127.0.0.1:9091",
                    "skill_names": ["navigate_to", "approach_object"],
                    "backend": "internvla_n1_system2",
                    "control_mode": "planner_only",
                    "mock_mode": True,
                    **dict(settings or {}),
                }
            }
        }
    )
    return config.capability_services["vln_nav"]


class _FakeS2Model:
    def __init__(self, output) -> None:
        self.output = output
        self.calls: list[dict] = []

    def s2_step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        self.calls.append(
            {
                "rgb": rgb,
                "depth": depth,
                "pose": pose,
                "instruction": instruction,
                "intrinsic": intrinsic,
                "look_down": look_down,
            }
        )
        return self.output


def _real_spec(settings: dict | None = None):
    return _spec(
        {"mock_mode": False, "model_path": "dummy-model", **dict(settings or {})}
    )


def _write_rgb(path, *, size=(8, 6)) -> None:
    data = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    data[:, :, 0] = 128
    Image.fromarray(data).save(path)


def test_internvla_n1_system2_mock_health_is_loaded() -> None:
    executor = InternVLAN1System2Executor("vln_nav", _spec())

    health = executor.health()

    assert health["online"] is True
    assert health["loaded"] is True
    assert health["metrics"]["backend"] == "internvla_n1_system2"
    assert health["metrics"]["control_mode"] == "planner_only"
    assert health["metrics"]["mock_mode"] is True


def test_internvla_n1_system2_mock_returns_center_pixel_goal() -> None:
    executor = InternVLAN1System2Executor(
        "vln_nav",
        _spec({"image_width": 640, "image_height": 480}),
    )

    result = executor.execute({"arguments": {"target": "desk"}})

    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["metrics"]["vln"]["mode"] == "pixel_goal"
    assert result["metrics"]["vln"]["pixel_goal"] == [320, 240]
    assert result["metrics"]["vln"]["stop"] is False


def test_internvla_n1_system2_mock_can_return_stop() -> None:
    executor = InternVLAN1System2Executor("vln_nav", _spec())

    result = executor.execute({"arguments": {"instruction": "stop when done"}})

    assert result["success"] is True
    assert result["metrics"]["vln"]["mode"] == "stop"
    assert result["metrics"]["vln"]["stop"] is True


def test_internvla_n1_system2_rejects_non_planner_control_mode() -> None:
    executor = InternVLAN1System2Executor(
        "vln_nav",
        _spec({"control_mode": "direct_velocity"}),
    )

    result = executor.execute({"arguments": {"target": "desk"}})

    assert result["success"] is False
    assert result["failure_mode"] == "unsupported_control_mode"


def test_internvla_n1_system2_real_path_adapts_image_path_and_pixel_output(
    tmp_path,
) -> None:
    image_path = tmp_path / "front.png"
    _write_rgb(image_path)
    model = _FakeS2Model(SimpleNamespace(output_pixel=np.asarray([3, 4])))
    executor = InternVLAN1System2Executor("vln_nav", _real_spec({"hfov": 90}))
    executor._model = model

    result = executor.execute(
        {
            "objective": "go to the desk",
            "arguments": {
                "image_path": str(image_path),
                "pose": [1.0, 2.0, 0.5],
                "look_down": True,
            },
        }
    )

    assert result["success"] is True
    assert result["metrics"]["vln"]["mode"] == "pixel_goal"
    assert result["metrics"]["vln"]["pixel_goal"] == [3, 4]
    assert result["metrics"]["vln"]["image_source"] == str(image_path)
    call = model.calls[0]
    assert call["rgb"].shape == (6, 8, 3)
    assert call["depth"].shape == (6, 8)
    assert call["pose"] == (1.0, 2.0, 0.5)
    assert call["instruction"] == "go to the desk"
    assert call["intrinsic"].shape == (4, 4)
    assert call["look_down"] is True


def test_internvla_n1_system2_real_path_selects_matching_observation_camera(
    tmp_path,
) -> None:
    front = tmp_path / "front.png"
    wrist = tmp_path / "wrist.png"
    _write_rgb(front)
    _write_rgb(wrist)
    model = _FakeS2Model(SimpleNamespace(output_pixel=np.asarray([1, 2])))
    executor = InternVLAN1System2Executor("vln_nav", _real_spec({"camera": "front"}))
    executor._model = model

    result = executor.execute(
        {
            "arguments": {
                "target": "charging dock",
                "observation": {
                    "images": [
                        {"camera": "wrist", "uri": str(wrist)},
                        {"camera": "front", "uri": str(front)},
                    ]
                },
            }
        }
    )

    assert result["success"] is True
    assert result["metrics"]["vln"]["image_source"] == str(front)


def test_internvla_n1_system2_real_path_clamps_out_of_bounds_pixel(tmp_path) -> None:
    image_path = tmp_path / "front.png"
    _write_rgb(image_path, size=(8, 6))
    model = _FakeS2Model(SimpleNamespace(output_pixel=np.asarray([99, -3])))
    executor = InternVLAN1System2Executor("vln_nav", _real_spec())
    executor._model = model

    result = executor.execute(
        {"arguments": {"target": "desk", "image_path": str(image_path)}}
    )

    assert result["success"] is True
    assert result["metrics"]["vln"]["pixel_goal"] == [7, 0]
    assert "clamped" in result["metrics"]["vln"]["reason"]


def test_internvla_n1_system2_real_path_maps_stop_action(tmp_path) -> None:
    image_path = tmp_path / "front.png"
    _write_rgb(image_path)
    model = _FakeS2Model(SimpleNamespace(output_action=[0], output_pixel=None))
    executor = InternVLAN1System2Executor("vln_nav", _real_spec())
    executor._model = model

    result = executor.execute(
        {"arguments": {"target": "desk", "image_path": str(image_path)}}
    )

    assert result["success"] is True
    assert result["metrics"]["vln"]["mode"] == "stop"
    assert result["metrics"]["vln"]["stop"] is True


def test_internvla_n1_system2_real_path_requires_image() -> None:
    executor = InternVLAN1System2Executor("vln_nav", _real_spec())
    executor._model = _FakeS2Model(SimpleNamespace(output_pixel=np.asarray([1, 2])))

    result = executor.execute({"arguments": {"target": "desk"}})

    assert result["success"] is False
    assert result["failure_mode"] == "image_unavailable"


def test_internvla_n1_system2_real_path_maps_non_stop_action_to_heading(
    tmp_path,
) -> None:
    image_path = tmp_path / "front.png"
    _write_rgb(image_path)
    model = _FakeS2Model(SimpleNamespace(output_action=[1], output_pixel=None))
    executor = InternVLAN1System2Executor("vln_nav", _real_spec())
    executor._model = model

    result = executor.execute(
        {"arguments": {"target": "desk", "image_path": str(image_path)}}
    )

    assert result["success"] is True
    assert result["metrics"]["vln"]["mode"] == "heading"
    assert result["metrics"]["vln"]["heading_deg"] == 0.0


def test_vln_adapter_maps_center_pixel_to_forward_step() -> None:
    command = planner_output_to_primitive(
        {"mode": "pixel_goal", "pixel_goal": [320, 240]},
        image_width=640,
    )

    assert command.primitive == "move_base"
    assert command.arguments == {"direction": "forward", "distance_cm": 15.0}


def test_vln_adapter_maps_off_center_pixel_to_turn() -> None:
    command = planner_output_to_primitive(
        {"mode": "pixel_goal", "pixel_goal": [32, 240]},
        image_width=640,
    )

    assert command.primitive == "turn_base"
    assert command.arguments["direction"] == "left"
    assert 5.0 <= command.arguments["angle_deg"] <= 15.0


def test_vln_adapter_maps_heading_and_stop() -> None:
    turn = planner_output_to_primitive({"mode": "heading", "heading_deg": 30.0})
    stop = planner_output_to_primitive({"mode": "stop", "stop": True})

    assert turn.primitive == "turn_base"
    assert turn.arguments == {"direction": "right", "angle_deg": 15.0}
    assert stop.primitive == "stop_motion"
    assert stop.arguments == {}


def test_vln_adapter_rejects_empty_planner_output() -> None:
    with pytest.raises(ValueError, match="planner output"):
        planner_output_to_primitive({})


def test_action_to_heading_maps_direction_codes() -> None:
    from hey_robot.foundation.backends.vln.internvla_n1_system2.executor import (
        _action_to_heading,
    )

    assert _action_to_heading([1]) == 0.0
    assert _action_to_heading([2]) == -90.0
    assert _action_to_heading([3]) == 90.0
    assert _action_to_heading([5]) == 180.0
    assert _action_to_heading([0]) is None
    assert _action_to_heading(None) is None


def test_to_float_list_handles_iterables() -> None:
    from hey_robot.foundation.backends.vln.internvla_n1_system2.executor import (
        _to_float_list,
    )

    assert _to_float_list([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    assert _to_float_list((4.0, 5.0, 6.0)) == [4.0, 5.0, 6.0]
    assert _to_float_list("not a list") == []
