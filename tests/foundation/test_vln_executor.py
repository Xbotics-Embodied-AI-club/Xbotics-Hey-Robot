from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from hey_robot.config import DeploymentConfig
from hey_robot.foundation.backends.vln import VLNPlannerExecutor
from hey_robot.foundation.backends.vln.control import build_base_action_chunk
from hey_robot.foundation.backends.vln.input import to_float_list
from hey_robot.foundation.backends.vln.internvla_n1 import (
    InternVLAN1Runtime,
    action_to_heading,
)
from hey_robot.foundation.backends.vln.models import VLNPlannerResult


class InternVLAN1DualVLNExecutor(VLNPlannerExecutor):
    """Test adapter that injects a tiny dual-system model into the runtime."""

    @property
    def _model(self):
        runtime = self._runtime
        return getattr(runtime, "_model", None)

    @_model.setter
    def _model(self, value) -> None:
        runtime = InternVLAN1Runtime(dict(self.spec.settings))
        runtime._model = value
        self._runtime = runtime


def _spec(settings: dict | None = None):
    config = DeploymentConfig.from_dict(
        {
            "model_services": {
                "vln_nav": {
                    "type": "vln_planner",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "target": "127.0.0.1:9091",
                    "provides": ["navigate_to", "approach_object"],
                    "backend": "internvla_n1_dualvln",
                    "control_mode": "base_action_chunk",
                    "base_linear_speed": 0.25,
                    "base_angular_speed": 0.3,
                    "discrete_forward_cm": 25,
                    "discrete_turn_deg": 15,
                    "max_action_chunk_steps": 4,
                    "system1_replans_per_waypoint": 4,
                    "mock_mode": True,
                    **dict(settings or {}),
                }
            }
        }
    )
    return config.model_services["vln_nav"]


class _FakeS2Model:
    def __init__(
        self, output, *, llm_output: str = "", system1_actions: list[int] | None = None
    ) -> None:
        self.output = output
        self.llm_output = llm_output
        self.device = "cpu"
        self.system1_actions = system1_actions or [1, 1, 1, 1]
        self.calls: list[dict] = []
        self.system1_calls: list[dict] = []
        self.no_infer_calls: list[dict] = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

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

    def step_no_infer(self, rgb, depth, pose) -> None:
        self.no_infer_calls.append({"rgb": rgb, "depth": depth, "pose": pose})

    def s1_step_latent(self, rgbs, depths, latent):
        self.system1_calls.append({"rgbs": rgbs, "depths": depths, "latent": latent})
        return SimpleNamespace(idx=list(self.system1_actions))


def _real_spec(settings: dict | None = None):
    return _spec(
        {
            "mock_mode": False,
            "model_path": "dummy-model",
            "internnav_repo": "third_party/InternNav",
            "media_root": ".",
            **dict(settings or {}),
        }
    )


def _write_rgb(path, *, size=(8, 6)) -> None:
    data = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    data[:, :, 0] = 128
    Image.fromarray(data).save(path)


def test_internvla_n1_dualvln_mock_health_is_loaded() -> None:
    executor = InternVLAN1DualVLNExecutor("vln_nav", _spec())

    health = executor.health()

    assert health["online"] is True
    assert health["loaded"] is True
    assert health["metrics"]["backend"] == "internvla_n1_dualvln"
    assert health["metrics"]["control_mode"] == "base_action_chunk"
    assert health["metrics"]["mock_mode"] is True


def test_real_vln_health_is_not_loaded_before_runtime_load() -> None:
    executor = VLNPlannerExecutor("vln_nav", _real_spec())

    health = executor.health()

    assert health["online"] is True
    assert health["loaded"] is False
    assert health["error"] is None


def test_runtime_requires_complete_dual_system_checkpoint() -> None:
    vlm = SimpleNamespace(
        latent_queries=None,
        traj_dit=None,
        action_encoder=None,
        rgb_model=None,
        memory_encoder=None,
        cond_projector=None,
    )
    inner = SimpleNamespace(
        config=SimpleNamespace(system1=None),
        get_model=lambda: vlm,
    )
    runtime = InternVLAN1Runtime({"n_query": 4})

    with pytest.raises(RuntimeError, match="DualVLN checkpoint is required"):
        runtime._validate_dual_system(SimpleNamespace(model=inner))


def test_internvla_n1_dualvln_mock_returns_center_pixel_goal() -> None:
    executor = InternVLAN1DualVLNExecutor(
        "vln_nav",
        _spec({"image_width": 640, "image_height": 480}),
    )

    result = executor.execute({"arguments": {"target": "desk"}})

    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["metrics"]["vln"]["mode"] == "pixel_goal"
    assert result["metrics"]["vln"]["pixel_goal"] == [240, 320]
    assert result["metrics"]["vln"]["stop"] is False
    assert result["metrics"]["vln"]["policy_result"]["kind"] == "local_goal"
    assert result["metrics"]["vln"]["local_goal"]["pixel_goal"] == [240, 320]


def test_internvla_n1_dualvln_mock_can_return_stop() -> None:
    executor = InternVLAN1DualVLNExecutor("vln_nav", _spec())

    result = executor.execute({"arguments": {"instruction": "stop when done"}})

    assert result["success"] is True
    assert result["metrics"]["vln"]["mode"] == "stop"
    assert result["metrics"]["vln"]["stop"] is True


def test_base_action_chunk_publishes_calibrated_control_contract() -> None:
    executor = InternVLAN1DualVLNExecutor(
        "vln_nav",
        _spec(
            {
                "control_mode": "base_action_chunk",
                "base_linear_speed": 0.25,
                "base_angular_speed": 0.3,
                "discrete_forward_cm": 25,
                "discrete_turn_deg": 15,
                "max_action_chunk_steps": 4,
            }
        ),
    )

    result = executor.execute({"arguments": {"target": "desk"}})

    assert result["success"] is True
    assert result["metrics"]["vln"]["base_control"] == {
        "linear_speed": 0.25,
        "angular_speed": 0.3,
        "forward_distance_cm": 25.0,
        "turn_angle_deg": 15.0,
        "max_chunk_steps": 4,
    }
    assert result["metrics"]["vln"]["control_chunk"]["kind"] == ("base_velocity_chunk")
    assert len(result["metrics"]["vln"]["control_chunk"]["actions"]) == 1


def test_base_action_chunk_preserves_native_motion_scale() -> None:
    chunk = build_base_action_chunk(
        VLNPlannerResult(mode="heading", action_sequence=[2, 2, 1, 3]),
        {
            "base_linear_speed": 0.25,
            "base_angular_speed": 0.3,
            "discrete_forward_cm": 25,
            "discrete_turn_deg": 15,
            "max_action_chunk_steps": 4,
        },
    )

    assert chunk["kind"] == "base_velocity_chunk"
    assert [action["source"] for action in chunk["actions"]] == [
        "system2_left",
        "system2_left",
        "system2_forward",
        "system2_right",
    ]
    assert [action["wz"] for action in chunk["actions"]] == [0.3, 0.3, 0.0, -0.3]


def test_base_action_chunk_preserves_stop_after_motion() -> None:
    chunk = build_base_action_chunk(
        VLNPlannerResult(
            mode="trajectory_chunk",
            action_sequence=[1, 0],
            policy_stage="system1",
        ),
        {
            "base_linear_speed": 0.25,
            "base_angular_speed": 0.3,
            "discrete_forward_cm": 25,
            "discrete_turn_deg": 15,
            "max_action_chunk_steps": 4,
        },
    )

    assert chunk["stop"] is False
    assert chunk["stop_after_actions"] is True
    assert [action["source"] for action in chunk["actions"]] == ["system1_forward"]


def test_internvla_n1_dualvln_rejects_unknown_control_mode() -> None:
    executor = InternVLAN1DualVLNExecutor(
        "vln_nav",
        _spec({"control_mode": "direct_velocity"}),
    )

    result = executor.execute({"arguments": {"target": "desk"}})

    assert result["success"] is False
    assert result["failure_mode"] == "unsupported_control_mode"


def test_internvla_n1_dualvln_runs_system1_for_pixel_plan(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "hey_robot.foundation.backends.vln.internvla_n1._system1_rgb_pair",
        lambda goal_rgb, current_rgb, **_kwargs: np.stack([goal_rgb, current_rgb]),
    )
    image_path = tmp_path / "front.png"
    _write_rgb(image_path)
    model = _FakeS2Model(
        SimpleNamespace(output_pixel=np.asarray([3, 4]), output_latent=np.asarray([9]))
    )
    executor = InternVLAN1DualVLNExecutor("vln_nav", _real_spec({"hfov": 90}))
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
    assert result["metrics"]["vln"]["mode"] == "trajectory_chunk"
    assert result["metrics"]["vln"]["pixel_goal"] == [3, 4]
    assert result["metrics"]["vln"]["latent_available"] is True
    assert result["metrics"]["vln"]["policy_stage"] == "system1"
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
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "hey_robot.foundation.backends.vln.internvla_n1._system1_rgb_pair",
        lambda goal_rgb, current_rgb, **_kwargs: np.stack([goal_rgb, current_rgb]),
    )
    front = tmp_path / "front.png"
    wrist = tmp_path / "wrist.png"
    _write_rgb(front)
    _write_rgb(wrist)
    model = _FakeS2Model(
        SimpleNamespace(output_pixel=np.asarray([1, 2]), output_latent=np.asarray([9]))
    )
    executor = InternVLAN1DualVLNExecutor("vln_nav", _real_spec({"camera": "front"}))
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


def test_internvla_n1_system2_real_path_clamps_out_of_bounds_pixel(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "hey_robot.foundation.backends.vln.internvla_n1._system1_rgb_pair",
        lambda goal_rgb, current_rgb, **_kwargs: np.stack([goal_rgb, current_rgb]),
    )
    image_path = tmp_path / "front.png"
    _write_rgb(image_path, size=(8, 6))
    model = _FakeS2Model(
        SimpleNamespace(
            output_pixel=np.asarray([99, -3]), output_latent=np.asarray([9])
        )
    )
    executor = InternVLAN1DualVLNExecutor("vln_nav", _real_spec())
    executor._model = model

    result = executor.execute(
        {"arguments": {"target": "desk", "image_path": str(image_path)}}
    )

    assert result["success"] is True
    assert result["metrics"]["vln"]["pixel_goal"] == [99, 0]
    assert result["metrics"]["vln"]["image_width"] == 384
    assert result["metrics"]["vln"]["image_height"] == 384


def test_internvla_n1_system2_real_path_maps_stop_action(tmp_path) -> None:
    image_path = tmp_path / "front.png"
    _write_rgb(image_path)
    model = _FakeS2Model(SimpleNamespace(output_action=[0], output_pixel=None))
    executor = InternVLAN1DualVLNExecutor("vln_nav", _real_spec())
    executor._model = model

    result = executor.execute(
        {"arguments": {"target": "desk", "image_path": str(image_path)}}
    )

    assert result["success"] is True
    assert result["metrics"]["vln"]["mode"] == "stop"
    assert result["metrics"]["vln"]["stop"] is True


def test_internvla_n1_system2_real_path_requires_image() -> None:
    executor = InternVLAN1DualVLNExecutor("vln_nav", _real_spec())
    executor._model = _FakeS2Model(
        SimpleNamespace(output_pixel=np.asarray([1, 2]), output_latent=np.asarray([9]))
    )

    result = executor.execute({"arguments": {"target": "desk"}})

    assert result["success"] is False
    assert result["failure_mode"] == "image_unavailable"


def test_vln_rejects_unsafe_local_media_uri(tmp_path) -> None:
    executor = InternVLAN1DualVLNExecutor(
        "vln_nav", _real_spec({"media_root": str(tmp_path)})
    )
    executor._model = _FakeS2Model(
        SimpleNamespace(output_pixel=np.asarray([1, 2]), output_latent=np.asarray([9]))
    )

    result = executor.execute(
        {
            "arguments": {
                "target": "desk",
                "image_ref": "media://local/../outside.png",
            }
        }
    )

    assert result["success"] is False
    assert result["failure_mode"] == "image_unavailable"


def test_internvla_n1_system2_real_path_maps_non_stop_action_to_heading(
    tmp_path,
) -> None:
    image_path = tmp_path / "front.png"
    _write_rgb(image_path)
    model = _FakeS2Model(SimpleNamespace(output_action=[1], output_pixel=None))
    executor = InternVLAN1DualVLNExecutor("vln_nav", _real_spec())
    executor._model = model

    result = executor.execute(
        {"arguments": {"target": "desk", "image_path": str(image_path)}}
    )

    assert result["success"] is True
    assert result["metrics"]["vln"]["mode"] == "heading"
    assert result["metrics"]["vln"]["heading_deg"] == 0.0
    assert result["metrics"]["vln"]["action_code"] == 1
    assert result["metrics"]["vln"]["forward_distance_cm"] == 25.0


def test_internvla_n1_system2_loads_base64_observation_image(monkeypatch) -> None:
    monkeypatch.setattr(
        "hey_robot.foundation.backends.vln.internvla_n1._system1_rgb_pair",
        lambda goal_rgb, current_rgb, **_kwargs: np.stack([goal_rgb, current_rgb]),
    )
    image = Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    model = _FakeS2Model(
        SimpleNamespace(output_pixel=np.asarray([3, 4]), output_latent=np.asarray([9]))
    )
    executor = InternVLAN1DualVLNExecutor("vln_nav", _real_spec({"camera": "front"}))
    executor._model = model

    result = executor.execute(
        {
            "skill_id": "nav1",
            "arguments": {
                "target": "desk",
                "observation": {
                    "frame_id": 7,
                    "images": [
                        {
                            "camera": "front",
                            "format": "jpeg",
                            "data": base64.b64encode(buf.getvalue()).decode("ascii"),
                        }
                    ],
                },
            },
        }
    )

    assert result["success"] is True
    assert model.calls[0]["rgb"].shape == (6, 8, 3)
    assert result["metrics"]["vln"]["policy_session_id"] == "nav1"


def test_internvla_n1_system2_action_five_requires_look_down(tmp_path) -> None:
    image_path = tmp_path / "front.png"
    _write_rgb(image_path)
    model = _FakeS2Model(SimpleNamespace(output_action=[5], output_pixel=None))
    executor = InternVLAN1DualVLNExecutor("vln_nav", _real_spec())
    executor._model = model

    result = executor.execute(
        {"arguments": {"target": "desk", "image_path": str(image_path)}}
    )

    assert result["success"] is True
    assert result["metrics"]["vln"]["mode"] == "look_down_required"
    assert result["metrics"]["vln"]["requires_secondary_observation"] is True
    assert result["metrics"]["vln"]["heading_deg"] is None


def test_dualvln_returns_complete_native_system2_action_chunk(tmp_path) -> None:
    image_path = tmp_path / "front.png"
    _write_rgb(image_path)
    model = _FakeS2Model(
        SimpleNamespace(output_action=[1, 0], output_pixel=None),
        llm_output="↑STOP",
    )
    executor = InternVLAN1DualVLNExecutor("vln_nav", _real_spec())
    executor._model = model

    first = executor.execute(
        {"arguments": {"target": "desk", "image_path": str(image_path)}}
    )
    assert first["success"] is True
    assert first["metrics"]["vln"]["mode"] == "heading"
    assert first["metrics"]["vln"]["action_sequence"] == [1, 0]
    assert first["metrics"]["vln"]["remaining_action_count"] == 1
    assert first["metrics"]["vln"]["raw_output"] == "↑STOP"
    assert len(model.calls) == 1
    assert len(model.no_infer_calls) == 0


def test_internvla_n1_system2_resets_on_new_policy_session(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "hey_robot.foundation.backends.vln.internvla_n1._system1_rgb_pair",
        lambda goal_rgb, current_rgb, **_kwargs: np.stack([goal_rgb, current_rgb]),
    )
    image_path = tmp_path / "front.png"
    _write_rgb(image_path)
    model = _FakeS2Model(
        SimpleNamespace(output_pixel=np.asarray([3, 4]), output_latent=np.asarray([9]))
    )
    executor = InternVLAN1DualVLNExecutor("vln_nav", _real_spec())
    executor._model = model

    payload = {
        "arguments": {
            "target": "desk",
            "image_path": str(image_path),
            "policy_session_id": "session-a",
        }
    }
    result1 = executor.execute(payload)
    result2 = executor.execute(payload)
    result3 = executor.execute(
        {
            "arguments": {
                "target": "door",
                "image_path": str(image_path),
                "policy_session_id": "session-b",
            }
        }
    )

    assert result1["success"] is True
    assert result2["success"] is True
    assert result3["success"] is True
    assert model.reset_calls == 2
    assert len(model.calls) == 2
    assert len(model.system1_calls) == 3
    assert result3["metrics"]["vln"]["policy_session_id"] == "session-b"


def test_action_to_heading_maps_direction_codes() -> None:
    assert action_to_heading([1]) == 0.0
    assert action_to_heading([2]) == -15.0
    assert action_to_heading([3]) == 15.0
    assert action_to_heading([2], turn_angle_deg=30.0) == -30.0
    assert action_to_heading([5]) is None
    assert action_to_heading([1, 0]) == 0.0
    assert action_to_heading([0]) is None
    assert action_to_heading(None) is None


def test_to_float_list_handles_iterables() -> None:
    assert to_float_list([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    assert to_float_list((4.0, 5.0, 6.0)) == [4.0, 5.0, 6.0]
    assert to_float_list("not a list") == []
