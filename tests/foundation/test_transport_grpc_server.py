from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from hey_robot.cli.main import CLI_ACTIONS
from hey_robot.config import DeploymentConfig
from hey_robot.foundation.backends.vla.lerobot.executor import (
    DEFAULT_ARM_CALIBRATION_DIR,
    LeRobotVLAExecutor,
    LeRobotVLAPolicyExecutor,
)
from hey_robot.foundation.contract.v1 import model_service_pb2
from hey_robot.foundation.transport.grpc.server import (
    ModelServiceServicer,
    VLAPolicyService,
    VLNPlannerService,
    build_model_service,
)


def _spec(settings: dict):
    config = DeploymentConfig.from_dict(
        {
            "model_services": {
                "arm_vla": {
                    "type": "vla_policy",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "target": "127.0.0.1:9090",
                    "provides": ["manipulate"],
                    "timeout_sec": 5,
                    **settings,
                }
            }
        }
    )
    return config.model_services["arm_vla"]


def test_vla_executor_health_reports_missing_configuration() -> None:
    executor = LeRobotVLAExecutor("arm_vla", _spec({}))

    health = executor.health()

    assert health["online"] is True
    assert health["loaded"] is False
    assert "missing VLA configuration" in health["error"]


def test_vla_executor_runs_lerobot_client(monkeypatch) -> None:
    calls: list[str] = []

    class FakeRobotConfig:
        def __init__(self, port, cameras) -> None:
            self.port = port
            self.cameras = cameras

    class FakeCameraConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeRuntimeConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeRobotClient:
        def __init__(self, config) -> None:
            self.config = config

        def start(self) -> bool:
            calls.append("policy.start")
            return True

        def receive_actions(self) -> None:
            calls.append("policy.receive")

        def control_loop(self, *, task: str) -> None:
            calls.append(f"policy.control:{task}")

        def stop(self) -> None:
            calls.append("policy.stop")

    monkeypatch.setattr(
        LeRobotVLAExecutor,
        "_lerobot_classes",
        staticmethod(
            lambda: (
                FakeRobotClient,
                FakeRuntimeConfig,
                FakeRobotConfig,
                FakeCameraConfig,
            )
        ),
    )
    executor = LeRobotVLAExecutor(
        "arm_vla",
        _spec(
            {
                "server_address": "127.0.0.1:8080",
                "model_path": "org/policy",
                "policy_type": "pi05",
                "arm_port": "COM5",
                "camera_config": {"camera1": {"device_id": 0}},
            }
        ),
    )

    result = executor.execute({"arguments": {"task": "pick up cup"}, "timeout_sec": 5})

    assert result["success"] is True
    assert "policy.start" in calls
    assert "policy.control:pick up cup" in calls
    assert calls[-1] == "policy.stop"


def test_vla_executor_accepts_lerobot_single_arm_settings(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRobotConfig:
        def __init__(self, port, cameras) -> None:
            self.port = port
            self.cameras = cameras

    class FakeCameraConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeRuntimeConfig:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    class FakeRobotClient:
        def __init__(self, config) -> None:
            self.config = config

        def start(self) -> bool:
            return True

        def receive_actions(self) -> None:
            return None

        def control_loop(self, *, task: str) -> None:
            captured["control_task"] = task

        def stop(self) -> None:
            captured["stopped"] = True

    monkeypatch.setattr(
        LeRobotVLAExecutor,
        "_lerobot_classes",
        staticmethod(
            lambda: (
                FakeRobotClient,
                FakeRuntimeConfig,
                FakeRobotConfig,
                FakeCameraConfig,
            )
        ),
    )
    executor = LeRobotVLAExecutor(
        "arm_vla",
        _spec(
            {
                "server_address": "127.0.0.1:8080",
                "policy_name": "Grigorij/pi05_collect_tissue_23_02",
                "policy_type": "pi05",
                "arm_port": "/dev/arm_right",
                "task_prompt": "Pick up tissue.",
                "execution_time": 30,
                "camera_source": "opencv",
                "camera_config": {
                    "camera1": {"index_or_path": "/dev/camera_center"},
                    "camera2": {"index_or_path": "/dev/camera_right"},
                },
            }
        ),
    )

    result = executor.execute({})

    assert result["success"] is True
    assert captured["task"] == "Pick up tissue."
    assert captured["control_task"] == "Pick up tissue."
    assert captured["policy_type"] == "pi05"
    assert captured["pretrained_name_or_path"] == "Grigorij/pi05_collect_tissue_23_02"
    robot = cast(Any, captured["robot"])
    assert robot.id == "robot_arm"
    assert robot.port == "/dev/arm_right"
    assert str(robot.calibration_dir).endswith("so_follower")
    assert result["metrics"]["vla"]["arm_side"] == "right"
    assert result["summary"] == "Arm manipulation done"


def test_vla_executor_defaults_to_project_calibration_dir() -> None:
    executor = LeRobotVLAExecutor(
        "arm_vla",
        _spec(
            {
                "server_address": "127.0.0.1:8080",
                "policy_name": "org/policy",
                "policy_type": "pi05",
                "arm_port": "COM5",
                "task_prompt": "Pick up object.",
                "camera_config": {"camera1": {"device_id": 0}},
            }
        ),
    )

    config = executor._base_config({})

    assert config["calibration_dir"] == DEFAULT_ARM_CALIBRATION_DIR
    assert config["robot_id"] == "robot_arm"
    assert config["camera_source"] == "opencv"


def test_vla_executor_base_config_uses_payload_objective_and_infers_arm_side() -> None:
    executor = LeRobotVLAExecutor(
        "arm_vla",
        _spec(
            {
                "server_address": "127.0.0.1:8080",
                "policy_name": "org/policy",
                "policy_type": "pi05",
                "arm_port": "/dev/arm_right",
                "camera_config": {"camera1": {"device_id": 0}},
            }
        ),
    )

    config = executor._base_config(
        {
            "objective": "fallback objective",
            "timeout_sec": 9,
            "arguments": {"execution_time": 12},
        }
    )

    assert config["task"] == "fallback objective"
    assert config["timeout_sec"] == 12.0
    assert config["arm_side"] == "right"


def test_vla_executor_requires_camera_config() -> None:
    executor = LeRobotVLAExecutor(
        "arm_vla",
        _spec(
            {
                "server_address": "127.0.0.1:8080",
                "policy_name": "org/policy",
                "policy_type": "pi05",
                "arm_port": "COM5",
                "task_prompt": "Pick up object.",
                "camera_source": "opencv",
            }
        ),
    )

    missing_with_camera = executor._missing_config(executor._base_config({}))
    assert "camera_config" in missing_with_camera


def test_vla_executor_reports_missing_dependency(monkeypatch) -> None:
    executor = LeRobotVLAExecutor(
        "arm_vla",
        _spec(
            {
                "server_address": "127.0.0.1:8080",
                "policy_name": "org/policy",
                "policy_type": "pi05",
                "arm_port": "COM5",
                "task_prompt": "Pick up object.",
                "camera_config": {"camera1": {"device_id": 0}},
            }
        ),
    )

    monkeypatch.setattr(
        LeRobotVLAExecutor,
        "_lerobot_classes",
        staticmethod(lambda: (_raise_import_error(), None, None, None)),
    )
    result = executor.execute({})
    assert result["success"] is False
    assert result["failure_mode"] == "missing_dependency"


def test_vla_executor_reports_policy_server_unavailable(monkeypatch) -> None:
    class FakeRobotConfig:
        def __init__(self, port, cameras) -> None:
            self.port = port
            self.cameras = cameras

    class FakeCameraConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeRuntimeConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeRobotClient:
        def __init__(self, config) -> None:
            self.config = config

        def start(self) -> bool:
            return False

        def stop(self) -> None:
            return None

    monkeypatch.setattr(
        LeRobotVLAExecutor,
        "_lerobot_classes",
        staticmethod(
            lambda: (
                FakeRobotClient,
                FakeRuntimeConfig,
                FakeRobotConfig,
                FakeCameraConfig,
            )
        ),
    )
    executor = LeRobotVLAExecutor(
        "arm_vla",
        _spec(
            {
                "server_address": "127.0.0.1:8080",
                "policy_name": "org/policy",
                "policy_type": "pi05",
                "arm_port": "COM5",
                "task_prompt": "Pick up object.",
                "camera_config": {"camera1": {"device_id": 0}},
            }
        ),
    )

    result = executor.execute({})

    assert result["success"] is False
    assert result["failure_mode"] == "policy_server_unavailable"


def test_vla_executor_reports_control_loop_failure(monkeypatch) -> None:
    class FakeRobotConfig:
        def __init__(self, port, cameras) -> None:
            self.port = port
            self.cameras = cameras

    class FakeCameraConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeRuntimeConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeRobotClient:
        def __init__(self, config) -> None:
            self.config = config

        def start(self) -> bool:
            return True

        def receive_actions(self) -> None:
            return None

        def control_loop(self, *, task: str) -> None:
            raise ValueError(f"bad task: {task}")

        def stop(self) -> None:
            return None

    monkeypatch.setattr(
        LeRobotVLAExecutor,
        "_lerobot_classes",
        staticmethod(
            lambda: (
                FakeRobotClient,
                FakeRuntimeConfig,
                FakeRobotConfig,
                FakeCameraConfig,
            )
        ),
    )
    executor = LeRobotVLAExecutor(
        "arm_vla",
        _spec(
            {
                "server_address": "127.0.0.1:8080",
                "policy_name": "org/policy",
                "policy_type": "pi05",
                "arm_port": "COM5",
                "task_prompt": "Pick up object.",
                "camera_config": {"camera1": {"device_id": 0}},
            }
        ),
    )

    result = executor.execute({})

    assert result["success"] is False
    assert result["failure_mode"] == "execution_failed"
    assert "ValueError" in result["summary"]


def test_vla_policy_executor_requires_observation_for_action_chunk_policy() -> None:
    executor = LeRobotVLAPolicyExecutor(
        "arm_vla",
        _spec(
            {
                "backend_mode": "action_chunk_policy",
                "server_address": "127.0.0.1:8080",
                "model_path": "org/policy",
            }
        ),
    )

    result = executor.execute({"arguments": {"task_prompt": "pick cup"}})

    assert result["success"] is False
    assert result["failure_mode"] == "observation_unavailable"


def test_vla_policy_executor_calls_action_chunk_endpoint_without_lerobot(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_exc: object):
            return None

        def read(self) -> bytes:
            return (
                b'{"actions":[{"joints":{"shoulder_pan":0.1},"gripper":0.4}],'
                b'"horizon":1,"dt":0.033,"done":false,"confidence":0.8}'
            )

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = request.data
        return FakeResponse()

    monkeypatch.setattr(
        "hey_robot.foundation.backends.vla.lerobot.executor.urllib_request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        LeRobotVLAPolicyExecutor,
        "_lerobot_classes",
        staticmethod(lambda: (_raise_import_error(), None, None, None)),
    )
    executor = LeRobotVLAPolicyExecutor(
        "arm_vla",
        _spec(
            {
                "backend_mode": "action_chunk_policy",
                "action_chunk_endpoint": "http://127.0.0.1:8088/policy_step",
            }
        ),
    )

    result = executor.execute(
        {
            "skill_id": "pick-1",
            "robot_id": "xlerobot",
            "arguments": {
                "skill_name": "manipulate",
                "task_prompt": "pick cup",
                "policy_session_id": "pick-1",
                "observation": {
                    "frame_id": 12,
                    "images": [{"camera": "front", "format": "jpeg", "data": "abc"}],
                    "proprioception": {"arm": [0.0]},
                },
            },
            "timeout_sec": 2.0,
        }
    )

    assert result["success"] is True
    assert result["metrics"]["policy_result"]["kind"] == "action_chunk"
    assert result["metrics"]["vla"]["joint_angles"] == {"shoulder_pan": 0.1}
    assert result["metrics"]["vla"]["gripper_action"] == 0.4
    assert result["metrics"]["vla"]["hardware_ownership"] == "none"
    assert captured["url"] == "http://127.0.0.1:8088/policy_step"


def test_vla_policy_executor_rejects_invalid_action_chunk_response(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_exc: object):
            return None

        def read(self) -> bytes:
            return b'{"policy_result":{"kind":"local_goal","actions":[]}}'

    def fake_urlopen(_request, timeout):
        del timeout
        return FakeResponse()

    monkeypatch.setattr(
        "hey_robot.foundation.backends.vla.lerobot.executor.urllib_request.urlopen",
        fake_urlopen,
    )
    executor = LeRobotVLAPolicyExecutor(
        "arm_vla",
        _spec(
            {
                "backend_mode": "action_chunk_policy",
                "action_chunk_endpoint": "http://127.0.0.1:8088/policy_step",
            }
        ),
    )

    result = executor.execute(
        {
            "arguments": {
                "task_prompt": "pick cup",
                "observation": {
                    "frame_id": 12,
                    "images": [{"camera": "front", "format": "jpeg", "data": "abc"}],
                },
            }
        }
    )

    assert result["success"] is False
    assert result["failure_mode"] == "action_chunk_policy_invalid_response"


def test_vla_service_selects_legacy_control_loop_backend() -> None:
    service = VLAPolicyService(
        DeploymentConfig.from_dict(
            {
                "model_services": {
                    "arm_vla": {
                        "type": "vla_policy",
                        "enabled": True,
                        "robot_id": "xlerobot",
                        "target": "127.0.0.1:9090",
                        "provides": ["manipulate"],
                        "backend_mode": "lerobot_control_loop",
                    }
                }
            }
        ),
        service_id="arm_vla",
    )

    assert isinstance(service.executor, LeRobotVLAExecutor)


def test_vla_model_servicer_health_execute_cancel() -> None:
    class FakeExecutor:
        def __init__(self) -> None:
            self.executed: list[dict[str, Any]] = []
            self.cancelled = 0

        def health(self) -> dict[str, Any]:
            return {
                "name": "arm_vla",
                "online": True,
                "loaded": True,
                "robot_id": "xlerobot",
                "error": None,
                "metrics": {"policy_type": "pi05"},
            }

        def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.executed.append(payload)
            return {
                "success": True,
                "status": "completed",
                "summary": "ok",
                "error": None,
                "metrics": {"frames": 3},
            }

        def cancel(self) -> None:
            self.cancelled += 1

    service = VLAPolicyService(
        DeploymentConfig.from_dict(
            {
                "model_services": {
                    "arm_vla": {
                        "type": "vla_policy",
                        "enabled": True,
                        "robot_id": "xlerobot",
                        "target": "127.0.0.1:9090",
                        "provides": ["manipulate"],
                        "port": 9191,
                        "host": "127.0.0.1",
                        "policy_type": "pi05",
                        "model_path": "org/policy",
                        "arm_port": "COM5",
                        "task_prompt": "Pick up cup",
                        "camera_config": {"camera1": {"device_id": 0}},
                    }
                }
            }
        ),
        service_id="arm_vla",
    )
    fake_executor = FakeExecutor()
    servicer = ModelServiceServicer(service.state, cast(Any, fake_executor))

    async def run() -> None:
        health = await servicer.GetHealth(
            model_service_pb2.GetHealthRequest(service_id="arm_vla"), None
        )
        assert health.busy is False
        assert dict(health.metrics)["policy_type"] == "pi05"

        service.state.busy = True
        busy = await servicer.ExecuteSkill(
            model_service_pb2.ExecuteSkillRequest(
                service_id="arm_vla", skill_id="skill-1"
            ),
            None,
        )
        assert busy.failure_mode == "model_service_busy"

        service.state.busy = False
        result = await servicer.ExecuteSkill(
            model_service_pb2.ExecuteSkillRequest(
                service_id="arm_vla",
                skill_id="skill-2",
                skill_name="manipulate",
                objective="pick",
                arguments=_struct(task="pick"),
            ),
            None,
        )
        assert result.success is True
        assert fake_executor.executed[0]["skill_id"] == "skill-2"

        health2 = await servicer.GetHealth(
            model_service_pb2.GetHealthRequest(service_id="arm_vla"), None
        )
        assert dict(health2.metrics)["last_result"]["summary"] == "ok"

        cancelled = await servicer.CancelSkill(
            model_service_pb2.CancelSkillRequest(
                service_id="arm_vla", skill_id="skill-2"
            ),
            None,
        )
        assert cancelled.accepted is True
        assert fake_executor.cancelled == 1

    asyncio.run(run())


def test_vla_model_service_start_and_stop(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeServer:
        def __init__(self) -> None:
            self.port = None
            self.started = False
            self.stopped = None

        def add_insecure_port(self, target: str) -> None:
            self.port = target

        async def start(self) -> None:
            self.started = True

        async def wait_for_termination(self) -> None:
            captured["waited"] = True

        async def stop(self, grace: float) -> None:
            self.stopped = grace

    fake_server = FakeServer()
    monkeypatch.setattr(
        "hey_robot.foundation.transport.grpc.server.grpc.aio.server",
        lambda: fake_server,
    )
    added = {}
    monkeypatch.setattr(
        "hey_robot.foundation.transport.grpc.server.model_service_pb2_grpc.add_ModelServiceServicer_to_server",
        lambda servicer, server: added.update({"servicer": servicer, "server": server}),
    )

    service = VLAPolicyService(
        DeploymentConfig.from_dict(
            {
                "model_services": {
                    "arm_vla": {
                        "type": "vla_policy",
                        "enabled": True,
                        "robot_id": "xlerobot",
                        "target": "127.0.0.1:9090",
                        "provides": ["manipulate"],
                        "port": 9191,
                        "host": "127.0.0.1",
                    }
                }
            }
        ),
        service_id="arm_vla",
    )

    async def run() -> None:
        await service.start()
        assert fake_server.port == "127.0.0.1:9191"
        assert fake_server.started is True
        assert added["server"] is fake_server
        await service.stop()
        assert fake_server.stopped == 0.5

    asyncio.run(run())


def test_build_model_service_supports_vln_planner() -> None:
    config = DeploymentConfig.from_dict(
        {
            "model_services": {
                "vln_nav": {
                    "type": "vln_planner",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "target": "127.0.0.1:9091",
                    "provides": ["navigate_to", "approach_object"],
                    "backend": "internvla_n1_system2",
                    "control_mode": "planner_only",
                    "mock_mode": True,
                }
            }
        }
    )

    service = build_model_service(config, service_id="vln_nav")

    assert isinstance(service, VLNPlannerService)
    assert service.service_id == "vln_nav"
    assert service.port == 9091


def test_build_model_service_rejects_unknown_type() -> None:
    config = DeploymentConfig.from_dict(
        {
            "model_services": {
                "bad": {
                    "type": "unknown_service",
                    "enabled": True,
                    "robot_id": "xlerobot",
                }
            }
        }
    )

    with pytest.raises(ValueError, match="unsupported model service type"):
        build_model_service(config, service_id="bad")


def test_model_service_cli_action_is_registered() -> None:
    assert CLI_ACTIONS["model-service"] == "hey_robot.cli.model_service:main"


def _raise_import_error():
    raise ImportError("missing lerobot")


def _struct(**kwargs):
    message = model_service_pb2.ExecuteSkillRequest().arguments
    message.update(kwargs)
    return message
