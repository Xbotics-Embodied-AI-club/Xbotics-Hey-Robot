from __future__ import annotations

import asyncio
from typing import Any, cast

from hey_robot.capability.contract.v1 import capability_pb2
from hey_robot.capability.transport.grpc.server import (
    DEFAULT_ARM_CALIBRATION_DIR,
    LeRobotVLAExecutor,
    VLACapabilityService,
    VLACapabilityServicer,
)
from hey_robot.cli.main import CLI_ACTIONS
from hey_robot.config import DeploymentConfig


def _spec(settings: dict):
    config = DeploymentConfig.from_dict(
        {
            "capability_services": {
                "arm_vla": {
                    "type": "vla_service",
                    "enabled": True,
                    "robot_id": "xlerobot",
                    "target": "127.0.0.1:9090",
                    "skill_names": ["vla_manipulation"],
                    "timeout_sec": 5,
                    **settings,
                }
            }
        }
    )
    return config.capability_services["arm_vla"]


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
                "camera_source": "camera_observation",
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
                "camera_source": "camera_observation",
            }
        ),
    )

    config = executor._base_config({})

    assert config["calibration_dir"] == DEFAULT_ARM_CALIBRATION_DIR
    assert config["robot_id"] == "robot_arm"
    assert config["camera_source"] == "camera_observation"


def test_vla_executor_base_config_uses_payload_objective_and_infers_arm_side() -> None:
    executor = LeRobotVLAExecutor(
        "arm_vla",
        _spec(
            {
                "server_address": "127.0.0.1:8080",
                "policy_name": "org/policy",
                "policy_type": "pi05",
                "arm_port": "/dev/arm_right",
                "camera_source": "camera_observation",
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


def test_vla_executor_missing_config_depends_on_camera_source() -> None:
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

    config = executor._base_config({})
    config["camera_source"] = "camera_observation"
    config["camera_config"] = {}

    assert "camera_config" not in executor._missing_config(config)


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
                "camera_source": "camera_observation",
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
                "camera_source": "camera_observation",
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
                "camera_source": "camera_observation",
            }
        ),
    )

    result = executor.execute({})

    assert result["success"] is False
    assert result["failure_mode"] == "execution_failed"
    assert "ValueError" in result["summary"]


def test_vla_capability_servicer_health_execute_cancel() -> None:
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

    service = VLACapabilityService(
        DeploymentConfig.from_dict(
            {
                "capability_services": {
                    "arm_vla": {
                        "type": "vla_service",
                        "enabled": True,
                        "robot_id": "xlerobot",
                        "target": "127.0.0.1:9090",
                        "skill_names": ["vla_manipulation"],
                        "port": 9191,
                        "host": "127.0.0.1",
                        "policy_type": "pi05",
                        "model_path": "org/policy",
                        "arm_port": "COM5",
                        "task_prompt": "Pick up cup",
                        "camera_source": "camera_observation",
                    }
                }
            }
        ),
        service_id="arm_vla",
    )
    fake_executor = FakeExecutor()
    servicer = VLACapabilityServicer(service.state, cast(Any, fake_executor))

    async def run() -> None:
        health = await servicer.GetHealth(
            capability_pb2.GetHealthRequest(service_id="arm_vla"), None
        )
        assert health.busy is False
        assert dict(health.metrics)["policy_type"] == "pi05"

        service.state.busy = True
        busy = await servicer.ExecuteCapability(
            capability_pb2.ExecuteCapabilityRequest(
                service_id="arm_vla", skill_id="skill-1"
            ),
            None,
        )
        assert busy.failure_mode == "capability_busy"

        service.state.busy = False
        result = await servicer.ExecuteCapability(
            capability_pb2.ExecuteCapabilityRequest(
                service_id="arm_vla",
                skill_id="skill-2",
                skill_name="vla_manipulation",
                objective="pick",
                arguments=_struct(task="pick"),
            ),
            None,
        )
        assert result.success is True
        assert fake_executor.executed[0]["skill_id"] == "skill-2"

        health2 = await servicer.GetHealth(
            capability_pb2.GetHealthRequest(service_id="arm_vla"), None
        )
        assert dict(health2.metrics)["last_result"]["summary"] == "ok"

        cancelled = await servicer.CancelCapability(
            capability_pb2.CancelCapabilityRequest(
                service_id="arm_vla", skill_id="skill-2"
            ),
            None,
        )
        assert cancelled.accepted is True
        assert fake_executor.cancelled == 1

    asyncio.run(run())


def test_vla_capability_service_start_and_stop(monkeypatch) -> None:
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
        "hey_robot.capability.transport.grpc.server.grpc.aio.server",
        lambda: fake_server,
    )
    added = {}
    monkeypatch.setattr(
        "hey_robot.capability.transport.grpc.server.capability_pb2_grpc.add_CapabilityServiceServicer_to_server",
        lambda servicer, server: added.update({"servicer": servicer, "server": server}),
    )

    service = VLACapabilityService(
        DeploymentConfig.from_dict(
            {
                "capability_services": {
                    "arm_vla": {
                        "type": "vla_service",
                        "enabled": True,
                        "robot_id": "xlerobot",
                        "target": "127.0.0.1:9090",
                        "skill_names": ["vla_manipulation"],
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


def test_capability_service_cli_action_is_registered() -> None:
    assert CLI_ACTIONS["capability-service"] == "hey_robot.cli.capability_service:main"


def _raise_import_error():
    raise ImportError("missing lerobot")


def _struct(**kwargs):
    message = capability_pb2.ExecuteCapabilityRequest().arguments
    message.update(kwargs)
    return message
