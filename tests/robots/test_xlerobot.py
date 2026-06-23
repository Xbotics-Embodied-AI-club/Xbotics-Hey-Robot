from __future__ import annotations

import time
from types import SimpleNamespace

from hey_robot.config import DeploymentConfig
from hey_robot.protocol import Envelope, SkillIntent
from hey_robot.robots import RobotManager, get_embodiment_profile
from hey_robot.robots.base import RobotDriverContext
from hey_robot.robots.classic.primitives import SUPPORTED_CLASSIC_PRIMITIVES
from hey_robot.robots.lekiwi import LeKiwiDriver
from hey_robot.robots.lekiwi.base import LeKiwiBase
from hey_robot.robots.so101 import SO101Driver
from hey_robot.robots.xlerobot import XLeRobotDriver
from hey_robot.robots.xlerobot.executor import XLeRobotSkillExecutor
from hey_robot.robots.xlerobot.hardware.native import _service_diagnostic
from hey_robot.skills import RobotSkillAction, SkillPlanner


def test_robot_manager_supports_xlerobot() -> None:
    config = DeploymentConfig.from_dict(
        {
            "robots": {
                "xlerobot": {
                    "type": "xlerobot",
                    "serial_bus": {"port": "COM5", "baudrate": 1000000},
                    "components": {"camera": {"device_id": 1}},
                }
            }
        }
    )

    driver = RobotManager(config).require("xlerobot")

    assert isinstance(driver, XLeRobotDriver)
    assert driver.robot_id == "xlerobot"


def test_robot_manager_supports_decoupled_hardware_bodies() -> None:
    config = DeploymentConfig.from_dict(
        {
            "robots": {
                "arm0": {
                    "type": "so101",
                    "serial_bus": {"port": "COM6", "baudrate": 1000000},
                    "components": {"arm": {"type": "so101_arm"}},
                },
                "base0": {
                    "type": "lekiwi",
                    "serial_bus": {"port": "COM7", "baudrate": 1000000},
                    "components": {"base": {"type": "lekiwi_base"}},
                },
            }
        }
    )

    manager = RobotManager(config)

    assert isinstance(manager.require("arm0"), SO101Driver)
    assert isinstance(manager.require("base0"), LeKiwiDriver)


def test_robot_manager_supports_explicit_family_environment_driver_identity() -> None:
    config = DeploymentConfig.from_dict(
        {
            "robots": {
                "sim0": {
                    "type": "xlerobot_sim",
                    "family": "xlerobot",
                    "environment": "sim",
                    "driver": "mujoco",
                }
            }
        }
    )

    driver = RobotManager(config).require("sim0")

    from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

    assert isinstance(driver, XLeRobotSimDriver)


def test_skill_planner_maps_chinese_forward_motion() -> None:
    assert SkillPlanner().plan("往前走10cm") == RobotSkillAction(
        "move_base",
        {"direction": "forward", "distance_cm": 10.0},
        expected_duration_sec=1.0,
    )


def test_xlerobot_deployment_uses_native_skill_policy() -> None:
    config = DeploymentConfig.from_yaml("configs/xlerobot.real.windows.yaml")

    assert config.robots["xlerobot"].type == "xlerobot"
    assert config.robots["xlerobot"].robot_family == "xlerobot"
    assert config.robots["xlerobot"].robot_environment == "real"
    assert config.robots["xlerobot"].driver_kind == "native"
    assert config.robots["xlerobot"].embodiment_profile == "xlerobot_real"
    assert (
        config.robots["xlerobot"].settings["components"]["base"]["type"]
        == "lekiwi_base"
    )
    assert (
        config.robots["xlerobot"].settings["components"]["arm"]["type"] == "so101_arm"
    )
    assert "vla" not in config.robots["xlerobot"].settings["components"]
    assert config.capability_services == {}
    assert config.policies["embodied_skills"].freq_hz == 1.0


def test_xlerobot_hardware_config_supports_multi_arm_and_multi_camera_shape() -> None:
    from hey_robot.robots.xlerobot.hardware.config import hardware_config_from_settings

    config = hardware_config_from_settings(
        {
            "serial_bus": {"port": "COM5", "baudrate": 1000000},
            "components": {
                "default_arm": "right",
                "default_camera": "front",
                "arms": {
                    "left": {
                        "joint_ids": {
                            "base": 11,
                            "shoulder": 12,
                            "elbow": 13,
                            "wrist_flex": 14,
                            "wrist_roll": 15,
                            "gripper": 16,
                        }
                    },
                    "right": {
                        "joint_ids": {
                            "base": 1,
                            "shoulder": 2,
                            "elbow": 3,
                            "wrist_flex": 4,
                            "wrist_roll": 5,
                            "gripper": 6,
                        }
                    },
                },
                "cameras": {
                    "front": {"device_id": 1, "owner": "camera_service"},
                    "left_wrist": {"device_id": 2},
                    "right_wrist": {"device_id": 3},
                },
            },
        }
    )

    assert config.default_arm == "right"
    assert set(config.arms) == {"left", "right"}
    assert config.arms["left"].joint_ids["base"] == 11
    assert config.default_camera == "front"
    assert set(config.cameras) == {"front", "left_wrist", "right_wrist"}
    assert config.camera_owners["front"] == "camera_service"
    assert config.cameras["left_wrist"].device_id == 2


def test_xlerobot_executor_rejects_direct_perception_execution() -> None:
    class FakeClient:
        pass

    result = XLeRobotSkillExecutor(FakeClient()).execute(
        RobotSkillAction("inspect_scene", {"question": "look ahead"})
    )  # type: ignore[arg-type]

    assert result.success is False
    assert "RobotRuntime perception pipeline" in result.message
    assert result.data["failure_mode"] == "wrong_execution_boundary"


def test_xlerobot_executor_rejects_legacy_skill_names() -> None:
    class FakeClient:
        pass

    result = XLeRobotSkillExecutor(FakeClient()).execute(  # type: ignore[arg-type]
        RobotSkillAction("vla_manipulation", {"task": "pick up cup"})
    )

    assert result.success is False
    assert "unsupported classic primitive" in result.message


def test_xlerobot_supported_skills_come_from_catalog() -> None:
    assert XLeRobotSkillExecutor.supported_skills == SUPPORTED_CLASSIC_PRIMITIVES


def test_xlerobot_velocity_step_streams_small_velocity_and_renews_watchdog() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def set_velocity(self, vx: float, vz: float, *, vy: float = 0.0):
            self.calls.append(("set_velocity", vx, vy, vz))
            return {"success": True, "message": "velocity applied"}

        def base_stop(self):
            self.calls.append(("base_stop",))
            return {"success": True, "message": "stopped"}

    client = FakeClient()
    executor = XLeRobotSkillExecutor(client)  # type: ignore[arg-type]
    action = RobotSkillAction(
        "base_velocity_step",
        {"vx": -0.0047, "vy": 0.0, "wz": 0.0121, "duration_ms": 120},
    )

    first = executor.execute(action)
    time.sleep(0.06)
    second = executor.execute(action)
    time.sleep(0.08)

    assert first.success is True
    assert second.success is True
    assert client.calls == [
        ("set_velocity", -0.0047, 0.0, 0.0121),
        ("set_velocity", -0.0047, 0.0, 0.0121),
    ]

    time.sleep(0.08)
    assert client.calls[-1] == ("base_stop",)


def test_xlerobot_executor_dispatches_atomic_motion_and_arm_skills() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def move_forward_cm(self, distance_cm: float):
            self.calls.append(("move_forward_cm", distance_cm))
            return {"success": True, "distance_cm": distance_cm}

        def turn_right_deg(self, angle_deg: float):
            self.calls.append(("turn_right_deg", angle_deg))
            return {"success": True, "angle_deg": angle_deg}

        def set_joints_delta(
            self, joints: dict[str, float], arm_name: str | None = None
        ):
            self.calls.append(("set_joints_delta", joints, arm_name))
            return {"success": True, "joints": joints}

        def move_named_pose(self, pose_name: str, arm_name: str | None = None):
            self.calls.append(("move_named_pose", pose_name, arm_name))
            return {"success": True, "pose_name": pose_name}

        def set_gripper_opening_pct(self, pct: float, arm_name: str | None = None):
            self.calls.append(("set_gripper_opening_pct", pct, arm_name))
            return {"success": True, "pct": pct}

    client = FakeClient()
    executor = XLeRobotSkillExecutor(client)  # type: ignore[arg-type]

    assert executor.execute(
        RobotSkillAction("move_base", {"direction": "forward", "distance_cm": 18})
    ).success
    assert executor.execute(
        RobotSkillAction("turn_base", {"direction": "right", "angle_deg": 45})
    ).success
    assert executor.execute(
        RobotSkillAction(
            "move_arm_joints", {"mode": "delta", "joints": {"wrist_roll": 12}}
        )
    ).success
    assert executor.execute(
        RobotSkillAction("set_arm_pose", {"pose_name": "pregrasp", "arm": "left"})
    ).success
    assert executor.execute(
        RobotSkillAction("set_gripper", {"opening_pct": 35, "arm": "right"})
    ).success

    assert client.calls == [
        ("move_forward_cm", 18.0),
        ("turn_right_deg", 45.0),
        ("set_joints_delta", {"wrist_roll": 12.0}, None),
        ("move_named_pose", "pregrasp", "left"),
        ("set_gripper_opening_pct", 35.0, "right"),
    ]


def test_xlerobot_service_diagnostic_preserves_startup_failure_reason() -> None:
    diagnostic = _service_diagnostic(
        {
            "ok": False,
            "issue": "missing arm servos: [1, 2]",
            "response": {"success": False, "message": "missing arm servos: [1, 2]"},
        },
        {"success": False, "enabled": True, "initialized": False, "joint_states": {}},
        joint_count=0,
    )

    assert diagnostic["ok"] is False
    assert diagnostic["issue"] == "missing arm servos: [1, 2]"
    assert diagnostic["startup_response"]["message"] == "missing arm servos: [1, 2]"
    assert diagnostic["status_response"]["initialized"] is False


async def test_xlerobot_driver_rejects_action_when_contract_readiness_fails() -> None:
    config = DeploymentConfig.from_dict({"robots": {"xlerobot": {"type": "xlerobot"}}})
    driver = XLeRobotDriver(
        RobotDriverContext("xlerobot", config.robots["xlerobot"], "test")
    )
    driver.state = "idle"
    driver.last_arm_status = {"success": False, "message": "arm not initialized"}
    driver.last_battery = {"ok": True, "status": "normal"}

    intent = SkillIntent(
        envelope=Envelope(robot_id="xlerobot"),
        skill_id="skill1",
        name="set_gripper",
        arguments={"action": "open"},
        objective="open gripper",
    )
    action = RobotSkillAction("set_gripper", {"action": "open"}).to_robot_action(intent)

    status = await driver.apply_action(action)

    assert status.success is False
    assert status.state == "failed"
    assert "gripper is not ready" in (status.error or "")
    assert status.metrics["last_skill_result"]["failure_mode"] == "readiness_failed"
    assert status.metrics["readiness"]["base"]["ok"] is True
    assert status.metrics["readiness"]["camera"]["owner"] is not None


async def test_xlerobot_status_exposes_multi_arm_and_multi_camera_shape() -> None:
    config = DeploymentConfig.from_dict({"robots": {"xlerobot": {"type": "xlerobot"}}})
    driver = XLeRobotDriver(
        RobotDriverContext("xlerobot", config.robots["xlerobot"], "test")
    )
    driver.state = "idle"
    driver.last_arm_status = {"success": True, "message": "default arm ok"}
    driver.last_arms_status = {
        "left": {"success": True, "message": "left ok"},
        "right": {"success": True, "message": "right ok"},
    }
    driver.last_camera = {
        "frame_available": True,
        "frame_id": 1,
        "image_shape": [1, 1, 3],
    }
    driver.last_cameras_status = {
        "front": {"frame_available": True, "owner": "camera_service", "ok": True},
        "left_wrist": {"frame_available": True, "owner": "robot_driver", "ok": True},
    }
    driver.last_battery = {"status": "normal", "ok": True}
    driver.startup_diagnostics = {
        "base": {"ok": True},
        "arm": {"ok": True},
        "camera": {"ok": True},
    }
    driver.client = SimpleNamespace(base_control_diagnostics=dict)  # type: ignore[assignment]

    status = await driver.status()

    assert "arms" in status.metrics
    assert "cameras" in status.metrics
    assert "left_arm" in status.metrics["readiness"]
    assert "left_wrist_camera" in status.metrics["readiness"]


async def test_xlerobot_observe_merges_camera_frames_into_readiness() -> None:
    class _Image:
        shape = (2, 3, 3)

    def _capture_frames(*, timeout_ms: int = 2000) -> dict[str, dict[str, object]]:
        _ = timeout_ms
        return {"front": {"frame_id": 10, "image": _Image()}}

    config = DeploymentConfig.from_dict({"robots": {"xlerobot": {"type": "xlerobot"}}})
    driver = XLeRobotDriver(
        RobotDriverContext("xlerobot", config.robots["xlerobot"], "test")
    )
    driver.state = "idle"
    driver.client = SimpleNamespace(
        capture_frames=_capture_frames,
        arm_status=lambda: {"success": True, "arms": {"arm": {"success": True}}},
        camera_status=lambda: {
            "ok": False,
            "default_camera": "front",
            "cameras": {
                "front": {
                    "success": False,
                    "ok": False,
                    "owner": "robot_driver",
                    "issue": "stale diagnostic",
                }
            },
        },
        battery_status=lambda: {"status": "normal", "ok": True},
        base_control_diagnostics=dict,
    )  # type: ignore[assignment]

    observation = await driver.observe()
    readiness = observation.metadata["readiness"]

    assert driver.last_cameras_status["front"]["frame_available"] is True
    assert driver.last_cameras_status["front"]["issue"] == "stale diagnostic"
    assert readiness["camera"]["ok"] is True
    assert readiness["front_camera"]["ok"] is True


async def test_xlerobot_observe_reuses_cached_telemetry_between_video_frames() -> None:
    class _Image:
        shape = (2, 3, 3)

    calls = {"arm": 0, "battery": 0}

    def arm_status():
        calls["arm"] += 1
        return {"success": True, "arms": {"arm": {"success": True}}}

    def battery_status():
        calls["battery"] += 1
        return {"status": "normal", "ok": True}

    config = DeploymentConfig.from_dict({"robots": {"xlerobot": {"type": "xlerobot"}}})
    driver = XLeRobotDriver(
        RobotDriverContext("xlerobot", config.robots["xlerobot"], "test")
    )
    driver.client = SimpleNamespace(
        capture_frames=lambda **_kwargs: {"front": {"frame_id": 10, "image": _Image()}},
        camera_status=lambda: {"cameras": {"front": {"success": True, "ok": True}}},
        arm_status=arm_status,
        battery_status=battery_status,
    )  # type: ignore[assignment]

    first = await driver.observe()
    second = await driver.observe()

    assert first.frame_id == 10
    assert second.frame_id == 10
    assert calls == {"arm": 1, "battery": 1}


def test_lekiwi_base_records_failed_wheel_write_for_monitoring() -> None:
    class FakeBus:
        connected = True

        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def write_speed(self, servo_id: int, speed: int) -> bool:
            self.calls.append((servo_id, speed))
            return servo_id != 7

    config = SimpleNamespace(
        enabled=True,
        left_front_id=7,
        right_front_id=9,
        rear_id=8,
        max_linear_speed_mps=0.4,
        max_angular_speed_radps=1.0,
        chassis_radius_m=0.12,
        default_wheel_speed=1000,
    )
    base = LeKiwiBase(FakeBus(), config)  # type: ignore[arg-type]
    base._initialized = True

    response = base.set_velocity(0.1, 0.0, 0.0)

    assert response["success"] is False
    assert response["control"]["failed_servo_id"] == 7
    assert response["control"]["wheel_writes"][0]["servo_id"] == 7
    assert (
        response["control"]["stop_response"]["control"]["wheel_writes"][0][
            "servo_speed"
        ]
        == 0
    )


async def test_xlerobot_status_exposes_base_control_diagnostics() -> None:
    config = DeploymentConfig.from_dict({"robots": {"xlerobot": {"type": "xlerobot"}}})
    driver = XLeRobotDriver(
        RobotDriverContext("xlerobot", config.robots["xlerobot"], "test")
    )
    driver.state = "idle"
    driver.client = SimpleNamespace(
        base_control_diagnostics=lambda: {
            "last_motion_report": {
                "kind": "pulse_velocity",
                "stop_reason": "write_failed",
            },
            "base": {"last_stop_command": {"success": False}},
        }
    )

    status = await driver.status()

    assert (
        status.metrics["base_control"]["last_motion_report"]["stop_reason"]
        == "write_failed"
    )
    assert (
        status.metrics["base_control"]["base"]["last_stop_command"]["success"] is False
    )


async def test_so101_status_exposes_embodiment_readiness_shape() -> None:
    config = DeploymentConfig.from_dict({"robots": {"arm0": {"type": "so101"}}})
    spec = config.robots["arm0"]
    driver = SO101Driver(
        RobotDriverContext(
            "arm0", spec, "test", embodiment=get_embodiment_profile(spec)
        )
    )
    driver.state = "idle"
    driver.last_arm_status = {"success": True}
    driver.last_battery = {"status": "normal", "ok": True}
    driver.startup_diagnostics = {
        "arm": {"ok": True},
        "camera": {"ok": True, "frame_available": True},
    }

    status = await driver.status()
    capabilities = await driver.capabilities()

    assert capabilities.metadata["embodiment_profile"] == "so101_real"
    assert status.metrics["readiness"]["arm"]["ok"] is True
    assert status.metrics["readiness"]["gripper"]["ok"] is True
    assert "camera" in status.metrics["readiness"]


async def test_lekiwi_status_exposes_embodiment_readiness_shape() -> None:
    config = DeploymentConfig.from_dict({"robots": {"base0": {"type": "lekiwi"}}})
    spec = config.robots["base0"]
    driver = LeKiwiDriver(
        RobotDriverContext(
            "base0", spec, "test", embodiment=get_embodiment_profile(spec)
        )
    )
    driver.state = "idle"
    driver.last_battery = {"status": "normal", "ok": True}
    driver.startup_diagnostics = {
        "base": {"ok": True},
        "camera": {"ok": True, "frame_available": True},
    }

    status = await driver.status()
    capabilities = await driver.capabilities()

    assert capabilities.metadata["embodiment_profile"] == "lekiwi_real"
    assert status.metrics["readiness"]["base"]["ok"] is True
    assert "camera" in status.metrics["readiness"]
