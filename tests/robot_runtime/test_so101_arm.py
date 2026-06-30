from __future__ import annotations

import pytest

from hey_robot.robot_runtime.so101 import SO101Arm, SO101ArmConfig


class FakeBus:
    def __init__(
        self,
        config: SO101ArmConfig,
        *,
        connected: bool = True,
        missing_servos: set[int] | None = None,
        write_ok: bool = True,
        torque_disable_ok: bool = True,
    ) -> None:
        self.config = config
        self.connected = connected
        self.missing_servos = missing_servos or set()
        self.write_ok = write_ok
        self.torque_disable_ok = torque_disable_ok
        self.torque_enabled = False
        self.torque_disabled = False
        self.positions = {
            servo_id: int(
                config.angle_offset + config.rest_position[joint] * config.angle_scale
            )
            for joint, servo_id in config.joint_ids.items()
        }

    def ping(self, servo_id: int) -> bool:
        return servo_id not in self.missing_servos

    def torque_enable(self) -> bool:
        self.torque_enabled = True
        return True

    def torque_disable(self, servo_id: int = 254) -> bool:
        _ = servo_id
        self.torque_disabled = True
        return self.torque_disable_ok

    def sync_read_positions(self, servo_ids: list[int]) -> dict[int, int]:
        return {servo_id: self.positions[servo_id] for servo_id in servo_ids}

    def sync_write_positions(self, positions: dict[int, tuple[int, int, int]]) -> bool:
        if not self.write_ok:
            return False
        for servo_id, payload in positions.items():
            self.positions[servo_id] = payload[0]
        return True


def test_so101_arm_supports_relative_joint_moves_named_poses_and_gripper_pct() -> None:
    config = SO101ArmConfig(
        rest_position={
            "base": 0.0,
            "shoulder": 0.0,
            "elbow": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
        named_poses={
            "pregrasp": {
                "base": -60.0,
                "shoulder": 20.0,
                "elbow": 120.0,
                "wrist_flex": 15.0,
                "wrist_roll": 0.0,
                "gripper": 80.0,
            }
        },
    )
    arm = SO101Arm(FakeBus(config), config)  # type: ignore[arg-type]

    init = arm.initialize()
    delta = arm.set_joint_delta("wrist_roll", 15.0)
    pose = arm.move_named_pose("pregrasp")
    gripper = arm.set_gripper_opening_pct(25.0)

    assert init["success"] is True
    assert delta["success"] is True
    assert round(delta["joint_states"]["wrist_roll"], 1) == 15.0
    assert pose["success"] is True
    assert round(pose["joint_states"]["shoulder"], 1) == 20.0
    assert gripper["success"] is True
    assert round(gripper["joint_states"]["gripper"], 1) == 22.5


def test_so101_arm_named_pose_rejects_unknown_pose() -> None:
    config = SO101ArmConfig()
    arm = SO101Arm(FakeBus(config), config)  # type: ignore[arg-type]
    arm.initialize()

    result = arm.move_named_pose("missing_pose")

    assert result["success"] is False
    assert "unknown named pose" in result["message"]


def test_so101_arm_initialization_reports_disabled_disconnected_and_missing_servos() -> (
    None
):
    disabled_config = SO101ArmConfig(enabled=False)
    disabled_arm = SO101Arm(FakeBus(disabled_config), disabled_config)  # type: ignore[arg-type]
    assert disabled_arm.initialize() == {
        "success": True,
        "message": "arm disabled",
        "enabled": False,
    }
    assert disabled_arm.status()["success"] is True

    disconnected_config = SO101ArmConfig()
    disconnected_arm = SO101Arm(
        FakeBus(disconnected_config, connected=False),  # type: ignore[arg-type]
        disconnected_config,
    )
    assert disconnected_arm.initialize() == {
        "success": False,
        "message": "servo bus is not connected",
    }

    config = SO101ArmConfig()
    missing_id = next(iter(config.joint_ids.values()))
    missing_arm = SO101Arm(
        FakeBus(config, missing_servos={missing_id}),  # type: ignore[arg-type]
        config,
    )
    result = missing_arm.initialize()

    assert result["success"] is False
    assert result["missing_servos"] == [missing_id]

    auto_home_config = SO101ArmConfig(auto_home_on_startup=True)
    auto_home_arm = SO101Arm(FakeBus(auto_home_config), auto_home_config)  # type: ignore[arg-type]
    assert auto_home_arm.initialize()["message"] == "arm joints applied"


def test_so101_arm_rejects_uninitialized_unknown_and_failed_writes() -> None:
    config = SO101ArmConfig()
    bus = FakeBus(config)
    arm = SO101Arm(bus, config)  # type: ignore[arg-type]

    assert arm.set_joint("base", 10.0)["message"] == "arm is not initialized"

    assert arm.initialize()["success"] is True
    unknown = arm.set_joint("not_a_joint", 10.0)
    assert unknown["success"] is False
    assert "unknown joint" in unknown["message"]

    bus.write_ok = False
    failed = arm.set_joint("base", 10.0)
    assert failed["success"] is False
    assert failed["message"] == "failed to write arm joint positions"


def test_so101_arm_batch_delta_home_pose_and_gripper_shortcuts() -> None:
    config = SO101ArmConfig()
    arm = SO101Arm(FakeBus(config), config)  # type: ignore[arg-type]
    arm.initialize()

    delta = arm.set_joints_delta({"base": 10.0, "shoulder": -5.0}, speed=222)
    home = arm.move_named_pose("home")
    opened = arm.open_gripper()
    closed = arm.close_gripper()
    clamped_opening = arm.set_gripper_opening_pct(999.0)

    assert delta["success"] is True
    assert round(delta["joint_states"]["base"], 1) == 10.0
    assert delta["joint_states"]["shoulder"] == pytest.approx(
        config.rest_position["shoulder"] - 5.0,
        abs=0.1,
    )
    assert home["success"] is True
    assert opened["joint_states"]["gripper"] == 90.0
    assert closed["joint_states"]["gripper"] == 0.0
    assert clamped_opening["joint_states"]["gripper"] == 90.0


def test_so101_arm_disabled_set_joints_is_noop() -> None:
    config = SO101ArmConfig(enabled=False)
    arm = SO101Arm(FakeBus(config), config)  # type: ignore[arg-type]

    assert arm.set_joints({"base": 30.0}) == {
        "success": True,
        "message": "arm disabled",
    }


def test_so101_arm_clamps_angles_uses_empty_home_and_close_policy() -> None:
    config = SO101ArmConfig(
        auto_home_on_startup=False,
        home_on_close=True,
        joint_limits={"base": (-30.0, 30.0)},
        rest_position={"base": 5.0},
        joint_ids={"base": 1},
    )
    bus = FakeBus(config)
    arm = SO101Arm(bus, config)  # type: ignore[arg-type]

    assert arm.initialize()["success"] is True
    high = arm.set_joint("base", 90.0, speed=321)
    high_position = bus.positions[1]
    low = arm.set_joints({"base": -90.0})
    low_position = bus.positions[1]
    empty = arm.set_joints({})

    assert high["joint_states"]["base"] == 30.0
    assert high_position == arm._angle_to_position(30.0)
    assert low["joint_states"]["base"] == -30.0
    assert low_position == arm._angle_to_position(-30.0)
    assert empty["joint_states"]["base"] == 5.0

    arm.close()
    assert arm.initialized is False
    assert bus.positions[1] == arm._angle_to_position(5.0)


def test_so101_arm_emergency_stop_and_diagnostics_expose_runtime_state() -> None:
    config = SO101ArmConfig()
    bus = FakeBus(config)
    arm = SO101Arm(bus, config)  # type: ignore[arg-type]
    arm.initialize()

    diagnostics = arm.diagnostics()
    stop = arm.emergency_stop()

    assert diagnostics["config"]["joint_ids"] == config.joint_ids
    assert stop == {"success": True, "message": "arm torque disabled"}
    assert bus.torque_disabled is True

    failing = SO101Arm(
        FakeBus(config, torque_disable_ok=False),  # type: ignore[arg-type]
        config,
    )
    assert failing.emergency_stop() == {
        "success": False,
        "message": "failed to disable arm torque",
    }
