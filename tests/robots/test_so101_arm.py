from __future__ import annotations

from hey_robot.robots.so101 import SO101Arm, SO101ArmConfig


class FakeBus:
    connected = True

    def __init__(self, config: SO101ArmConfig) -> None:
        self.config = config
        self.positions = {
            servo_id: int(
                config.angle_offset + config.rest_position[joint] * config.angle_scale
            )
            for joint, servo_id in config.joint_ids.items()
        }

    def ping(self, servo_id: int) -> bool:
        _ = servo_id
        return True

    def torque_enable(self) -> bool:
        return True

    def sync_read_positions(self, servo_ids: list[int]) -> dict[int, int]:
        return {servo_id: self.positions[servo_id] for servo_id in servo_ids}

    def sync_write_positions(self, positions: dict[int, tuple[int, int, int]]) -> bool:
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
