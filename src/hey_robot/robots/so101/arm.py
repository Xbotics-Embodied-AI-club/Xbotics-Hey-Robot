from __future__ import annotations

from hey_robot.robots.components import ServoBus
from hey_robot.robots.so101.config import SO101ArmConfig


class SO101Arm:
    """SO101 arm and gripper component."""

    def __init__(self, bus: ServoBus, config: SO101ArmConfig) -> None:
        self.bus = bus
        self.config = config
        self._initialized = False
        self._current_angles: dict[str, float] = {}

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> dict:
        if not self.config.enabled:
            return {"success": True, "message": "arm disabled", "enabled": False}
        if not self.bus.connected:
            return {"success": False, "message": "servo bus is not connected"}
        missing = [
            servo_id
            for servo_id in self.config.joint_ids.values()
            if not self.bus.ping(servo_id)
        ]
        if missing:
            return {
                "success": False,
                "message": f"missing arm servos: {missing}",
                "missing_servos": missing,
            }
        self.bus.torque_enable()
        self._read_positions()
        self._initialized = True
        if self.config.auto_home_on_startup:
            return self.home()
        return {
            "success": True,
            "message": "arm initialized",
            "joint_states": dict(self._current_angles),
        }

    def status(self) -> dict:
        if self.config.enabled and self._initialized:
            self._read_positions()
        return {
            "success": self._initialized or not self.config.enabled,
            "enabled": self.config.enabled,
            "initialized": self._initialized,
            "joint_states": dict(self._current_angles),
            "joint_ids": dict(self.config.joint_ids),
        }

    def home(self, *, speed: int | None = None) -> dict:
        return self.set_joints(dict(self.config.rest_position), speed=speed)

    def set_joint(self, name: str, angle: float, *, speed: int | None = None) -> dict:
        return self.set_joints({name: angle}, speed=speed)

    def set_joint_delta(
        self, name: str, delta_angle: float, *, speed: int | None = None
    ) -> dict:
        self._read_positions()
        current = self._current_angles.get(
            name, self.config.rest_position.get(name, 0.0)
        )
        return self.set_joint(name, current + float(delta_angle), speed=speed)

    def set_joints(self, joints: dict[str, float], *, speed: int | None = None) -> dict:
        if not self.config.enabled:
            return {"success": True, "message": "arm disabled"}
        if not self._initialized:
            return {"success": False, "message": "arm is not initialized"}
        if not joints:
            joints = dict(self.config.rest_position)
        speed = int(speed if speed is not None else self.config.default_speed)
        positions: dict[int, tuple[int, int, int]] = {}
        accepted: dict[str, float] = {}
        for joint, angle in joints.items():
            if joint not in self.config.joint_ids:
                return {"success": False, "message": f"unknown joint: {joint}"}
            clamped = self._clamp_angle(joint, float(angle))
            positions[self.config.joint_ids[joint]] = (
                self._angle_to_position(clamped),
                speed,
                self.config.default_acc,
            )
            accepted[joint] = clamped
        if not self.bus.sync_write_positions(positions):
            return {"success": False, "message": "failed to write arm joint positions"}
        self._current_angles.update(accepted)
        return {
            "success": True,
            "message": "arm joints applied",
            "joint_states": dict(self._current_angles),
        }

    def set_joints_delta(
        self, joints: dict[str, float], *, speed: int | None = None
    ) -> dict:
        self._read_positions()
        absolute = {
            joint: self._current_angles.get(
                joint, self.config.rest_position.get(joint, 0.0)
            )
            + float(delta)
            for joint, delta in joints.items()
        }
        return self.set_joints(absolute, speed=speed)

    def move_named_pose(self, pose_name: str, *, speed: int | None = None) -> dict:
        if pose_name == "home":
            return self.home(speed=speed)
        pose = self.config.named_poses.get(pose_name)
        if pose is None:
            return {"success": False, "message": f"unknown named pose: {pose_name}"}
        return self.set_joints(dict(pose), speed=speed)

    def open_gripper(self) -> dict:
        return self.set_joint("gripper", 90.0)

    def close_gripper(self) -> dict:
        return self.set_joint("gripper", 0.0)

    def set_gripper_opening_pct(self, pct: float, *, speed: int | None = None) -> dict:
        pct = max(0.0, min(100.0, float(pct)))
        angle = 90.0 * (pct / 100.0)
        return self.set_joint("gripper", angle, speed=speed)

    def emergency_stop(self) -> dict:
        ok = self.bus.torque_disable()
        return {
            "success": ok,
            "message": "arm torque disabled" if ok else "failed to disable arm torque",
        }

    def close(self) -> None:
        if self._initialized and self.config.home_on_close:
            self.home()
        self._initialized = False

    def diagnostics(self) -> dict:
        status = self.status()
        status["config"] = {
            "joint_ids": dict(self.config.joint_ids),
            "joint_limits": dict(self.config.joint_limits),
        }
        return status

    def _read_positions(self) -> None:
        positions = self.bus.sync_read_positions(list(self.config.joint_ids.values()))
        for joint, servo_id in self.config.joint_ids.items():
            position = positions.get(servo_id)
            if position is not None:
                self._current_angles[joint] = self._position_to_angle(int(position))

    def _angle_to_position(self, angle: float) -> int:
        position = int(self.config.angle_offset + angle * self.config.angle_scale)
        return max(0, min(4095, position))

    def _position_to_angle(self, position: int) -> float:
        return (position - self.config.angle_offset) / self.config.angle_scale

    def _clamp_angle(self, joint: str, angle: float) -> float:
        lower, upper = self.config.joint_limits.get(joint, (-180.0, 180.0))
        return max(lower, min(upper, angle))
