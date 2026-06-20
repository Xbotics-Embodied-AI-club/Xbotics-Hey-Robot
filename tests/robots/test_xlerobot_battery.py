from __future__ import annotations

from hey_robot.protocol import Envelope
from hey_robot.protocol.messages import RobotAction
from hey_robot.robots.base import RobotCapabilities, RobotHealth
from hey_robot.robots.components import (
    ServoBusBattery,
    ServoBusBatteryConfig,
    ServoState,
)
from hey_robot.robots.safety import RobotSafetySupervisor


class FakeBus:
    connected = True

    def __init__(self, state: ServoState) -> None:
        self.state = state

    def read_state(self, servo_id: int) -> ServoState:
        _ = servo_id
        return self.state


def test_xlerobot_battery_reports_voltage_status() -> None:
    monitor = ServoBusBattery(
        FakeBus(ServoState(servo_id=1, voltage=10.4, temperature=31)),  # type: ignore[arg-type]
        ServoBusBatteryConfig(servo_ids=[1], low_voltage=10.5, critical_voltage=9.5),
    )

    state = monitor.read()

    assert state.ok is True
    assert state.status == "low"
    assert state.voltage == 10.4
    assert state.temperature == 31


def test_safety_blocks_non_stop_action_on_critical_battery() -> None:
    supervisor = RobotSafetySupervisor()
    action = RobotAction(
        envelope=Envelope(robot_id="xlerobot"),
        skill_id="cmd1",
        values=[],
        metadata={"action_type": "skill", "skill": {"name": "move_base"}},
    )

    decision = supervisor.evaluate_action(
        action,
        capabilities=RobotCapabilities(robot_id="xlerobot", driver_type="xlerobot"),
        health=RobotHealth(
            robot_id="xlerobot",
            online=True,
            state="idle",
            metrics={"battery": {"status": "critical", "voltage": 9.2}},
        ),
    )

    assert decision.allowed is False
    assert "battery critical" in (decision.reason or "")


def test_safety_allows_emergency_stop_on_critical_battery() -> None:
    supervisor = RobotSafetySupervisor()
    action = RobotAction(
        envelope=Envelope(robot_id="xlerobot"),
        skill_id="cmd1",
        values=[],
        metadata={"action_type": "skill", "skill": {"name": "stop_motion"}},
    )

    decision = supervisor.evaluate_action(
        action,
        capabilities=RobotCapabilities(robot_id="xlerobot", driver_type="xlerobot"),
        health=RobotHealth(
            robot_id="xlerobot",
            online=True,
            state="idle",
            metrics={"battery": {"status": "critical", "voltage": 9.2}},
        ),
    )

    assert decision.allowed is True
