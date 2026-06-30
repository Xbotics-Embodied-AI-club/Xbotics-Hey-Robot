from __future__ import annotations

from typing import ClassVar

# ServoBus wraps Feetech SDK camelCase methods. Fakes intentionally mirror that
# API so tests verify the exact hardware-adapter boundary.
# ruff: noqa: N802, ARG005
import pytest

from hey_robot.robot_runtime.components import servo_bus as servo_bus_module
from hey_robot.robot_runtime.components.battery import (
    ServoBusBattery,
    ServoBusBatteryConfig,
)
from hey_robot.robot_runtime.components.scservo_sdk import COMM_SUCCESS
from hey_robot.robot_runtime.components.servo_bus import ServoBus, ServoState


class FakePortHandler:
    instances: ClassVar[list[FakePortHandler]] = []
    open_result: ClassVar[bool] = True
    baud_result: ClassVar[bool] = True

    def __init__(self, port: str) -> None:
        self.port = port
        self.closed = False
        self.baudrates: list[int] = []
        FakePortHandler.instances.append(self)

    def openPort(self) -> bool:
        return self.open_result

    def setBaudRate(self, baudrate: int) -> bool:
        self.baudrates.append(baudrate)
        return self.baud_result

    def closePort(self) -> None:
        self.closed = True


class FakePacketHandler:
    instances: ClassVar[list[FakePacketHandler]] = []

    def __init__(self, _port_handler: FakePortHandler) -> None:
        self.calls: list[tuple] = []
        self.ping_result = COMM_SUCCESS
        self.write_result = COMM_SUCCESS
        self.positions = {1: 1234}
        self.sync_positions = {1: 1234, 2: None}
        self.reads = {
            "ReadSpeed": (22, COMM_SUCCESS, 0),
            "ReadLoad": (-5, COMM_SUCCESS, 0),
            "ReadCurrent": (130, COMM_SUCCESS, 0),
            "ReadVoltage": (121, COMM_SUCCESS, 0),
            "ReadTemperature": (32, COMM_SUCCESS, 0),
            "ReadMoving": (1, COMM_SUCCESS, 0),
        }
        FakePacketHandler.instances.append(self)

    def ping(self, servo_id: int):
        self.calls.append(("ping", servo_id))
        return 0x1234, self.ping_result, 0

    def WriteSpec(self, servo_id: int, speed: int, acc: int):
        self.calls.append(("WriteSpec", servo_id, speed, acc))
        return self.write_result, 0

    def WritePosEx(self, servo_id: int, position: int, speed: int, acc: int):
        self.calls.append(("WritePosEx", servo_id, position, speed, acc))
        return self.write_result, 0

    def SyncWritePosEx(self, positions: dict[int, tuple[int, int, int]]):
        self.calls.append(("SyncWritePosEx", positions))
        return self.write_result

    def ReadPos(self, servo_id: int):
        self.calls.append(("ReadPos", servo_id))
        return self.positions.get(servo_id, 0), self.write_result, 0

    def SyncReadPos(self, servo_ids: list[int]):
        self.calls.append(("SyncReadPos", list(servo_ids)))
        return {servo_id: self.sync_positions.get(servo_id) for servo_id in servo_ids}

    def write1ByteTxRx(self, servo_id: int, address: int, value: int):
        self.calls.append(("write1ByteTxRx", servo_id, address, value))
        return self.write_result, 0

    def __getattr__(self, name: str):
        if name.startswith("Read"):
            return lambda servo_id: self.reads[name]
        raise AttributeError(name)


@pytest.fixture(autouse=True)
def fake_servo_sdk(monkeypatch):
    FakePortHandler.instances = []
    FakePacketHandler.instances = []
    FakePortHandler.open_result = True
    FakePortHandler.baud_result = True
    monkeypatch.setattr(servo_bus_module, "PortHandler", FakePortHandler)
    monkeypatch.setattr(servo_bus_module, "sms_sts", FakePacketHandler)


def _connected_bus() -> tuple[ServoBus, FakePacketHandler, FakePortHandler]:
    bus = ServoBus("COM_TEST", 1_000_000)
    assert bus.connect() is True
    return bus, FakePacketHandler.instances[-1], FakePortHandler.instances[-1]


def test_servo_bus_connect_close_and_failure_paths_are_explicit() -> None:
    bus = ServoBus("COM_TEST", 1_000_000)

    assert bus.connected is False
    assert bus.close() is None
    assert bus.connect() is True
    assert bus.connected is True
    assert bus.connect() is True
    assert len(FakePortHandler.instances) == 1

    port = FakePortHandler.instances[-1]
    bus.close()

    assert port.closed is True
    assert bus.connected is False

    FakePortHandler.open_result = False
    assert ServoBus("COM_BAD", 1_000_000).connect() is False

    FakePortHandler.open_result = True
    FakePortHandler.baud_result = False
    bus = ServoBus("COM_BAD_BAUD", 1_000_000)
    assert bus.connect() is False
    assert FakePortHandler.instances[-1].closed is True


def test_servo_bus_returns_safe_defaults_when_disconnected() -> None:
    bus = ServoBus("COM_TEST", 1_000_000)

    assert bus.ping(1) is False
    assert bus.write_speed(1, 100) is False
    assert bus.write_position(1, 2000, 100, 50) is False
    assert bus.sync_write_positions({1: (2000, 100, 50)}) is False
    assert bus.read_position(1) is None
    assert bus.sync_read_positions([1, 2]) == {1: None, 2: None}
    assert bus.write_u8(1, 40, 1) is False


def test_servo_bus_ping_speed_position_and_batch_write_normalize_values() -> None:
    bus, packet, _port = _connected_bus()

    assert bus.ping(1) is True
    packet.ping_result = -1
    assert bus.ping(1) is False
    packet.ping_result = COMM_SUCCESS

    assert bus.write_speed(1, 999999, acc=7) is True
    assert packet.calls[-1] == ("WriteSpec", 1, 32767, 7)
    assert bus.write_speed(1, -999999, acc=8) is True
    assert packet.calls[-1] == ("WriteSpec", 1, -32767, 8)

    assert bus.write_position(1, -100, speed=200, acc=3) is True
    assert packet.calls[-1] == ("WritePosEx", 1, 0, 200, 3)
    assert bus.write_position(1, 9000, speed=201, acc=4) is True
    assert packet.calls[-1] == ("WritePosEx", 1, 4095, 201, 4)

    assert bus.sync_write_positions({1: (-1, 10, 1), 2: (9000, 11, 2)}) is True
    assert packet.calls[-1] == (
        "SyncWritePosEx",
        {1: (0, 10, 1), 2: (4095, 11, 2)},
    )


def test_servo_bus_torque_and_wheel_mode_write_expected_register_sequence() -> None:
    bus, packet, _port = _connected_bus()

    assert bus.torque_enable() is True
    assert packet.calls[-1] == ("write1ByteTxRx", 254, 40, 1)
    assert bus.torque_disable(3) is True
    assert packet.calls[-1] == ("write1ByteTxRx", 3, 40, 0)

    assert bus.set_wheel_mode(7) is True
    assert packet.calls[-8:] == [
        ("write1ByteTxRx", 7, 40, 0),
        ("write1ByteTxRx", 7, 55, 0),
        ("write1ByteTxRx", 7, 9, 0),
        ("write1ByteTxRx", 7, 10, 0),
        ("write1ByteTxRx", 7, 11, 0),
        ("write1ByteTxRx", 7, 12, 0),
        ("write1ByteTxRx", 7, 33, 1),
        ("write1ByteTxRx", 7, 55, 1),
    ]

    packet.write_result = -1
    assert bus.set_wheel_mode(8) is False


def test_servo_bus_read_state_aggregates_position_telemetry_and_voltage() -> None:
    bus, packet, _port = _connected_bus()

    state = bus.read_state(1)

    assert state == ServoState(
        servo_id=1,
        position=1234,
        speed=22,
        load=-5,
        current=130,
        voltage=12.1,
        temperature=32,
        moving=True,
    )
    assert bus.sync_read_positions([1, 2, 3]) == {1: 1234, 2: None, 3: None}

    packet.write_result = -1
    assert bus.read_position(1) is None
    assert bus.read_state(1).position is None


def test_servo_bus_requires_packet_handler_after_partial_internal_failure() -> None:
    bus = ServoBus("COM_TEST", 1_000_000)
    bus._connected = True
    bus._packet_handler = None

    with pytest.raises(RuntimeError, match="servo packet handler is not initialized"):
        bus.ping(1)


class FakeBatteryBus:
    def __init__(self, *, connected: bool, states: dict[int, ServoState]) -> None:
        self.connected = connected
        self.states = states

    def read_state(self, servo_id: int) -> ServoState:
        return self.states.get(servo_id, ServoState(servo_id=servo_id))


def test_servo_bus_battery_reports_disabled_disconnected_and_missing_voltage() -> None:
    disabled = ServoBusBattery(
        FakeBatteryBus(connected=True, states={}),  # type: ignore[arg-type]
        ServoBusBatteryConfig(enabled=False),
    )
    assert disabled.read().to_dict()["status"] == "disabled"

    disconnected = ServoBusBattery(
        FakeBatteryBus(connected=False, states={}),  # type: ignore[arg-type]
        ServoBusBatteryConfig(),
    )
    disconnected_state = disconnected.read()
    assert disconnected_state.ok is False
    assert disconnected_state.issue == "servo bus is not connected"

    missing = ServoBusBattery(
        FakeBatteryBus(connected=True, states={}),  # type: ignore[arg-type]
        ServoBusBatteryConfig(servo_ids=[1, 2]),
    )
    missing_state = missing.read()
    assert missing_state.ok is False
    assert missing_state.status == "unknown"
    assert "voltage unavailable" in (missing_state.issue or "")


def test_servo_bus_battery_clamps_percentage_and_classifies_thresholds() -> None:
    critical = ServoBusBattery(
        FakeBatteryBus(
            connected=True,
            states={1: ServoState(servo_id=1, voltage=8.0, temperature=40)},
        ),  # type: ignore[arg-type]
        ServoBusBatteryConfig(
            full_voltage=12.0,
            low_voltage=10.5,
            critical_voltage=9.5,
            min_voltage=9.0,
        ),
    ).read()
    assert critical.status == "critical"
    assert critical.percentage == 0.0

    normal = ServoBusBattery(
        FakeBatteryBus(
            connected=True,
            states={1: ServoState(servo_id=1, voltage=13.0, temperature=30)},
        ),  # type: ignore[arg-type]
        ServoBusBatteryConfig(full_voltage=12.0, min_voltage=9.0),
    ).read()
    assert normal.status == "normal"
    assert normal.percentage == 100.0

    invalid_range = ServoBusBattery(
        FakeBatteryBus(
            connected=True,
            states={1: ServoState(servo_id=1, voltage=10.0)},
        ),  # type: ignore[arg-type]
        ServoBusBatteryConfig(full_voltage=9.0, min_voltage=9.0),
    ).read()
    assert invalid_range.percentage == 0.0
