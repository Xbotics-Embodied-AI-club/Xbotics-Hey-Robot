from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from hey_robot.robots.components.scservo_sdk import (
    BROADCAST_ID,
    COMM_SUCCESS,
    SMS_STS_LOCK,
    SMS_STS_MAX_ANGLE_LIMIT_H,
    SMS_STS_MAX_ANGLE_LIMIT_L,
    SMS_STS_MIN_ANGLE_LIMIT_H,
    SMS_STS_MIN_ANGLE_LIMIT_L,
    SMS_STS_MODE,
    SMS_STS_TORQUE_ENABLE,
    PortHandler,
    sms_sts,
)


@dataclass(frozen=True)
class ServoState:
    servo_id: int
    position: int | None = None
    speed: int | None = None
    load: int | None = None
    current: int | None = None
    voltage: float | None = None
    temperature: int | None = None
    moving: bool | None = None


class _PortHandlerProtocol(Protocol):
    openPort: Callable[[], bool]
    setBaudRate: Callable[[int], bool]
    closePort: Callable[[], None]


class _PacketHandlerProtocol(Protocol):
    ping: Callable[[int], tuple[Any, int, Any]]
    WriteSpec: Callable[[int, int, int], tuple[int, Any]]
    WritePosEx: Callable[[int, int, int, int], tuple[int, Any]]
    SyncWritePosEx: Callable[[dict[int, tuple[int, int, int]]], Any]
    ReadPos: Callable[[int], tuple[Any, int, Any]]
    SyncReadPos: Callable[[list[int]], dict[int, int | None]]
    write1ByteTxRx: Callable[[int, int, int], tuple[int, Any]]


class ServoBus:
    """Thread-safe Feetech STS servo bus shared by SO101 and LeKiwi components."""

    def __init__(self, port: str, baudrate: int) -> None:
        self.port = port
        self.baudrate = baudrate
        self._port_handler: _PortHandlerProtocol | None = None
        self._packet_handler: _PacketHandlerProtocol | None = None
        self._connected = False
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        with self._lock:
            if self._connected:
                return True
            self._port_handler = cast(_PortHandlerProtocol, PortHandler(self.port))
            if not self._port_handler.openPort():
                return False
            if not self._port_handler.setBaudRate(self.baudrate):
                self._port_handler.closePort()
                self._port_handler = None
                return False
            self._packet_handler = cast(
                _PacketHandlerProtocol, sms_sts(self._port_handler)
            )
            self._connected = True
            return True

    def close(self) -> None:
        with self._lock:
            if not self._connected:
                return
            if self._port_handler is not None:
                self._port_handler.closePort()
            self._packet_handler = None
            self._port_handler = None
            self._connected = False

    def ping(self, servo_id: int) -> bool:
        with self._lock:
            if not self._connected:
                return False
            packet_handler = self._require_packet_handler()
            _model, comm_result, _error = packet_handler.ping(servo_id)
            return comm_result == COMM_SUCCESS

    def torque_enable(self, servo_id: int = -1) -> bool:
        target = BROADCAST_ID if servo_id == -1 else servo_id
        return self.write_u8(target, SMS_STS_TORQUE_ENABLE, 1)

    def torque_disable(self, servo_id: int = -1) -> bool:
        target = BROADCAST_ID if servo_id == -1 else servo_id
        return self.write_u8(target, SMS_STS_TORQUE_ENABLE, 0)

    def set_wheel_mode(self, servo_id: int) -> bool:
        for _attempt in range(3):
            self.write_u8(servo_id, SMS_STS_TORQUE_ENABLE, 0)
            time.sleep(0.03)
            if not self.write_u8(servo_id, SMS_STS_LOCK, 0):
                continue
            for address in (
                SMS_STS_MIN_ANGLE_LIMIT_L,
                SMS_STS_MIN_ANGLE_LIMIT_H,
                SMS_STS_MAX_ANGLE_LIMIT_L,
                SMS_STS_MAX_ANGLE_LIMIT_H,
            ):
                if not self.write_u8(servo_id, address, 0):
                    break
            else:
                if self.write_u8(servo_id, SMS_STS_MODE, 1):
                    self.write_u8(servo_id, SMS_STS_LOCK, 1)
                    return True
            time.sleep(0.05)
        return False

    def write_speed(self, servo_id: int, speed: int, acc: int = 50) -> bool:
        with self._lock:
            if not self._connected:
                return False
            speed = max(-32767, min(32767, int(speed)))
            packet_handler = self._require_packet_handler()
            comm_result, _error = packet_handler.WriteSpec(servo_id, speed, acc)
            return comm_result == COMM_SUCCESS

    def write_position(
        self, servo_id: int, position: int, speed: int, acc: int
    ) -> bool:
        with self._lock:
            if not self._connected:
                return False
            position = max(0, min(4095, int(position)))
            packet_handler = self._require_packet_handler()
            comm_result, _error = packet_handler.WritePosEx(
                servo_id, position, int(speed), int(acc)
            )
            return comm_result == COMM_SUCCESS

    def sync_write_positions(self, positions: dict[int, tuple[int, int, int]]) -> bool:
        with self._lock:
            if not self._connected:
                return False
            normalized = {
                int(servo_id): (max(0, min(4095, int(pos))), int(speed), int(acc))
                for servo_id, (pos, speed, acc) in positions.items()
            }
            packet_handler = self._require_packet_handler()
            packet_handler.SyncWritePosEx(normalized)
            return True

    def read_position(self, servo_id: int) -> int | None:
        with self._lock:
            if not self._connected:
                return None
            packet_handler = self._require_packet_handler()
            position, comm_result, _error = packet_handler.ReadPos(servo_id)
            return int(position) if comm_result == COMM_SUCCESS else None

    def sync_read_positions(self, servo_ids: list[int]) -> dict[int, int | None]:
        with self._lock:
            if not self._connected:
                return dict.fromkeys(servo_ids)
            packet_handler = self._require_packet_handler()
            return dict(
                packet_handler.SyncReadPos([int(servo_id) for servo_id in servo_ids])
            )

    def read_state(self, servo_id: int) -> ServoState:
        return ServoState(
            servo_id=servo_id,
            position=self.read_position(servo_id),
            speed=self._read("ReadSpeed", servo_id),
            load=self._read("ReadLoad", servo_id),
            current=self._read("ReadCurrent", servo_id),
            voltage=self._read_voltage(servo_id),
            temperature=self._read("ReadTemperature", servo_id),
            moving=_as_bool(self._read("ReadMoving", servo_id)),
        )

    def write_u8(self, servo_id: int, address: int, value: int) -> bool:
        with self._lock:
            if not self._connected:
                return False
            packet_handler = self._require_packet_handler()
            comm_result, _error = packet_handler.write1ByteTxRx(
                servo_id, address, int(value)
            )
            return comm_result == COMM_SUCCESS

    def _read(self, method: str, servo_id: int) -> int | None:
        with self._lock:
            if not self._connected:
                return None
            packet_handler = self._require_packet_handler()
            value, comm_result, _error = getattr(packet_handler, method)(servo_id)
            return int(value) if comm_result == COMM_SUCCESS else None

    def _read_voltage(self, servo_id: int) -> float | None:
        value = self._read("ReadVoltage", servo_id)
        return None if value is None else value / 10.0

    def _require_packet_handler(self) -> _PacketHandlerProtocol:
        if self._packet_handler is None:
            raise RuntimeError("servo packet handler is not initialized")
        return self._packet_handler


def _as_bool(value: int | None) -> bool | None:
    return None if value is None else bool(value)
