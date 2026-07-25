"""Feetech SMS/STS servo protocol adapter for native robots.

Wraps the stateless ``feetech-servo-sdk`` pip package to provide the
stateful API expected by :class:`ServoBus` and the project's calibration
scripts.

The pip SDK is untyped; adapter wrappers propagate ``Any`` through the
dynamically-generated shim.  Disabling ``no-any-return`` here is correct
because we own the bridging layer but not the upstream signatures.
"""

# mypy: disable-error-code="no-any-return"

from __future__ import annotations

import logging
from typing import Any

from scservo_sdk.protocol_packet_handler import protocol_packet_handler
from scservo_sdk.scservo_def import (
    BROADCAST_ID,
    COMM_NOT_AVAILABLE,
    COMM_RX_CORRUPT,
    COMM_RX_FAIL,
    COMM_RX_TIMEOUT,
    COMM_SUCCESS,
)

_logger = logging.getLogger(__name__)

# ── adapter: stateless pip API → stateful vendor API ────────────────────

_PORT_METHODS = [
    "txPacket",
    "rxPacket",
    "txRxPacket",
    "ping",
    "action",
    "readTx",
    "readRx",
    "readTxRx",
    "read1ByteTx",
    "read1ByteRx",
    "read1ByteTxRx",
    "read2ByteTx",
    "read2ByteRx",
    "read2ByteTxRx",
    "read4ByteTx",
    "read4ByteRx",
    "read4ByteTxRx",
    "writeTxOnly",
    "writeTxRx",
    "write1ByteTxOnly",
    "write1ByteTxRx",
    "write2ByteTxOnly",
    "write2ByteTxRx",
    "write4ByteTxOnly",
    "write4ByteTxRx",
    "regWriteTxOnly",
    "regWriteTxRx",
    "syncReadTx",
    "syncWriteTxOnly",
]


def _make_stateful_adapter(base_cls: Any) -> type:
    """Create a stateful adapter class wrapping *base_cls*.

    The pip ``feetech-servo-sdk`` package uses a stateless design where
    every I/O method takes ``port`` as its first argument.  The vendor
    SDK this project was built against stored the port on the instance.
    This adapter restores the stateful convention by injecting
    ``self.port_handler`` into every port-taking method.
    """

    class _StatefulAdapter(base_cls):
        def __init__(self, port_handler: Any, protocol_end: int = 0) -> None:
            super().__init__()
            self.port_handler = port_handler
            self._end = protocol_end

        # ── instance helpers (global functions in pip, methods in vendor) ──
        def scs_getend(self) -> int:
            return self._end

        def scs_setend(self, e: int) -> None:
            self._end = e

        @staticmethod
        def scs_tohost(a: int, b: int) -> int:
            if a & (1 << b):
                return -(a & ~(1 << b))
            return a

        @staticmethod
        def scs_toscs(a: int, b: int) -> int:
            if a < 0:
                return -a | (1 << b)
            return a

        def scs_makeword(self, a: int, b: int) -> int:
            if self.scs_getend() == 0:
                return (a & 0xFF) | ((b & 0xFF) << 8)
            return (b & 0xFF) | ((a & 0xFF) << 8)

        @staticmethod
        def scs_makedword(a: int, b: int) -> int:
            return (a & 0xFFFF) | (b & 0xFFFF) << 16

        @staticmethod
        def scs_loword(word: int) -> int:
            return word & 0xFFFF

        @staticmethod
        def scs_hiword(word: int) -> int:
            return (word >> 16) & 0xFFFF

        def scs_lobyte(self, w: int) -> int:
            if self.scs_getend() == 0:
                return w & 0xFF
            return (w >> 8) & 0xFF

        def scs_hibyte(self, w: int) -> int:
            if self.scs_getend() == 0:
                return (w >> 8) & 0xFF
            return w & 0xFF

        # ── vendor-only: syncReadRx was removed in pip ──
        def syncReadRx(
            self, data_length: int, param_length: int
        ) -> tuple[int, list[int]]:
            wait_length = (6 + data_length) * param_length
            self.port_handler.setPacketTimeout(wait_length)
            rxpacket: list[int] = []
            rx_length = 0
            while True:
                rxpacket.extend(self.port_handler.readPort(wait_length - rx_length))
                rx_length = len(rxpacket)
                if rx_length >= wait_length:
                    result = COMM_SUCCESS
                    break
                if self.port_handler.isPacketTimeout():
                    if rx_length == 0:
                        result = COMM_RX_TIMEOUT
                    else:
                        result = COMM_RX_CORRUPT
                    break
            self.port_handler.is_using = False
            return result, rxpacket

        # ── leaf methods: pip internally passes (self, port, ...) but
        # external callers omit port.  Arg-count dispatch avoids
        # double-injection when pip methods call each other.

        def txPacket(self, *args: Any) -> int:
            if len(args) == 1:  # external: only txpacket
                return base_cls.txPacket(self, self.port_handler, *args)  # type: ignore[no-any-return]
            return base_cls.txPacket(self, *args)  # type: ignore[no-any-return]

        def rxPacket(self, *args: Any) -> tuple[list[int], int]:
            if len(args) == 0:  # external: no args
                return base_cls.rxPacket(self, self.port_handler, *args)  # type: ignore[no-any-return]
            return base_cls.rxPacket(self, *args)  # type: ignore[no-any-return]

    # ── wrap remaining stateless pip methods ──
    # Each wrapper auto-detects whether port was already provided (internal
    # call from another pip method) or needs injection (external call).
    import inspect as _inspect

    for _name in _PORT_METHODS:
        if _name in {"txPacket", "rxPacket"}:
            continue

        _base = getattr(base_cls, _name)
        _self_arg_count = len(_inspect.signature(_base).parameters)

        def _wrapper(
            self: Any,
            *args: Any,
            _method: str = _name,
            _base_fn: Any = _base,
            _expected: int = _self_arg_count,
            **kwargs: Any,
        ) -> Any:
            if len(args) == _expected - 2:  # external (without port)
                return _base_fn(self, self.port_handler, *args, **kwargs)
            return _base_fn(self, *args, **kwargs)

        _wrapper.__name__ = _name
        setattr(_StatefulAdapter, _name, _wrapper)

    return _StatefulAdapter


_StatefulAdapter = _make_stateful_adapter(protocol_packet_handler)

# ── sync helpers (compatible with stateful adapter) ────────────────────


class GroupSyncRead:
    def __init__(self, ph: Any, start_address: int, data_length: int) -> None:
        self.ph = ph
        self.start_address = start_address
        self.data_length = data_length
        self.last_result = False
        self.is_param_changed = False
        self.param: list[int] = []
        self.data_dict: dict[int, list[int]] = {}
        self.clearParam()

    def makeParam(self) -> None:
        if not self.data_dict:
            return
        self.param = []
        for scs_id in self.data_dict:
            self.param.append(scs_id)

    def addParam(self, scs_id: int) -> bool:
        if scs_id in self.data_dict:
            return False
        self.data_dict[scs_id] = []
        self.is_param_changed = True
        return True

    def removeParam(self, scs_id: int) -> None:
        if scs_id not in self.data_dict:
            return
        del self.data_dict[scs_id]
        self.is_param_changed = True

    def clearParam(self) -> None:
        self.data_dict.clear()

    def txPacket(self) -> int:
        if len(self.data_dict.keys()) == 0:
            return COMM_NOT_AVAILABLE
        if self.is_param_changed is True or not self.param:
            self.makeParam()
        return self.ph.syncReadTx(
            self.start_address,
            self.data_length,
            self.param,
            len(self.data_dict.keys()),
        )

    def rxPacket(self) -> int:
        self.last_result = True
        result = COMM_RX_FAIL
        if len(self.data_dict.keys()) == 0:
            return COMM_NOT_AVAILABLE
        result, rxpacket = self.ph.syncReadRx(
            self.data_length, len(self.data_dict.keys())
        )
        if len(rxpacket) >= (self.data_length + 6):
            for scs_id in self.data_dict:
                raw, result = self.readRx(rxpacket, scs_id, self.data_length)
                if raw is not None:
                    self.data_dict[scs_id] = raw
                if result != COMM_SUCCESS:
                    self.last_result = False
        else:
            self.last_result = False
        return result

    def txRxPacket(self) -> int:
        result = self.txPacket()
        if result != COMM_SUCCESS:
            return result
        return self.rxPacket()

    def readRx(
        self, rxpacket: list[int], scs_id: int, data_length: int
    ) -> tuple[list[int] | None, int]:
        data: list[int] = []
        rx_length = len(rxpacket)
        rx_index = 0
        while (rx_index + 6 + data_length) <= rx_length:
            headpacket = [0x00, 0x00, 0x00]
            while rx_index < rx_length:
                headpacket[2] = headpacket[1]
                headpacket[1] = headpacket[0]
                headpacket[0] = rxpacket[rx_index]
                rx_index += 1
                if (
                    (headpacket[2] == 0xFF)
                    and (headpacket[1] == 0xFF)
                    and headpacket[0] == scs_id
                ):
                    break
            if (rx_index + 3 + data_length) > rx_length:
                break
            if rxpacket[rx_index] != (data_length + 2):
                rx_index += 1
                continue
            rx_index += 1
            error = rxpacket[rx_index]
            rx_index += 1
            checksum = scs_id + (data_length + 2) + error
            data = [error]
            data.extend(rxpacket[rx_index : rx_index + data_length])
            for _ in range(data_length):
                checksum += rxpacket[rx_index]
                rx_index += 1
            checksum = ~checksum & 0xFF
            if checksum != rxpacket[rx_index]:
                return None, COMM_RX_CORRUPT
            return data, COMM_SUCCESS
        return None, COMM_RX_CORRUPT

    def isAvailable(
        self, scs_id: int, address: int, data_length: int
    ) -> tuple[bool, int]:
        if scs_id not in self.data_dict:
            return False, 0
        if (address < self.start_address) or (
            self.start_address + self.data_length - data_length < address
        ):
            return False, 0
        if not self.data_dict[scs_id]:
            return False, 0
        if len(self.data_dict[scs_id]) < (data_length + 1):
            return False, 0
        return True, self.data_dict[scs_id][0]

    def getData(self, scs_id: int, address: int, data_length: int) -> int:
        if data_length == 1:
            return self.data_dict[scs_id][address - self.start_address + 1]
        if data_length == 2:
            return self.ph.scs_makeword(
                self.data_dict[scs_id][address - self.start_address + 1],
                self.data_dict[scs_id][address - self.start_address + 2],
            )
        if data_length == 4:
            return self.ph.scs_makedword(
                self.ph.scs_makeword(
                    self.data_dict[scs_id][address - self.start_address + 1],
                    self.data_dict[scs_id][address - self.start_address + 2],
                ),
                self.ph.scs_makeword(
                    self.data_dict[scs_id][address - self.start_address + 3],
                    self.data_dict[scs_id][address - self.start_address + 4],
                ),
            )
        return 0


class GroupSyncWrite:
    def __init__(self, ph: Any, start_address: int, data_length: int) -> None:
        self.ph = ph
        self.start_address = start_address
        self.data_length = data_length
        self.is_param_changed = False
        self.param: list[int] = []
        self.data_dict: dict[int, list[int]] = {}
        self.clearParam()

    def makeParam(self) -> None:
        if not self.data_dict:
            return
        self.param = []
        for scs_id in self.data_dict:
            if not self.data_dict[scs_id]:
                return
            self.param.append(scs_id)
            self.param.extend(self.data_dict[scs_id])

    def addParam(self, scs_id: int, data: list[int]) -> bool:
        if scs_id in self.data_dict:
            return False
        if len(data) > self.data_length:
            return False
        self.data_dict[scs_id] = data
        self.is_param_changed = True
        return True

    def removeParam(self, scs_id: int) -> None:
        if scs_id not in self.data_dict:
            return
        del self.data_dict[scs_id]
        self.is_param_changed = True

    def changeParam(self, scs_id: int, data: list[int]) -> bool:
        if scs_id not in self.data_dict:
            return False
        if len(data) > self.data_length:
            return False
        self.data_dict[scs_id] = data
        self.is_param_changed = True
        return True

    def clearParam(self) -> None:
        self.data_dict.clear()

    def txPacket(self) -> int:
        if len(self.data_dict.keys()) == 0:
            return COMM_NOT_AVAILABLE
        if self.is_param_changed is True or not self.param:
            self.makeParam()
        return self.ph.syncWriteTxOnly(
            self.start_address,
            self.data_length,
            self.param,
            len(self.data_dict.keys()) * (1 + self.data_length),
        )


# ── SMS / STS register map ─────────────────────────────────────────────

# 内存表定义
# -------EPROM(读写)--------
SMS_STS_MIN_ANGLE_LIMIT_L = 9
SMS_STS_MIN_ANGLE_LIMIT_H = 10
SMS_STS_MAX_ANGLE_LIMIT_L = 11
SMS_STS_MAX_ANGLE_LIMIT_H = 12
SMS_STS_MODE = 33

# -------SRAM(读写)--------
SMS_STS_TORQUE_ENABLE = 40
SMS_STS_ACC = 41
SMS_STS_LOCK = 55

# -------SRAM(只读)--------
SMS_STS_PRESENT_POSITION_L = 56
SMS_STS_PRESENT_SPEED_L = 58
SMS_STS_PRESENT_LOAD_L = 60
SMS_STS_PRESENT_VOLTAGE = 62
SMS_STS_PRESENT_TEMPERATURE = 63
SMS_STS_MOVING = 66
SMS_STS_PRESENT_CURRENT_L = 69


# ── SmsSts class ────────────────────────────────────────────────────────


class SmsSts(_StatefulAdapter):  # type: ignore[valid-type,misc]
    """Stateful protocol handler for Feetech SMS/STS-series servo motors."""

    def __init__(self, port_handler: Any) -> None:
        super().__init__(port_handler, 0)
        self.groupSyncWrite = GroupSyncWrite(self, SMS_STS_ACC, 7)
        self.groupSyncRead = GroupSyncRead(self, SMS_STS_PRESENT_POSITION_L, 4)

    def torque_enable(self, scs_id: int = -1) -> tuple[int, int]:
        if scs_id == -1:
            return self.writeTxRx(BROADCAST_ID, SMS_STS_TORQUE_ENABLE, 1, [1])
        return self.writeTxRx(scs_id, SMS_STS_TORQUE_ENABLE, 1, [1])

    def torque_disable(self, scs_id: int = -1) -> tuple[int, int]:
        if scs_id == -1:
            return self.writeTxRx(BROADCAST_ID, SMS_STS_TORQUE_ENABLE, 1, [0])
        return self.writeTxRx(scs_id, SMS_STS_TORQUE_ENABLE, 1, [0])

    def set_midpoint(self, scs_id: int = -1) -> tuple[int, int]:
        if scs_id == -1:
            return self.writeTxRx(BROADCAST_ID, SMS_STS_TORQUE_ENABLE, 1, [128])
        return self.writeTxRx(scs_id, SMS_STS_TORQUE_ENABLE, 1, [128])

    def WritePosEx(
        self, scs_id: int, position: int, speed: int, acc: int
    ) -> tuple[int, int]:
        position = self.scs_toscs(position, 15)
        txpacket = [
            acc,
            self.scs_lobyte(position),
            self.scs_hibyte(position),
            0,
            0,
            self.scs_lobyte(speed),
            self.scs_hibyte(speed),
        ]
        return self.writeTxRx(scs_id, SMS_STS_ACC, len(txpacket), txpacket)

    def ReadPos(self, scs_id: int) -> tuple[int, int, int]:
        scs_present_position, scs_comm_result, scs_error = self.read2ByteTxRx(
            scs_id, SMS_STS_PRESENT_POSITION_L
        )
        return self.scs_tohost(scs_present_position, 15), scs_comm_result, scs_error

    def ReadSpeed(self, scs_id: int) -> tuple[int, int, int]:
        raw, comm_result, scs_error = self.read2ByteTxRx(
            scs_id, SMS_STS_PRESENT_SPEED_L
        )
        return self.scs_tohost(raw, 15), comm_result, scs_error

    def ReadPosSpeed(self, scs_id: int) -> tuple[int, int, int, int]:
        raw, comm_result, scs_error = self.read4ByteTxRx(
            scs_id, SMS_STS_PRESENT_POSITION_L
        )
        scs_present_position = self.scs_loword(raw)
        scs_present_speed = self.scs_hiword(raw)
        return (
            self.scs_tohost(scs_present_position, 15),
            self.scs_tohost(scs_present_speed, 15),
            comm_result,
            scs_error,
        )

    def ReadMoving(self, scs_id: int) -> tuple[int, int, int]:
        moving, scs_comm_result, scs_error = self.read1ByteTxRx(scs_id, SMS_STS_MOVING)
        return moving, scs_comm_result, scs_error

    def ReadVoltage(self, scs_id: int) -> tuple[int, int, int]:
        voltage, scs_comm_result, scs_error = self.read1ByteTxRx(
            scs_id, SMS_STS_PRESENT_VOLTAGE
        )
        return voltage, scs_comm_result, scs_error

    def ReadTemperature(self, scs_id: int) -> tuple[int, int, int]:
        temp, scs_comm_result, scs_error = self.read1ByteTxRx(
            scs_id, SMS_STS_PRESENT_TEMPERATURE
        )
        return temp, scs_comm_result, scs_error

    def ReadLoad(self, scs_id: int) -> tuple[int, int, int]:
        raw, comm_result, scs_error = self.read2ByteTxRx(scs_id, SMS_STS_PRESENT_LOAD_L)
        return self.scs_tohost(raw, 15), comm_result, scs_error

    def ReadCurrent(self, scs_id: int) -> tuple[int, int, int]:
        current, scs_comm_result, scs_error = self.read2ByteTxRx(
            scs_id, SMS_STS_PRESENT_CURRENT_L
        )
        return current, scs_comm_result, scs_error

    def SyncWritePosEx(self, pos_dict: dict[int, tuple[int, int, int]]) -> int:
        self.groupSyncWrite.clearParam()
        for scs_id, params in pos_dict.items():
            position, speed, acc = params
            position = self.scs_toscs(position, 15)
            txpacket = [
                acc,
                self.scs_lobyte(position),
                self.scs_hibyte(position),
                0,
                0,
                self.scs_lobyte(speed),
                self.scs_hibyte(speed),
            ]
            self.groupSyncWrite.addParam(scs_id, txpacket)
        return self.groupSyncWrite.txPacket()

    def SyncReadPos(self, scs_ids: list[int]) -> dict[int, int | None]:
        for scs_id in scs_ids:
            self.groupSyncRead.addParam(scs_id)
        scs_comm_result = self.groupSyncRead.txRxPacket()
        if scs_comm_result != COMM_SUCCESS:
            _logger.warning("%s", self.getTxRxResult(scs_comm_result))
        pos_dict: dict[int, int | None] = {}
        for scs_id in scs_ids:
            scs_data_result, scs_error = self.groupSyncRead.isAvailable(
                scs_id, SMS_STS_PRESENT_POSITION_L, 4
            )
            if scs_data_result is True:
                scs_present_position = self.groupSyncRead.getData(
                    scs_id, SMS_STS_PRESENT_POSITION_L, 2
                )
                pos_dict[scs_id] = scs_present_position
            else:
                _logger.warning("[ID:%03d] groupSyncRead getdata failed", scs_id)
                continue
            if scs_error != 0:
                _logger.warning("%s", self.getRxPacketError(scs_error))
        self.groupSyncRead.clearParam()
        return pos_dict

    def RegWritePosEx(
        self, scs_id: int, position: int, speed: int, acc: int
    ) -> tuple[int, int]:
        position = self.scs_toscs(position, 15)
        txpacket = [
            acc,
            self.scs_lobyte(position),
            self.scs_hibyte(position),
            0,
            0,
            self.scs_lobyte(speed),
            self.scs_hibyte(speed),
        ]
        return self.regWriteTxRx(scs_id, SMS_STS_ACC, len(txpacket), txpacket)

    def RegAction(self) -> int:
        return self.action(BROADCAST_ID)

    def WheelMode(self, scs_id: int) -> int:
        return self.write1ByteTxRx(scs_id, SMS_STS_MODE, 1)

    def WriteSpec(self, scs_id: int, speed: int, acc: int) -> tuple[int, int]:
        speed = self.scs_toscs(speed, 15)
        txpacket = [
            acc,
            0,
            0,
            0,
            0,
            self.scs_lobyte(speed),
            self.scs_hibyte(speed),
        ]
        return self.writeTxRx(scs_id, SMS_STS_ACC, len(txpacket), txpacket)

    def LockEprom(self, scs_id: int) -> int:
        return self.write1ByteTxRx(scs_id, SMS_STS_LOCK, 1)

    def unLockEprom(self, scs_id: int) -> int:
        return self.write1ByteTxRx(scs_id, SMS_STS_LOCK, 0)
