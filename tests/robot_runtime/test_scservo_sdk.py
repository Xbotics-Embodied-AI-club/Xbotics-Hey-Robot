from __future__ import annotations

# The SDK mirrors Feetech's camelCase protocol API. Tests intentionally use the
# same method names on fakes to exercise call boundaries exactly.
# ruff: noqa: N802, ANN205, ARG002
import pytest

from hey_robot.robot_runtime.components.scservo_sdk.group_sync_read import GroupSyncRead
from hey_robot.robot_runtime.components.scservo_sdk.group_sync_write import (
    GroupSyncWrite,
)
from hey_robot.robot_runtime.components.scservo_sdk.protocol_packet_handler import (
    ERRBIT_ANGLE,
    ERRBIT_OVERLOAD,
    ERRBIT_VOLTAGE,
    PKT_HEADER0,
    PKT_HEADER1,
    PKT_ID,
    PKT_INSTRUCTION,
    PKT_LENGTH,
    PKT_PARAMETER0,
    RXPACKET_MAX_LEN,
    TXPACKET_MAX_LEN,
    protocol_packet_handler,
)
from hey_robot.robot_runtime.components.scservo_sdk.scservo_def import (
    BROADCAST_ID,
    COMM_NOT_AVAILABLE,
    COMM_PORT_BUSY,
    COMM_RX_CORRUPT,
    COMM_RX_TIMEOUT,
    COMM_SUCCESS,
    COMM_TX_ERROR,
    COMM_TX_FAIL,
    INST_ACTION,
    INST_PING,
    INST_READ,
    INST_REG_WRITE,
    INST_SYNC_READ,
    INST_SYNC_WRITE,
    INST_WRITE,
)
from hey_robot.robot_runtime.components.scservo_sdk.sms_sts import (
    SMS_STS_ACC,
    SMS_STS_LOCK,
    SMS_STS_MODE,
    SMS_STS_PRESENT_LOAD_L,
    SMS_STS_PRESENT_POSITION_L,
    SMS_STS_PRESENT_SPEED_L,
    SMS_STS_PRESENT_VOLTAGE,
    SMS_STS_TORQUE_ENABLE,
    sms_sts,
)


def _status_packet(scs_id: int, error: int = 0, params: list[int] | None = None):
    params = params or []
    packet = [0xFF, 0xFF, scs_id, len(params) + 2, error, *params, 0]
    packet[-1] = (~sum(packet[2:-1])) & 0xFF
    return packet


class FakePort:
    def __init__(self, reads: list[list[int]] | None = None) -> None:
        self.is_using = False
        self.reads = list(reads or [])
        self.writes: list[list[int]] = []
        self.timeouts: list[int] = []
        self.clear_count = 0
        self.short_write = False
        self.timeout = False

    def clearPort(self):
        self.clear_count += 1

    def writePort(self, packet):
        self.writes.append(list(packet))
        return max(0, len(packet) - 1) if self.short_write else len(packet)

    def readPort(self, length):
        if not self.reads:
            return []
        chunk = self.reads.pop(0)
        head, tail = chunk[:length], chunk[length:]
        if tail:
            self.reads.insert(0, tail)
        return head

    def setPacketTimeout(self, length):
        self.timeouts.append(length)

    def isPacketTimeout(self):
        return self.timeout or not self.reads


class RecordingProtocol(protocol_packet_handler):
    def __init__(self):
        super().__init__(FakePort(), 0)
        self.calls: list[tuple] = []
        self.read_results: dict[tuple[int, int], tuple[int, int, int]] = {}
        self.write_result = (COMM_SUCCESS, 0)
        self.action_result = COMM_SUCCESS

    def writeTxRx(self, scs_id, address, length, data):
        self.calls.append(("writeTxRx", scs_id, address, length, list(data)))
        return self.write_result

    def write1ByteTxRx(self, scs_id, address, data):
        self.calls.append(("write1ByteTxRx", scs_id, address, data))
        return self.write_result

    def read1ByteTxRx(self, scs_id, address):
        return self.read_results.get((scs_id, address), (0, COMM_SUCCESS, 0))

    def read2ByteTxRx(self, scs_id, address):
        return self.read_results.get((scs_id, address), (0, COMM_SUCCESS, 0))

    def read4ByteTxRx(self, scs_id, address):
        return self.read_results.get((scs_id, address), (0, COMM_SUCCESS, 0))

    def action(self, scs_id):
        self.calls.append(("action", scs_id))
        return self.action_result


def test_protocol_byte_order_and_signed_conversion_cover_servo_value_encoding():
    little = protocol_packet_handler(FakePort(), 0)
    big = protocol_packet_handler(FakePort(), 1)

    assert little.scs_makeword(0x34, 0x12) == 0x1234
    assert big.scs_makeword(0x34, 0x12) == 0x3412
    assert little.scs_lobyte(0x1234) == 0x34
    assert little.scs_hibyte(0x1234) == 0x12
    assert big.scs_lobyte(0x1234) == 0x12
    assert big.scs_hibyte(0x1234) == 0x34
    assert little.scs_makedword(0x5678, 0x1234) == 0x12345678
    assert little.scs_loword(0x12345678) == 0x5678
    assert little.scs_hiword(0x12345678) == 0x1234
    assert little.scs_toscs(-12, 15) == 0x800C
    assert little.scs_tohost(0x800C, 15) == -12
    little.scs_setend(1)
    assert little.scs_getend() == 1


def test_protocol_result_and_error_messages_are_stable_for_diagnostics():
    ph = protocol_packet_handler(FakePort(), 0)

    assert "success" in ph.getTxRxResult(COMM_SUCCESS)
    assert "Port is in use" in ph.getTxRxResult(COMM_PORT_BUSY)
    assert "Failed transmit" in ph.getTxRxResult(COMM_TX_FAIL)
    assert "Incorrect status" in ph.getTxRxResult(COMM_RX_CORRUPT)
    assert ph.getTxRxResult(12345) == ""
    assert "voltage" in ph.getRxPacketError(ERRBIT_VOLTAGE)
    assert "Angle" in ph.getRxPacketError(ERRBIT_ANGLE)
    assert "Overload" in ph.getRxPacketError(ERRBIT_OVERLOAD)
    assert ph.getRxPacketError(0) == ""


def test_tx_packet_adds_headers_checksum_and_resets_busy_on_failures():
    port = FakePort()
    ph = protocol_packet_handler(port, 0)
    packet = [0] * 8
    packet[PKT_ID] = 3
    packet[PKT_LENGTH] = 4
    packet[PKT_INSTRUCTION] = INST_READ
    packet[PKT_PARAMETER0] = 42
    packet[PKT_PARAMETER0 + 1] = 2

    assert ph.txPacket(packet) == COMM_SUCCESS
    assert port.writes[-1][PKT_HEADER0] == 0xFF
    assert port.writes[-1][PKT_HEADER1] == 0xFF
    assert port.writes[-1][-1] == (~sum(port.writes[-1][2:-1])) & 0xFF
    assert port.is_using is True

    port.is_using = True
    assert ph.txPacket(packet) == COMM_PORT_BUSY

    port.is_using = False
    oversized = [0] * (TXPACKET_MAX_LEN + 10)
    oversized[PKT_LENGTH] = TXPACKET_MAX_LEN
    assert ph.txPacket(oversized) == COMM_TX_ERROR
    assert port.is_using is False

    port.short_write = True
    assert ph.txPacket(packet) == COMM_TX_FAIL
    assert port.is_using is False


def test_rx_packet_recovers_from_noise_and_rejects_corrupt_or_timeout_packets():
    good = _status_packet(7, params=[0x34, 0x12])
    port = FakePort(reads=[[0x00, 0x12], good[:3], good[3:]])
    ph = protocol_packet_handler(port, 0)

    packet, result = ph.rxPacket()

    assert result == COMM_SUCCESS
    assert packet == good
    assert port.is_using is False

    corrupt = _status_packet(8, params=[1])
    corrupt[-1] ^= 0xFF
    packet, result = protocol_packet_handler(FakePort(reads=[corrupt]), 0).rxPacket()
    assert packet == corrupt
    assert result == COMM_RX_CORRUPT

    timeout_port = FakePort(reads=[[]])
    timeout_port.timeout = True
    packet, result = protocol_packet_handler(timeout_port, 0).rxPacket()
    assert packet == []
    assert result == COMM_RX_TIMEOUT


def test_rx_packet_discards_invalid_header_candidates_before_valid_status():
    invalid = [0xFF, 0xFF, 0xFE, RXPACKET_MAX_LEN + 1, 0, 0]
    valid = _status_packet(2, params=[9])
    packet, result = protocol_packet_handler(
        FakePort(reads=[invalid + valid]), 0
    ).rxPacket()

    assert result == COMM_SUCCESS
    assert packet == valid


def test_tx_rx_packet_handles_broadcast_and_filters_wrong_servo_response():
    correct = _status_packet(3, params=[0xAA])
    wrong = _status_packet(4, params=[0xBB])
    port = FakePort(reads=[wrong, correct])
    ph = protocol_packet_handler(port, 0)
    tx = [0] * 8
    tx[PKT_ID] = 3
    tx[PKT_LENGTH] = 4
    tx[PKT_INSTRUCTION] = INST_READ
    tx[PKT_PARAMETER0] = 56
    tx[PKT_PARAMETER0 + 1] = 1

    rx, result, error = ph.txRxPacket(tx)

    assert rx == correct
    assert result == COMM_SUCCESS
    assert error == 0
    assert port.timeouts == [7]

    broadcast = [0] * 6
    broadcast[PKT_ID] = BROADCAST_ID
    broadcast[PKT_LENGTH] = 2
    broadcast[PKT_INSTRUCTION] = INST_ACTION
    rx, result, error = protocol_packet_handler(FakePort(), 0).txRxPacket(broadcast)
    assert rx is None
    assert result == COMM_SUCCESS
    assert error == 0


def test_read_and_write_helpers_encode_registers_and_decode_values():
    port = FakePort(reads=[_status_packet(2, params=[0x34, 0x12])])
    ph = protocol_packet_handler(port, 0)

    value, result, error = ph.read2ByteTxRx(2, 56)

    assert value == 0x1234
    assert result == COMM_SUCCESS
    assert error == 0
    assert port.writes[-1][PKT_INSTRUCTION] == INST_READ
    assert port.writes[-1][PKT_PARAMETER0 : PKT_PARAMETER0 + 2] == [56, 2]

    assert ph.readTx(BROADCAST_ID, 56, 1) == COMM_NOT_AVAILABLE
    data, result, error = ph.readTxRx(BROADCAST_ID, 56, 1)
    assert data == []
    assert result == COMM_NOT_AVAILABLE
    assert error == 0

    port = FakePort(reads=[_status_packet(2)])
    ph = protocol_packet_handler(port, 0)
    result, error = ph.write2ByteTxRx(2, 42, 0x1234)
    assert (result, error) == (COMM_SUCCESS, 0)
    assert port.writes[-1][PKT_INSTRUCTION] == INST_WRITE
    assert port.writes[-1][PKT_PARAMETER0 : PKT_PARAMETER0 + 3] == [42, 0x34, 0x12]

    port = FakePort()
    ph = protocol_packet_handler(port, 0)
    assert ph.write4ByteTxOnly(2, 42, 0x12345678) == COMM_SUCCESS
    assert port.writes[-1][PKT_PARAMETER0 : PKT_PARAMETER0 + 5] == [
        42,
        0x78,
        0x56,
        0x34,
        0x12,
    ]


def test_ping_action_reg_write_and_sync_packets_use_protocol_instructions():
    port = FakePort(reads=[_status_packet(2), _status_packet(2, params=[0x34, 0x12])])
    ph = protocol_packet_handler(port, 0)

    model, result, error = ph.ping(2)
    assert (model, result, error) == (0x1234, COMM_SUCCESS, 0)
    assert port.writes[0][PKT_INSTRUCTION] == INST_PING
    assert ph.ping(BROADCAST_ID + 1) == (0, COMM_NOT_AVAILABLE, 0)

    port = FakePort(reads=[_status_packet(BROADCAST_ID)])
    ph = protocol_packet_handler(port, 0)
    assert ph.action(BROADCAST_ID) == COMM_SUCCESS
    assert port.writes[-1][PKT_INSTRUCTION] == INST_ACTION

    port = FakePort(reads=[_status_packet(2)])
    ph = protocol_packet_handler(port, 0)
    assert ph.regWriteTxRx(2, 42, 2, [1, 2]) == (COMM_SUCCESS, 0)
    assert port.writes[-1][PKT_INSTRUCTION] == INST_REG_WRITE

    port = FakePort()
    ph = protocol_packet_handler(port, 0)
    assert ph.syncReadTx(56, 4, [1, 2, 3], 3) == COMM_SUCCESS
    assert port.writes[-1][PKT_ID] == BROADCAST_ID
    assert port.writes[-1][PKT_INSTRUCTION] == INST_SYNC_READ
    assert port.writes[-1][PKT_PARAMETER0 : PKT_PARAMETER0 + 5] == [56, 4, 1, 2, 3]

    port = FakePort()
    ph = protocol_packet_handler(port, 0)
    assert ph.syncWriteTxOnly(42, 2, [1, 0x34, 0x12], 3) == COMM_SUCCESS
    assert port.writes[-1][PKT_INSTRUCTION] == INST_SYNC_WRITE


def test_sync_read_rx_returns_raw_bulk_response_or_timeout_state():
    packet = _status_packet(1, params=[0x34, 0x12])
    port = FakePort(reads=[packet])
    ph = protocol_packet_handler(port, 0)

    result, rx = ph.syncReadRx(data_length=2, param_length=1)

    assert result == COMM_SUCCESS
    assert rx == packet
    assert port.timeouts == [8]

    timeout_port = FakePort(reads=[packet[:3]])
    result, rx = protocol_packet_handler(timeout_port, 0).syncReadRx(2, 1)
    assert result == COMM_RX_CORRUPT
    assert rx == packet[:3]


def test_group_sync_write_validates_ids_lengths_and_transmits_flattened_payload():
    class FakeProtocol:
        def __init__(self):
            self.calls = []

        def syncWriteTxOnly(self, start_address, data_length, param, param_length):
            self.calls.append((start_address, data_length, list(param), param_length))
            return COMM_SUCCESS

    ph = FakeProtocol()
    sync = GroupSyncWrite(ph, start_address=42, data_length=2)

    assert sync.txPacket() == COMM_NOT_AVAILABLE
    assert sync.addParam(1, [0x34, 0x12]) is True
    assert sync.addParam(1, [0, 0]) is False
    assert sync.addParam(2, [1, 2, 3]) is False
    assert sync.changeParam(1, [0x78, 0x56]) is True
    assert sync.changeParam(9, [0]) is False
    assert sync.txPacket() == COMM_SUCCESS
    assert ph.calls[-1] == (42, 2, [1, 0x78, 0x56], 3)
    sync.removeParam(1)
    assert sync.txPacket() == COMM_NOT_AVAILABLE


def test_group_sync_read_parses_data_checks_availability_and_combines_widths():
    class FakeProtocol:
        def __init__(self):
            self.calls = []

        def syncReadTx(self, start_address, data_length, param, param_length):
            self.calls.append(
                ("tx", start_address, data_length, list(param), param_length)
            )
            return COMM_SUCCESS

        def syncReadRx(self, data_length, param_length):
            packet = _status_packet(1, params=[0x34, 0x12, 0x78, 0x56])
            return COMM_SUCCESS, packet

        @staticmethod
        def scs_makeword(a, b):
            return (a & 0xFF) | ((b & 0xFF) << 8)

        def scs_makedword(self, a, b):
            return (a & 0xFFFF) | ((b & 0xFFFF) << 16)

    ph = FakeProtocol()
    sync = GroupSyncRead(ph, start_address=56, data_length=4)

    assert sync.txPacket() == COMM_NOT_AVAILABLE
    assert sync.addParam(1) is True
    assert sync.addParam(1) is False
    assert sync.txPacket() == COMM_SUCCESS
    assert ph.calls[-1] == ("tx", 56, 4, [1], 1)
    assert sync.rxPacket() == COMM_SUCCESS
    assert sync.last_result is True
    assert sync.isAvailable(1, 56, 2) == (True, 0)
    assert sync.getData(1, 56, 1) == 0x34
    assert sync.getData(1, 56, 2) == 0x1234
    assert sync.getData(1, 56, 4) == 0x56781234
    assert sync.getData(1, 56, 3) == 0
    assert sync.isAvailable(1, 55, 1) == (False, 0)
    sync.removeParam(1)
    assert sync.isAvailable(1, 56, 1) == (False, 0)


def test_group_sync_read_marks_corrupt_packets_and_empty_bulk_responses():
    class CorruptProtocol:
        def syncReadRx(self, data_length, param_length):
            return COMM_SUCCESS, _status_packet(1, params=[0x34])[:-1]

    sync = GroupSyncRead(CorruptProtocol(), start_address=56, data_length=2)
    sync.addParam(1)

    assert sync.rxPacket() == COMM_SUCCESS
    assert sync.last_result is False

    data, result = sync.readRx(_status_packet(1, params=[0x34])[:-1], 1, 1)
    assert data is None
    assert result == COMM_RX_CORRUPT


def test_sms_sts_maps_high_level_methods_to_expected_register_operations():
    servo = sms_sts(FakePort())
    recorder = RecordingProtocol()
    servo.writeTxRx = recorder.writeTxRx
    servo.write1ByteTxRx = recorder.write1ByteTxRx
    servo.read1ByteTxRx = recorder.read1ByteTxRx
    servo.read2ByteTxRx = recorder.read2ByteTxRx
    servo.read4ByteTxRx = recorder.read4ByteTxRx
    servo.action = recorder.action

    assert servo.torque_enable(3) == (COMM_SUCCESS, 0)
    assert recorder.calls[-1] == ("writeTxRx", 3, SMS_STS_TORQUE_ENABLE, 1, [1])
    assert servo.torque_disable() == (COMM_SUCCESS, 0)
    assert recorder.calls[-1] == (
        "writeTxRx",
        BROADCAST_ID,
        SMS_STS_TORQUE_ENABLE,
        1,
        [0],
    )
    assert servo.set_midpoint(4) == (COMM_SUCCESS, 0)
    assert recorder.calls[-1] == ("writeTxRx", 4, SMS_STS_TORQUE_ENABLE, 1, [128])
    assert servo.WheelMode(5) == (COMM_SUCCESS, 0)
    assert recorder.calls[-1] == ("write1ByteTxRx", 5, SMS_STS_MODE, 1)
    assert servo.LockEprom(5) == (COMM_SUCCESS, 0)
    assert recorder.calls[-1] == ("write1ByteTxRx", 5, SMS_STS_LOCK, 1)
    assert servo.unLockEprom(5) == (COMM_SUCCESS, 0)
    assert recorder.calls[-1] == ("write1ByteTxRx", 5, SMS_STS_LOCK, 0)
    assert servo.RegAction() == COMM_SUCCESS
    assert recorder.calls[-1] == ("action", BROADCAST_ID)


def test_sms_sts_position_speed_and_sensor_reads_apply_signed_conversion():
    servo = sms_sts(FakePort())
    recorder = RecordingProtocol()
    servo.read1ByteTxRx = recorder.read1ByteTxRx
    servo.read2ByteTxRx = recorder.read2ByteTxRx
    servo.read4ByteTxRx = recorder.read4ByteTxRx

    recorder.read_results[(2, SMS_STS_PRESENT_POSITION_L)] = (0x800C, COMM_SUCCESS, 0)
    recorder.read_results[(2, SMS_STS_PRESENT_SPEED_L)] = (0x0010, COMM_SUCCESS, 1)
    recorder.read_results[(2, SMS_STS_PRESENT_VOLTAGE)] = (120, COMM_SUCCESS, 0)
    recorder.read_results[(2, SMS_STS_PRESENT_LOAD_L)] = (0x8005, COMM_SUCCESS, 0)

    assert servo.ReadPos(2) == (-12, COMM_SUCCESS, 0)
    assert servo.ReadSpeed(2) == (16, COMM_SUCCESS, 1)
    assert servo.ReadVoltage(2) == (120, COMM_SUCCESS, 0)
    assert servo.ReadLoad(2) == (-5, COMM_SUCCESS, 0)

    packed = servo.scs_makedword(0x800C, 0x0010)
    recorder.read_results[(2, SMS_STS_PRESENT_POSITION_L)] = (packed, COMM_SUCCESS, 0)
    assert servo.ReadPosSpeed(2) == (-12, 16, COMM_SUCCESS, 0)


def test_sms_sts_write_and_sync_methods_build_motion_payloads():
    servo = sms_sts(FakePort())
    recorder = RecordingProtocol()
    servo.writeTxRx = recorder.writeTxRx
    servo.regWriteTxRx = recorder.writeTxRx

    assert servo.WritePosEx(2, -12, 0x1234, 9) == (COMM_SUCCESS, 0)
    assert recorder.calls[-1] == (
        "writeTxRx",
        2,
        SMS_STS_ACC,
        7,
        [9, 0x0C, 0x80, 0, 0, 0x34, 0x12],
    )
    assert servo.WriteSpec(2, -3, 7) == (COMM_SUCCESS, 0)
    assert recorder.calls[-1] == (
        "writeTxRx",
        2,
        SMS_STS_ACC,
        7,
        [7, 0, 0, 0, 0, 3, 0x80],
    )
    assert servo.RegWritePosEx(2, 0x1234, 0x5678, 1) == (COMM_SUCCESS, 0)
    assert recorder.calls[-1][0:4] == ("writeTxRx", 2, SMS_STS_ACC, 7)

    class FakeGroupWrite:
        def __init__(self):
            self.cleared = False
            self.params = []

        def clearParam(self):
            self.cleared = True

        def addParam(self, scs_id, data):
            self.params.append((scs_id, list(data)))
            return True

        def txPacket(self):
            return COMM_SUCCESS

    group = FakeGroupWrite()
    servo.groupSyncWrite = group

    assert servo.SyncWritePosEx({1: (100, 200, 3), 2: (-100, 300, 4)}) == COMM_SUCCESS
    assert group.cleared is True
    assert group.params[0] == (1, [3, 100, 0, 0, 0, 200, 0])
    assert group.params[1] == (2, [4, 100, 0x80, 0, 0, 44, 1])


def test_sms_sts_sync_read_returns_only_available_positions_and_clears_params():
    servo = sms_sts(FakePort())

    class FakeGroupRead:
        def __init__(self):
            self.ids = []
            self.cleared = False

        def addParam(self, scs_id):
            self.ids.append(scs_id)
            return True

        def txRxPacket(self):
            return COMM_SUCCESS

        def isAvailable(self, scs_id, address, data_length):
            assert address == SMS_STS_PRESENT_POSITION_L
            assert data_length == 4
            return (scs_id == 1), 0

        def getData(self, scs_id, address, data_length):
            assert (scs_id, address, data_length) == (1, SMS_STS_PRESENT_POSITION_L, 2)
            return 2048

        def clearParam(self):
            self.cleared = True

    group = FakeGroupRead()
    servo.groupSyncRead = group

    assert servo.SyncReadPos([1, 2]) == {1: 2048}
    assert group.ids == [1, 2]
    assert group.cleared is True


def test_port_handler_guard_reports_clear_error_before_opening_serial_port():
    from hey_robot.robot_runtime.components.scservo_sdk.port_handler import PortHandler

    port = PortHandler("COM_TEST")

    with pytest.raises(RuntimeError, match="serial port is not open"):
        port.clearPort()
    with pytest.raises(RuntimeError, match="serial port is not open"):
        port.closePort()
