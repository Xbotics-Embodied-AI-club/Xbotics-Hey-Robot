"""调试 XLeRobot SO101 机械臂状态和安全维护动作。

用法:
    uv run python scripts/robots/xlerobot/debug_arm.py --status
    uv run python scripts/robots/xlerobot/debug_arm.py --disable-torque
    uv run python scripts/robots/xlerobot/debug_arm.py --reset --yes
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

TOOL_ROOT = Path(__file__).resolve().parent
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from common import angle_from_position, load_hardware_config, print_json_or_text

from hey_robot.robots.components import ServoBus

_SEPARATOR = "-" * 60


class ArmDebugReport:
    def __init__(self, report: dict[str, Any]) -> None:
        self._r = report

    def render(self) -> str:
        if self._r.get("error"):
            return (
                "\n  XLeRobot 机械臂调试\n"
                "  ===================\n\n"
                "  状态: 失败\n\n"
                f"  {self._r['error']}\n"
            )
        return self._title() + self._joint_table() + self._summary()

    def _title(self) -> str:
        return (
            "\n  XLeRobot arm debug\n"
            "  ===================\n\n"
            f"  动作: {_action_text(self._r['action'])}\n"
            f"  串口: {self._r['serial_port']} @ {self._r['baudrate']} baud\n"
        )

    def _joint_table(self) -> str:
        lines = ["\n  关节\n"]
        for item in self._r["joints"]:
            joint = item["joint"]
            sid = item["servo_id"]
            position = "-" if item.get("position") is None else str(item["position"])
            angle = (
                "-" if item.get("angle") is None else f"{float(item['angle']):+.1f} 度"
            )
            rest = (
                "-"
                if item.get("rest_angle") is None
                else f"{float(item['rest_angle']):+.1f} 度"
            )
            detail = item.get("detail") or ""
            lines.append(
                f"    {_joint_text(joint):<10} ID {sid:>2}  位置={position:<5}  "
                f"角度={angle:<10} rest={rest:<10} {_status_text(item['status'])}"
                f"{('  ' + detail) if detail else ''}"
            )
        return "\n".join(lines)

    def _summary(self) -> str:
        ok = all(item["ok"] for item in self._r["joints"])
        return f"\n\n  {_SEPARATOR}\n\n  结果: {'通过' if ok else '失败'}\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="调试 XLeRobot SO101 机械臂。",
        formatter_class=ChineseHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出。")
    parser.add_argument("--config", default=None, help="部署配置 YAML 路径。")
    parser.add_argument("--robot", default="xlerobot", help="配置里的机器人 ID。")
    parser.add_argument(
        "--serial-port",
        default=None,
        help="临时覆盖串口，例如 COM5 或 /dev/ttyUSB0。",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true", help="读取关节状态。")
    action.add_argument(
        "--disable-torque",
        action="store_true",
        help="失能机械臂舵机扭矩，方便手动摆位。",
    )
    action.add_argument("--enable-torque", action="store_true", help="使能舵机扭矩。")
    action.add_argument(
        "--reset",
        action="store_true",
        help="移动机械臂到配置里的 rest_position。",
    )
    parser.add_argument("--speed", type=int, default=None, help="覆盖复位速度。")
    parser.add_argument("--acc", type=int, default=None, help="覆盖复位加速度。")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过写入/运动动作的确认。只建议维护人员使用。",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON。")
    args = parser.parse_args()
    if args.config is None:
        args.config = (
            "configs/xlerobot.real.windows.yaml"
            if sys.platform.startswith("win")
            else "configs/xlerobot.real.ubuntu.yaml"
        )
    if not any((args.status, args.disable_torque, args.enable_torque, args.reset)):
        args.status = True
    report = run(args)
    print_json_or_text(report, ArmDebugReport(report).render(), as_json=args.json)
    ok = report.get("bus_ok") and all(item["ok"] for item in report.get("joints", []))
    raise SystemExit(0 if ok else 2)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _settings, hardware = load_hardware_config(
        args.config, args.robot, serial_port=args.serial_port
    )
    action = _action_name(args)
    port = hardware.serial_bus.port
    if not _serial_port_exists(port):
        return _error_report(
            action=action,
            port=port,
            error=f"串口 {port} 不存在。",
        )
    if (
        action in {"disable_torque", "enable_torque", "reset"}
        and not args.yes
        and not _confirm(action)
    ):
        return _error_report(
            action=action,
            port=port,
            error="操作者取消了动作。",
        )

    bus = ServoBus(port, hardware.serial_bus.baudrate)
    try:
        bus_ok = bus.connect()
    except Exception as exc:
        return _error_report(
            action=action,
            port=port,
            error=f"打开串口 {port} 失败: {exc}",
        )
    joints: list[dict[str, Any]] = []
    try:
        if action == "disable_torque":
            joints = _torque(bus, hardware.arm.joint_ids, enable=False)
        elif action == "enable_torque":
            joints = _torque(bus, hardware.arm.joint_ids, enable=True)
        elif action == "reset":
            joints = _reset(bus, hardware.arm, speed=args.speed, acc=args.acc)
        else:
            joints = _status(bus, hardware.arm)
    finally:
        bus.close()
    return {
        "bus_ok": bus_ok,
        "action": action,
        "serial_port": port,
        "baudrate": hardware.serial_bus.baudrate,
        "joints": joints,
    }


def _status(bus: ServoBus, arm: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for joint, servo_id in arm.joint_ids.items():
        position = bus.read_position(servo_id)
        if position is None:
            rows.append(_joint_row(joint, servo_id, ok=False, status="unreadable"))
            continue
        angle = angle_from_position(
            position,
            offset=arm.angle_offset,
            scale=arm.angle_scale,
        )
        rest = arm.rest_position.get(joint)
        lower, upper = arm.joint_limits.get(joint, (-180.0, 180.0))
        within = lower <= angle <= upper
        rows.append(
            _joint_row(
                joint,
                servo_id,
                ok=within,
                status="ok" if within else "out_of_limit",
                position=position,
                angle=angle,
                rest_angle=rest,
                detail="" if rest is None else f"距 rest 偏差={angle - rest:+.1f} 度",
            )
        )
    return rows


def _torque(
    bus: ServoBus, joint_ids: dict[str, int], *, enable: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for joint, servo_id in joint_ids.items():
        ok = bus.torque_enable(servo_id) if enable else bus.torque_disable(servo_id)
        rows.append(
            _joint_row(
                joint,
                servo_id,
                ok=ok,
                status="torque_enabled"
                if enable and ok
                else "torque_disabled"
                if ok
                else "failed",
            )
        )
    return rows


def _reset(
    bus: ServoBus, arm: Any, *, speed: int | None, acc: int | None
) -> list[dict[str, Any]]:
    target_speed = int(speed if speed is not None else arm.default_speed)
    target_acc = int(acc if acc is not None else arm.default_acc)
    positions: dict[int, tuple[int, int, int]] = {}
    for joint, servo_id in arm.joint_ids.items():
        angle = arm.rest_position.get(joint)
        if angle is None:
            continue
        positions[servo_id] = (
            _angle_to_position(angle, offset=arm.angle_offset, scale=arm.angle_scale),
            target_speed,
            target_acc,
        )
    for servo_id in positions:
        bus.torque_enable(servo_id)
    ok = bus.sync_write_positions(positions)
    time.sleep(2.0)
    rows = _status(bus, arm)
    for row in rows:
        row["status"] = "reset_ok" if ok and row["ok"] else "reset_check_failed"
    return rows


def _angle_to_position(angle: float, *, offset: int, scale: float) -> int:
    return max(0, min(4095, round(offset + angle * scale)))


def _joint_row(
    joint: str,
    servo_id: int,
    *,
    ok: bool,
    status: str,
    position: int | None = None,
    angle: float | None = None,
    rest_angle: float | None = None,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "joint": joint,
        "servo_id": servo_id,
        "position": position,
        "angle": angle,
        "rest_angle": rest_angle,
        "ok": ok,
        "status": status,
        "detail": detail,
    }


def _action_name(args: argparse.Namespace) -> str:
    if args.disable_torque:
        return "disable_torque"
    if args.enable_torque:
        return "enable_torque"
    if args.reset:
        return "reset"
    return "status"


def _confirm(action: str) -> bool:
    print()
    print(f"警告：机械臂动作 {_action_text(action)!r} 会写入舵机或移动机械臂。")
    print("请确保人手、线缆和物体远离机械臂活动范围。")
    try:
        answer = input("输入 ARM 继续: ").strip()
    except EOFError:
        return False
    return answer == "ARM"


def _error_report(*, action: str, port: str, error: str) -> dict[str, Any]:
    return {
        "bus_ok": False,
        "action": action,
        "serial_port": port,
        "baudrate": None,
        "joints": [],
        "error": error,
    }


def _serial_port_exists(port: str) -> bool:
    if sys.platform.startswith("win") and port.upper().startswith("COM"):
        return True
    return Path(port).exists()


def _joint_text(joint: str) -> str:
    return {
        "base": "底座",
        "shoulder": "肩部",
        "elbow": "肘部",
        "wrist_flex": "腕部俯仰",
        "wrist_roll": "腕部旋转",
        "gripper": "夹爪",
    }.get(joint, joint)


def _action_text(action: str) -> str:
    return {
        "status": "读取状态",
        "disable_torque": "失能扭矩",
        "enable_torque": "使能扭矩",
        "reset": "复位到 rest_position",
    }.get(action, action)


def _status_text(status: str) -> str:
    return {
        "ok": "正常",
        "unreadable": "无法读取",
        "out_of_limit": "超出限位",
        "torque_enabled": "已使能扭矩",
        "torque_disabled": "已失能扭矩",
        "failed": "失败",
        "reset_ok": "复位正常",
        "reset_check_failed": "复位后检查失败",
    }.get(status, status)


class ChineseHelpFormatter(argparse.HelpFormatter):
    def _format_usage(
        self,
        usage: str | None,
        actions: Any,
        groups: Any,
        prefix: str | None,
    ) -> str:
        return super()._format_usage(usage, actions, groups, prefix or "用法: ")

    def start_section(self, heading: str | None) -> None:
        headings = {
            "positional arguments": "位置参数",
            "options": "选项",
            "optional arguments": "选项",
        }
        super().start_section(headings.get(heading or "", heading))


if __name__ == "__main__":
    main()
