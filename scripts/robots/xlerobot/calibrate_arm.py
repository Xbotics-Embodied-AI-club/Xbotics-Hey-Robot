"""校准 XLeRobot SO101 机械臂舵机零点。

用法:
    uv run python scripts/robots/xlerobot/calibrate_arm.py --verify
    uv run python scripts/robots/xlerobot/calibrate_arm.py --joint base
    uv run python scripts/robots/xlerobot/calibrate_arm.py --all --yes

本脚本会写入飞特舵机 EEPROM。只有在机械臂已经手动摆到正确机械零点后才允许执行校准。
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
from hey_robot.robots.components.scservo_sdk import (
    SMS_STS_LOCK,
    SMS_STS_MAX_ANGLE_LIMIT_H,
    SMS_STS_MAX_ANGLE_LIMIT_L,
    SMS_STS_MIN_ANGLE_LIMIT_H,
    SMS_STS_MIN_ANGLE_LIMIT_L,
    SMS_STS_TORQUE_ENABLE,
)

_SEPARATOR = "-" * 60
_CENTER_POSITION = 2048
_DEFAULT_MIN_POSITION = 0
_DEFAULT_MAX_POSITION = 4095

_JOINT_LABELS: dict[str, str] = {
    "base": "底座",
    "shoulder": "肩部",
    "elbow": "肘部",
    "wrist_flex": "腕部俯仰",
    "wrist_roll": "腕部旋转",
    "gripper": "夹爪",
}


class CalibrationReport:
    def __init__(self, report: dict[str, Any]) -> None:
        self._r = report

    def render(self) -> str:
        if self._r.get("error"):
            return (
                "\n  XLeRobot 机械臂零点校准\n"
                "  ========================\n\n"
                "  状态: 失败\n\n"
                f"  {self._r['error']}\n"
            )
        return self._title() + self._joint_table() + self._summary()

    def _title(self) -> str:
        mode = "只读检查" if self._r["verify_only"] else "写入校准"
        return (
            "\n  XLeRobot 机械臂零点校准\n"
            "  ========================\n\n"
            f"  模式: {mode}\n"
            f"  串口: {self._r['serial_port']} @ {self._r['baudrate']} baud\n"
        )

    def _joint_table(self) -> str:
        lines = ["\n  关节\n"]
        for item in self._r["joints"]:
            joint = item["joint"]
            sid = item["servo_id"]
            before = _position_text(item.get("before_position"))
            after = _position_text(item.get("after_position"))
            status = _status_text(item["status"])
            detail = item.get("detail") or ""
            lines.append(
                f"    {_joint_label(joint):<10} ID {sid:>2}  校准前={before:<5}  "
                f"校准后={after:<5}  {status}{('  ' + detail) if detail else ''}"
            )
        return "\n".join(lines)

    def _summary(self) -> str:
        ok = all(item["ok"] for item in self._r["joints"])
        return f"\n\n  {_SEPARATOR}\n\n  结果: {'通过' if ok else '失败'}\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="校准 XLeRobot SO101 机械臂零点。注意：本命令会写入舵机 EEPROM。",
        formatter_class=ChineseHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出。")
    parser.add_argument(
        "--config",
        default=None,
        help="部署配置 YAML 路径。默认按当前系统选择真机配置。",
    )
    parser.add_argument("--robot", default="xlerobot", help="部署配置中的机器人 ID。")
    parser.add_argument(
        "--serial-port",
        default=None,
        help="临时覆盖串口，例如 COM5 或 /dev/ttyUSB0。",
    )
    parser.add_argument(
        "--joint",
        action="append",
        default=[],
        help=(
            "要校准的关节名或舵机 ID，可重复传入。示例：--joint base --joint gripper"
        ),
    )
    parser.add_argument("--all", action="store_true", help="校准所有机械臂关节。")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="只读取当前位置，不写入校准。",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过交互式安全确认。只建议维护人员在确认姿态正确后使用。",
    )
    parser.add_argument(
        "--set-limits",
        action="store_true",
        help="校准后写入每个舵机的最小/最大位置限制。",
    )
    parser.add_argument(
        "--min-position",
        type=int,
        default=_DEFAULT_MIN_POSITION,
        help="配合 --set-limits 写入的最小位置，默认 0。",
    )
    parser.add_argument(
        "--max-position",
        type=int,
        default=_DEFAULT_MAX_POSITION,
        help="配合 --set-limits 写入的最大位置，默认 4095。",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON。")
    args = parser.parse_args()
    if args.config is None:
        args.config = (
            "configs/xlerobot.real.windows.yaml"
            if sys.platform.startswith("win")
            else "configs/xlerobot.real.ubuntu.yaml"
        )
    report = run(args)
    print_json_or_text(report, CalibrationReport(report).render(), as_json=args.json)
    ok = report.get("bus_ok") and all(item["ok"] for item in report.get("joints", []))
    raise SystemExit(0 if ok else 2)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _settings, hardware = load_hardware_config(
        args.config, args.robot, serial_port=args.serial_port
    )
    port = hardware.serial_bus.port
    if not _serial_port_exists(port):
        return {
            "bus_ok": False,
            "serial_port": port,
            "verify_only": args.verify,
            "joints": [],
            "error": (
                f"串口 {port} 不存在。请先连接机器人 USB，并确认配置里的串口号。"
            ),
        }
    selected = _selected_joints(args, hardware.arm.joint_ids)
    if not selected:
        return {
            "bus_ok": False,
            "serial_port": port,
            "verify_only": args.verify,
            "joints": [],
            "error": "未选择关节。请使用 --verify、--joint <名称|ID> 或 --all。",
        }
    confirmed = True if args.verify or args.yes else _confirm(selected)
    if not confirmed:
        return {
            "bus_ok": False,
            "serial_port": port,
            "verify_only": args.verify,
            "joints": [],
            "error": "操作者取消了校准。",
        }

    bus = ServoBus(port, hardware.serial_bus.baudrate)
    try:
        bus_ok = bus.connect()
    except Exception as exc:
        msg = str(exc).lower()
        if "permission denied" in msg or "errno 13" in msg:
            error = f"串口 {port} 没有访问权限。Linux 下请把当前用户加入 dialout 组后重新登录。"
        else:
            error = f"打开串口 {port} 失败: {exc}"
        return {
            "bus_ok": False,
            "serial_port": port,
            "verify_only": args.verify,
            "joints": [],
            "error": error,
        }
    joints: list[dict[str, Any]] = []
    try:
        for joint, servo_id in selected:
            before = bus.read_position(servo_id) if bus_ok else None
            item: dict[str, Any] = {
                "joint": joint,
                "servo_id": servo_id,
                "before_position": before,
                "after_position": None,
                "ok": False,
                "status": "failed",
            }
            if not bus_ok:
                item["detail"] = "总线连接失败"
            elif before is None:
                item["detail"] = "校准前无法读取位置"
            elif args.verify:
                item.update(
                    {
                        "after_position": before,
                        "angle": angle_from_position(
                            before,
                            offset=hardware.arm.angle_offset,
                            scale=hardware.arm.angle_scale,
                        ),
                        "ok": True,
                        "status": "read",
                    }
                )
            else:
                item.update(
                    _calibrate_joint(
                        bus,
                        servo_id,
                        set_limits=args.set_limits,
                        min_position=args.min_position,
                        max_position=args.max_position,
                    )
                )
            joints.append(item)
            time.sleep(0.2)
    finally:
        bus.close()
    return {
        "bus_ok": bus_ok,
        "serial_port": port,
        "baudrate": hardware.serial_bus.baudrate,
        "verify_only": args.verify,
        "set_limits": bool(args.set_limits),
        "joints": joints,
    }


def _selected_joints(
    args: argparse.Namespace, configured: dict[str, int]
) -> list[tuple[str, int]]:
    if args.verify or args.all:
        return list(configured.items())
    if not args.joint:
        return []
    by_id = {sid: name for name, sid in configured.items()}
    selected: list[tuple[str, int]] = []
    for raw in args.joint:
        value = str(raw).strip()
        if value.isdigit():
            sid = int(value)
            name = by_id.get(sid, f"id_{sid}")
            selected.append((name, sid))
            continue
        if value not in configured:
            valid = ", ".join(configured)
            raise SystemExit(f"未知关节 {value!r}。可用关节: {valid}")
        selected.append((value, configured[value]))
    return selected


def _confirm(joints: list[tuple[str, int]]) -> bool:
    print()
    print("警告：本命令会写入飞特舵机 EEPROM。")
    print("继续前必须先把选中的关节手动摆到正确机械零点。")
    print("将要校准的关节：")
    for joint, sid in joints:
        label = _JOINT_LABELS.get(joint, joint)
        print(f"  - {label} (ID {sid})")
    try:
        answer = input("输入 CALIBRATE 继续: ").strip()
    except EOFError:
        return False
    return answer == "CALIBRATE"


def _calibrate_joint(
    bus: ServoBus,
    servo_id: int,
    *,
    set_limits: bool,
    min_position: int,
    max_position: int,
) -> dict[str, Any]:
    if not bus.torque_disable(servo_id):
        return {"ok": False, "status": "failed", "detail": "失能扭矩失败"}
    time.sleep(0.1)
    if not bus.write_u8(servo_id, SMS_STS_TORQUE_ENABLE, 128):
        return {"ok": False, "status": "failed", "detail": "零点写入失败"}
    time.sleep(0.2)
    if not bus.torque_enable(servo_id):
        return {"ok": False, "status": "failed", "detail": "使能扭矩失败"}
    time.sleep(0.1)
    after = bus.read_position(servo_id)
    if after is None:
        return {
            "ok": False,
            "status": "failed",
            "after_position": None,
            "detail": "校准后无法读取位置",
        }
    centered = abs(after - _CENTER_POSITION) <= 10
    limits_ok = True
    if set_limits:
        limits_ok = _set_angle_limits(bus, servo_id, min_position, max_position)
    return {
        "ok": centered and limits_ok,
        "status": "calibrated" if centered and limits_ok else "failed",
        "after_position": after,
        "detail": ""
        if centered and limits_ok
        else f"距中心偏差={after - _CENTER_POSITION}, 限位写入={limits_ok}",
    }


def _set_angle_limits(
    bus: ServoBus, servo_id: int, min_position: int, max_position: int
) -> bool:
    min_position = max(0, min(4095, int(min_position)))
    max_position = max(0, min(4095, int(max_position)))
    if min_position >= max_position:
        return False
    if not bus.write_u8(servo_id, SMS_STS_LOCK, 0):
        return False
    ok = (
        bus.write_u8(servo_id, SMS_STS_MIN_ANGLE_LIMIT_L, min_position & 0xFF)
        and bus.write_u8(servo_id, SMS_STS_MIN_ANGLE_LIMIT_H, min_position >> 8)
        and bus.write_u8(servo_id, SMS_STS_MAX_ANGLE_LIMIT_L, max_position & 0xFF)
        and bus.write_u8(servo_id, SMS_STS_MAX_ANGLE_LIMIT_H, max_position >> 8)
    )
    bus.write_u8(servo_id, SMS_STS_LOCK, 1)
    return ok


def _position_text(value: Any) -> str:
    return "-" if value is None else str(value)


def _joint_label(joint: str) -> str:
    return _JOINT_LABELS.get(joint, joint)


def _status_text(status: str) -> str:
    return {
        "read": "已读取",
        "calibrated": "已校准",
        "failed": "失败",
    }.get(status, status)


def _serial_port_exists(port: str) -> bool:
    if sys.platform.startswith("win") and port.upper().startswith("COM"):
        return True
    return Path(port).exists()


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
