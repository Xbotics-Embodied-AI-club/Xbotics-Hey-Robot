"""检查 XLeRobot 机械臂各关节舵机角度（Windows / Ubuntu 通用）。

用法:
    uv run python scripts/robots/xlerobot/check_arm.py                    # 自动选配置
    uv run python scripts/robots/xlerobot/check_arm.py --serial-port /dev/ttyACM0
    uv run python scripts/robots/xlerobot/check_arm.py --json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

TOOL_ROOT = Path(__file__).resolve().parent
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from common import angle_from_position, load_hardware_config, print_json_or_text

from hey_robot.robots.components import ServoBus

_SEPARATOR = "─" * 60

_JOINT_NAMES: dict[str, str] = {
    "base": "底座",
    "shoulder": "肩部",
    "elbow": "肘部",
    "wrist_flex": "腕俯仰",
    "wrist_roll": "腕旋转",
    "gripper": "夹爪",
}


def _display_width(text: str) -> int:
    w = 0
    for ch in text:
        cp = ord(ch)
        if (
            0x1100 <= cp <= 0x115F
            or 0x2E80 <= cp <= 0xA4CF
            or 0xAC00 <= cp <= 0xD7A3
            or 0xF900 <= cp <= 0xFAFF
            or 0xFF01 <= cp <= 0xFF60
            or 0xFFE0 <= cp <= 0xFFE6
        ):
            w += 2
        else:
            w += 1
    return w


def _pad_cjk(text: str, width: int) -> str:
    dw = _display_width(text)
    return text + " " * (width - dw) if dw < width else text


class ArmReport:
    def __init__(self, report: dict[str, Any]) -> None:
        self._r = report

    def render(self) -> str:
        if self._r.get("error"):
            return f"\n  XLeRobot 机械臂\n  {'=' * 14}\n\n  状态: 异常\n\n  {self._r['error']}\n"
        return self._title() + self._joint_table() + self._summary()

    def _title(self) -> str:
        r = self._r
        bus_icon = "✓" if r["bus_ok"] else "✗"
        return (
            f"\n  XLeRobot 机械臂\n  {'=' * 14}\n\n"
            f"  串口 {r['serial_port']} @ {r['baudrate']} baud"
            f"    总线 {bus_icon}\n"
        )

    def _joint_table(self) -> str:
        joints = self._r["joints"]
        if not joints:
            return "\n  (无关节数据)\n"
        lines = ["\n  ▸ 关节状态\n"]
        # Header
        lines.append(
            f"    {'关节':<10s}{'ID':>4s}  {'原始位置':>6s}  {'角度':>8s}  {'限位':>20s}  {'距rest':>8s}"
        )
        for item in joints:
            if not item["ok"]:
                lines.append(
                    f"    ✗ {item['joint']:<8s}  ID={item['servo_id']}  读不到位置"
                )
                continue
            cn = _JOINT_NAMES.get(item["joint"], item["joint"])
            flag = " ✗超限" if not item["within_limit"] else ""
            delta = item["delta_from_rest"]
            delta_s = f"{delta:+.1f}°" if delta is not None else "—"
            limit = item["limit"]
            limit_s = f"[{limit[0]:.0f}, {limit[1]:.0f}]"
            lines.append(
                f"    {_pad_cjk(cn, 10)}{item['servo_id']:>4d}"
                f"  {item['position']:>6d}"
                f"  {item['angle']:+8.1f}°"
                f"  {_pad_cjk(limit_s, 18)}"
                f"  {delta_s:>8s}"
                f"{flag}"
            )
        return "\n".join(lines)

    def _summary(self) -> str:
        r = self._r
        missing = r["missing_joints"]
        lines: list[str] = [
            "",
            f"  {_SEPARATOR}",
            "",
        ]
        if missing:
            lines.append(f"  ✗ 读不到的关节: {missing}")
            lines.append("    检查这些舵机的接线和 ID 是否冲突。")
        else:
            over_limit = [j for j in r["joints"] if j["ok"] and not j["within_limit"]]
            if over_limit:
                joints_s = ", ".join(
                    _JOINT_NAMES.get(j["joint"], j["joint"]) for j in over_limit
                )
                lines.append(f"  ⚠ 超限关节: {joints_s}")
            else:
                lines.append("  全部关节正常。")
        lines.append("")
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 XLeRobot 机械臂各关节角度。")
    parser.add_argument(
        "--config", default=None, help="部署配置 YAML 路径；不指定则自动选择"
    )
    parser.add_argument("--robot", default="xlerobot", help="部署配置中的机器人 ID")
    parser.add_argument(
        "--serial-port",
        default=None,
        help="临时覆盖串口（Win: COM5, Ubuntu: /dev/ttyUSB0）",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args()
    if args.config is None:
        args.config = (
            "configs/xlerobot.real.windows.yaml"
            if sys.platform.startswith("win")
            else "configs/xlerobot.real.ubuntu.yaml"
        )
    report = check(args)
    print_json_or_text(report, ArmReport(report).render(), as_json=args.json)
    raise SystemExit(0 if report["bus_ok"] and not report["missing_joints"] else 2)


def check(args: argparse.Namespace) -> dict[str, Any]:
    _settings, hardware = load_hardware_config(
        args.config, args.robot, serial_port=args.serial_port
    )
    port = hardware.serial_bus.port
    if not Path(port).exists():
        return {
            "bus_ok": False,
            "serial_port": port,
            "missing_joints": [],
            "error": (
                f"串口 {port} 不存在。请先连接机器人 USB，确认: ls /dev/ttyUSB* /dev/ttyACM*\n"
                f'  如果串口存在但是 Permission denied，用: sg dialout -c "..."'
            ),
        }
    bus = ServoBus(port, hardware.serial_bus.baudrate)
    try:
        bus_ok = bus.connect()
    except Exception as exc:
        msg = str(exc).lower()
        if "permission denied" not in msg and "errno 13" not in msg:
            raise
        return {
            "bus_ok": False,
            "serial_port": port,
            "missing_joints": [],
            "error": (
                f"串口 {port} 无权限。\n"
                f"  方法一: sudo usermod -a -G dialout $USER  然后重新登录\n"
                f'  方法二: sg dialout -c "uv run python scripts/robots/xlerobot/check_arm.py"'
            ),
        }
    joints = []
    missing = []
    try:
        if bus_ok:
            for joint, sid in hardware.arm.joint_ids.items():
                position = bus.read_position(sid)
                if position is None:
                    missing.append(joint)
                    joints.append({"joint": joint, "servo_id": sid, "ok": False})
                    continue
                angle = angle_from_position(
                    position,
                    offset=hardware.arm.angle_offset,
                    scale=hardware.arm.angle_scale,
                )
                lower, upper = hardware.arm.joint_limits.get(joint, (-180.0, 180.0))
                rest = hardware.arm.rest_position.get(joint)
                joints.append(
                    {
                        "joint": joint,
                        "servo_id": sid,
                        "ok": True,
                        "position": position,
                        "angle": angle,
                        "limit": [lower, upper],
                        "within_limit": lower <= angle <= upper,
                        "rest_angle": rest,
                        "delta_from_rest": None if rest is None else angle - rest,
                    }
                )
    finally:
        bus.close()
    return {
        "bus_ok": bus_ok,
        "serial_port": port,
        "baudrate": hardware.serial_bus.baudrate,
        "joints": joints,
        "missing_joints": missing,
    }


if __name__ == "__main__":
    main()
