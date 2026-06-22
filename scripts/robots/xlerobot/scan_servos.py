"""扫描 XLeRobot 串行总线上的舵机（Windows / Ubuntu 通用）。

用法:
    uv run python scripts/robots/xlerobot/scan_servos.py                    # 自动选配置
    uv run python scripts/robots/xlerobot/scan_servos.py --serial-port /dev/ttyACM0
    uv run python scripts/robots/xlerobot/scan_servos.py --start-id 1 --end-id 30
    uv run python scripts/robots/xlerobot/scan_servos.py --json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

TOOL_ROOT = Path(__file__).resolve().parent
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from common import load_hardware_config, print_json_or_text

from hey_robot.robots.components import ServoBus

_SEPARATOR = "─" * 60


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


class ServoReport:
    def __init__(self, report: dict[str, Any]) -> None:
        self._r = report

    def render(self) -> str:
        if self._r.get("error"):
            return f"\n  XLeRobot 舵机扫描\n  {'=' * 16}\n\n  状态: 异常\n\n  {self._r['error']}\n"
        return self._title() + self._servo_list() + self._summary()

    def _title(self) -> str:
        r = self._r
        online = sum(1 for s in r["servos"] if s["ok"])
        total = len(r["servos"])
        return (
            f"\n  XLeRobot 舵机扫描\n  {'=' * 16}\n\n"
            f"  串口 {r['serial_port']} @ {r['baudrate']} baud"
            f"    范围 ID {r['scan_range'][0]}..{r['scan_range'][1]}"
            f"    在线 {online}/{total}\n"
        )

    def _servo_list(self) -> str:
        servos = self._r["servos"]
        if not servos:
            return "\n  (无舵机在线)\n"
        lines = ["\n  ▸ 在线舵机\n"]
        kw = 10
        for s in servos:
            if not s["ok"]:
                continue
            sid = f"ID {s['servo_id']:>2}"
            pos = f"pos={s['position']}" if s["position"] is not None else "pos=—"
            lines.append(f"    ✓ {_pad_cjk(sid, kw)}{pos}")
        return "\n".join(lines)

    def _summary(self) -> str:
        r = self._r
        missing = r["missing_expected"]
        lines: list[str] = [
            "",
            f"  {_SEPARATOR}",
            "",
            "  预期舵机",
            f"    底盘    {r['expected']['base']}    缺失: {missing['base'] or '无'}",
            f"    机械臂  {r['expected']['arm']}    缺失: {missing['arm'] or '无'}",
            "",
        ]
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描 XLeRobot 串行总线上的舵机。")
    parser.add_argument(
        "--config", default=None, help="部署配置 YAML 路径；不指定则自动选择"
    )
    parser.add_argument("--robot", default="xlerobot", help="部署配置中的机器人 ID")
    parser.add_argument(
        "--serial-port",
        default=None,
        help="临时覆盖串口（Win: COM5, Ubuntu: /dev/ttyUSB0）",
    )
    parser.add_argument(
        "--start-id", type=int, default=1, help="扫描起始舵机 ID（含），默认 1"
    )
    parser.add_argument(
        "--end-id", type=int, default=20, help="扫描结束舵机 ID（含），默认 20"
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args()
    if args.config is None:
        args.config = (
            "configs/xlerobot.real.windows.yaml"
            if sys.platform.startswith("win")
            else "configs/xlerobot.real.ubuntu.yaml"
        )
    report = scan(args)
    print_json_or_text(report, ServoReport(report).render(), as_json=args.json)
    raise SystemExit(0 if report["bus_ok"] else 2)


def scan(args: argparse.Namespace) -> dict[str, Any]:
    _settings, hardware = load_hardware_config(
        args.config, args.robot, serial_port=args.serial_port
    )
    port = hardware.serial_bus.port
    if not Path(port).exists():
        return {
            "bus_ok": False,
            "serial_port": port,
            "error": (
                f"串口 {port} 不存在。请先连接机器人 USB，确认: ls /dev/ttyUSB* /dev/ttyACM*\n"
                f"  如果串口存在但是 Permission denied:\n"
                f"  方法一: sudo usermod -a -G dialout $USER  然后重新登录\n"
                f'  方法二: sg dialout -c "..."'
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
            "error": (
                f"串口 {port} 无权限。\n"
                f"  方法一: sudo usermod -a -G dialout $USER  然后重新登录\n"
                f'  方法二: sg dialout -c "uv run python scripts/robots/xlerobot/scan_servos.py"'
            ),
        }
    servos = []
    try:
        if bus_ok:
            for sid in range(args.start_id, args.end_id + 1):
                ok = bus.ping(sid)
                servos.append(
                    {
                        "servo_id": sid,
                        "ok": ok,
                        "position": bus.read_position(sid) if ok else None,
                    }
                )
    finally:
        bus.close()
    expected = {
        "base": [
            hardware.base.left_front_id,
            hardware.base.right_front_id,
            hardware.base.rear_id,
        ],
        "arm": list(hardware.arm.joint_ids.values()),
    }
    found_ids = {s["servo_id"] for s in servos if s["ok"]}
    return {
        "bus_ok": bus_ok,
        "serial_port": port,
        "baudrate": hardware.serial_bus.baudrate,
        "scan_range": [args.start_id, args.end_id],
        "servos": servos,
        "expected": expected,
        "missing_expected": {
            g: [i for i in ids if i not in found_ids] for g, ids in expected.items()
        },
    }


if __name__ == "__main__":
    main()
