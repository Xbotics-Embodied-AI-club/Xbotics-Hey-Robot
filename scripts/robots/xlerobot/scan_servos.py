"""扫描 XLeRobot 串行总线上的 Feetech 舵机。

用途：
  - 不知道接了哪些舵机、ID 怎么分布时，用这个工具在指定 ID 范围内 ping 一遍。
  - 验证底盘和机械臂上每个预期的舵机 ID 都能响应。
  - 找出 ID 冲突或接线异常（不响应）的舵机。

常见用法：
  # 默认扫描 ID 1..20
  uv run python scripts/robots/xlerobot/scan_servos.py

  # 扩大扫描范围
  uv run python scripts/robots/xlerobot/scan_servos.py --start-id 1 --end-id 30

  # 临时覆盖串口
  uv run python scripts/robots/xlerobot/scan_servos.py --serial-port COM6

  # 机器可读 JSON 输出
  uv run python scripts/robots/xlerobot/scan_servos.py --json

输出说明：
  - 每个在线舵机的 ID 和当前位置
  - 配置文件预期的底盘/机械臂舵机 ID，标出哪些没找到

退出码：
  - 0：串口连接成功（即使没找到全部预期舵机也算成功）
  - 2：串口连接失败

更多选项：uv run python scripts/robots/xlerobot/scan_servos.py --help
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="扫描 XLeRobot 串行总线上的舵机，列出每个在线舵机的 ID 和位置。"
    )
    parser.add_argument(
        "--config",
        default="configs/xlerobot.real.windows.yaml",
        help="部署配置 YAML 路径",
    )
    parser.add_argument("--robot", default="xlerobot", help="部署配置中的机器人 ID")
    parser.add_argument("--serial-port", default=None, help="临时覆盖串口，例如 COM5")
    parser.add_argument(
        "--start-id", type=int, default=1, help="扫描起始舵机 ID（含），默认 1"
    )
    parser.add_argument(
        "--end-id", type=int, default=20, help="扫描结束舵机 ID（含），默认 20"
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args()

    report = scan(args)
    print_json_or_text(report, format_report(report), as_json=args.json)
    raise SystemExit(0 if report["bus_ok"] else 2)


def scan(args: argparse.Namespace) -> dict[str, Any]:
    _settings, hardware = load_hardware_config(
        args.config, args.robot, serial_port=args.serial_port
    )
    bus = ServoBus(hardware.serial_bus.port, hardware.serial_bus.baudrate)
    bus_ok = bus.connect()
    servos = []
    try:
        if bus_ok:
            for servo_id in range(args.start_id, args.end_id + 1):
                ok = bus.ping(servo_id)
                servos.append(
                    {
                        "servo_id": servo_id,
                        "ok": ok,
                        "position": bus.read_position(servo_id) if ok else None,
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
    found_ids = {item["servo_id"] for item in servos if item["ok"]}
    return {
        "bus_ok": bus_ok,
        "serial_port": hardware.serial_bus.port,
        "baudrate": hardware.serial_bus.baudrate,
        "scan_range": [args.start_id, args.end_id],
        "servos": servos,
        "expected": expected,
        "missing_expected": {
            group: [servo_id for servo_id in ids if servo_id not in found_ids]
            for group, ids in expected.items()
        },
    }


def format_report(report: dict[str, Any]) -> str:
    bus_status = "正常" if report["bus_ok"] else "异常"
    lines = [
        f"XLeRobot 舵机扫描：串口={report['serial_port']} 波特率={report['baudrate']} 总线={bus_status}",
        f"扫描范围：ID {report['scan_range'][0]}..{report['scan_range'][1]}",
        "",
        "在线舵机：",
    ]
    lines.extend(
        f"  - ID {item['servo_id']:2d}：在线  位置={item['position']}"
        for item in report["servos"]
        if item["ok"]
    )
    missing = report["missing_expected"]
    lines.extend(
        [
            "",
            f"底盘预期 ID：{report['expected']['base']}  缺失：{missing['base'] or '无'}",
            f"机械臂预期 ID：{report['expected']['arm']}  缺失：{missing['arm'] or '无'}",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
