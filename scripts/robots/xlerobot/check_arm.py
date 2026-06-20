"""检查 XLeRobot 机械臂各关节的舵机角度是否正常。

用途：
  - 机械臂配置好舵机 ID 后，用这个工具读取每个关节的实际角度。
  - 验证舵机接线、ID 设置、角度限位是否正确。
  - 发现某个关节读不到位置（接线松动或 ID 冲突），会单独标出来。

常见用法：
  # 最简单：用配置文件中的串口和关节 ID 检查
  uv run python scripts/robots/xlerobot/check_arm.py

  # 临时覆盖串口
  uv run python scripts/robots/xlerobot/check_arm.py --serial-port COM6

  # 机器可读 JSON 输出
  uv run python scripts/robots/xlerobot/check_arm.py --json

输出说明：
  - 每个关节：关节名 / 舵机 ID / 原始位置 / 角度 / 限位 / 距离 rest 位偏移
  - 超出限位的关节会标 [超限]
  - 读不到位置的关节会标 [失败]

退出码：
  - 0：所有关节读数正常且在限位内
  - 2：有读不到的关节，或某个关节角度超限

更多选项：uv run python scripts/robots/xlerobot/check_arm.py --help
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查 XLeRobot 机械臂各关节角度是否正常。"
    )
    parser.add_argument(
        "--config",
        default="configs/xlerobot.real.windows.yaml",
        help="部署配置 YAML 路径",
    )
    parser.add_argument("--robot", default="xlerobot", help="部署配置中的机器人 ID")
    parser.add_argument("--serial-port", default=None, help="临时覆盖串口，例如 COM5")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args()

    report = check(args)
    print_json_or_text(report, format_report(report), as_json=args.json)
    raise SystemExit(0 if report["bus_ok"] and not report["missing_joints"] else 2)


def check(args: argparse.Namespace) -> dict[str, Any]:
    _settings, hardware = load_hardware_config(
        args.config, args.robot, serial_port=args.serial_port
    )
    bus = ServoBus(hardware.serial_bus.port, hardware.serial_bus.baudrate)
    bus_ok = bus.connect()
    joints = []
    missing = []
    try:
        if bus_ok:
            for joint, servo_id in hardware.arm.joint_ids.items():
                position = bus.read_position(servo_id)
                if position is None:
                    missing.append(joint)
                    joints.append({"joint": joint, "servo_id": servo_id, "ok": False})
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
                        "servo_id": servo_id,
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
        "serial_port": hardware.serial_bus.port,
        "baudrate": hardware.serial_bus.baudrate,
        "joints": joints,
        "missing_joints": missing,
    }


def format_report(report: dict[str, Any]) -> str:
    bus_status = "正常" if report["bus_ok"] else "异常"
    lines = [
        f"XLeRobot 机械臂检查：串口={report['serial_port']} 波特率={report['baudrate']} 总线={bus_status}",
        "",
        "各关节状态：",
    ]
    for item in report["joints"]:
        if not item["ok"]:
            lines.append(
                f"  - {item['joint']}：[失败] 舵机 ID={item['servo_id']} 读不到位置"
            )
            continue
        limit_state = "[正常]" if item["within_limit"] else "[超限]"
        delta = item["delta_from_rest"]
        delta_text = "无" if delta is None else f"{delta:+.1f}"
        lines.append(
            f"  - {item['joint']:<10} 舵机ID={item['servo_id']:<2} "
            f"原始={item['position']:<4} 角度={item['angle']:+7.1f}° "
            f"限位={item['limit']} 距rest={delta_text}° {limit_state}"
        )
    if report["missing_joints"]:
        lines.extend(
            [
                "",
                f"读不到的关节：{report['missing_joints']}",
                "请检查这些舵机的接线和 ID 设置是否冲突。",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
