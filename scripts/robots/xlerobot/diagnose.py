"""XLeRobot 真机硬件诊断工具。

用途：
  - 一键检查底盘、机械臂、摄像头、电池、舵机总线是否正常工作。
  - 配置好 configs/xlerobot.real.windows.yaml 后启动 runtime 之前，先用这个工具确认硬件链路通畅。
  - 出问题时给出具体的中文排查建议（串口、电源、device_id 等）。
  - 配合 --scan-cameras 会把每个摄像头的视角截图保存到 outputs/diagnostic/cameras/，
    方便确认 device_id 对应的物理摄像头朝向。

常见用法：
  # 最简单：跑一次完整诊断
  uv run python scripts/robots/xlerobot/diagnose.py

  # 顺带扫描所有摄像头并保存视角截图（推荐第一次配置时用）
  uv run python scripts/robots/xlerobot/diagnose.py --scan-cameras

  # 摄像头打不开？临时换 device_id 或后端试试
  uv run python scripts/robots/xlerobot/diagnose.py --camera-device 0 --camera-backend dshow

  # 机器可读 JSON 输出
  uv run python scripts/robots/xlerobot/diagnose.py --json

输出说明：
  - 整体状态：就绪/异常
  - 总线 / 底盘 / 机械臂 / 摄像头 / 电池：每项正常或异常
  - 多摄像头会列出每个摄像头的 device_id、画面分辨率、归属
  - 异常时附"下一步排查建议"

退出码：
  - 0：全部正常（就绪）
  - 2：有异常（具体原因看输出和建议）

更多选项：uv run python scripts/robots/xlerobot/diagnose.py --help
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
from scan_cameras import (
    DEFAULT_SAMPLES_DIR as DEFAULT_CAMERA_SAMPLES_DIR,
    format_camera_scan,
    scan_cameras,
)

from hey_robot.robots.xlerobot.hardware.native import NativeXLeRobotClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="诊断 XLeRobot 真机硬件（底盘、机械臂、摄像头、电池、总线）。"
    )
    parser.add_argument(
        "--config",
        default="configs/xlerobot.real.windows.yaml",
        help="部署配置 YAML 路径",
    )
    parser.add_argument("--robot", default="xlerobot", help="部署配置中的机器人 ID")
    parser.add_argument("--serial-port", default=None, help="临时覆盖串口，例如 COM5")
    parser.add_argument(
        "--camera-device",
        type=int,
        default=None,
        help="临时覆盖默认摄像头 device_id",
    )
    parser.add_argument(
        "--camera-backend",
        default=None,
        help="临时覆盖 OpenCV 后端：auto/dshow/msmf/v4l2",
    )
    parser.add_argument(
        "--video-timeout-ms", type=int, default=1000, help="摄像头取帧超时（毫秒）"
    )
    parser.add_argument(
        "--scan-camera-limit",
        type=int,
        default=5,
        help="摄像头扫描时尝试 device 0..N-1，默认 5",
    )
    parser.add_argument(
        "--scan-cameras",
        action="store_true",
        help="强制扫描所有摄像头（即使配置的摄像头正常）",
    )
    parser.add_argument(
        "--save-camera-samples",
        default=None,
        nargs="?",
        const=str(DEFAULT_CAMERA_SAMPLES_DIR),
        help="扫描时保存视角截图的目录；不传值则使用默认目录；用 --no-camera-samples 关闭",
    )
    parser.add_argument(
        "--no-camera-samples",
        action="store_true",
        help="扫描时不保存视角截图",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args()

    report = diagnose(args)
    print_json_or_text(report, format_report(report), as_json=args.json)
    raise SystemExit(0 if report["ready"] else 2)


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    settings, hardware = load_hardware_config(
        args.config,
        args.robot,
        serial_port=args.serial_port,
        camera_device=args.camera_device,
        camera_backend=args.camera_backend,
    )
    client = NativeXLeRobotClient(
        hardware,
        default_linear_speed=float(settings.get("default_linear_speed", 0.2)),
        default_angular_speed=float(settings.get("default_angular_speed", 0.45)),
        motion_time_scale=float(settings.get("motion_time_scale", 2.0)),
    )
    try:
        client.connect()
        diagnostics = client.diagnose(video_timeout_ms=args.video_timeout_ms)
    finally:
        client.close()

    camera_scan = []
    camera = diagnostics.get("camera") or {}
    if (args.scan_cameras or not bool(camera.get("ok"))) and hardware.camera.enabled:
        if args.no_camera_samples:
            sample_dir = None
        elif args.save_camera_samples is not None:
            sample_dir = Path(args.save_camera_samples)
        else:
            sample_dir = DEFAULT_CAMERA_SAMPLES_DIR
        if sample_dir is not None:
            sample_dir.mkdir(parents=True, exist_ok=True)
        camera_scan = scan_cameras(
            limit=max(args.scan_camera_limit, 0),
            sample_dir=sample_dir,
            backend=hardware.camera.backend,
        )

    ready = all(
        bool((diagnostics.get(service) or {}).get("ok"))
        for service in ("base", "arm", "camera")
    )
    return {
        "ready": ready,
        "robot": args.robot,
        "config": str(Path(args.config)),
        "hardware": {
            "serial_port": hardware.serial_bus.port,
            "baudrate": hardware.serial_bus.baudrate,
            "camera_device_id": hardware.camera.device_id,
            "camera_backend": hardware.camera.backend,
            "base_type": hardware.base.type,
            "base_wheel_ids": [
                hardware.base.left_front_id,
                hardware.base.right_front_id,
                hardware.base.rear_id,
            ],
            "arm_type": hardware.arm.type,
            "arm_joint_ids": dict(hardware.arm.joint_ids),
        },
        "diagnostics": diagnostics,
        "camera_scan": camera_scan,
    }


_SERVICE_LABELS = {
    "base": "底盘",
    "arm": "机械臂",
}


def _format_details(response: dict[str, Any]) -> str:
    key_labels = {
        "enabled": "已启用",
        "initialized": "已初始化",
        "opened": "已打开",
        "device_id": "设备号",
        "backend": "后端",
        "message": "消息",
    }
    parts = []
    for key, label in key_labels.items():
        if key in response:
            parts.append(f"{label}={response[key]}")
    return "  ".join(parts)


def _format_camera_block(camera: dict[str, Any]) -> list[str]:
    cameras = camera.get("cameras") or {}
    default_camera = camera.get("default_camera")
    overall_ok = bool(camera.get("ok"))

    if not cameras:
        status = "正常" if overall_ok else "异常"
        lines = [f"摄像头：{status}"]
        response = camera.get("response")
        if isinstance(response, dict):
            details = _format_details(response)
            if details:
                lines.append(f"  详情：{details}")
        return lines

    count = len(cameras)
    status = "正常" if overall_ok else "异常"
    header = f"摄像头：{status}（默认={default_camera}，共 {count} 路）"
    lines = [header]

    max_name = max(len(name) for name in cameras)
    for name, info in cameras.items():
        cam_ok = bool(info.get("ok"))
        cam_status = "正常" if cam_ok else "异常"
        response = info.get("response") or {}
        parts = []
        if "device_id" in response:
            parts.append(f"设备号={response['device_id']}")
        shape = info.get("image_shape")
        if shape and len(shape) >= 2:
            parts.append(f"画面={shape[1]}x{shape[0]}")
        owner = info.get("owner")
        if owner:
            parts.append(f"归属={owner}")
        if not cam_ok:
            issue = info.get("issue")
            if issue:
                parts.append(f"原因={issue}")
        lines.append(f"  {name.ljust(max_name)} : {cam_status}  " + "  ".join(parts))

    return lines


def format_report(report: dict[str, Any]) -> str:
    hardware = report["hardware"]
    diagnostics = report["diagnostics"]
    ready = bool(report["ready"])
    lines = [
        f"XLeRobot 真机诊断报告：robot={report['robot']} 整体状态={'就绪' if ready else '异常'}",
        (
            "硬件信息："
            f"串口={hardware['serial_port']} 波特率={hardware['baudrate']} "
            f"默认摄像头=device {hardware['camera_device_id']} 后端={hardware['camera_backend']}"
        ),
        f"底盘类型：{hardware['base_type']} 轮子舵机={hardware['base_wheel_ids']}",
        f"机械臂类型：{hardware['arm_type']} 关节舵机={hardware['arm_joint_ids']}",
        "",
    ]

    bus = diagnostics.get("bus") or {}
    bus_messages = {
        "servo bus connected": "舵机总线已连接",
        "servo bus connection failed": "舵机总线连接失败",
    }
    bus_raw = bus.get("message")
    bus_msg = bus_messages.get(bus_raw, bus_raw)
    if bus.get("ok"):
        lines.append(f"总线：正常（{bus_msg}）")
    else:
        lines.append(f"总线：异常（{bus_msg}）")

    for service in ("base", "arm"):
        item = diagnostics.get(service) or {}
        label = _SERVICE_LABELS[service]
        if item.get("ok"):
            lines.append(f"{label}：正常")
        else:
            issue = item.get("issue")
            lines.append(f"{label}：异常" + (f"（{issue}）" if issue else ""))
        response = item.get("response")
        if isinstance(response, dict):
            details = _format_details(response)
            if details:
                lines.append(f"  详情：{details}")

    camera = diagnostics.get("camera") or {}
    lines.extend(_format_camera_block(camera))

    battery = diagnostics.get("battery") or {}
    if battery:
        bat_ok = battery.get("ok")
        status = "正常" if bat_ok else "异常"
        voltage = battery.get("voltage")
        percent = battery.get("percentage")
        voltage_str = f"{voltage}V" if voltage is not None else "未知"
        percent_str = f"{percent:.1f}%" if percent is not None else "未知"
        lines.append(f"电池：{status} 电压={voltage_str} 电量={percent_str}")

    camera_scan = report.get("camera_scan") or []
    if camera_scan:
        lines.extend(
            [
                "",
                "摄像头扫描（依次尝试打开 device 0 到 N）：",
                *format_camera_scan(camera_scan),
            ]
        )

    if not ready:
        lines.extend(
            [
                "",
                "下一步排查建议：",
                "  - 确认没有其他程序占用 COM5 串口",
                "  - 确认 XLeRobot 舵机电源已开启，且串口号与 Windows 设备管理器一致",
                "  - 确认摄像头 device_id 正确；可用 --camera-device N 临时覆盖，或修改 configs/xlerobot.real.windows.yaml",
                "  - 确认依赖已安装：uv sync 或 uv sync --extra agent",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
