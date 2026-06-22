"""XLeRobot 真机硬件诊断工具（Windows / Ubuntu 通用）。

用法:
    uv run python scripts/robots/xlerobot/diagnose.py                        # 自动选配置
    uv run python scripts/robots/xlerobot/diagnose.py --scan-cameras         # 含摄像头扫描
    uv run python scripts/robots/xlerobot/diagnose.py --serial-port /dev/ttyACM0  # 指定串口
    uv run python scripts/robots/xlerobot/diagnose.py --json                 # 机器可读
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

_SEPARATOR = "─" * 60

_LABELS: dict[str, str] = {
    "bus": "舵机总线",
    "base": "底盘",
    "arm": "机械臂",
    "camera": "摄像头",
    "camera_scan": "摄像头扫描",
    "battery": "电池",
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
    if dw >= width:
        return text
    return text + " " * (width - dw)


class DiagReport:
    """诊断报告的格式化输出。"""

    def __init__(self, report: dict[str, Any]) -> None:
        self._r = report

    def render(self) -> str:
        if self._r.get("error"):
            return self._error_block()
        sec: list[str] = []
        sec.append(self._title())
        sec.append(self._hardware_info())
        sec.append(self._subsystems())
        sec.append(self._advice())
        return "\n".join(sec)

    def _error_block(self) -> str:
        return (
            f"\n  XLeRobot 真机诊断\n  {'=' * 16}\n\n"
            f"  状态: 异常\n\n"
            f"  {self._r['message']}\n"
        )

    def _title(self) -> str:
        ready = bool(self._r["ready"])
        status = "✓ 就绪" if ready else "✗ 异常"
        return f"\n  XLeRobot 真机诊断\n  {'=' * 16}\n\n  整体状态: {status}\n"

    def _hardware_info(self) -> str:
        hw = self._r["hardware"]
        kw = 8  # label max display width (摄像头 = 6, but use 8 for spacing)
        return (
            f"\n  ▸ 硬件信息\n"
            f"    {_pad_cjk('串口', kw)}{hw['serial_port']} @ {hw['baudrate']} baud\n"
            f"    {_pad_cjk('摄像头', kw)}device {hw['camera_device_id']}  后端 {hw['camera_backend']}\n"
            f"    {_pad_cjk('底盘', kw)}{hw['base_type']}  舵机 ID: {hw['base_wheel_ids']}\n"
            f"    {_pad_cjk('机械臂', kw)}{hw['arm_type']}  舵机 ID: {hw['arm_joint_ids']}\n"
        )

    def _subsystems(self) -> str:
        diag = self._r["diagnostics"]
        lines: list[str] = ["\n  ▸ 子系统\n"]
        kw = 10  # label max display width (舵机总线 = 8)
        for key in ("bus", "base", "arm", "camera", "battery"):
            item = diag.get(key)
            if item is None:
                continue
            label = _LABELS.get(key, key)
            ok = bool(item.get("ok")) if isinstance(item, dict) else False
            icon = "✓" if ok else "✗"
            detail = self._sub_detail(key, item) if isinstance(item, dict) else ""
            lines.append(f"    {icon} {_pad_cjk(label, kw)}{detail}")
        return "\n".join(lines)

    def _sub_detail(self, key: str, item: dict[str, Any]) -> str:
        if key == "bus":
            msg = item.get("message", "")
            msg_cn = {
                "servo bus connected": "已连接",
                "servo bus connection failed": "连接失败",
            }.get(msg, msg)
            return msg_cn
        if key in ("base", "arm"):
            issue = item.get("issue", "")
            resp = item.get("response") or {}
            enabled = resp.get("enabled")
            initialized = resp.get("initialized")
            parts = []
            if issue:
                parts.append(issue)
            if enabled is not None:
                parts.append(f"enabled={enabled}")
            if initialized is not None:
                parts.append(f"init={initialized}")
            return "  ".join(parts) if parts else "—"
        if key == "camera":
            cameras = item.get("cameras") or {}
            if not cameras:
                return "—"
            names = []
            for name, info in cameras.items():
                cam_ok = "✓" if info.get("ok") else "✗"
                resp = info.get("response") or {}
                dev = resp.get("device_id", "?")
                shape = info.get("image_shape")
                res = f"{shape[1]}x{shape[0]}" if shape and len(shape) >= 2 else "—"
                names.append(f"{name}({cam_ok} dev={dev} {res})")
            return "  ".join(names)
        if key == "battery":
            voltage = item.get("voltage")
            percent = item.get("percentage")
            v = f"{voltage:.1f}V" if voltage is not None else "—"
            p = f"{percent:.0f}%" if percent is not None else "—"
            return f"{v}  {p}"
        return ""

    def _advice(self) -> str:
        lines: list[str] = [
            "",
            f"  {_SEPARATOR}",
            "",
        ]
        camera_scan = self._r.get("camera_scan") or []
        if camera_scan:
            lines.append("  摄像头扫描:")
            lines.extend(f"    {line}" for line in format_camera_scan(camera_scan))
            lines.append("")

        ready = bool(self._r["ready"])
        if ready:
            lines.append("  全部正常，可以启动 runtime。")
        else:
            hw = self._r["hardware"]
            diag = self._r["diagnostics"]
            lines.append("  排查建议:")
            bus_ok = (diag.get("bus") or {}).get("ok")
            if not bus_ok:
                lines.append(f"    - 确认串口 {hw['serial_port']} 未被占用")
                lines.append("    - 确认机器人电源已开启")
                lines.append("    - 确认 USB 线缆连接正常")
            base_ok = (diag.get("base") or {}).get("ok")
            arm_ok = (diag.get("arm") or {}).get("ok")
            if not base_ok or not arm_ok:
                lines.append("    - 舵机无响应: 检查电源、ID 配置、接线")
                lines.append(
                    "    - 可单独扫描舵机: uv run python scripts/robots/xlerobot/scan_servos.py"
                )
            cam_ok = (diag.get("camera") or {}).get("ok")
            if not cam_ok:
                lines.append(
                    "    - 摄像头异常: 用 --scan-cameras 扫描, --camera-device N 重试"
                )
            bat_ok = (diag.get("battery") or {}).get("ok")
            if bat_ok is not None and not bat_ok:
                lines.append("    - 电池读数异常: 检查电池接线和舵机总线")
        lines.append("")
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="诊断 XLeRobot 真机硬件（底盘、机械臂、摄像头、电池、总线）。"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="部署配置 YAML 路径；不指定则自动选择（Windows/Ubuntu）",
    )
    parser.add_argument("--robot", default="xlerobot", help="部署配置中的机器人 ID")
    parser.add_argument(
        "--serial-port",
        default=None,
        help="临时覆盖串口（Win: COM5, Ubuntu: /dev/ttyUSB0）",
    )
    parser.add_argument(
        "--camera-device", type=int, default=None, help="临时覆盖默认摄像头 device_id"
    )
    parser.add_argument(
        "--camera-backend",
        default=None,
        help="临时覆盖 OpenCV 后端（Win: dshow/msmf, Ubuntu: v4l2, 通用: auto）",
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
        "--scan-cameras", action="store_true", help="强制扫描所有摄像头"
    )
    parser.add_argument(
        "--save-camera-samples",
        default=None,
        nargs="?",
        const=str(DEFAULT_CAMERA_SAMPLES_DIR),
        help="扫描时保存视角截图的目录",
    )
    parser.add_argument(
        "--no-camera-samples", action="store_true", help="扫描时不保存视角截图"
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args()

    if args.config is None:
        args.config = (
            "configs/xlerobot.real.windows.yaml"
            if sys.platform.startswith("win")
            else "configs/xlerobot.real.ubuntu.yaml"
        )
    report = diagnose(args)
    print_json_or_text(report, DiagReport(report).render(), as_json=args.json)
    raise SystemExit(0 if report["ready"] else 2)


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    settings, hardware = load_hardware_config(
        args.config,
        args.robot,
        serial_port=args.serial_port,
        camera_device=args.camera_device,
        camera_backend=args.camera_backend,
    )
    port = hardware.serial_bus.port

    if not Path(port).exists():
        return _hw_error(
            args,
            hardware,
            port,
            "serial_port_not_found",
            f"串口 {port} 不存在。\n"
            f"  请先连接机器人 USB，确认: ls /dev/ttyUSB* /dev/ttyACM*\n"
            f"  如果串口不是 {port}，用 --serial-port 指定实际路径。",
        )

    client = NativeXLeRobotClient(
        hardware,
        default_linear_speed=float(settings.get("default_linear_speed", 0.2)),
        default_angular_speed=float(settings.get("default_angular_speed", 0.45)),
        motion_time_scale=float(settings.get("motion_time_scale", 2.0)),
    )
    try:
        try:
            client.connect()
        except Exception as exc:
            msg = str(exc).lower()
            if "permission denied" in msg or "errno 13" in msg:
                return _hw_error(
                    args,
                    hardware,
                    port,
                    "serial_port_permission",
                    f"没有权限访问串口 {port}。\n"
                    f"  方法一（永久）: sudo usermod -a -G dialout $USER  然后重新登录\n"
                    f'  方法二（本次）: sg dialout -c "uv run python scripts/robots/xlerobot/diagnose.py --serial-port {port}"',
                )
            raise
        diagnostics = client.diagnose(video_timeout_ms=args.video_timeout_ms)
    finally:
        client.close()

    camera_scan = _maybe_scan_cameras(args, hardware, diagnostics)
    ready = all(
        bool((diagnostics.get(s) or {}).get("ok")) for s in ("base", "arm", "camera")
    )

    return {
        "ready": ready,
        "robot": args.robot,
        "config": str(Path(args.config)),
        "hardware": {
            "serial_port": port,
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


def _hw_error(
    args: argparse.Namespace, hardware: Any, port: str, error: str, message: str
) -> dict[str, Any]:
    return {
        "ready": False,
        "config": args.config,
        "robot": args.robot,
        "serial_port": port,
        "hardware": {
            "serial_port": port,
            "baudrate": hardware.serial_bus.baudrate,
            "camera_device_id": hardware.camera.device_id,
            "camera_backend": hardware.camera.backend,
            "base_type": hardware.base.type,
            "base_wheel_ids": [],
            "arm_type": hardware.arm.type,
            "arm_joint_ids": {},
        },
        "error": error,
        "message": message,
        "diagnostics": {},
        "camera_scan": [],
    }


def _maybe_scan_cameras(
    args: argparse.Namespace, hardware: Any, diagnostics: dict[str, Any]
) -> list[dict[str, Any]]:
    camera = diagnostics.get("camera") or {}
    if (
        not (args.scan_cameras or not bool(camera.get("ok")))
        or not hardware.camera.enabled
    ):
        return []
    if args.no_camera_samples:
        sample_dir = None
    elif args.save_camera_samples is not None:
        sample_dir = Path(args.save_camera_samples)
    else:
        sample_dir = DEFAULT_CAMERA_SAMPLES_DIR
    if sample_dir is not None:
        sample_dir.mkdir(parents=True, exist_ok=True)
    return scan_cameras(
        limit=max(args.scan_camera_limit, 0),
        sample_dir=sample_dir,
        backend=hardware.camera.backend,
    )


if __name__ == "__main__":
    main()
