"""列出本机串口、摄像头和音频设备，辅助 Hey Robot 环境配置。

用法:
    uv run python scripts/ops/list_devices.py
    uv run python scripts/ops/list_devices.py --config configs/xlerobot.real.windows.yaml
    uv run python scripts/ops/list_devices.py --test-camera --json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hey_robot.config.model import DeploymentConfig


@dataclass(frozen=True)
class SerialDevice:
    port: str
    description: str
    hwid: str
    vid: str | None = None
    pid: str | None = None
    manufacturer: str | None = None


@dataclass(frozen=True)
class CameraDevice:
    device_id: int
    opened: bool
    backend: str
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_ok: bool | None = None


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_default_input: bool = False
    is_default_output: bool = False


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(
        description="列出串口、摄像头和音频设备。",
        formatter_class=ChineseHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出。")
    parser.add_argument("--config", default=None, help="部署配置 YAML 路径。")
    parser.add_argument(
        "--camera-limit", type=int, default=8, help="要扫描的摄像头编号数量。"
    )
    parser.add_argument(
        "--camera-backend",
        default=None,
        help="覆盖 OpenCV 摄像头后端：dshow、msmf、v4l2 或 auto。",
    )
    parser.add_argument(
        "--test-camera",
        action="store_true",
        help="从每个可打开的摄像头读取一帧。",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON。")
    args = parser.parse_args()

    config = _load_config(args.config)
    backend = args.camera_backend or _default_backend()
    report = {
        "config": config,
        "serial": [asdict(item) for item in list_serial_devices()],
        "cameras": [
            asdict(item)
            for item in list_camera_devices(
                limit=args.camera_limit,
                backend=backend,
                test_frame=args.test_camera,
            )
        ],
        "audio": [asdict(item) for item in list_audio_devices()],
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print(_render(report))


def list_serial_devices() -> list[SerialDevice]:
    try:
        from serial.tools import list_ports
    except Exception:
        return []
    return [
        SerialDevice(
            port=str(port.device),
            description=str(port.description or ""),
            hwid=str(port.hwid or ""),
            vid=f"{port.vid:04X}" if port.vid is not None else None,
            pid=f"{port.pid:04X}" if port.pid is not None else None,
            manufacturer=getattr(port, "manufacturer", None),
        )
        for port in sorted(list_ports.comports(), key=lambda item: item.device)
    ]


def list_camera_devices(
    *, limit: int, backend: str, test_frame: bool
) -> list[CameraDevice]:
    try:
        import cv2
    except Exception:
        return []
    backend_id = _opencv_backend_id(cv2, backend)
    cameras: list[CameraDevice] = []
    for device_id in range(max(0, limit)):
        cap = cv2.VideoCapture(device_id, backend_id)
        try:
            opened = bool(cap.isOpened())
            if not opened:
                continue
            frame_ok: bool | None = None
            if test_frame:
                frame_ok, _frame = cap.read()
                frame_ok = bool(frame_ok)
            cameras.append(
                CameraDevice(
                    device_id=device_id,
                    opened=opened,
                    backend=cap.getBackendName()
                    if hasattr(cap, "getBackendName")
                    else backend,
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
                    fps=float(cap.get(cv2.CAP_PROP_FPS)) or None,
                    frame_ok=frame_ok,
                )
            )
        finally:
            cap.release()
    return cameras


def list_audio_devices() -> list[AudioDevice]:
    try:
        import sounddevice as sd
    except Exception:
        return []
    try:
        raw = sd.query_devices()
        default_in, default_out = sd.default.device
    except Exception:
        return []
    devices: list[AudioDevice] = []
    for index, item in enumerate(raw):
        devices.append(
            AudioDevice(
                index=index,
                name=str(item.get("name", "")),
                max_input_channels=int(item.get("max_input_channels") or 0),
                max_output_channels=int(item.get("max_output_channels") or 0),
                default_samplerate=float(item.get("default_samplerate") or 0),
                is_default_input=isinstance(default_in, int) and index == default_in,
                is_default_output=isinstance(default_out, int) and index == default_out,
            )
        )
    return devices


def _load_config(config_path: str | None) -> dict[str, Any] | None:
    if not config_path:
        return None
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return {"path": str(path), "error": "配置文件不存在"}
    deployment = DeploymentConfig.from_yaml(path)
    config: dict[str, Any] = {
        "path": str(path),
        "deployment_id": deployment.deployment.id,
        "serial_ports": [],
        "cameras": [],
        "web_ports": [],
        "audio": [],
    }
    for robot_id, robot in deployment.robots.items():
        settings = robot.settings
        serial_bus = settings.get("serial_bus")
        if isinstance(serial_bus, dict):
            config["serial_ports"].append(
                {"robot": robot_id, "port": str(serial_bus.get("port", ""))}
            )
        components = settings.get("components")
        if isinstance(components, dict):
            cameras = components.get("cameras")
            if isinstance(cameras, dict):
                for role, camera in cameras.items():
                    if isinstance(camera, dict):
                        config["cameras"].append(
                            {
                                "robot": robot_id,
                                "role": str(role),
                                "device_id": camera.get("device_id"),
                                "backend": camera.get("backend"),
                            }
                        )
    for channel_id, channel in deployment.channels.items():
        if channel_id == "web" and channel.enabled:
            config["web_ports"].append(
                {
                    "channel": channel_id,
                    "host": channel.settings.get("host", "127.0.0.1"),
                    "port": channel.settings.get("port", 8080),
                }
            )
        if channel_id == "voice" and channel.enabled:
            recorder = channel.settings.get("recorder")
            tts = channel.settings.get("tts")
            config["audio"].append(
                {
                    "channel": channel_id,
                    "input_device": recorder.get("input_device")
                    if isinstance(recorder, dict)
                    else None,
                    "output_device": tts.get("output_device")
                    if isinstance(tts, dict)
                    else None,
                }
            )
    return config


def _render(report: dict[str, Any]) -> str:
    lines = ["\n  Hey Robot 设备列表", "  ==================", ""]
    config = report.get("config")
    if config:
        lines.append(f"  配置文件: {config.get('path')}")
        serial_lines = [
            f"    配置串口: {item['robot']} -> {item['port']}"
            for item in config.get("serial_ports", [])
        ]
        camera_lines = [
            "    配置摄像头: "
            f"{item['robot']}:{item['role']} -> "
            f"device_id={item['device_id']} backend={item['backend']}"
            for item in config.get("cameras", [])
        ]
        lines.extend(serial_lines)
        lines.extend(camera_lines)
        lines.append("")

    lines.append(f"  串口 ({len(report['serial'])})")
    if report["serial"]:
        for item in report["serial"]:
            vidpid = (
                f" VID/PID={item['vid']}:{item['pid']}"
                if item.get("vid") and item.get("pid")
                else ""
            )
            lines.append(f"    {item['port']:<12} {item['description']}{vidpid}")
    else:
        lines.append("    未检测到")

    lines.append("")
    lines.append(f"  摄像头 ({len(report['cameras'])})")
    if report["cameras"]:
        for item in report["cameras"]:
            size = (
                f"{item['width']}x{item['height']}"
                if item.get("width") and item.get("height")
                else "unknown"
            )
            frame = ""
            if item.get("frame_ok") is not None:
                frame = f" 取帧={'成功' if item['frame_ok'] else '失败'}"
            lines.append(
                f"    设备 {item['device_id']:<2} {size:<12} "
                f"backend={item['backend']}{frame}"
            )
    else:
        lines.append("    未检测到")

    inputs = [d for d in report["audio"] if d["max_input_channels"] > 0]
    outputs = [d for d in report["audio"] if d["max_output_channels"] > 0]
    lines.append("")
    lines.append(f"  麦克风 ({len(inputs)})")
    for item in inputs:
        mark = " 默认" if item["is_default_input"] else ""
        lines.append(
            f"    [{item['index']}] {item['name']} "
            f"{item['max_input_channels']}ch {item['default_samplerate']:.0f}Hz{mark}"
        )
    if not inputs:
        lines.append("    未检测到")

    lines.append("")
    lines.append(f"  扬声器/音频输出 ({len(outputs)})")
    for item in outputs:
        mark = " 默认" if item["is_default_output"] else ""
        lines.append(
            f"    [{item['index']}] {item['name']} "
            f"{item['max_output_channels']}ch {item['default_samplerate']:.0f}Hz{mark}"
        )
    if not outputs:
        lines.append("    未检测到")
    return "\n".join(lines) + "\n"


def _opencv_backend_id(cv2: Any, backend: str) -> int:
    normalized = backend.lower()
    if normalized == "auto":
        normalized = _default_backend()
    return {
        "dshow": getattr(cv2, "CAP_DSHOW", 0),
        "msmf": getattr(cv2, "CAP_MSMF", 0),
        "v4l2": getattr(cv2, "CAP_V4L2", 0),
    }.get(normalized, 0)


def _default_backend() -> str:
    return "dshow" if sys.platform.startswith("win") else "v4l2"


def _configure_stdout() -> None:
    with contextlib.suppress(AttributeError, OSError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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
