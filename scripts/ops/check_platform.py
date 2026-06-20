"""检查运行 hey-robot 所需的平台环境是否就绪。

用途：
  - 第一次部署或换机器时，先跑这个工具确认 Python 版本、NATS 服务、可选依赖等都到位。
  - 可选参数 --config 顺带校验部署配置文件里的串口、机器人类型等是否符合当前平台。
  - 启动 runtime 之前推荐先跑一次，环境异常时直接给出修复提示。

常见用法：
  # 最简单：只检查平台基础项（Python、NATS）
  uv run python scripts/ops/check_platform.py

  # 顺带校验部署配置文件
  uv run python scripts/ops/check_platform.py --config configs/xlerobot.real.windows.yaml

  # 机器可读 JSON 输出
  uv run python scripts/ops/check_platform.py --json

输出说明：
  - 平台信息：操作系统 / 版本 / Python 版本
  - 基础检查项：Python 版本 / NATS 服务 / 可选依赖
  - 配置检查项（如果传了 --config）：串口有效性 / 机器人类型匹配
  - 推荐配置文件列表

退出码：
  - 0：所有必要项就绪
  - 2：有检查项失败（看输出找原因）

更多选项：uv run python scripts/ops/check_platform.py --help
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hey_robot.config.model import DeploymentConfig

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查运行 hey-robot 所需的 Python、NATS、依赖等平台环境。"
    )
    parser.add_argument("--config", default=None, help="顺带校验的部署配置 YAML 路径")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args()

    report = build_report(config_path=args.config)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
    raise SystemExit(0 if report["ready"] else 2)


def build_report(*, config_path: str | None) -> dict[str, Any]:
    config_report = inspect_config(config_path) if config_path else None
    enabled_channels = (
        set(config_report.get("enabled_channels", [])) if config_report else set()
    )
    enabled_robot_types = (
        set(config_report.get("enabled_robot_types", [])) if config_report else set()
    )
    checks: list[CheckResult] = [check_python(), check_nats_server()]
    checks.extend(
        build_import_checks(
            enabled_channels=enabled_channels,
            enabled_robot_types=enabled_robot_types,
        )
    )
    ready = all(item.ok for item in checks)
    if config_report is not None:
        ready = ready and config_report["ready"]
    return {
        "ready": ready,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version.split()[0],
        },
        "checks": [asdict(item) for item in checks],
        "config": config_report,
    }


def check_python() -> CheckResult:
    ok = sys.version_info[:2] == (3, 12)
    detail = f"python={sys.version.split()[0]} required=3.12.x"
    return CheckResult(name="python", ok=ok, detail=detail)


def check_nats_server() -> CheckResult:
    resolved = shutil.which("nats-server")
    return CheckResult(
        name="nats_server",
        ok=resolved is not None,
        detail=resolved or "nats-server not found on PATH",
    )


def check_optional_import(module_name: str) -> CheckResult:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return CheckResult(
            name=f"import:{module_name}",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
        )
    return CheckResult(name=f"import:{module_name}", ok=True, detail="available")


def build_import_checks(
    *, enabled_channels: set[str], enabled_robot_types: set[str]
) -> list[CheckResult]:
    modules = set[str]()
    if not enabled_channels and not enabled_robot_types:
        modules.update({"cv2", "serial"})
    if enabled_robot_types:
        modules.update({"cv2", "serial"})
    if "web" in enabled_channels:
        modules.update({"fastapi", "uvicorn"})
    if "voice" in enabled_channels:
        modules.add("sounddevice")
    return [check_optional_import(module_name) for module_name in sorted(modules)]


def inspect_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return {
            "ready": False,
            "path": str(path),
            "checks": [
                asdict(
                    CheckResult(name="config_path", ok=False, detail="file not found")
                )
            ],
        }

    config = DeploymentConfig.from_yaml(path)
    checks: list[CheckResult] = [
        CheckResult(name="config_path", ok=True, detail=str(path))
    ]
    checks.extend(check_robot_platform_constraints(config))
    return {
        "ready": all(item.ok for item in checks),
        "path": str(path),
        "deployment_id": config.deployment.id,
        "enabled_channels": [
            channel_id
            for channel_id, channel in config.channels.items()
            if channel.enabled
        ],
        "enabled_robot_types": [
            robot.type for robot in config.robots.values() if robot.enabled
        ],
        "checks": [asdict(item) for item in checks],
    }


def check_robot_platform_constraints(config: DeploymentConfig) -> list[CheckResult]:
    results: list[CheckResult] = []
    is_windows = platform.system().lower().startswith("win")
    expected_backends = {"dshow", "msmf", "auto"} if is_windows else {"v4l2", "auto"}

    for robot_id, robot in config.robots.items():
        settings = robot.settings
        serial_bus = settings.get("serial_bus")
        if isinstance(serial_bus, dict):
            port = str(serial_bus.get("port", "")).strip()
            results.append(
                CheckResult(
                    name=f"{robot_id}:serial_port",
                    ok=is_valid_serial_port(port, is_windows=is_windows),
                    detail=port or "missing serial_bus.port",
                )
            )

        components = settings.get("components")
        if not isinstance(components, dict):
            continue
        camera = components.get("camera")
        if isinstance(camera, dict) and bool(camera.get("enabled", True)):
            backend = str(camera.get("backend", "auto")).strip().lower()
            device_id = camera.get("device_id")
            results.append(
                CheckResult(
                    name=f"{robot_id}:camera_backend",
                    ok=backend in expected_backends,
                    detail=f"backend={backend} expected={sorted(expected_backends)}",
                )
            )
            results.append(
                CheckResult(
                    name=f"{robot_id}:camera_device",
                    ok=isinstance(device_id, int) and device_id >= 0,
                    detail=f"device_id={device_id}",
                )
            )
    return results


def is_valid_serial_port(port: str, *, is_windows: bool) -> bool:
    if not port:
        return False
    if is_windows:
        normalized = port.upper()
        return normalized.startswith("COM") and normalized[3:].isdigit()
    return port.startswith("/dev/")


def format_report(report: dict[str, Any]) -> str:
    ready_status = "就绪" if report["ready"] else "异常"
    lines = [
        f"hey-robot 平台环境检查：整体={ready_status}",
        (
            "平台信息："
            f"{report['platform']['system']} {report['platform']['release']} "
            f"python={report['platform']['python']}"
        ),
        "",
        "基础检查项：",
    ]
    for item in report["checks"]:
        status = "正常" if item["ok"] else "失败"
        lines.append(f"  - {item['name']}：{status}  {item['detail']}")

    config = report.get("config")
    if config:
        lines.extend(["", f"配置检查：{config['path']}"])
        for item in config["checks"]:
            status = "正常" if item["ok"] else "失败"
            lines.append(f"  - {item['name']}：{status}  {item['detail']}")

    lines.extend(
        [
            "",
            "推荐配置文件：",
            "  - Windows 真机运行：configs/xlerobot.real.windows.yaml",
            "  - 本地开发（mock）：configs/mock.dev.yaml",
            "  - 测试（mock）：configs/mock.test.yaml",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
